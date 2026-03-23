/// @file src/ensemble.cpp
/// @brief Implementation of EnsembleVoter and SignalEnsemble.

#include "ensemble.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// @brief Bit-cast a double to uint64_t (C++20).
[[nodiscard]] uint64_t double_to_u64(double v) noexcept {
    return std::bit_cast<uint64_t>(v);
}

/// @brief Bit-cast a uint64_t back to double.
[[nodiscard]] double u64_to_double(uint64_t u) noexcept {
    return std::bit_cast<double>(u);
}

constexpr uint64_t kNoVote = 0xFF'FF'FF'FF'FF'FF'FF'FFull; ///< Sentinel "no vote".

} // anonymous namespace

// ---------------------------------------------------------------------------
// EnsembleVoter
// ---------------------------------------------------------------------------

EnsembleVoter::EnsembleVoter(std::size_t n_engines, FusionMode mode)
    : n_engines_(n_engines)
{
    if (n_engines_ > kMaxEngines) {
        throw std::invalid_argument(
            "EnsembleVoter: n_engines exceeds kMaxEngines=" +
            std::to_string(kMaxEngines));
    }
    mode_.store(static_cast<uint8_t>(mode), std::memory_order_relaxed);
    // Initialise all vote slots to "no vote" sentinel.
    for (std::size_t i = 0; i < kMaxEngines; ++i) {
        pending_bias_[i].store(kNoVote, std::memory_order_relaxed);
        pending_vol_[i].store(kNoVote,  std::memory_order_relaxed);
        pending_conf_[i].store(kNoVote, std::memory_order_relaxed);
        vote_present_[i].store(0,       std::memory_order_relaxed);
    }
}

void EnsembleVoter::vote(uint32_t engine_id, const TradeSignal& signal) noexcept {
    if (engine_id >= n_engines_) return;

    pending_bias_[engine_id].store(
        double_to_u64(signal.delta_bias_shift),  std::memory_order_relaxed);
    pending_vol_[engine_id].store(
        double_to_u64(signal.volatility_adjustment), std::memory_order_relaxed);
    pending_conf_[engine_id].store(
        double_to_u64(signal.confidence),        std::memory_order_relaxed);

    // Mark vote present with release to ensure values are visible before flag.
    vote_present_[engine_id].store(1, std::memory_order_release);
    vote_count_.fetch_add(1, std::memory_order_relaxed);
}

std::optional<EnsembleVoter::FusedSignal>
EnsembleVoter::fuse(const std::vector<EnsembleWeight>& weights) noexcept {
    std::vector<double> biases, vols, confs;
    biases.reserve(n_engines_);
    vols.reserve(n_engines_);
    confs.reserve(n_engines_);

    // Collect votes.  Acquire fence ensures values written before vote_present
    // store are visible.
    for (std::size_t i = 0; i < n_engines_; ++i) {
        uint8_t present = vote_present_[i].load(std::memory_order_acquire);
        if (!present) continue;

        double b = u64_to_double(pending_bias_[i].load(std::memory_order_relaxed));
        double v = u64_to_double(pending_vol_[i].load(std::memory_order_relaxed));
        double c = u64_to_double(pending_conf_[i].load(std::memory_order_relaxed));

        // Clear the slot.
        vote_present_[i].store(0, std::memory_order_relaxed);

        // Guard against NaN/Inf (can happen with bit-cast sentinel if not cleared).
        if (!std::isfinite(b) || !std::isfinite(v) || !std::isfinite(c)) continue;

        biases.push_back(b);
        vols.push_back(v);
        confs.push_back(c);
    }

    if (biases.empty()) return std::nullopt;

    // Normalise weights for the votes we received.
    // Pad the weights vector if needed (use 1.0 for missing).
    std::vector<EnsembleWeight> active_weights;
    active_weights.reserve(biases.size());
    for (std::size_t i = 0; i < biases.size() && i < weights.size(); ++i) {
        active_weights.push_back(weights[i]);
    }
    while (active_weights.size() < biases.size()) {
        EnsembleWeight ew;
        ew.weight = 1.0;
        ew.normalised_weight = 1.0;
        active_weights.push_back(ew);
    }

    // Compute normalised weights.
    double w_sum = 0;
    for (auto& ew : active_weights) w_sum += std::max(0.0, ew.weight);
    if (w_sum < 1e-12) {
        for (auto& ew : active_weights) ew.normalised_weight = 1.0 / active_weights.size();
    } else {
        for (auto& ew : active_weights)
            ew.normalised_weight = std::max(0.0, ew.weight) / w_sum;
    }

    fuse_count_.fetch_add(1, std::memory_order_relaxed);

    const auto m = static_cast<FusionMode>(mode_.load(std::memory_order_relaxed));
    switch (m) {
        case FusionMode::MajorityVote:
            return fuse_majority(active_weights, biases, vols, confs);
        case FusionMode::KalmanFusion:
            return fuse_kalman(active_weights, biases, vols, confs);
        default:
            return fuse_weighted_average(active_weights, biases, vols, confs);
    }
}

EnsembleVoter::FusedSignal
EnsembleVoter::fuse_weighted_average(const std::vector<EnsembleWeight>& weights,
                                      const std::vector<double>& biases,
                                      const std::vector<double>& vols,
                                      const std::vector<double>& confs) const noexcept
{
    FusedSignal out;
    out.mode = FusionMode::WeightedAverage;
    out.engine_count = static_cast<uint32_t>(biases.size());

    double sum_bias = 0, sum_vol = 0, sum_conf = 0;
    for (std::size_t i = 0; i < biases.size(); ++i) {
        const double w = weights[i].normalised_weight;
        sum_bias += w * biases[i];
        sum_vol  += w * vols[i];
        sum_conf += w * confs[i];
    }
    out.bias       = sum_bias;
    out.volatility = sum_vol;
    out.confidence = sum_conf;

    // Agreement: fraction of engines on the same side as consensus.
    if (biases.size() == 1) {
        out.agreement = 1.0;
    } else {
        const double sign_c = (out.bias > 0) ? 1.0 : (out.bias < 0) ? -1.0 : 0.0;
        uint32_t same = 0;
        for (double b : biases) {
            const double sign_b = (b > 0) ? 1.0 : (b < 0) ? -1.0 : 0.0;
            if (sign_b == sign_c) ++same;
        }
        out.agreement = static_cast<double>(same) / static_cast<double>(biases.size());
    }
    return out;
}

EnsembleVoter::FusedSignal
EnsembleVoter::fuse_majority(const std::vector<EnsembleWeight>& weights,
                              const std::vector<double>& biases,
                              const std::vector<double>& vols,
                              const std::vector<double>& confs) const noexcept
{
    FusedSignal out;
    out.mode = FusionMode::MajorityVote;
    out.engine_count = static_cast<uint32_t>(biases.size());

    double vote_pos = 0, vote_neg = 0;
    double sum_vol = 0, sum_conf = 0;

    for (std::size_t i = 0; i < biases.size(); ++i) {
        const double w = weights[i].normalised_weight;
        if (biases[i] > 0) vote_pos += w;
        else if (biases[i] < 0) vote_neg += w;
        sum_vol  += w * vols[i];
        sum_conf += w * confs[i];
    }

    out.bias       = (vote_pos > vote_neg) ?  (vote_pos - vote_neg)
                   : (vote_neg > vote_pos) ? -(vote_neg - vote_pos)
                   : 0.0;
    out.volatility = sum_vol;
    out.confidence = sum_conf;
    out.agreement  = std::max(vote_pos, vote_neg);  // fraction of weight on winning side

    return out;
}

EnsembleVoter::FusedSignal
EnsembleVoter::fuse_kalman(const std::vector<EnsembleWeight>& weights,
                            const std::vector<double>& biases,
                            const std::vector<double>& vols,
                            const std::vector<double>& confs) noexcept
{
    FusedSignal out;
    out.mode = FusionMode::KalmanFusion;
    out.engine_count = static_cast<uint32_t>(biases.size());

    // Process each engine as a measurement with noise ~ 1 - recent_accuracy.
    // Measurement noise floor: 0.01 (avoids division by zero).
    double x = kalman_x_;
    double p = kalman_p_;

    for (std::size_t i = 0; i < biases.size(); ++i) {
        // Measurement noise: inverse of accuracy (more accurate = less noise).
        const double acc = std::clamp(weights[i].recent_accuracy, 0.01, 0.99);
        const double r = 1.0 - acc + 0.01; // measurement noise variance
        // Kalman gain.
        const double k = p / (p + r);
        // Update.
        x = x + k * (biases[i] - x);
        p = (1.0 - k) * p;
    }

    // Process noise (small drift).
    p += 0.001;

    kalman_x_ = x;
    kalman_p_ = p;

    out.bias = x;

    // Volatility and confidence: weighted average.
    double sum_vol = 0, sum_conf = 0;
    for (std::size_t i = 0; i < biases.size(); ++i) {
        const double w = weights[i].normalised_weight;
        sum_vol  += w * vols[i];
        sum_conf += w * confs[i];
    }
    out.volatility = sum_vol;
    out.confidence = sum_conf * (1.0 - std::clamp(p, 0.0, 1.0)); // reduce by uncertainty

    // Agreement: proportion of engines within 0.1 of the Kalman estimate.
    uint32_t near = 0;
    for (double b : biases) {
        if (std::abs(b - x) < 0.1) ++near;
    }
    out.agreement = static_cast<double>(near) / static_cast<double>(biases.size());

    return out;
}

void EnsembleVoter::record_outcome(uint32_t                     engine_id,
                                    double                       signal_bias,
                                    double                       realised_pnl,
                                    std::vector<EnsembleWeight>& weights,
                                    double                       alpha) noexcept
{
    if (engine_id >= weights.size()) return;

    // Correct = signal direction matches PnL direction.
    const bool correct =
        (signal_bias > 0 && realised_pnl > 0) ||
        (signal_bias < 0 && realised_pnl < 0);

    auto& ew = weights[engine_id];
    ew.recent_accuracy = alpha * (correct ? 1.0 : 0.0) + (1.0 - alpha) * ew.recent_accuracy;
    ew.signals_evaluated++;

    // Reweight proportional to accuracy (rescaled from [0,1] to [0.1, 2.0]).
    ew.weight = 0.1 + 1.9 * ew.recent_accuracy;
}

void EnsembleVoter::set_mode(FusionMode mode) noexcept {
    mode_.store(static_cast<uint8_t>(mode), std::memory_order_relaxed);
}

EnsembleVoter::FusionMode EnsembleVoter::mode() const noexcept {
    return static_cast<FusionMode>(mode_.load(std::memory_order_relaxed));
}

std::size_t EnsembleVoter::n_engines() const noexcept {
    return n_engines_;
}

uint64_t EnsembleVoter::fuse_count() const noexcept {
    return fuse_count_.load(std::memory_order_relaxed);
}

uint64_t EnsembleVoter::vote_count() const noexcept {
    return vote_count_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// SignalEnsemble
// ---------------------------------------------------------------------------

SignalEnsemble::Config SignalEnsemble::make_default_config() {
    Config cfg;
    cfg.fusion_mode = EnsembleVoter::FusionMode::WeightedAverage;
    cfg.weight_update_alpha = 0.1;

    // Fast timescale: decays quickly, captures flash events.
    EngineSpec fast;
    fast.name                = "fast";
    fast.decay_rate          = 0.80;
    fast.half_life_ms        = 100.0;
    fast.bias_sensitivity    = 1.2;
    fast.volatility_sensitivity = 1.0;
    fast.initial_weight      = 1.0;

    // Medium timescale: standard decay, captures sustained narratives.
    EngineSpec medium;
    medium.name              = "medium";
    medium.decay_rate        = 0.95;
    medium.half_life_ms      = 1000.0;
    medium.bias_sensitivity  = 1.0;
    medium.volatility_sensitivity = 1.0;
    medium.initial_weight    = 1.0;

    // Slow timescale: high persistence, captures macro trends.
    EngineSpec slow;
    slow.name                = "slow";
    slow.decay_rate          = 0.99;
    slow.half_life_ms        = 5000.0;
    slow.bias_sensitivity    = 0.8;
    slow.volatility_sensitivity = 0.9;
    slow.initial_weight      = 1.0;

    cfg.engines = {fast, medium, slow};
    return cfg;
}

SignalEnsemble::SignalEnsemble()
    : SignalEnsemble(make_default_config()) {}

SignalEnsemble::SignalEnsemble(Config cfg)
    : cfg_(std::move(cfg))
    , voter_(cfg_.engines.empty() ? 3 : cfg_.engines.size(),
             cfg_.fusion_mode)
{
    // If caller passed no engine specs, substitute the default 3.
    if (cfg_.engines.empty()) cfg_ = make_default_config();
    init_engines();
}

void SignalEnsemble::init_engines() {
    const std::size_t n = cfg_.engines.size();
    engines_.clear();
    engines_.reserve(n);
    weights_.clear();
    weights_.reserve(n);
    last_engine_bias_.assign(n, 0.0);
    engine_last_signals_.assign(n, std::nullopt);

    for (std::size_t i = 0; i < n; ++i) {
        const auto& spec = cfg_.engines[i];

        // Build per-engine config from base + spec overrides.
        TradeSignalEngine::Config ecfg = cfg_.base_engine_config;
        ecfg.signal_decay_rate       = spec.decay_rate;
        ecfg.time_decay_half_life_ms = spec.half_life_ms;
        ecfg.bias_sensitivity        = spec.bias_sensitivity;
        ecfg.volatility_sensitivity  = spec.volatility_sensitivity;

        engines_.emplace_back(ecfg);

        // Register a per-engine callback that captures the engine index.
        // The lambda stores the signal in engine_last_signals_[i].
        const uint32_t eid = static_cast<uint32_t>(i);
        engines_.back().set_signal_callback([this, eid](const TradeSignal& sig) {
            engine_last_signals_[eid] = sig;
        });

        EnsembleWeight ew;
        ew.engine_id        = static_cast<uint32_t>(i);
        ew.weight           = spec.initial_weight;
        ew.recent_accuracy  = 0.5;
        ew.signals_evaluated= 0;
        ew.normalised_weight= 1.0;
        weights_.push_back(ew);
    }
}

void SignalEnsemble::process_semantic_weight(const SemanticWeight& weight) {
    tokens_processed_.fetch_add(1, std::memory_order_relaxed);

    // Clear per-engine signal slots before processing.
    for (auto& slot : engine_last_signals_) slot = std::nullopt;

    // Feed each engine.  Per-engine callbacks (set in init_engines) populate
    // engine_last_signals_ when the engine emits a signal.
    bool any_vote = false;
    for (std::size_t i = 0; i < engines_.size(); ++i) {
        engines_[i].process_semantic_weight(weight);

        if (engine_last_signals_[i].has_value()) {
            const auto& emitted = *engine_last_signals_[i];
            last_engine_bias_[i] = emitted.delta_bias_shift;
            voter_.vote(static_cast<uint32_t>(i), emitted);
            any_vote = true;
        }
    }

    if (!any_vote) return;

    // Fuse votes.
    auto fused = voter_.fuse(weights_);
    if (!fused) return;

    fuse_count_.fetch_add(1, std::memory_order_relaxed);

    // Store atomically.
    auto sig_ptr = std::make_shared<EnsembleVoter::FusedSignal>(*fused);
    {
        std::lock_guard lock(last_signal_mtx_);
        last_signal_ = sig_ptr;
    }

    // Fire callback.
    std::function<void(const EnsembleVoter::FusedSignal&)> cb;
    {
        std::lock_guard lock(cb_mtx_);
        cb = callback_;
    }
    if (cb) cb(*fused);
}

std::optional<EnsembleVoter::FusedSignal>
SignalEnsemble::last_fused_signal() const noexcept {
    std::lock_guard lock(last_signal_mtx_);
    if (!last_signal_) return std::nullopt;
    return *last_signal_;
}

void SignalEnsemble::set_signal_callback(
    std::function<void(const EnsembleVoter::FusedSignal&)> cb)
{
    std::lock_guard lock(cb_mtx_);
    callback_ = std::move(cb);
}

void SignalEnsemble::record_outcome(double realised_pnl) noexcept {
    for (std::size_t i = 0; i < engines_.size(); ++i) {
        voter_.record_outcome(static_cast<uint32_t>(i),
                              last_engine_bias_[i],
                              realised_pnl,
                              weights_,
                              cfg_.weight_update_alpha);
    }
}

std::vector<EnsembleWeight> SignalEnsemble::weights() const {
    return weights_;
}

std::size_t SignalEnsemble::engine_count() const noexcept {
    return engines_.size();
}

const std::string& SignalEnsemble::engine_name(std::size_t i) const {
    return cfg_.engines.at(i).name;
}

const TradeSignalEngine& SignalEnsemble::engine(std::size_t i) const {
    return engines_.at(i);
}

uint64_t SignalEnsemble::tokens_processed() const noexcept {
    return tokens_processed_.load(std::memory_order_relaxed);
}

uint64_t SignalEnsemble::fuse_count() const noexcept {
    return fuse_count_.load(std::memory_order_relaxed);
}

const SignalEnsemble::Config& SignalEnsemble::config() const noexcept {
    return cfg_;
}

void SignalEnsemble::set_fusion_mode(EnsembleVoter::FusionMode mode) noexcept {
    voter_.set_mode(mode);
}

} // namespace llmquant
