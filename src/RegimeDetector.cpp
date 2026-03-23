#include "RegimeDetector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>
#include <iomanip>

namespace llmquant {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr double kMinStd  = 1e-6;
static constexpr double kMinProb = 1e-300;
static constexpr double kTwoPi   = 6.283185307179586;

double RegimeDetectorHMM::gauss_pdf(double x, double mu, double sigma) noexcept {
    double s = (sigma < kMinStd) ? kMinStd : sigma;
    double z = (x - mu) / s;
    return std::exp(-0.5 * z * z) / (s * std::sqrt(kTwoPi));
}

double RegimeDetectorHMM::normalise(std::array<double, kStates>& v) noexcept {
    double sum = v[0] + v[1] + v[2];
    if (sum < kMinProb) {
        v[0] = v[1] = v[2] = 1.0 / kStates;
        return 1.0 / kStates;
    }
    v[0] /= sum; v[1] /= sum; v[2] /= sum;
    return sum;
}

double RegimeDetectorHMM::kl_div(const std::array<double, kStates>& p,
                                  const std::array<double, kStates>& q) noexcept {
    double kl = 0.0;
    for (int s = 0; s < kStates; ++s) {
        double pi = (p[s] < kMinProb) ? kMinProb : p[s];
        double qi = (q[s] < kMinProb) ? kMinProb : q[s];
        kl += pi * std::log(pi / qi);
    }
    return kl;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RegimeDetectorHMM::RegimeDetectorHMM(Config cfg)
    : cfg_(cfg)
{
    emu_mean_ = cfg_.state_mean;
    emu_std_  = cfg_.state_std;
    trans_    = cfg_.transition;

    // Row-normalise the transition matrix to handle user-supplied configs.
    for (int i = 0; i < kStates; ++i) {
        double row_sum = trans_[i][0] + trans_[i][1] + trans_[i][2];
        if (row_sum > kMinProb) {
            trans_[i][0] /= row_sum;
            trans_[i][1] /= row_sum;
            trans_[i][2] /= row_sum;
        }
    }

    alpha_      = cfg_.init_prob;
    prev_gamma_ = cfg_.init_prob;
    normalise(alpha_);
}

// ---------------------------------------------------------------------------
// Online emission parameter update (called under mutex)
// ---------------------------------------------------------------------------

void RegimeDetectorHMM::update_emission(int s, double bias) {
    double lr  = cfg_.emission_learning_rate;
    double old_mu  = emu_mean_[s];
    double old_std = emu_std_[s];

    // Online mean update.
    double new_mu  = (1.0 - lr) * old_mu + lr * bias;
    // Online variance update (Welford-style EMA of squared deviation).
    double dev     = bias - old_mu;
    double new_var = (1.0 - lr) * (old_std * old_std) + lr * (dev * dev);
    double new_std = std::sqrt(new_var);

    emu_mean_[s] = new_mu;
    emu_std_[s]  = (new_std < kMinStd) ? kMinStd : new_std;
}

// ---------------------------------------------------------------------------
// Online transition matrix update (called under mutex)
// ---------------------------------------------------------------------------

void RegimeDetectorHMM::update_transition(
    const std::array<double, kStates>& prev_gamma,
    const std::array<double, kStates>& gamma)
{
    double lr = cfg_.transition_learning_rate;
    // Soft increment A[from][to] proportional to prev_gamma[from] * gamma[to].
    for (int from = 0; from < kStates; ++from) {
        double row_sum = 0.0;
        for (int to = 0; to < kStates; ++to) {
            double increment = lr * prev_gamma[from] * gamma[to];
            trans_[from][to] = (1.0 - lr) * trans_[from][to] + increment;
            row_sum += trans_[from][to];
        }
        // Row-normalise.
        if (row_sum > kMinProb) {
            for (int to = 0; to < kStates; ++to) {
                trans_[from][to] /= row_sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Regime classification (called under mutex)
// ---------------------------------------------------------------------------

TokenRegime RegimeDetectorHMM::classify(
    const std::array<double, kStates>& gamma,
    const std::array<double, kStates>& prev_gm) const
{
    double p_bear = gamma[0];
    double p_neu  = gamma[1];
    double p_bull = gamma[2];
    double max_p  = std::max({p_bear, p_neu, p_bull});

    // HIGH_UNCERTAINTY: no state has sufficient posterior mass.
    if (max_p < cfg_.uncertainty_threshold) {
        return TokenRegime::HIGH_UNCERTAINTY;
    }

    // BREAKOUT: KL divergence between current and previous posterior is large.
    double kl = kl_div(gamma, prev_gm);
    if (kl > cfg_.breakout_kl_threshold) {
        return TokenRegime::BREAKOUT;
    }

    // BULL_TRENDING: bullish state clearly dominant.
    if (p_bull >= cfg_.trend_threshold && p_bull > p_bear && p_bull > p_neu) {
        return TokenRegime::BULL_TRENDING;
    }

    // BEAR_TRENDING: bearish state clearly dominant.
    if (p_bear >= cfg_.trend_threshold && p_bear > p_bull && p_bear > p_neu) {
        return TokenRegime::BEAR_TRENDING;
    }

    // CONSOLIDATION: neutral state dominant or ambiguous.
    return TokenRegime::CONSOLIDATION;
}

// ---------------------------------------------------------------------------
// Hot-path update
// ---------------------------------------------------------------------------

void RegimeDetectorHMM::update(double token_bias, int64_t timestamp) {
    total_.fetch_add(1, std::memory_order_relaxed);

    // Copy regime change callback without holding mutex during call.
    TokenRegime  new_regime;
    TokenRegime  old_regime = current_regime();
    double       new_conf;
    double       new_bull, new_bear, new_neu;
    std::function<void(TokenRegime, TokenRegime, int64_t)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // ---- Forward prediction step ----------------------------------------
        // alpha_pred[j] = sum_i  alpha[i] * A[i][j]
        std::array<double, kStates> alpha_pred{};
        for (int j = 0; j < kStates; ++j) {
            alpha_pred[j] = 0.0;
            for (int i = 0; i < kStates; ++i) {
                alpha_pred[j] += alpha_[i] * trans_[i][j];
            }
        }

        // ---- Emission update step -------------------------------------------
        // alpha[j] = alpha_pred[j] * P(bias | state=j)
        for (int j = 0; j < kStates; ++j) {
            double p_emit = gauss_pdf(token_bias, emu_mean_[j], emu_std_[j]);
            // Guard against numeric underflow.
            alpha_pred[j] *= (p_emit < kMinProb) ? kMinProb : p_emit;
        }

        // ---- Normalise → posterior γ ----------------------------------------
        std::array<double, kStates> gamma = alpha_pred;
        normalise(gamma);

        // ---- Online parameter updates ---------------------------------------
        uint64_t n = total_.load(std::memory_order_relaxed);
        if (n > static_cast<uint64_t>(cfg_.min_samples)) {
            // Update emission params for each state weighted by posterior.
            for (int s = 0; s < kStates; ++s) {
                // Only update if state has meaningful posterior mass.
                if (gamma[s] > 0.1) {
                    update_emission(s, token_bias);
                }
            }
            update_transition(prev_gamma_, gamma);
        }

        // ---- Regime classification ------------------------------------------
        int min_s = cfg_.min_samples;
        TokenRegime regime_raw;
        if (static_cast<int>(n) < min_s) {
            regime_raw = TokenRegime::CONSOLIDATION;
        } else {
            regime_raw = classify(gamma, prev_gamma_);
        }

        // ---- Confidence: max posterior probability --------------------------
        double conf = std::max({gamma[0], gamma[1], gamma[2]});

        // ---- Store updated state -------------------------------------------
        alpha_      = gamma;
        prev_gamma_ = gamma;
        last_ts_    = timestamp;

        new_regime = regime_raw;
        new_conf   = conf;
        new_bull   = gamma[2];
        new_bear   = gamma[0];
        new_neu    = gamma[1];

        if (new_regime != old_regime) {
            transitions_.fetch_add(1, std::memory_order_relaxed);
            cb = regime_cb_;
        }
    }

    // Publish atomics (outside lock).
    regime_.store(static_cast<int>(new_regime), std::memory_order_release);
    confidence_.store(new_conf,  std::memory_order_relaxed);
    p_bull_.store(new_bull,      std::memory_order_relaxed);
    p_bear_.store(new_bear,      std::memory_order_relaxed);
    p_neutral_.store(new_neu,    std::memory_order_relaxed);

    if (cb) {
        spdlog::debug("[RegimeDetectorHMM] {} → {} (conf={:.3f} bull={:.3f} bear={:.3f})",
                      regime_name(old_regime), regime_name(new_regime),
                      new_conf, new_bull, new_bear);
        try { cb(new_regime, old_regime, timestamp); } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// Static utilities
// ---------------------------------------------------------------------------

const char* RegimeDetectorHMM::regime_name(TokenRegime r) noexcept {
    switch (r) {
        case TokenRegime::BULL_TRENDING:    return "BULL_TRENDING";
        case TokenRegime::BEAR_TRENDING:    return "BEAR_TRENDING";
        case TokenRegime::HIGH_UNCERTAINTY: return "HIGH_UNCERTAINTY";
        case TokenRegime::CONSOLIDATION:    return "CONSOLIDATION";
        case TokenRegime::BREAKOUT:         return "BREAKOUT";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Configuration and reset
// ---------------------------------------------------------------------------

void RegimeDetectorHMM::set_regime_change_callback(
    std::function<void(TokenRegime, TokenRegime, int64_t)> cb)
{
    std::lock_guard<std::mutex> lk(mutex_);
    regime_cb_ = std::move(cb);
}

void RegimeDetectorHMM::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    // Re-seed emission and transition from new config (preserves alpha).
    emu_mean_ = cfg_.state_mean;
    emu_std_  = cfg_.state_std;
    trans_    = cfg_.transition;
    for (int i = 0; i < kStates; ++i) {
        double s = trans_[i][0] + trans_[i][1] + trans_[i][2];
        if (s > kMinProb) {
            trans_[i][0] /= s;
            trans_[i][1] /= s;
            trans_[i][2] /= s;
        }
    }
}

RegimeDetectorHMM::Config RegimeDetectorHMM::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return cfg_;
}

void RegimeDetectorHMM::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    emu_mean_ = cfg_.state_mean;
    emu_std_  = cfg_.state_std;
    trans_    = cfg_.transition;
    for (int i = 0; i < kStates; ++i) {
        double s = trans_[i][0] + trans_[i][1] + trans_[i][2];
        if (s > kMinProb) {
            trans_[i][0] /= s; trans_[i][1] /= s; trans_[i][2] /= s;
        }
    }
    alpha_      = cfg_.init_prob;
    prev_gamma_ = cfg_.init_prob;
    normalise(alpha_);
    last_ts_ = 0;

    regime_.store(static_cast<int>(TokenRegime::CONSOLIDATION), std::memory_order_release);
    confidence_.store(0.0,   std::memory_order_relaxed);
    p_bull_.store(0.333,     std::memory_order_relaxed);
    p_bear_.store(0.333,     std::memory_order_relaxed);
    p_neutral_.store(0.334,  std::memory_order_relaxed);
    total_.store(0,          std::memory_order_relaxed);
    transitions_.store(0,    std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// JSON diagnostics
// ---------------------------------------------------------------------------

std::string RegimeDetectorHMM::to_stats_json() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    o << "{"
      << "\"regime\":\""      << regime_name(current_regime()) << "\""
      << ",\"confidence\":"   << regime_confidence()
      << ",\"p_bullish\":"    << p_bullish()
      << ",\"p_bearish\":"    << p_bearish()
      << ",\"p_neutral\":"    << p_neutral()
      << ",\"total_updates\":" << total_updates()
      << ",\"transitions\":"  << total_transitions()
      << "}";
    return o.str();
}

// ===========================================================================
// RegimeFilter
// ===========================================================================

RegimeFilter::RegimeFilter(RegimeDetectorHMM::Config detector_cfg,
                           Config                    filter_cfg)
    : owned_detector_(std::move(detector_cfg))
    , detector_ptr_(&owned_detector_)
    , cfg_(std::move(filter_cfg))
{}

RegimeFilter::RegimeFilter(RegimeDetectorHMM& detector,
                           Config             filter_cfg)
    : owned_detector_()           // default-constructed but unused
    , detector_ptr_(&detector)
    , cfg_(std::move(filter_cfg))
{}

void RegimeFilter::feed(double token_bias, int64_t timestamp) {
    detector_ptr_->update(token_bias, timestamp);
}

bool RegimeFilter::evaluate(int signal_direction, double /*signal_bias*/) const noexcept {
    evaluated_.fetch_add(1, std::memory_order_relaxed);

    TokenRegime r    = detector_ptr_->current_regime();
    double      conf = detector_ptr_->regime_confidence();

    Config local_cfg;
    {
        std::lock_guard<std::mutex> lk(cfg_mutex_);
        local_cfg = cfg_;
    }

    // Confidence gate.
    if (conf < local_cfg.min_confidence) {
        blocked_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // HIGH_UNCERTAINTY: always block (unless flag is off).
    if (local_cfg.block_high_uncertainty && r == TokenRegime::HIGH_UNCERTAINTY) {
        blocked_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // BREAKOUT: optional bypass (pass any signal direction).
    if (local_cfg.allow_breakout_bypass && r == TokenRegime::BREAKOUT) {
        passed_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool pass = false;
    switch (signal_direction) {
        case 1:   // buy
            pass = (r == TokenRegime::BULL_TRENDING) ||
                   (r == TokenRegime::BREAKOUT &&
                    detector_ptr_->p_bullish() >= detector_ptr_->p_bearish());
            break;
        case -1:  // sell
            pass = (r == TokenRegime::BEAR_TRENDING) ||
                   (r == TokenRegime::BREAKOUT &&
                    detector_ptr_->p_bearish() > detector_ptr_->p_bullish());
            break;
        case 0:   // neutral
            pass = (r == TokenRegime::CONSOLIDATION ||
                    r == TokenRegime::HIGH_UNCERTAINTY);
            break;
        default:
            pass = false;
    }

    if (pass) {
        passed_.fetch_add(1, std::memory_order_relaxed);
    } else {
        blocked_.fetch_add(1, std::memory_order_relaxed);
    }
    return pass;
}

double RegimeFilter::pass_rate() const noexcept {
    uint64_t ev = evaluated_.load(std::memory_order_relaxed);
    if (ev == 0) return 0.0;
    return static_cast<double>(passed_.load(std::memory_order_relaxed)) /
           static_cast<double>(ev);
}

void RegimeFilter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(cfg_mutex_);
    cfg_ = cfg;
}

std::string RegimeFilter::to_stats_json() const {
    uint64_t ev = evaluated_.load(std::memory_order_relaxed);
    uint64_t ps = passed_.load(std::memory_order_relaxed);
    uint64_t bl = blocked_.load(std::memory_order_relaxed);
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    o << "{"
      << "\"evaluated\":"  << ev
      << ",\"passed\":"    << ps
      << ",\"blocked\":"   << bl
      << ",\"pass_rate\":" << pass_rate()
      << ",\"detector\":"  << detector_ptr_->to_stats_json()
      << "}";
    return o.str();
}

} // namespace llmquant
