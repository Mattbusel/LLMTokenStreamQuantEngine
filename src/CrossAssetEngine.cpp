/**
 * @file CrossAssetEngine.cpp
 * @brief Implementation of the cross-asset signal correlation engine.
 *
 * Design notes
 * ============
 * Pearson r is maintained incrementally using two data structures per symbol:
 *
 *   1. WelfordState  — online mean and M2 (sum of squared deviations) for the
 *                      symbol's own scalar series, used to derive sigma.
 *
 *   2. PairState     — per ordered-pair running sum of products Σ(xi · yi),
 *                      combined with the per-symbol means to recover covariance.
 *
 * When the rolling window is full the oldest sample is evicted. The Welford
 * downdate subtracts the old value's contribution to mean and M2 before the
 * new value is added, keeping accumulators consistent without rescanning.
 *
 * Conviction multiplier
 * =====================
 * For the primary asset p, iterate over all peer assets q:
 *   r = pearson_r(p, q)
 *   if r > +CORR_THRESH and sign(last_q) == sign(last_p): boost += r * BOOST_SCALE
 *   if r > +CORR_THRESH and sign(last_q) != sign(last_p): boost -= r * BOOST_SCALE
 *   if r < -CORR_THRESH (anti-correlated) and signs differ: treat as confirmation.
 * Final multiplier is clamped to [0.5, 2.0].
 */

#include "CrossAssetEngine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------------------
namespace {
    constexpr double kEpsilon      = 1e-12;  ///< Guard against division by near-zero sigma.
    constexpr double kCorrThresh   = 0.25;   ///< Minimum |r| to consider a pair correlated.
    constexpr double kBoostScale   = 0.30;   ///< Conviction contribution per correlated peer.
    constexpr double kMultMin      = 0.50;
    constexpr double kMultMax      = 2.00;
} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

CrossAssetEngine::CrossAssetEngine(size_t window_size)
    : window_size_(window_size)
{
    if (window_size_ < 2)
        throw std::invalid_argument("CrossAssetEngine: window_size must be >= 2");
}

// ---------------------------------------------------------------------------
// update_signal
// ---------------------------------------------------------------------------

void CrossAssetEngine::update_signal(const AssetSignal& signal) {
    if (signal.symbol.empty())
        return;

    std::lock_guard<std::mutex> lock(mutex_);

    const std::string& sym = signal.symbol;
    const double value = signal.weight * signal.confidence;

    // ---- Initialise state for new symbols ----
    if (signal_windows_.find(sym) == signal_windows_.end()) {
        signal_windows_[sym] = {};
        signal_windows_[sym].reserve(window_size_);
        welford_states_[sym] = WelfordState{};
    }

    auto& window = signal_windows_[sym];
    auto& ws     = welford_states_[sym];

    // ---- Evict oldest sample if window is full ----
    if (window.size() >= window_size_) {
        const double old_value = window.front();
        window.erase(window.begin());
        evict_old_sample(sym, old_value);
    }

    // ---- Welford online update for this symbol ----
    ws.count += 1.0;
    const double delta  = value - ws.mean;
    ws.mean            += delta / ws.count;
    const double delta2 = value - ws.mean;
    ws.M2              += delta * delta2;

    // ---- Record the new value ----
    window.push_back(value);

    // ---- Update pair accumulators against all other tracked symbols ----
    update_pair_states(sym, value);

    ++total_samples_;
    if (signal.timestamp_ns > last_updated_ns_)
        last_updated_ns_ = signal.timestamp_ns;
}

// ---------------------------------------------------------------------------
// clear_symbol
// ---------------------------------------------------------------------------

void CrossAssetEngine::clear_symbol(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);

    signal_windows_.erase(symbol);
    welford_states_.erase(symbol);

    // Remove all pair entries that involve this symbol.
    auto it = pair_states_.begin();
    while (it != pair_states_.end()) {
        if (it->first.find(symbol) != std::string::npos)
            it = pair_states_.erase(it);
        else
            ++it;
    }
}

// ---------------------------------------------------------------------------
// get_correlations
// ---------------------------------------------------------------------------

CorrelationMatrix CrossAssetEngine::get_correlations() const {
    std::lock_guard<std::mutex> lock(mutex_);

    CorrelationMatrix result;
    result.last_updated_ns = last_updated_ns_;
    result.sample_count    = total_samples_;

    for (const auto& kv : signal_windows_)
        result.symbols.push_back(kv.first);

    std::sort(result.symbols.begin(), result.symbols.end());

    const size_t n = result.symbols.size();
    result.matrix.assign(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        result.matrix[i][i] = 1.0;  // diagonal is always 1
        for (size_t j = i + 1; j < n; ++j) {
            const double r = pearson_correlation_locked(result.symbols[i], result.symbols[j]);
            result.matrix[i][j] = r;
            result.matrix[j][i] = r;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// get_pairwise_correlation
// ---------------------------------------------------------------------------

double CrossAssetEngine::get_pairwise_correlation(const std::string& a, const std::string& b) const {
    if (a == b) return 1.0;
    std::lock_guard<std::mutex> lock(mutex_);
    return pearson_correlation_locked(a, b);
}

// ---------------------------------------------------------------------------
// compute_conviction_multiplier
// ---------------------------------------------------------------------------

double CrossAssetEngine::compute_conviction_multiplier(const std::string& primary_symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto wit = signal_windows_.find(primary_symbol);
    if (wit == signal_windows_.end() || wit->second.empty())
        return 1.0;

    const double primary_last = wit->second.back();
    const double primary_sign = (primary_last >= 0.0) ? 1.0 : -1.0;

    double boost = 0.0;
    int    peers = 0;

    for (const auto& kv : signal_windows_) {
        if (kv.first == primary_symbol || kv.second.empty())
            continue;

        const double r = pearson_correlation_locked(primary_symbol, kv.first);
        if (std::abs(r) < kCorrThresh)
            continue;

        ++peers;
        const double peer_last = kv.second.back();
        const double peer_sign = (peer_last >= 0.0) ? 1.0 : -1.0;

        if (r > kCorrThresh) {
            // Positively correlated pair: same direction = conviction, opposite = doubt.
            if (primary_sign == peer_sign)
                boost += r * kBoostScale;
            else
                boost -= r * kBoostScale;
        } else {
            // Anti-correlated pair: opposite direction = confirmation, same = doubt.
            if (primary_sign != peer_sign)
                boost += std::abs(r) * kBoostScale;
            else
                boost -= std::abs(r) * kBoostScale;
        }
    }

    const double multiplier = 1.0 + (peers > 0 ? boost / static_cast<double>(peers) : 0.0);
    return std::max(kMultMin, std::min(kMultMax, multiplier));
}

// ---------------------------------------------------------------------------
// hedge_ratio
// ---------------------------------------------------------------------------

double CrossAssetEngine::hedge_ratio(const std::string& long_sym, const std::string& short_sym) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const double r = pearson_correlation_locked(long_sym, short_sym);
    if (std::abs(r) < 0.20)
        return 0.0;  // insufficient correlation

    const auto it_a = welford_states_.find(long_sym);
    const auto it_b = welford_states_.find(short_sym);
    if (it_a == welford_states_.end() || it_b == welford_states_.end())
        return 0.0;

    const double n_a   = it_a->second.count;
    const double n_b   = it_b->second.count;
    const double var_a = (n_a > 1) ? (it_a->second.M2 / (n_a - 1.0)) : 0.0;
    const double var_b = (n_b > 1) ? (it_b->second.M2 / (n_b - 1.0)) : 0.0;

    const double sigma_a = std::sqrt(std::max(0.0, var_a));
    const double sigma_b = std::sqrt(std::max(0.0, var_b));

    if (sigma_b < kEpsilon)
        return 0.0;

    return r * (sigma_a / sigma_b);
}

// ---------------------------------------------------------------------------
// tracked_symbols / window_size_for
// ---------------------------------------------------------------------------

std::vector<std::string> CrossAssetEngine::tracked_symbols() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.reserve(signal_windows_.size());
    for (const auto& kv : signal_windows_)
        out.push_back(kv.first);
    return out;
}

size_t CrossAssetEngine::window_size_for(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = signal_windows_.find(symbol);
    return (it != signal_windows_.end()) ? it->second.size() : 0;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

// static
std::string CrossAssetEngine::pair_key(const std::string& a, const std::string& b) {
    if (a <= b)
        return a + '\0' + b;
    return b + '\0' + a;
}

double CrossAssetEngine::pearson_correlation_locked(const std::string& a, const std::string& b) const {
    const auto it_a = welford_states_.find(a);
    const auto it_b = welford_states_.find(b);
    if (it_a == welford_states_.end() || it_b == welford_states_.end())
        return 0.0;

    const WelfordState& wa = it_a->second;
    const WelfordState& wb = it_b->second;

    const double n = std::min(wa.count, wb.count);
    if (n < 2.0)
        return 0.0;

    const double var_a = wa.M2 / (n - 1.0);
    const double var_b = wb.M2 / (n - 1.0);
    const double sigma_a = std::sqrt(std::max(0.0, var_a));
    const double sigma_b = std::sqrt(std::max(0.0, var_b));

    if (sigma_a < kEpsilon || sigma_b < kEpsilon)
        return 0.0;

    // Recover covariance from pair state: cov = (Σxy/n - mean_a * mean_b)
    const std::string pk = pair_key(a, b);
    const auto pit = pair_states_.find(pk);
    if (pit == pair_states_.end() || pit->second.count < 2.0)
        return 0.0;

    const double pair_n  = pit->second.count;
    const double mean_xy = pit->second.sum_xy / pair_n;
    const double cov_ab  = mean_xy - wa.mean * wb.mean;

    const double r = cov_ab / (sigma_a * sigma_b);
    // Clamp to [-1, 1] to guard against floating-point overshoot.
    return std::max(-1.0, std::min(1.0, r));
}

void CrossAssetEngine::evict_old_sample(const std::string& symbol, double old_value) {
    auto& ws = welford_states_[symbol];
    if (ws.count < 1.0)
        return;

    // Welford downdate: remove old_value from running mean and M2.
    const double old_count = ws.count;
    ws.count -= 1.0;

    if (ws.count < 1.0) {
        ws.mean = 0.0;
        ws.M2   = 0.0;
        return;
    }

    const double old_mean = ws.mean;
    ws.mean = (old_mean * old_count - old_value) / ws.count;
    ws.M2  -= (old_value - old_mean) * (old_value - ws.mean);
    ws.M2   = std::max(0.0, ws.M2);  // Guard against tiny negative floats.

    // Downdate pair accumulators: subtract old_value * peer_old_value for each peer.
    for (const auto& kv : signal_windows_) {
        if (kv.first == symbol || kv.second.empty())
            continue;
        // Use the oldest peer sample if available (same age as the evicted sample).
        const double peer_old = kv.second.front();
        const std::string pk  = pair_key(symbol, kv.first);
        auto pit = pair_states_.find(pk);
        if (pit != pair_states_.end() && pit->second.count >= 1.0) {
            pit->second.sum_xy -= old_value * peer_old;
            pit->second.count  -= 1.0;
            if (pit->second.count < 0.0)
                pit->second.count = 0.0;
        }
    }
}

void CrossAssetEngine::update_pair_states(const std::string& sym, double new_value) {
    for (const auto& kv : signal_windows_) {
        if (kv.first == sym || kv.second.empty())
            continue;
        const double peer_last = kv.second.back();
        const std::string pk   = pair_key(sym, kv.first);
        auto& ps = pair_states_[pk];
        ps.sum_xy += new_value * peer_last;
        ps.count  += 1.0;
    }
}

} // namespace llmquant
