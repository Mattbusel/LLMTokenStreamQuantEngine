#include "TokenInformationBottleneck.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TokenInformationBottleneck::TokenInformationBottleneck() = default;

TokenInformationBottleneck::TokenInformationBottleneck(const Config& cfg)
    : cfg_(cfg) {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

double TokenInformationBottleneck::compute_relevance(
    const TokenStats& ts) const noexcept {
    if (ts.count < cfg_.min_observations) return 0.0;

    int n = ts.count;
    // Pearson r between weights and biases over the filled window.
    double sum_w = 0.0, sum_b = 0.0;
    for (int i = 0; i < n; ++i) { sum_w += ts.weights[i]; sum_b += ts.biases[i]; }
    double mean_w = sum_w / n;
    double mean_b = sum_b / n;

    double cov = 0.0, var_w = 0.0, var_b = 0.0;
    for (int i = 0; i < n; ++i) {
        double dw = ts.weights[i] - mean_w;
        double db = ts.biases[i]  - mean_b;
        cov   += dw * db;
        var_w += dw * dw;
        var_b += db * db;
    }
    double denom = std::sqrt(var_w * var_b);
    if (denom < cfg_.epsilon) return 0.0;
    double r = cov / denom;
    return std::abs(r);  // relevance = |correlation|
}

double TokenInformationBottleneck::compute_complexity(
    const TokenStats& ts) const noexcept {
    if (ts.count < 2) return 0.0;

    int n = ts.count;
    int B = cfg_.n_weight_bins;

    // Find range
    double lo = ts.weights[0], hi = ts.weights[0];
    for (int i = 1; i < n; ++i) {
        if (ts.weights[i] < lo) lo = ts.weights[i];
        if (ts.weights[i] > hi) hi = ts.weights[i];
    }
    double range = hi - lo;
    if (range < cfg_.epsilon) return 0.0;  // all same weight → zero entropy

    // Count per bin
    std::vector<int> bins(B, 0);
    for (int i = 0; i < n; ++i) {
        int b = static_cast<int>((ts.weights[i] - lo) / range * B);
        if (b >= B) b = B - 1;
        ++bins[b];
    }

    // Shannon entropy normalised by log2(B)
    double H = 0.0;
    for (int b = 0; b < B; ++b) {
        if (bins[b] == 0) continue;
        double p = static_cast<double>(bins[b]) / n;
        H -= p * std::log2(p);
    }
    double H_max = std::log2(static_cast<double>(B));
    return (H_max > 0.0) ? H / H_max : 0.0;
}

double TokenInformationBottleneck::compute_score(
    const TokenStats& ts) const noexcept {
    double R = compute_relevance(ts);
    double C = compute_complexity(ts);
    return R / (C + cfg_.epsilon);
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void TokenInformationBottleneck::record(const std::string& token,
                                         double weight, double bias_shift) {
    std::function<void(const std::string&, double)> cb;
    bool   do_cb     = false;
    double score_val = 0.0;
    bool   recovered = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        auto& ts = stats_[token];
        // Initialise window on first encounter.
        if (static_cast<int>(ts.weights.size()) < cfg_.window_size) {
            ts.weights.resize(cfg_.window_size, 0.0);
            ts.biases.resize(cfg_.window_size, 0.0);
        }
        ts.weights[ts.head] = weight;
        ts.biases[ts.head]  = bias_shift;
        ts.head = (ts.head + 1) % cfg_.window_size;
        if (ts.count < cfg_.window_size) ++ts.count;

        // Check flag status after sufficient observations.
        if (ts.count >= cfg_.min_observations) {
            double sc = compute_score(ts);
            score_val = sc;
            bool should_flag = (sc < cfg_.prune_threshold);

            if (should_flag && !ts.flagged) {
                ts.flagged = true;
                ++flagged_count_;
                do_cb = true;
                recovered = false;
                cb = cfg_.on_low_score;
            } else if (!should_flag && ts.flagged) {
                ts.flagged = false;
                --flagged_count_;
                do_cb = true;
                recovered = true;
                cb = cfg_.on_recovered;
            }
        }
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (do_cb && cb) cb(token, score_val);
}

// ---------------------------------------------------------------------------
// Per-token accessors
// ---------------------------------------------------------------------------

double TokenInformationBottleneck::ib_score(const std::string& token) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = stats_.find(token);
    if (it == stats_.end()) return 0.0;
    return compute_score(it->second);
}

double TokenInformationBottleneck::relevance(const std::string& token) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = stats_.find(token);
    if (it == stats_.end()) return 0.0;
    return compute_relevance(it->second);
}

double TokenInformationBottleneck::complexity(const std::string& token) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = stats_.find(token);
    if (it == stats_.end()) return 0.0;
    return compute_complexity(it->second);
}

int TokenInformationBottleneck::observation_count(const std::string& token) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = stats_.find(token);
    if (it == stats_.end()) return 0;
    return it->second.count;
}

// ---------------------------------------------------------------------------
// Aggregate accessors
// ---------------------------------------------------------------------------

uint64_t TokenInformationBottleneck::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

int TokenInformationBottleneck::distinct_tokens() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(stats_.size());
}

int TokenInformationBottleneck::flagged_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return flagged_count_;
}

std::vector<std::pair<std::string, double>>
TokenInformationBottleneck::top_k_tokens() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::pair<std::string, double>> scored;
    scored.reserve(stats_.size());
    for (const auto& [tok, ts] : stats_) {
        if (ts.count >= cfg_.min_observations)
            scored.emplace_back(tok, compute_score(ts));
    }
    int K = std::min(cfg_.top_k, static_cast<int>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
                      [](const auto& a, const auto& b){ return a.second > b.second; });
    scored.resize(K);
    return scored;
}

std::vector<std::pair<std::string, double>>
TokenInformationBottleneck::bottom_k_tokens() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::pair<std::string, double>> scored;
    scored.reserve(stats_.size());
    for (const auto& [tok, ts] : stats_) {
        if (ts.count >= cfg_.min_observations)
            scored.emplace_back(tok, compute_score(ts));
    }
    int K = std::min(cfg_.top_k, static_cast<int>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
                      [](const auto& a, const auto& b){ return a.second < b.second; });
    scored.resize(K);
    return scored;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void TokenInformationBottleneck::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    stats_.clear();
    flagged_count_ = 0;
    total_.store(0, std::memory_order_relaxed);
}

void TokenInformationBottleneck::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    stats_.clear();
    flagged_count_ = 0;
    total_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string TokenInformationBottleneck::to_stats_json() const {
    auto tot  = total_records();
    int  dist = distinct_tokens();
    int  flag = flagged_count();

    std::ostringstream os;
    os << "{"
       << "\"total_records\":"  << tot  << ","
       << "\"distinct_tokens\":" << dist << ","
       << "\"flagged_count\":"  << flag
       << "}";
    return os.str();
}

}  // namespace llmquant
