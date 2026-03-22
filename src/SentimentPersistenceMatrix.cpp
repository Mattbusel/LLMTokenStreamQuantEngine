#include "SentimentPersistenceMatrix.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SentimentPersistenceMatrix::SentimentPersistenceMatrix() {
    int n = cfg_.n_states;
    counts_.assign(n, std::vector<double>(n, 0.0));
    marginal_.assign(n, 0.0);
}

SentimentPersistenceMatrix::SentimentPersistenceMatrix(const Config& cfg)
    : cfg_(cfg) {
    int n = cfg_.n_states;
    counts_.assign(n, std::vector<double>(n, 0.0));
    marginal_.assign(n, 0.0);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

int SentimentPersistenceMatrix::bias_to_state(double bias) const noexcept {
    int n = static_cast<int>(counts_.size());
    if (n <= 0) return 0;
    double frac = (bias - cfg_.range_min) / (cfg_.range_max - cfg_.range_min);
    int s = static_cast<int>(frac * n);
    if (s < 0) s = 0;
    if (s >= n) s = n - 1;
    return s;
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void SentimentPersistenceMatrix::record(double bias) {
    std::function<void(int, int, double)> cb;
    int    old_s   = 0, new_s = 0;
    int    pred    = -1;
    double sticky  = 0.0;
    double trans_p = 0.0;
    bool   changed = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        int n = static_cast<int>(counts_.size());
        new_s = bias_to_state(bias);
        marginal_[new_s] += 1.0;

        if (seeded_) {
            old_s = current_state_val_;
            counts_[old_s][new_s] += 1.0;

            // Row sum for the outgoing state.
            double row_sum = 0.0;
            for (int j = 0; j < n; ++j) row_sum += counts_[old_s][j];
            trans_p = (row_sum > 0.0) ? (counts_[old_s][new_s] / row_sum) : 0.0;

            // Stickiness and predicted state use the NEW current state's row.
            double new_row_sum = 0.0;
            for (int j = 0; j < n; ++j) new_row_sum += counts_[new_s][j];
            if (new_row_sum >= static_cast<double>(cfg_.min_row_count)) {
                int best_j = 0;
                double best_p = -1.0;
                for (int j = 0; j < n; ++j) {
                    double p = counts_[new_s][j] / new_row_sum;
                    if (p > best_p) { best_p = p; best_j = j; }
                }
                pred   = best_j;
                sticky = counts_[new_s][new_s] / new_row_sum;
            }

            if (new_s != old_s) {
                changed = true;
                changes_.fetch_add(1, std::memory_order_relaxed);
                cb = cfg_.on_state_change;
            }
        } else {
            seeded_ = true;
        }

        current_state_val_ = new_s;
    }  // mutex released here

    current_state_.store(new_s,   std::memory_order_relaxed);
    predicted_state_.store(pred,  std::memory_order_relaxed);
    stickiness_.store(sticky,     std::memory_order_relaxed);
    total_.fetch_add(1,           std::memory_order_relaxed);

    if (changed && cb) cb(old_s, new_s, trans_p);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int SentimentPersistenceMatrix::current_state() const noexcept {
    return current_state_.load(std::memory_order_relaxed);
}

int SentimentPersistenceMatrix::predicted_state() const noexcept {
    return predicted_state_.load(std::memory_order_relaxed);
}

double SentimentPersistenceMatrix::state_probability(int s) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int cur = current_state_val_;
    int n   = static_cast<int>(counts_.size());
    if (s < 0 || s >= n || cur < 0 || cur >= n) return 0.0;
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) row_sum += counts_[cur][j];
    if (row_sum < static_cast<double>(cfg_.min_row_count)) return 0.0;
    return counts_[cur][s] / row_sum;
}

double SentimentPersistenceMatrix::transition_probability(int from, int to) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(counts_.size());
    if (from < 0 || from >= n || to < 0 || to >= n) return 0.0;
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) row_sum += counts_[from][j];
    if (row_sum == 0.0) return 0.0;
    return counts_[from][to] / row_sum;
}

double SentimentPersistenceMatrix::stickiness() const noexcept {
    return stickiness_.load(std::memory_order_relaxed);
}

double SentimentPersistenceMatrix::stationary_estimate(int s) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(marginal_.size());
    if (s < 0 || s >= n) return 0.0;
    double total = 0.0;
    for (double m : marginal_) total += m;
    if (total == 0.0) return 0.0;
    return marginal_[s] / total;
}

uint64_t SentimentPersistenceMatrix::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

uint64_t SentimentPersistenceMatrix::state_changes() const noexcept {
    return changes_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void SentimentPersistenceMatrix::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(counts_.size());
    for (auto& row : counts_) std::fill(row.begin(), row.end(), 0.0);
    std::fill(marginal_.begin(), marginal_.end(), 0.0);
    current_state_val_ = 0;
    seeded_            = false;
    current_state_.store(0,    std::memory_order_relaxed);
    predicted_state_.store(-1, std::memory_order_relaxed);
    stickiness_.store(0.0,     std::memory_order_relaxed);
    total_.store(0,            std::memory_order_relaxed);
    changes_.store(0,          std::memory_order_relaxed);
    (void)n;
}

void SentimentPersistenceMatrix::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int n = cfg_.n_states;
    counts_.assign(n, std::vector<double>(n, 0.0));
    marginal_.assign(n, 0.0);
    current_state_val_ = 0;
    seeded_            = false;
    current_state_.store(0,    std::memory_order_relaxed);
    predicted_state_.store(-1, std::memory_order_relaxed);
    stickiness_.store(0.0,     std::memory_order_relaxed);
    total_.store(0,            std::memory_order_relaxed);
    changes_.store(0,          std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string SentimentPersistenceMatrix::to_stats_json() const {
    int     cur  = current_state();
    int     pred = predicted_state();
    double  stk  = stickiness();
    auto    tot  = total_records();
    auto    chg  = state_changes();

    std::ostringstream os;
    os << "{"
       << "\"current_state\":"  << cur  << ","
       << "\"predicted_state\":" << pred << ","
       << "\"stickiness\":"     << stk  << ","
       << "\"state_changes\":"  << chg  << ","
       << "\"total_records\":"  << tot
       << "}";
    return os.str();
}

} // namespace llmquant
