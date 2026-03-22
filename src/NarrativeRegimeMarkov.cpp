#include "NarrativeRegimeMarkov.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llmquant {

NarrativeRegimeMarkov::NarrativeRegimeMarkov(Config cfg)
    : cfg_(std::move(cfg))
    , trans_counts_(cfg_.n_regimes, std::vector<double>(cfg_.n_regimes, 0.0))
    , steady_state_(cfg_.n_regimes, 1.0 / cfg_.n_regimes)
{}

int NarrativeRegimeMarkov::bias_to_state(double bias) const {
    // Bin: state = clamp(floor((bias + n/2 * bias_range) / bias_range), 0, n-1)
    int s = static_cast<int>(std::floor(bias / cfg_.bias_range + cfg_.n_regimes / 2.0));
    if (s < 0) s = 0;
    if (s >= cfg_.n_regimes) s = cfg_.n_regimes - 1;
    return s;
}

void NarrativeRegimeMarkov::update_steady_state() {
    // Power iteration: π' = π × T (row-normalised)
    int n = cfg_.n_regimes;
    // Build row-stochastic matrix
    std::vector<std::vector<double>> T(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) row_sum += trans_counts_[i][j];
        if (row_sum > 0) {
            for (int j = 0; j < n; ++j) T[i][j] = trans_counts_[i][j] / row_sum;
        } else {
            T[i][i] = 1.0; // self-loop if no transitions
        }
    }
    // 10 power iterations
    std::vector<double> pi = steady_state_;
    std::vector<double> pi_new(n, 0.0);
    for (int iter = 0; iter < 10; ++iter) {
        std::fill(pi_new.begin(), pi_new.end(), 0.0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                pi_new[j] += pi[i] * T[i][j];
        double s = 0.0;
        for (double v : pi_new) s += v;
        if (s > 1e-12) for (double& v : pi_new) v /= s;
        pi = pi_new;
    }
    steady_state_ = pi;
}

void NarrativeRegimeMarkov::record(double bias) {
    int  new_state   = 0;
    bool change_ev   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        new_state = bias_to_state(bias);

        if (prev_state_ >= 0) {
            trans_counts_[prev_state_][new_state] += 1.0;
            // Update steady state every 10 transitions
            if (static_cast<int>(total_.load(std::memory_order_relaxed)) % 10 == 0) {
                update_steady_state();
            }
        }

        bool has_min = total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples);
        if (has_min && prev_state_ >= 0 && new_state != prev_state_) {
            change_ev = true;
            change_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_state_ = new_state;
    }

    current_state_.store(new_state, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (change_ev && cfg_.on_regime_change) cfg_.on_regime_change(prev_state_, new_state);
}

int NarrativeRegimeMarkov::current_state() const noexcept {
    return current_state_.load(std::memory_order_relaxed);
}

double NarrativeRegimeMarkov::steady_state(int regime) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (regime < 0 || regime >= cfg_.n_regimes) return 0.0;
    return steady_state_[regime];
}

double NarrativeRegimeMarkov::transition(int from, int to) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (from < 0 || from >= cfg_.n_regimes || to < 0 || to >= cfg_.n_regimes) return 0.0;
    double row_sum = 0.0;
    for (int j = 0; j < cfg_.n_regimes; ++j) row_sum += trans_counts_[from][j];
    return (row_sum > 0) ? trans_counts_[from][to] / row_sum : 0.0;
}

std::size_t NarrativeRegimeMarkov::regime_change_events() const noexcept {
    return change_events_.load(std::memory_order_relaxed);
}
std::size_t NarrativeRegimeMarkov::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

std::string NarrativeRegimeMarkov::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"current_state\":" << current_state_.load(std::memory_order_relaxed)
       << ",\"regime_change_events\":" << change_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeRegimeMarkov::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& row : trans_counts_) std::fill(row.begin(), row.end(), 0.0);
    std::fill(steady_state_.begin(), steady_state_.end(), 1.0 / cfg_.n_regimes);
    prev_state_ = -1;
    current_state_.store(0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeRegimeMarkov::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    trans_counts_.assign(cfg_.n_regimes, std::vector<double>(cfg_.n_regimes, 0.0));
    steady_state_.assign(cfg_.n_regimes, 1.0 / cfg_.n_regimes);
    prev_state_ = -1;
}

} // namespace llmquant
