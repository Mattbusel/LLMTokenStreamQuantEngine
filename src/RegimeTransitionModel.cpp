#include "RegimeTransitionModel.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

void RegimeTransitionModel::record_transition(Regime from, Regime to) {
    total_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    ++count_[regime_index(from)][regime_index(to)];
}

RegimeTransitionModel::Prediction
RegimeTransitionModel::predict_next(Regime current) const {
    Prediction pred;
    pred.regime = Regime::Neutral;
    pred.prob   = 0.0;

    int from = regime_index(current);
    uint64_t row_total = 0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (int j = 0; j < kRegimes; ++j) row_total += count_[from][j];
        if (row_total == 0) return pred;

        int best_j   = 0;
        uint32_t best = 0;
        for (int j = 0; j < kRegimes; ++j) {
            if (count_[from][j] > best) {
                best   = count_[from][j];
                best_j = j;
            }
        }
        pred.regime  = static_cast<Regime>(best_j);
        pred.prob    = static_cast<double>(best) / static_cast<double>(row_total);
    }

    pred.reliable = (total_.load(std::memory_order_relaxed) >=
                     static_cast<uint64_t>(min_transitions_for_reliability));
    return pred;
}

RegimeTransitionModel::ProbMatrix
RegimeTransitionModel::transition_probabilities() const {
    ProbMatrix P{};
    std::lock_guard<std::mutex> lk(mutex_);
    for (int i = 0; i < kRegimes; ++i) {
        uint64_t row_sum = 0;
        for (int j = 0; j < kRegimes; ++j) row_sum += count_[i][j];
        if (row_sum == 0) continue;
        double inv = 1.0 / static_cast<double>(row_sum);
        for (int j = 0; j < kRegimes; ++j)
            P[i][j] = static_cast<double>(count_[i][j]) * inv;
    }
    return P;
}

RegimeTransitionModel::SteadyState
RegimeTransitionModel::steady_state(int max_iter) const {
    SteadyState pi;
    pi.fill(1.0 / kRegimes);  // uniform start

    if (total_.load(std::memory_order_relaxed) <
        static_cast<uint64_t>(min_transitions_for_reliability))
        return pi;

    ProbMatrix P = transition_probabilities();

    // Power iteration: π ← π P, normalise each step.
    SteadyState next;
    for (int iter = 0; iter < max_iter; ++iter) {
        next.fill(0.0);
        for (int i = 0; i < kRegimes; ++i)
            for (int j = 0; j < kRegimes; ++j)
                next[j] += pi[i] * P[i][j];

        // Normalise.
        double s = 0.0;
        for (double v : next) s += v;
        if (s < 1e-12) break;
        double inv = 1.0 / s;
        for (double& v : next) v *= inv;

        // Convergence check.
        double diff = 0.0;
        for (int i = 0; i < kRegimes; ++i)
            diff += std::abs(next[i] - pi[i]);
        pi = next;
        if (diff < 1e-9) break;
    }
    return pi;
}

void RegimeTransitionModel::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& row : count_) row.fill(0);
    total_.store(0, std::memory_order_relaxed);
}

std::string RegimeTransitionModel::to_stats_json() const {
    static const char* names[] = {"Neutral", "Bull", "Bear", "Volatile", "RiskOff"};

    ProbMatrix P = transition_probabilities();
    SteadyState ss = steady_state();
    auto pred_neutral = predict_next(Regime::Neutral);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"total_transitions\":" << total_.load(std::memory_order_relaxed);
    o << ",\"steady_state\":{";
    for (int i = 0; i < kRegimes; ++i) {
        if (i) o << ",";
        o << "\"" << names[i] << "\":" << ss[i];
    }
    o << "},\"transition_matrix\":{";
    for (int i = 0; i < kRegimes; ++i) {
        if (i) o << ",";
        o << "\"" << names[i] << "\":{";
        for (int j = 0; j < kRegimes; ++j) {
            if (j) o << ",";
            o << "\"" << names[j] << "\":" << P[i][j];
        }
        o << "}";
    }
    o << "}}";
    return o.str();
}

} // namespace llmquant
