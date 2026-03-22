#include "ConfidenceCalibrator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>

namespace llmquant {

ConfidenceCalibrator::ConfidenceCalibrator(Config cfg) : cfg_(std::move(cfg)) {}

// ---------------------------------------------------------------------------

double ConfidenceCalibrator::sigmoid(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
}

double ConfidenceCalibrator::calibrate(double raw_confidence) const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (static_cast<int>(history_.size()) < cfg_.min_samples) return raw_confidence;
    double val = sigmoid(platt_a_ * raw_confidence + platt_b_);
    return std::clamp(val, 0.0, 1.0);
}

void ConfidenceCalibrator::sgd_step_locked(double predicted, bool won) {
    // Logistic loss gradient for parameters a and b.
    double y     = won ? 1.0 : 0.0;
    double score = platt_a_ * predicted + platt_b_;
    double prob  = sigmoid(score);
    double err   = prob - y;  // gradient of cross-entropy

    platt_a_ -= cfg_.learning_rate * err * predicted;
    platt_b_ -= cfg_.learning_rate * err;
}

void ConfidenceCalibrator::record_outcome(double predicted_confidence, bool won) {
    std::lock_guard<std::mutex> lk(mtx_);
    history_.push_back({predicted_confidence, won});
    if (static_cast<int>(history_.size()) > cfg_.window_size) {
        history_.pop_front();
    }
    sgd_step_locked(predicted_confidence, won);
    total_outcomes_.fetch_add(1, std::memory_order_relaxed);
}

bool ConfidenceCalibrator::is_calibrated() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return static_cast<int>(history_.size()) >= cfg_.min_samples;
}

double ConfidenceCalibrator::empirical_win_rate() const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (history_.empty()) return 0.0;
    int wins = 0;
    for (const auto& r : history_) if (r.won) ++wins;
    return static_cast<double>(wins) / static_cast<double>(history_.size());
}

std::pair<double, double> ConfidenceCalibrator::platt_params() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return {platt_a_, platt_b_};
}

void ConfidenceCalibrator::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    history_.clear();
    platt_a_ = 1.0;
    platt_b_ = 0.0;
    total_outcomes_.store(0, std::memory_order_relaxed);
}

void ConfidenceCalibrator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_     = cfg;
    history_.clear();
    platt_a_ = 1.0;
    platt_b_ = 0.0;
    total_outcomes_.store(0, std::memory_order_relaxed);
}

std::string ConfidenceCalibrator::to_stats_json() const {
    auto [a, b] = platt_params();
    double wr   = empirical_win_rate();
    char buf[512];
    std::lock_guard<std::mutex> lk(mtx_);
    std::snprintf(buf, sizeof(buf),
        "{\"calibrated\":%s,\"platt_a\":%.6f,\"platt_b\":%.6f,"
        "\"empirical_win_rate\":%.4f,\"window_samples\":%zu,"
        "\"total_outcomes\":%llu,\"min_samples\":%d}",
        (static_cast<int>(history_.size()) >= cfg_.min_samples ? "true" : "false"),
        a, b, wr,
        history_.size(),
        static_cast<unsigned long long>(total_outcomes_.load(std::memory_order_relaxed)),
        cfg_.min_samples);
    return buf;
}

} // namespace llmquant
