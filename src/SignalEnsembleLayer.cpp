#include "SignalEnsembleLayer.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llmquant {

SignalEnsembleLayer::SignalEnsembleLayer(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// register_source
// ---------------------------------------------------------------------------
int SignalEnsembleLayer::register_source(const std::string& name) {
    std::lock_guard<std::mutex> lk(mutex_);
    int id = static_cast<int>(names_.size());
    names_.push_back(name);
    values_.push_back(0.0);
    weights_.push_back(config_.initial_weight);
    project_weights();
    return id;
}

// ---------------------------------------------------------------------------
// update_source
// ---------------------------------------------------------------------------
void SignalEnsembleLayer::update_source(int source_id, double value) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (source_id < 0 || source_id >= static_cast<int>(values_.size())) return;
    values_[static_cast<std::size_t>(source_id)] = value;
}

// ---------------------------------------------------------------------------
// ensemble_output
// ---------------------------------------------------------------------------
double SignalEnsembleLayer::ensemble_output() const {
    std::lock_guard<std::mutex> lk(mutex_);
    double out = 0.0;
    for (std::size_t i = 0; i < weights_.size(); ++i)
        out += weights_[i] * values_[i];
    return out;
}

// ---------------------------------------------------------------------------
// record_outcome  — SGD weight update
// ---------------------------------------------------------------------------
void SignalEnsembleLayer::record_outcome(double target) {
    total_outcomes_.fetch_add(1, std::memory_order_relaxed);

    std::vector<double> weights_copy;
    std::function<void(const std::vector<double>&)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (weights_.empty()) return;

        // Compute current output under lock (values may be updated concurrently).
        double out = 0.0;
        for (std::size_t i = 0; i < weights_.size(); ++i)
            out += weights_[i] * values_[i];

        double err = out - target;
        double lr  = config_.learning_rate;

        for (std::size_t i = 0; i < weights_.size(); ++i)
            weights_[i] -= lr * err * values_[i];

        project_weights();

        if (config_.on_weight_update) {
            weights_copy = weights_;
            cb           = config_.on_weight_update;
        }
    }

    if (cb) cb(weights_copy);
}

// ---------------------------------------------------------------------------
// project_weights — non-negative simplex projection (called under lock)
// ---------------------------------------------------------------------------
void SignalEnsembleLayer::project_weights() {
    if (weights_.empty()) return;

    // Clamp to min_weight.
    double min_w = config_.min_weight;
    for (auto& w : weights_) w = std::max(min_w, w);

    // Normalise to sum-to-one.
    double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sum < 1e-14) {
        // All weights collapsed — reinstate uniform.
        double uni = 1.0 / static_cast<double>(weights_.size());
        for (auto& w : weights_) w = uni;
    } else {
        for (auto& w : weights_) w /= sum;
    }
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------
int SignalEnsembleLayer::source_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(names_.size());
}

double SignalEnsembleLayer::weight(int source_id) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (source_id < 0 || source_id >= static_cast<int>(weights_.size())) return 0.0;
    return weights_[static_cast<std::size_t>(source_id)];
}

std::vector<double> SignalEnsembleLayer::weights() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return weights_;
}

void SignalEnsembleLayer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    double uni = names_.empty() ? 1.0 : 1.0 / static_cast<double>(names_.size());
    for (auto& w : weights_) w = uni;
    for (auto& v : values_)  v = 0.0;
    total_outcomes_.store(0, std::memory_order_relaxed);
}

void SignalEnsembleLayer::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

std::string SignalEnsembleLayer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << std::fixed;
    os << "{"
       << "\"source_count\":"   << names_.size() << ","
       << "\"total_outcomes\":" << total_outcomes_.load(std::memory_order_relaxed) << ","
       << "\"sources\":[";
    for (std::size_t i = 0; i < names_.size(); ++i) {
        if (i > 0) os << ",";
        os << "{"
           << "\"name\":\"" << names_[i] << "\","
           << "\"weight\":"  << weights_[i] << ","
           << "\"value\":"   << values_[i]
           << "}";
    }
    os << "]}";
    return os.str();
}

} // namespace llmquant
