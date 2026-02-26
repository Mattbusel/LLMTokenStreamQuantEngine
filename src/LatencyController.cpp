#include "LatencyController.h"
#include <algorithm>
#include <numeric>

namespace llmquant {

LatencyController::LatencyController(const Config& config) : config_(config) {
    latency_samples_.reserve(config_.sample_window);
}

void LatencyController::start_measurement() {
    measurement_start_ = std::chrono::high_resolution_clock::now();
}

void LatencyController::end_measurement() {
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - measurement_start_);
    record_latency(latency);
}

void LatencyController::record_latency(std::chrono::microseconds latency) {
    uint64_t latency_us = latency.count();
    
    // Update atomic stats
    total_measurements_++;
    total_latency_us_ += latency_us;
    
    uint64_t current_min = min_latency_us_.load();
    while (latency_us < current_min && !min_latency_us_.compare_exchange_weak(current_min, latency_us));
    
    uint64_t current_max = max_latency_us_.load();
    while (latency_us > current_max && !max_latency_us_.compare_exchange_weak(current_max, latency_us));
    
    // Store sample for percentile calculation
    if (config_.enable_profiling) {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        latency_samples_.push_back(latency);
        
        if (latency_samples_.size() > config_.sample_window) {
            latency_samples_.erase(latency_samples_.begin());
        }
    }
}

LatencyController::LatencyStats LatencyController::get_stats() const {
    LatencyStats stats;
    
    uint64_t measurements = total_measurements_.load();
    if (measurements == 0) return stats;
    
    stats.avg_latency = std::chrono::microseconds(total_latency_us_.load() / measurements);
    stats.min_latency = std::chrono::microseconds(min_latency_us_.load());
    stats.max_latency = std::chrono::microseconds(max_latency_us_.load());
    stats.measurements = measurements;
    
    // Calculate percentiles from samples
    if (config_.enable_profiling) {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        if (!latency_samples_.empty()) {
            auto samples_copy = latency_samples_;
            std::sort(samples_copy.begin(), samples_copy.end());
            
            size_t p95_idx = static_cast<size_t>(samples_copy.size() * 0.95);
            size_t p99_idx = static_cast<size_t>(samples_copy.size() * 0.99);
            
            stats.p95_latency = samples_copy[std::min(p95_idx, samples_copy.size() - 1)];
            stats.p99_latency = samples_copy[std::min(p99_idx, samples_copy.size() - 1)];
            
            // Calculate jitter (standard deviation)
            auto mean = stats.avg_latency.count();
            double variance = 0.0;
            for (const auto& sample : samples_copy) {
                double diff = sample.count() - mean;
                variance += diff * diff;
            }
            stats.jitter_ms = std::sqrt(variance / samples_copy.size()) / 1000.0;
        }
    }
    
    return stats;
}

void LatencyController::reset_stats() {
    total_measurements_ = 0;
    total_latency_us_ = 0;
    min_latency_us_ = UINT64_MAX;
    max_latency_us_ = 0;
    
    std::lock_guard<std::mutex> lock(samples_mutex_);
    latency_samples_.clear();
}

void LatencyController::profile_token_processing() {
    // Hook for detailed token processing profiling
    start_measurement();
}

void LatencyController::profile_signal_generation() {
    // Hook for signal generation profiling
}

void LatencyController::profile_queue_lag() {
    // Hook for queue lag profiling
}

void LatencyController::update_ingestion_pressure(double arrival_rate_tps,
                                                   double max_rate_tps) {
    double p = (max_rate_tps > 0.0)
                   ? std::clamp(arrival_rate_tps / max_rate_tps, 0.0, 1.0)
                   : 0.0;
    std::lock_guard<std::mutex> lock(pressure_mutex_);
    pressure_.ingestion_pressure = p;
    recompute_composite();
}

void LatencyController::update_semantic_pressure(double weight_variance) {
    // Variance > 0.25 saturates semantic pressure.
    double p = std::clamp(weight_variance / 0.25, 0.0, 1.0);
    std::lock_guard<std::mutex> lock(pressure_mutex_);
    pressure_.semantic_pressure = p;
    recompute_composite();
}

void LatencyController::update_queue_pressure(size_t queue_depth,
                                              size_t queue_capacity) {
    double p = (queue_capacity > 0)
                   ? std::clamp(static_cast<double>(queue_depth) / queue_capacity, 0.0, 1.0)
                   : 0.0;
    std::lock_guard<std::mutex> lock(pressure_mutex_);
    pressure_.queue_pressure = p;
    recompute_composite();
}

LatencyController::PressureState LatencyController::get_pressure() const {
    std::lock_guard<std::mutex> lock(pressure_mutex_);
    return pressure_;
}

double LatencyController::get_backoff_multiplier() const {
    return backoff_multiplier_.load();
}

void LatencyController::recompute_composite() {
    double c = std::max({pressure_.ingestion_pressure,
                         pressure_.semantic_pressure,
                         pressure_.queue_pressure});
    pressure_.composite = c;

    // Exponential backoff: ramps from 1x to 5x as composite exceeds 0.8.
    if (c >= 0.8) {
        double current = backoff_multiplier_.load();
        double next = std::min(current * 1.5, 5.0);
        backoff_multiplier_.store(next);
    } else if (c < 0.5) {
        backoff_multiplier_.store(1.0);
    }
}

} // namespace llmquant
