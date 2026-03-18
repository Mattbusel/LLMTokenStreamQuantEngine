#include "LatencyController.h"
#include <algorithm>
#include <numeric>

namespace llmquant {

LatencyController::LatencyController(const Config& config) : config_(config) {
    // Pre-allocate the ring buffer to its full capacity so writes never reallocate.
    latency_samples_.assign(config_.sample_window, std::chrono::microseconds{0});
}

void LatencyController::start_measurement() {
    latency_measurement_start_ = std::chrono::high_resolution_clock::now();
}

void LatencyController::end_measurement() {
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - latency_measurement_start_);
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
    
    // Store sample in ring buffer for percentile calculation — O(1) per insert.
    if (config_.enable_profiling) {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        latency_samples_[sample_head_] = latency;
        sample_head_ = (sample_head_ + 1) % config_.sample_window;
        if (sample_count_ < config_.sample_window) ++sample_count_;
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
        if (sample_count_ > 0) {
            // Reconstruct the valid samples in insertion order from the ring buffer.
            std::vector<std::chrono::microseconds> samples_copy(sample_count_);
            for (size_t k = 0; k < sample_count_; ++k) {
                size_t idx = (sample_head_ + config_.sample_window - sample_count_ + k)
                             % config_.sample_window;
                samples_copy[k] = latency_samples_[idx];
            }

            size_t N = samples_copy.size();
            size_t p95_idx = static_cast<size_t>(std::ceil(static_cast<double>(N) * 0.95));
            if (p95_idx > 0) p95_idx--;
            size_t p99_idx = static_cast<size_t>(std::ceil(static_cast<double>(N) * 0.99));
            if (p99_idx > 0) p99_idx--;
            p95_idx = std::min(p95_idx, N - 1);
            p99_idx = std::min(p99_idx, N - 1);

            // Run nth_element for p95 first (smaller index), then p99.
            // Both use the full range — the standard only guarantees correctness
            // when the range contains the nth element and all elements that
            // should be before/after it.
            std::nth_element(samples_copy.begin(),
                             samples_copy.begin() + static_cast<std::ptrdiff_t>(p95_idx),
                             samples_copy.end());
            stats.p95_latency = samples_copy[p95_idx];

            std::nth_element(samples_copy.begin(),
                             samples_copy.begin() + static_cast<std::ptrdiff_t>(p99_idx),
                             samples_copy.end());
            stats.p99_latency = samples_copy[p99_idx];

            // Calculate jitter (std dev) against the window mean, not global avg
            double window_mean = 0.0;
            for (const auto& s : samples_copy)
                window_mean += static_cast<double>(s.count());
            window_mean /= static_cast<double>(samples_copy.size());

            double variance = 0.0;
            for (const auto& sample : samples_copy) {
                double diff = static_cast<double>(sample.count()) - window_mean;
                variance += diff * diff;
            }
            // Divide by N-1 (Bessel's correction for unbiased sample std dev).
            if (N > 1) {
                stats.jitter_ms = std::sqrt(variance / static_cast<double>(N - 1)) / 1000.0;
            } else {
                stats.jitter_ms = 0.0;
            }
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
    std::fill(latency_samples_.begin(), latency_samples_.end(),
              std::chrono::microseconds{0});
    sample_head_  = 0;
    sample_count_ = 0;
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
