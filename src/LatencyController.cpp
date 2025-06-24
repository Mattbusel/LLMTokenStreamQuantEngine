#pragma once
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>

namespace llmquant {

class LatencyController {
public:
    struct Config {
        std::chrono::microseconds target_latency{10}; // 10μs target
        size_t sample_window{1000};
        bool enable_profiling{true};
    };

    explicit LatencyController(const Config& config);
    ~LatencyController() = default;

    // Timing measurement
    void start_measurement();
    void end_measurement();
    void record_latency(std::chrono::microseconds latency);
    
    // Performance analysis
    struct LatencyStats {
        std::chrono::microseconds avg_latency{0};
        std::chrono::microseconds min_latency{std::chrono::microseconds::max()};
        std::chrono::microseconds max_latency{0};
        std::chrono::microseconds p95_latency{0};
        std::chrono::microseconds p99_latency{0};
        double jitter_ms{0.0};
        uint64_t measurements{0};
    };
    
    LatencyStats get_stats() const;
    void reset_stats();
    
    // Profiling hooks
    void profile_token_processing();
    void profile_signal_generation();
    void profile_queue_lag();

private:
    void update_percentiles();
    
    Config config_;
    std::chrono::high_resolution_clock::time_point measurement_start_;
    std::vector<std::chrono::microseconds> latency_samples_;
    mutable std::mutex samples_mutex_;
    
    std::atomic<uint64_t> total_measurements_{0};
    std::atomic<uint64_t> total_latency_us_{0};
    std::atomic<uint64_t> min_latency_us_{UINT64_MAX};
    std::atomic<uint64_t> max_latency_us_{0};
};

} // namespace llmquant
