#include "TokenStreamHealthMonitor.h"

#include <algorithm>
#include <sstream>

namespace llmquant {

TokenStreamHealthMonitor::TokenStreamHealthMonitor(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int64_t to_ms(std::chrono::steady_clock::duration d) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}

TokenStreamHealthMonitor::Status
TokenStreamHealthMonitor::evaluate_locked(TimePoint now) {
    // Compute token rate over the rolling window.
    int64_t window_ms = config_.rate_window_ms;
    if (window_ms <= 0) window_ms = 1000;

    // Advance buckets to cover [now - window_ms, now].
    int64_t bucket_width_ms = window_ms / kBuckets;
    if (bucket_width_ms < 1) bucket_width_ms = 1;

    if (ever_received_) {
        int64_t elapsed_since_start = to_ms(now - bucket_start_tp_);
        int64_t buckets_to_advance  = elapsed_since_start / bucket_width_ms;
        if (buckets_to_advance > 0) {
            int clear = static_cast<int>(std::min<int64_t>(buckets_to_advance, kBuckets));
            for (int i = 0; i < clear; ++i) {
                bucket_head_ = (bucket_head_ + 1) % kBuckets;
                bucket_counts_[bucket_head_] = 0;
            }
            bucket_start_tp_ = now;
        }
    }

    // Sum all buckets.
    int64_t total_in_window = 0;
    for (int i = 0; i < kBuckets; ++i) total_in_window += bucket_counts_[i];
    double rate = (window_ms > 0)
                ? static_cast<double>(total_in_window) * 1000.0 / window_ms
                : 0.0;
    rate_.store(rate, std::memory_order_relaxed);

    // Determine new status.
    Status new_status = Status::Healthy;

    if (ever_received_) {
        int64_t gap_ms = to_ms(now - last_token_tp_);
        if (gap_ms > config_.stall_timeout_ms) {
            new_status = Status::Stalled;
        }
    }

    if (new_status == Status::Healthy &&
        config_.max_tokens_per_sec > 0.0 &&
        rate > config_.max_tokens_per_sec) {
        new_status = Status::Flooded;
    }

    return new_status;
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void TokenStreamHealthMonitor::ping() noexcept {
    total_tokens_.fetch_add(1, std::memory_order_relaxed);

    Status   new_status = Status::Healthy;
    Status   old_status = Status::Healthy;
    Config   cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        TimePoint now = Clock::now();
        if (!ever_received_) {
            ever_received_  = true;
            last_token_tp_  = now;
            bucket_start_tp_ = now;
        }

        // Increment the current bucket.
        bucket_counts_[bucket_head_]++;
        last_token_tp_ = now;

        old_status = prev_status_;
        new_status = evaluate_locked(now);
        status_.store(new_status, std::memory_order_relaxed);

        if (new_status != old_status) {
            prev_status_ = new_status;
            if (new_status == Status::Stalled)
                stall_count_.fetch_add(1, std::memory_order_relaxed);
            else if (new_status == Status::Flooded)
                flood_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Fire callbacks outside lock.
    if (new_status != old_status) {
        double rate_snap = rate_.load(std::memory_order_relaxed);
        if (new_status == Status::Flooded && cfg_copy.on_flood)
            cfg_copy.on_flood(rate_snap);
        else if (new_status == Status::Healthy &&
                 old_status != Status::Healthy && cfg_copy.on_recovery)
            cfg_copy.on_recovery();
    }
}

void TokenStreamHealthMonitor::poll() noexcept {
    Status   new_status = Status::Healthy;
    Status   old_status = Status::Healthy;
    int64_t  gap_ms     = 0;
    Config   cfg_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg_copy = config_;

        TimePoint now = Clock::now();
        old_status = prev_status_;
        new_status = evaluate_locked(now);
        status_.store(new_status, std::memory_order_relaxed);

        if (new_status != old_status) {
            prev_status_ = new_status;
            if (new_status == Status::Stalled) {
                stall_count_.fetch_add(1, std::memory_order_relaxed);
                gap_ms = ever_received_ ? to_ms(now - last_token_tp_) : 0;
            } else if (new_status == Status::Flooded) {
                flood_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Fire callbacks outside lock.
    if (new_status != old_status) {
        double rate_snap = rate_.load(std::memory_order_relaxed);
        if (new_status == Status::Stalled && cfg_copy.on_stall)
            cfg_copy.on_stall(gap_ms);
        else if (new_status == Status::Flooded && cfg_copy.on_flood)
            cfg_copy.on_flood(rate_snap);
        else if (new_status == Status::Healthy &&
                 old_status != Status::Healthy && cfg_copy.on_recovery)
            cfg_copy.on_recovery();
    }
}

int64_t TokenStreamHealthMonitor::ms_since_last_token() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!ever_received_) return INT64_MAX;
    return to_ms(Clock::now() - last_token_tp_);
}

void TokenStreamHealthMonitor::update_config(Config config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = std::move(config);
    // Reset state.
    ever_received_ = false;
    for (auto& b : bucket_counts_) b = 0;
    bucket_head_  = 0;
    status_.store(Status::Healthy, std::memory_order_relaxed);
    rate_.store(0.0, std::memory_order_relaxed);
    total_tokens_.store(0, std::memory_order_relaxed);
    stall_count_.store(0, std::memory_order_relaxed);
    flood_count_.store(0, std::memory_order_relaxed);
    prev_status_ = Status::Healthy;
}

void TokenStreamHealthMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ever_received_ = false;
    for (auto& b : bucket_counts_) b = 0;
    bucket_head_  = 0;
    status_.store(Status::Healthy, std::memory_order_relaxed);
    rate_.store(0.0, std::memory_order_relaxed);
    total_tokens_.store(0, std::memory_order_relaxed);
    stall_count_.store(0, std::memory_order_relaxed);
    flood_count_.store(0, std::memory_order_relaxed);
    prev_status_ = Status::Healthy;
}

std::string TokenStreamHealthMonitor::to_stats_json() const {
    Status s   = status_.load(std::memory_order_relaxed);
    double r   = rate_.load(std::memory_order_relaxed);
    uint64_t t = total_tokens_.load(std::memory_order_relaxed);
    uint64_t sc = stall_count_.load(std::memory_order_relaxed);
    uint64_t fc = flood_count_.load(std::memory_order_relaxed);

    const char* status_str = (s == Status::Healthy)  ? "healthy"
                           : (s == Status::Stalled)  ? "stalled"
                                                      : "flooded";
    std::ostringstream o;
    o << std::fixed;
    o.precision(2);
    o << "{\"status\":\"" << status_str << "\""
      << ",\"rate_tps\":"     << r
      << ",\"total_tokens\":" << t
      << ",\"stall_count\":"  << sc
      << ",\"flood_count\":"  << fc << "}";
    return o.str();
}

} // namespace llmquant
