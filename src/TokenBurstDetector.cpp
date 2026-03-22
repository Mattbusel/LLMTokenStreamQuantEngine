#include "TokenBurstDetector.h"

#include <algorithm>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TokenBurstDetector::TokenBurstDetector(Config config)
    : config_(std::move(config)) {
    // Pre-allocate ring buffer.  Maximum capacity = burst_threshold * window_s * 2
    // but cap at a sane upper bound to avoid huge allocations.
    int cap = std::max(256, static_cast<int>(config_.burst_threshold *
                                             config_.window_ms / 1000.0 * 4));
    cap = std::min(cap, 1 << 20); // 1 M entries max
    ring_.resize(static_cast<std::size_t>(cap), 0);
}

// ---------------------------------------------------------------------------
// evict_and_rate  (internal, called under lock)
// ---------------------------------------------------------------------------

double TokenBurstDetector::evict_and_rate(int64_t now_ns) {
    int64_t cutoff_ns = now_ns - static_cast<int64_t>(config_.window_ms) * 1'000'000LL;
    // Evict head entries older than cutoff.
    while (count_ > 0) {
        int64_t oldest = ring_[static_cast<std::size_t>(head_)];
        if (oldest >= cutoff_ns) break;
        head_ = (head_ + 1) % static_cast<int>(ring_.size());
        --count_;
    }
    double window_s = config_.window_ms / 1000.0;
    last_rate_ = (window_s > 0.0 && count_ > 0)
                 ? static_cast<double>(count_) / window_s
                 : 0.0;
    return last_rate_;
}

// ---------------------------------------------------------------------------
// record_at
// ---------------------------------------------------------------------------

double TokenBurstDetector::record_at(std::chrono::steady_clock::time_point tp) {
    total_tokens_.fetch_add(1, std::memory_order_relaxed);

    std::function<void(double)> start_cb, end_cb;
    bool fire_start = false, fire_end = false;
    double rate = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             tp.time_since_epoch()).count();

        // Grow ring if needed (shouldn't normally happen after construction).
        if (count_ == static_cast<int>(ring_.size())) {
            ring_.resize(ring_.size() * 2);
        }

        // Write new arrival at tail.
        int tail = (head_ + count_) % static_cast<int>(ring_.size());
        ring_[static_cast<std::size_t>(tail)] = now_ns;
        ++count_;

        rate = evict_and_rate(now_ns);

        bool currently_burst = in_burst_.load(std::memory_order_relaxed);
        double enter_thr = config_.burst_threshold;
        double exit_thr  = config_.burst_threshold * (1.0 - config_.hysteresis);

        if (!currently_burst && rate >= enter_thr) {
            in_burst_.store(true, std::memory_order_relaxed);
            burst_events_.fetch_add(1, std::memory_order_relaxed);
            fire_start = true;
            start_cb   = config_.on_burst_start;
        } else if (currently_burst && rate < exit_thr) {
            in_burst_.store(false, std::memory_order_relaxed);
            fire_end = true;
            end_cb   = config_.on_burst_end;
        }
    }

    if (fire_start) {
        spdlog::warn("[burst_det] BURST rate={:.1f} tok/s", rate);
        if (start_cb) start_cb(rate);
    } else if (fire_end) {
        spdlog::info("[burst_det] burst END rate={:.1f} tok/s", rate);
        if (end_cb) end_cb(rate);
    }

    return rate;
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

double TokenBurstDetector::record() {
    return record_at(std::chrono::steady_clock::now());
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void TokenBurstDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), 0LL);
    head_       = 0;
    count_      = 0;
    last_rate_  = 0.0;
    in_burst_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// current_rate
// ---------------------------------------------------------------------------

double TokenBurstDetector::current_rate() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_rate_;
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void TokenBurstDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_    = config;
    head_      = 0;
    count_     = 0;
    last_rate_ = 0.0;
    in_burst_.store(false, std::memory_order_relaxed);
    int cap = std::max(256, static_cast<int>(config_.burst_threshold *
                                             config_.window_ms / 1000.0 * 4));
    cap = std::min(cap, 1 << 20);
    ring_.assign(static_cast<std::size_t>(cap), 0LL);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string TokenBurstDetector::to_stats_json() const {
    double rate = 0.0;
    bool   burst = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        rate  = last_rate_;
        burst = in_burst_.load(std::memory_order_relaxed);
    }
    std::ostringstream s;
    s << "{\"current_rate\":"  << rate
      << ",\"is_burst\":"      << (burst ? "true" : "false")
      << ",\"total_tokens\":"  << total_tokens()
      << ",\"burst_events\":"  << burst_events()
      << "}";
    return s.str();
}

} // namespace llmquant
