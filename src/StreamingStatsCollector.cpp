#include "StreamingStatsCollector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace llmquant {

namespace {

/// Current time in microseconds since epoch.
int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()
           ).count();
}

/// Current time in milliseconds since epoch.
int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               steady_clock::now().time_since_epoch()
           ).count();
}

} // anonymous namespace

// ============================================================================
// Construction
// ============================================================================

StreamingStatsCollector::StreamingStatsCollector(Config cfg)
    : cfg_(std::move(cfg))
{
    hist_.fill(0);
    tp_ring_.fill(ThroughputSlot{});
    tp_slot_ms_ = std::max(int64_t{1},
        static_cast<int64_t>(cfg_.throughput_window_ms) / kThroughputSlots);
}

// ============================================================================
// Bucket mapping  —  quadratic spacing so that small latencies get finer
// resolution while still covering the full range to max_latency_us.
// ============================================================================

int StreamingStatsCollector::bucket_for(int64_t latency_us) const noexcept {
    if (latency_us <= 0)                           return 0;
    if (latency_us >= cfg_.max_latency_us)         return kHistogramBuckets - 1;
    // Normalise to [0,1], apply sqrt for sub-linear bucket growth.
    double norm = static_cast<double>(latency_us) /
                  static_cast<double>(cfg_.max_latency_us);
    double mapped = std::sqrt(norm);  // sqrt compresses large values
    int b = static_cast<int>(mapped * (kHistogramBuckets - 1));
    return std::clamp(b, 0, kHistogramBuckets - 1);
}

int64_t StreamingStatsCollector::bucket_lower(int bucket) const noexcept {
    if (bucket <= 0) return 0;
    if (bucket >= kHistogramBuckets) return cfg_.max_latency_us;
    double mapped = static_cast<double>(bucket) / (kHistogramBuckets - 1);
    // Inverse of sqrt mapping: (mapped)^2 * max_us
    double norm = mapped * mapped;
    return static_cast<int64_t>(norm * static_cast<double>(cfg_.max_latency_us));
}

// ============================================================================
// Throughput ring management
// ============================================================================

void StreamingStatsCollector::advance_throughput_epoch_locked() {
    int64_t now = now_ms();
    int64_t current_epoch = now / tp_slot_ms_;

    ThroughputSlot& cur = tp_ring_[static_cast<std::size_t>(tp_head_)];

    if (cur.epoch_ms == current_epoch) {
        // Still in the same slot — nothing to advance.
        return;
    }

    // How many slots have elapsed?
    int64_t elapsed_slots = current_epoch - cur.epoch_ms;

    if (elapsed_slots >= kThroughputSlots) {
        // All slots have aged out — zero the whole ring.
        tp_ring_.fill(ThroughputSlot{});
    } else {
        // Zero out stale intermediate slots.
        for (int64_t s = 1; s <= elapsed_slots; ++s) {
            int idx = static_cast<int>((static_cast<int64_t>(tp_head_) + s)
                      % kThroughputSlots);
            tp_ring_[static_cast<std::size_t>(idx)] = ThroughputSlot{};
        }
    }

    // Advance head to current epoch.
    tp_head_ = static_cast<int>(current_epoch % kThroughputSlots);
    tp_ring_[static_cast<std::size_t>(tp_head_)].epoch_ms = current_epoch;
}

// ============================================================================
// Data intake
// ============================================================================

void StreamingStatsCollector::record_latency(int64_t latency_us, int token_bytes) {
    bool fire_p99  = false;
    int64_t p99_snap = 0;
    double  tps_snap = 0.0;
    std::function<void(int64_t, double)> p99_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        p99_cb = cfg_.on_p99_alert;

        // Histogram update.
        if (latency_us >= cfg_.max_latency_us) {
            ++overflow_count_val_;
            overflow_count_.store(overflow_count_val_, std::memory_order_relaxed);
        } else {
            int b = bucket_for(latency_us);
            hist_[static_cast<std::size_t>(b)]++;
        }

        if (latency_us < hist_min_) hist_min_ = latency_us;
        if (latency_us > hist_max_) hist_max_ = latency_us;

        total_count_.fetch_add(1, std::memory_order_relaxed);
        if (token_bytes > 0) {
            total_bytes_.fetch_add(static_cast<uint64_t>(token_bytes),
                                   std::memory_order_relaxed);
        }

        // P99 alert check.
        if (p99_cb && cfg_.p99_alert_us > 0) {
            // Quick histogram p99 estimate.
            uint64_t tc    = total_count_.load(std::memory_order_relaxed);
            uint64_t tgt   = static_cast<uint64_t>(static_cast<double>(tc) * 0.99);
            uint64_t accum = 0;
            for (int i = 0; i < kHistogramBuckets; ++i) {
                accum += hist_[static_cast<std::size_t>(i)];
                if (accum >= tgt) {
                    p99_snap = bucket_lower(i);
                    break;
                }
            }
            if (p99_snap >= cfg_.p99_alert_us) {
                fire_p99 = true;
            }
        }
    }

    if (fire_p99 && p99_cb) {
        tps_snap = throughput_tps();
        p99_cb(p99_snap, tps_snap);
    }
}

void StreamingStatsCollector::record_arrival() {
    bool fire_gap = false;
    int64_t gap   = 0;
    std::function<void(int64_t)> gap_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        gap_cb = cfg_.on_high_gap;

        int64_t ts = now_us();
        if (last_arrival_us_ > 0) {
            gap = ts - last_arrival_us_;
            // EMA of gap.
            gap_ema_ += kGapAlpha * (static_cast<double>(gap) - gap_ema_);
            mean_gap_us_.store(gap_ema_, std::memory_order_relaxed);

            int64_t cur_max = max_gap_us_.load(std::memory_order_relaxed);
            if (gap > cur_max) {
                max_gap_us_.store(gap, std::memory_order_relaxed);
            }

            if (cfg_.gap_alert_us > 0 && gap >= cfg_.gap_alert_us) {
                fire_gap = true;
            }
        }
        last_arrival_us_ = ts;

        // Update throughput ring.
        advance_throughput_epoch_locked();
        ThroughputSlot& slot = tp_ring_[static_cast<std::size_t>(tp_head_)];
        slot.count++;
        slot.bytes += static_cast<uint64_t>(
            total_bytes_.load(std::memory_order_relaxed)); // placeholder
    }

    if (fire_gap && gap_cb) gap_cb(gap);
}

void StreamingStatsCollector::record(int64_t latency_us, int token_bytes) {
    record_arrival();
    record_latency(latency_us, token_bytes);
}

// ============================================================================
// Queries
// ============================================================================

StreamingStatsCollector::LatencyPercentiles
StreamingStatsCollector::latency_percentiles() const {
    std::lock_guard<std::mutex> lk(mutex_);

    LatencyPercentiles out;
    uint64_t tc = total_count_.load(std::memory_order_relaxed);
    out.count   = tc;
    if (tc == 0) return out;

    out.min_us = (hist_min_ == INT64_MAX) ? 0 : hist_min_;
    out.max_us = hist_max_;

    // Compute cumulative histogram for percentile lookup.
    // Percentile targets: 50, 75, 90, 95, 99.
    const double targets[] = {0.50, 0.75, 0.90, 0.95, 0.99};
    int64_t* results[]     = {&out.p50_us, &out.p75_us,
                               &out.p90_us, &out.p95_us, &out.p99_us};
    constexpr int N_TARGETS = 5;
    int target_idx = 0;

    uint64_t accum = 0;
    for (int b = 0; b < kHistogramBuckets && target_idx < N_TARGETS; ++b) {
        accum += hist_[static_cast<std::size_t>(b)];
        double frac = static_cast<double>(accum) / static_cast<double>(tc);
        while (target_idx < N_TARGETS && frac >= targets[target_idx]) {
            *results[target_idx] = bucket_lower(b);
            ++target_idx;
        }
    }
    // Fill remaining targets with max (overflow).
    while (target_idx < N_TARGETS) {
        *results[target_idx] = cfg_.max_latency_us;
        ++target_idx;
    }

    return out;
}

double StreamingStatsCollector::throughput_tps() const {
    std::lock_guard<std::mutex> lk(mutex_);

    int64_t now    = now_ms();
    int64_t window_start = now - static_cast<int64_t>(cfg_.throughput_window_ms);

    uint64_t total_in_window = 0;
    for (int i = 0; i < kThroughputSlots; ++i) {
        const auto& slot = tp_ring_[static_cast<std::size_t>(i)];
        int64_t slot_time_ms = slot.epoch_ms * tp_slot_ms_;
        if (slot_time_ms >= window_start) {
            total_in_window += slot.count;
        }
    }

    double window_sec = static_cast<double>(cfg_.throughput_window_ms) / 1000.0;
    if (window_sec <= 0.0) return 0.0;
    return static_cast<double>(total_in_window) / window_sec;
}

double StreamingStatsCollector::throughput_bps() const {
    std::lock_guard<std::mutex> lk(mutex_);

    int64_t now    = now_ms();
    int64_t window_start = now - static_cast<int64_t>(cfg_.throughput_window_ms);

    uint64_t bytes_in_window = 0;
    for (int i = 0; i < kThroughputSlots; ++i) {
        const auto& slot = tp_ring_[static_cast<std::size_t>(i)];
        int64_t slot_time_ms = slot.epoch_ms * tp_slot_ms_;
        if (slot_time_ms >= window_start) {
            bytes_in_window += slot.bytes;
        }
    }

    double window_sec = static_cast<double>(cfg_.throughput_window_ms) / 1000.0;
    if (window_sec <= 0.0) return 0.0;
    return static_cast<double>(bytes_in_window) / window_sec;
}

// ============================================================================
// Control
// ============================================================================

void StreamingStatsCollector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    hist_.fill(0);
    overflow_count_val_ = 0;
    hist_min_           = INT64_MAX;
    hist_max_           = 0;
    tp_ring_.fill(ThroughputSlot{});
    tp_head_            = 0;
    last_arrival_us_    = 0;
    gap_ema_            = 0.0;
    total_count_.store(0, std::memory_order_relaxed);
    total_bytes_.store(0, std::memory_order_relaxed);
    overflow_count_.store(0, std::memory_order_relaxed);
    mean_gap_us_.store(0.0, std::memory_order_relaxed);
    max_gap_us_.store(0, std::memory_order_relaxed);
}

void StreamingStatsCollector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_        = cfg;
    tp_slot_ms_ = std::max(int64_t{1},
        static_cast<int64_t>(cfg_.throughput_window_ms) / kThroughputSlots);
}

// ============================================================================
// JSON stats
// ============================================================================

std::string StreamingStatsCollector::to_stats_json() const {
    LatencyPercentiles pct = latency_percentiles();
    double tps = throughput_tps();
    double bps = throughput_bps();

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "{"
       << "\"total_count\":"     << pct.count        << ","
       << "\"p50_us\":"          << pct.p50_us        << ","
       << "\"p75_us\":"          << pct.p75_us        << ","
       << "\"p90_us\":"          << pct.p90_us        << ","
       << "\"p95_us\":"          << pct.p95_us        << ","
       << "\"p99_us\":"          << pct.p99_us        << ","
       << "\"min_us\":"          << pct.min_us        << ","
       << "\"max_us\":"          << pct.max_us        << ","
       << "\"overflow_count\":"  << overflow_count_.load(std::memory_order_relaxed) << ","
       << "\"throughput_tps\":"  << tps               << ","
       << "\"throughput_bps\":"  << bps               << ","
       << "\"mean_gap_us\":"     << mean_gap_us_.load(std::memory_order_relaxed) << ","
       << "\"max_gap_us\":"      << max_gap_us_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

} // namespace llmquant
