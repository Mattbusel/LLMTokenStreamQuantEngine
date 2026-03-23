#include "ExecutionQuality.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ExecutionQualityTracker::ExecutionQualityTracker() {
    // Zero-init all ring slots.
    for (auto& slot : ring_) {
        slot = ExecutionRecord{};
    }
}

// ---------------------------------------------------------------------------
// Slippage auto-computation
// ---------------------------------------------------------------------------

double ExecutionQualityTracker::compute_slippage(const ExecutionRecord& r) noexcept {
    if (r.slippage_bps != 0.0) return r.slippage_bps;
    if (r.expected_price <= 0.0 || r.actual_price <= 0.0) return 0.0;
    // slippage_bps = (actual - expected) / expected * 10000 * direction_sign
    // Positive = adverse fill (paid more than expected on a buy, or received
    // less than expected on a sell).
    double raw = (r.actual_price - r.expected_price) / r.expected_price * 10'000.0;
    return raw * static_cast<double>(r.direction);
}

// ---------------------------------------------------------------------------
// record() — lock-free ring-buffer insert
// ---------------------------------------------------------------------------

void ExecutionQualityTracker::record(ExecutionRecord rec) {
    // Auto-fill slippage if not supplied.
    if (rec.slippage_bps == 0.0) {
        rec.slippage_bps = compute_slippage(rec);
    }

    // Atomically reserve a slot.  Sequential consistency ensures visibility
    // ordering across producers without a mutex.
    uint64_t slot_idx =
        head_.fetch_add(1, std::memory_order_seq_cst) % kCapacity;

    // Write the record to the reserved slot.  This is safe because:
    //   1. Each slot_idx is produced by exactly one writer at a time.
    //   2. Consumers snapshot after reading head_, so they only read slots
    //      whose slot_idx < acquired_head.
    ring_[slot_idx] = rec;
    total_written_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Snapshot helper
// ---------------------------------------------------------------------------

void ExecutionQualityTracker::snapshot(std::vector<ExecutionRecord>& out) const {
    // Acquire the current write cursor.
    uint64_t h = head_.load(std::memory_order_acquire);
    if (h == 0) {
        out.clear();
        return;
    }

    // Clamp to ring size.
    uint64_t n = (h < kCapacity) ? h : kCapacity;
    out.resize(static_cast<size_t>(n));

    for (uint64_t i = 0; i < n; ++i) {
        // Read in chronological order (oldest first).
        uint64_t slot = (h >= kCapacity) ? ((h - n + i) % kCapacity) : i;
        out[i] = ring_[slot];  // relaxed read; may race with a concurrent writer
                                // on the same slot but will produce a complete
                                // struct value (no torn reads on modern ABIs).
    }
}

// ---------------------------------------------------------------------------
// mean_slippage_bps
// ---------------------------------------------------------------------------

double ExecutionQualityTracker::mean_slippage_bps() const {
    std::vector<ExecutionRecord> snap;
    snapshot(snap);
    if (snap.empty()) return 0.0;

    double sum = 0.0;
    int    cnt = 0;
    for (const auto& r : snap) {
        if (r.filled) {
            double slip = (r.slippage_bps != 0.0)
                              ? r.slippage_bps
                              : compute_slippage(r);
            sum += slip;
            ++cnt;
        }
    }
    return (cnt > 0) ? (sum / cnt) : 0.0;
}

// ---------------------------------------------------------------------------
// p99_latency_ns
// ---------------------------------------------------------------------------

int64_t ExecutionQualityTracker::p99_latency_ns() const {
    std::vector<ExecutionRecord> snap;
    snapshot(snap);

    std::vector<int64_t> lats;
    lats.reserve(snap.size());
    for (const auto& r : snap) {
        if (r.filled && r.execution_time_ns > r.signal_time_ns) {
            lats.push_back(r.execution_time_ns - r.signal_time_ns);
        }
    }
    if (lats.empty()) return 0;

    size_t p99_idx = static_cast<size_t>(0.99 * static_cast<double>(lats.size() - 1));
    // nth_element is O(N) and sufficient for a one-sided percentile.
    std::nth_element(lats.begin(), lats.begin() + static_cast<ptrdiff_t>(p99_idx), lats.end());
    return lats[p99_idx];
}

// ---------------------------------------------------------------------------
// fill_rate
// ---------------------------------------------------------------------------

double ExecutionQualityTracker::fill_rate() const noexcept {
    uint64_t tw = total_written_.load(std::memory_order_relaxed);
    if (tw == 0) return 0.0;

    // Count filled records in the current ring snapshot.
    uint64_t filled = filled_in_ring();
    uint64_t denom  = (tw < kCapacity) ? tw : kCapacity;
    return static_cast<double>(filled) / static_cast<double>(denom);
}

uint64_t ExecutionQualityTracker::filled_in_ring() const noexcept {
    uint64_t h = head_.load(std::memory_order_acquire);
    if (h == 0) return 0;
    uint64_t n = (h < kCapacity) ? h : kCapacity;
    uint64_t filled = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t slot = (h >= kCapacity) ? ((h - n + i) % kCapacity) : i;
        if (ring_[slot].filled) ++filled;
    }
    return filled;
}

// ---------------------------------------------------------------------------
// signal_alpha
// ---------------------------------------------------------------------------

double ExecutionQualityTracker::signal_alpha(int lookback_n) const {
    if (lookback_n <= 0) return 0.0;

    std::vector<ExecutionRecord> snap;
    snapshot(snap);

    // Collect filled records in reverse-chronological order (newest first).
    std::vector<double> alphas;
    alphas.reserve(static_cast<size_t>(std::min(lookback_n, static_cast<int>(kCapacity))));

    for (auto it = snap.rbegin(); it != snap.rend(); ++it) {
        if (!it->filled) continue;
        if (static_cast<int>(alphas.size()) >= lookback_n) break;

        double exp_px = it->expected_price;
        double act_px = it->actual_price;

        if (exp_px <= 0.0 || act_px <= 0.0) continue;

        // Fractional alpha: positive means favourable execution.
        double alpha = (act_px - exp_px) / exp_px *
                       static_cast<double>(it->direction);
        alphas.push_back(alpha);
    }

    if (alphas.empty()) return 0.0;
    return std::accumulate(alphas.begin(), alphas.end(), 0.0) /
           static_cast<double>(alphas.size());
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void ExecutionQualityTracker::reset() {
    head_.store(0, std::memory_order_seq_cst);
    total_written_.store(0, std::memory_order_relaxed);
    for (auto& slot : ring_) {
        slot = ExecutionRecord{};
    }
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string ExecutionQualityTracker::to_stats_json() const {
    double  mean_slip = mean_slippage_bps();
    int64_t p99_lat   = p99_latency_ns();
    double  fr        = fill_rate();
    uint64_t tw       = total_written_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    o << "{"
      << "\"total_recorded\":"    << tw
      << ",\"fill_rate\":"        << fr
      << ",\"mean_slippage_bps\":" << mean_slip
      << ",\"p99_latency_ns\":"   << p99_lat
      << ",\"signal_alpha_50\":"  << signal_alpha(50)
      << "}";
    return o.str();
}

} // namespace llmquant
