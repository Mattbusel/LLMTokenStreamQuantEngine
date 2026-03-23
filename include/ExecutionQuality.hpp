#pragma once

/// @file ExecutionQuality.hpp
/// @brief Lock-free ring-buffer execution quality tracker with nanosecond
///        latency percentiles, fill-rate accounting, and realized alpha.
///
/// ## Overview
/// `ExecutionQualityTracker` records completed signal executions into a
/// lock-free ring buffer of up to **10 000** entries and provides:
///
///   - `mean_slippage_bps()`  — mean fill slippage in basis points
///   - `p99_latency_ns()`     — 99th-percentile signal-to-fill latency (ns)
///   - `fill_rate()`          — fraction of signals that got an execution record
///   - `signal_alpha(n)`      — realised alpha of the last N completed trades
///
/// ## Lock-free ring buffer
/// Writes use a single atomic write-cursor (`head_`) advanced with
/// `fetch_add(1, seq_cst)`.  Each slot contains an `ExecutionRecord`
/// written in a single struct assignment (no cross-slot synchronisation
/// required because each slot is written exactly once per rotation and
/// `capacity` is a power of two).  Reads for statistics snapshot acquire
/// a consistent `head_` position and scan backwards without a mutex.
///
/// ## Signal alpha computation
/// For each `ExecutionRecord` the realised per-signal alpha is:
///   alpha_i = (actual_price - expected_price) * sign(direction)
/// where direction is inferred from `signal_id` parity (or from the
/// `direction` field in `ExecutionRecord`).  The mean over the last N
/// records is returned as a fractional return.
///
/// ## Thread safety
/// `record()` is safe to call from multiple threads simultaneously.
/// `mean_slippage_bps()`, `p99_latency_ns()`, `fill_rate()`, and
/// `signal_alpha()` take a consistent snapshot of the ring and are safe to
/// call concurrently with `record()`.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_EXECUTION_QUALITY` (default ON — shared with
/// ExecutionQualityMonitor.h).

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// ExecutionRecord
// ---------------------------------------------------------------------------

/**
 * @brief Complete record for a single signal → execution round-trip.
 */
struct ExecutionRecord {
    /// Unique signal identifier (from TradeSignal::timestamp_ns or a counter).
    uint64_t signal_id{0};

    /// Wall-clock nanoseconds when the trade signal was emitted.
    int64_t signal_time_ns{0};

    /// Wall-clock nanoseconds when the OMS acknowledged the fill.
    int64_t execution_time_ns{0};

    /// Mid-price or model price at signal emission time.
    double expected_price{0.0};

    /// Actual fill price reported by the OMS.
    double actual_price{0.0};

    /**
     * @brief Slippage in basis points.
     *
     * Positive = filled at worse price than expected.
     * Pre-computed for performance: slippage_bps = (actual - expected) /
     * expected * 10000 * direction_sign.
     * If not pre-computed by the caller, set to 0.0 and the tracker will
     * compute it from expected_price / actual_price.
     */
    double slippage_bps{0.0};

    /**
     * @brief Trade direction: +1 buy, -1 sell.  Used for alpha computation.
     * Default +1.
     */
    int8_t direction{1};

    /**
     * @brief True if this record represents a filled signal; false if the
     * signal was cancelled, rejected, or timed out (contributes to fill rate
     * denominator but not numerator).
     */
    bool filled{true};
};

// ---------------------------------------------------------------------------
// ExecutionQualityTracker
// ---------------------------------------------------------------------------

/**
 * @brief Lock-free ring-buffer execution quality tracker (max 10K records).
 */
class ExecutionQualityTracker {
public:
    /// Hard capacity of the ring buffer.
    static constexpr uint32_t kCapacity = 10'000;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    ExecutionQualityTracker();

    // Non-copyable (owns atomic state and fixed ring buffer).
    ExecutionQualityTracker(const ExecutionQualityTracker&)            = delete;
    ExecutionQualityTracker& operator=(const ExecutionQualityTracker&) = delete;

    // -----------------------------------------------------------------------
    // Record ingestion
    // -----------------------------------------------------------------------

    /**
     * @brief Record a completed execution into the ring buffer.
     *
     * Lock-free and safe to call from any thread.
     *
     * If `rec.slippage_bps == 0.0` and both prices are non-zero the tracker
     * auto-computes slippage from expected/actual prices.
     *
     * @param rec Execution record to store.
     */
    void record(ExecutionRecord rec);

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /**
     * @brief Mean slippage in basis points across all ring entries.
     *
     * Scans the current ring snapshot.  O(kCapacity).
     */
    [[nodiscard]] double mean_slippage_bps() const;

    /**
     * @brief 99th-percentile signal-to-fill latency in nanoseconds.
     *
     * Computed via partial sort over the current ring snapshot.  O(N log N).
     */
    [[nodiscard]] int64_t p99_latency_ns() const;

    /**
     * @brief Fill rate: fraction of recorded signals that were filled.
     *
     * fill_rate = filled_count / total_recorded.
     * Returns 0.0 if no records exist.
     */
    [[nodiscard]] double fill_rate() const noexcept;

    /**
     * @brief Realised alpha over the last `lookback_n` filled records.
     *
     * Alpha per record = (actual_price - expected_price) * direction / expected_price.
     * Returns the arithmetic mean across `min(lookback_n, available_filled)` records.
     *
     * @param lookback_n Number of most-recent filled records to include.
     *                   Clamped to kCapacity.
     * @return Mean fractional return (positive = favourable fills).
     */
    [[nodiscard]] double signal_alpha(int lookback_n) const;

    // -----------------------------------------------------------------------
    // Counters
    // -----------------------------------------------------------------------

    /// Total records ever written (including overwritten slots).
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_written_.load(std::memory_order_relaxed);
    }

    /// Number of filled records in the current ring snapshot.
    [[nodiscard]] uint64_t filled_in_ring() const noexcept;

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// Reset all ring-buffer slots and counters.
    void reset();

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

private:
    // -----------------------------------------------------------------------
    // Ring buffer
    // -----------------------------------------------------------------------

    /// Atomic write cursor.  Monotonically increasing; slot = head % kCapacity.
    alignas(64) std::atomic<uint64_t> head_{0};

    /// Total fills written (mirrors head_ semantically, kept for clarity).
    alignas(64) std::atomic<uint64_t> total_written_{0};

    /// Fixed-capacity ring buffer.  Slots are written with relaxed stores
    /// after the slot index is reserved via head_ fetch_add.
    alignas(64) ExecutionRecord ring_[kCapacity]{};

    // -----------------------------------------------------------------------
    // Snapshot helpers
    // -----------------------------------------------------------------------

    /// Take a consistent snapshot of up to kCapacity most-recent records.
    void snapshot(std::vector<ExecutionRecord>& out) const;

    /// Auto-compute slippage_bps from prices if not pre-provided.
    static double compute_slippage(const ExecutionRecord& r) noexcept;
};

} // namespace llmquant
