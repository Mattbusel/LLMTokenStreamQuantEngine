#pragma once

/// @file TokenImportanceDecayScheduler.h
/// @brief Applies exponential temporal decay to token weights, producing a
///        time-weighted effective sentiment signal that auto-discounts stale narrative.
///
/// ## Motivation
/// A breaking news token emitted 30 seconds ago should carry less signal weight
/// than the same token emitted just now.  Standard EMA conflates *time* with
/// *observation count*.  This scheduler applies a true wall-clock or sequence-based
/// half-life decay, so tokens become increasingly irrelevant as time passes,
/// regardless of whether new tokens arrive.
///
/// ## Algorithm
/// - `record(token, weight, timestamp_ns)` adds an entry to a sliding window.
/// - The effective weight at query time t is: `w_i * exp(-λ * (t - t_i))`
///   where λ = ln(2) / half_life_s.
/// - `effective_sentiment(query_ns)` returns Σ(decayed weights) / Σ(decayed |weights|).
/// - Window entries older than `max_age_s` are evicted.
/// - Fires `on_sentiment_flip(from, to)` when the sign of effective_sentiment changes.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_DECAY_SCHEDULER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenImportanceDecayScheduler {
public:
    struct Config {
        /// Half-life for weight decay in seconds. Default 10.0.
        double half_life_s{10.0};

        /// Maximum token age before eviction (seconds). Default 60.0.
        double max_age_s{60.0};

        /// Maximum window entries (prevents unbounded growth). Default 1024.
        int max_entries{1024};

        /// Fired when the effective sentiment sign flips.
        /// Parameters: (old_sentiment, new_sentiment).
        std::function<void(double old_sent, double new_sent)> on_sentiment_flip;
    };

    explicit TokenImportanceDecayScheduler(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a token weight at a given wall-clock timestamp.
    /// @param token        Token string (used only for diagnostics).
    /// @param weight       Raw semantic weight ∈ (−∞, ∞).
    /// @param timestamp_ns Monotonic timestamp in nanoseconds.
    void record(const std::string& token, double weight, uint64_t timestamp_ns);

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Compute the time-decayed effective sentiment at a given time.
    /// Returns 0 if the window is empty.
    [[nodiscard]] double effective_sentiment(uint64_t query_ns) const;

    /// Evict stale entries older than `max_age_s` relative to `now_ns`.
    /// Called automatically by `record`; also callable externally.
    void evict_stale(uint64_t now_ns);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of active (non-evicted) entries in the window.
    [[nodiscard]] std::size_t active_entries() const noexcept {
        return active_entries_.load(std::memory_order_relaxed);
    }

    /// Total tokens recorded (including evicted).
    [[nodiscard]] uint64_t total_recorded() const noexcept {
        return total_recorded_.load(std::memory_order_relaxed);
    }

    /// Total sentiment-sign flips observed.
    [[nodiscard]] uint64_t flip_count() const noexcept {
        return flip_count_.load(std::memory_order_relaxed);
    }

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json(uint64_t query_ns) const;

    /// Reset window.
    void reset();

    /// Update configuration (resets window).
    void update_config(const Config& cfg);

private:
    struct Entry {
        double   weight;
        uint64_t timestamp_ns;
    };

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<Entry> window_;
    double   prev_sentiment_{0.0};
    bool     sentiment_seeded_{false};

    std::atomic<std::size_t> active_entries_{0};
    std::atomic<uint64_t>    total_recorded_{0};
    std::atomic<uint64_t>    flip_count_{0};

    void evict_stale_locked(uint64_t now_ns);
};

} // namespace llmquant
