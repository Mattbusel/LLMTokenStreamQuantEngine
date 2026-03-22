#pragma once

/// @file SignalLifecycleTracker.h
/// @brief Tracks each emitted signal's lifecycle (birth → peak → decay → death)
///        and computes empirical signal half-lives.
///
/// ## Motivation
/// Not all signals age at the same rate.  An earnings-surprise signal may decay
/// in 30 seconds while a macro-narrative shift signal may persist for hours.
/// This module maintains a running record of active signals, updating their state
/// on each return observation.  When a signal's value decays below a threshold
/// (relative to its peak), it is declared "dead" and its half-life is recorded.
/// Persistently living signals ("zombies") are flagged when they exceed
/// `max_expected_halflife` without decaying.
///
/// ## Algorithm
/// - `record_signal(id, bias)` creates / updates a signal entry.
/// - `record_return(ret)` advances the decay clock for all active signals.
/// - A signal dies when `current_bias / peak_bias < decay_fraction`.
/// - Half-life is the time from birth to when bias first crosses 50% of peak.
/// - `on_zombie(id, age_s)` fires when an active signal exceeds `max_age_s`.
/// - `on_death(id, halflife_s, peak_bias)` fires on signal expiry.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_LIFECYCLE_TRACKER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llmquant {

class SignalLifecycleTracker {
public:
    struct Config {
        /// Fraction of peak bias at which a signal is declared dead. Default 0.1.
        double decay_fraction{0.1};

        /// Fraction of peak bias at which the half-life clock stops. Default 0.5.
        double halflife_fraction{0.5};

        /// Maximum signal age before flagging as zombie (seconds). Default 300.0.
        double max_age_s{300.0};

        /// Fired when a signal decays below decay_fraction × peak.
        /// Parameters: (signal_id, halflife_s, peak_bias).
        std::function<void(const std::string& id, double halflife_s, double peak_bias)>
            on_death;

        /// Fired when a signal exceeds max_age_s without decaying.
        /// Parameters: (signal_id, age_s).
        std::function<void(const std::string& id, double age_s)> on_zombie;
    };

    explicit SignalLifecycleTracker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record or update a signal's current bias.
    void record_signal(const std::string& id, double bias);

    /// Advance decay clocks and fire lifecycle callbacks.
    /// Call once per "tick" (e.g., every return observation).
    void tick();

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of currently active signals.
    [[nodiscard]] std::size_t active_signal_count() const noexcept {
        return active_count_.load(std::memory_order_relaxed);
    }

    /// Total signals that have died.
    [[nodiscard]] uint64_t dead_count() const noexcept {
        return dead_count_.load(std::memory_order_relaxed);
    }

    /// Total zombie signals flagged.
    [[nodiscard]] uint64_t zombie_count() const noexcept {
        return zombie_count_.load(std::memory_order_relaxed);
    }

    /// Mean empirical half-life of completed signals (seconds).
    [[nodiscard]] double mean_halflife_s() const noexcept {
        return mean_halflife_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    struct SignalEntry {
        double   birth_bias;
        double   peak_bias;
        double   current_bias;
        std::chrono::steady_clock::time_point birth_time;
        std::chrono::steady_clock::time_point halflife_time;
        bool     halflife_recorded;
        bool     zombie_fired;
    };

    mutable std::mutex mutex_;
    Config cfg_;

    std::unordered_map<std::string, SignalEntry> signals_;

    // Online mean for half-life.
    double   halflife_sum_{0.0};
    uint64_t halflife_n_{0};

    std::atomic<std::size_t> active_count_{0};
    std::atomic<uint64_t>    dead_count_{0};
    std::atomic<uint64_t>    zombie_count_{0};
    std::atomic<double>      mean_halflife_{0.0};
};

} // namespace llmquant
