#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief NYSE/NASDAQ market-hours gating for signal emission.
 *
 * Converts UTC wall-clock to US/Eastern (ET) using a rule-based DST
 * calculation (no OS timezone database required), then compares against
 * configurable session windows.  Signals outside market hours — or on
 * weekends and named holiday blackout dates — are flagged for suppression.
 *
 * ## Typical usage
 * @code
 * TradingHoursGuard guard;
 * // In the signal callback:
 * if (!guard.is_market_open()) { ++signals_blocked; return; }
 * // ... emit signal ...
 * @endcode
 *
 * ## DST rules applied
 * US DST begins at 02:00 ET on the second Sunday of March  (+1 h → EDT = UTC-4).
 * US DST ends   at 02:00 ET on the first  Sunday of November (–1 h → EST = UTC-5).
 * These rules have been stable since the Energy Policy Act of 2005.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TRADING_HOURS` (default ON).
 */
class TradingHoursGuard {
public:
    /**
     * @brief A single intra-day trading session window (in ET).
     */
    struct Session {
        int open_hour{9};      ///< Hour in ET (24-h clock).
        int open_minute{30};   ///< Minute in ET.
        int close_hour{16};    ///< Hour in ET.
        int close_minute{0};   ///< Minute in ET.
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Regular trading session (default: NYSE 09:30–16:00 ET).
         */
        Session regular{9, 30, 16, 0};

        /**
         * @brief Pre-market session (default: 04:00–09:30 ET).
         *
         * Only active when allow_pre_market is true.
         */
        Session pre_market{4, 0, 9, 30};

        /**
         * @brief Post-market session (default: 16:00–20:00 ET).
         *
         * Only active when allow_post_market is true.
         */
        Session post_market{16, 0, 20, 0};

        /**
         * @brief If true, allow signals during the pre-market session.
         *
         * Default false (regular session only).
         */
        bool allow_pre_market{false};

        /**
         * @brief If true, allow signals during the post-market session.
         *
         * Default false (regular session only).
         */
        bool allow_post_market{false};

        /**
         * @brief If true, block all signals on Saturday and Sunday.
         *
         * Default true.
         */
        bool block_weekends{true};

        /**
         * @brief Holiday blackout dates in YYYY-MM-DD format (ET calendar).
         *
         * Any date listed here is treated as a full market close regardless
         * of day-of-week.  Example: "2025-07-04" (Independence Day).
         */
        std::vector<std::string> holiday_dates;

        /**
         * @brief Callback invoked when the market transitions open → closed
         *        or closed → open.
         *
         * Parameter: true = market just opened; false = market just closed.
         * Called from the thread that calls is_market_open() or should_block().
         * Must not block.
         */
        std::function<void(bool)> on_session_change;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TradingHoursGuard(Config config = Config{});

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Return true if the market is currently open (signals should pass).
     *
     * Internally updates the open/closed state and fires on_session_change if
     * the state just transitioned.  Thread-safe.
     */
    [[nodiscard]] bool is_market_open() const noexcept;

    /**
     * @brief Return true if signals should be blocked (inverse of is_market_open).
     *
     * Increments the internal blocked counter.  Thread-safe.
     */
    [[nodiscard]] bool should_block() const noexcept;

    /**
     * @brief Seconds until the next session open or close event.
     *
     * Returns a positive number of seconds.  Returns 0 if the computation
     * overflows or the next event cannot be determined.
     */
    [[nodiscard]] int64_t seconds_until_next_event() const noexcept;

    /**
     * @brief Human-readable current ET time string "HH:MM:SS (EST|EDT) YYYY-MM-DD".
     */
    [[nodiscard]] std::string current_et_time_str() const;

    // -----------------------------------------------------------------------
    // Runtime configuration
    // -----------------------------------------------------------------------

    /**
     * @brief Replace the active configuration at runtime.
     *
     * Thread-safe.  The previous open/closed state is preserved so that a
     * reconfiguration mid-session does not spuriously re-fire on_session_change.
     */
    void update_config(const Config& config);

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total calls to should_block() that returned true. */
    [[nodiscard]] uint64_t signals_blocked() const noexcept {
        return signals_blocked_.load(std::memory_order_relaxed);
    }

    /** @brief Total calls to is_market_open() / should_block(). */
    [[nodiscard]] uint64_t total_checks() const noexcept {
        return total_checks_.load(std::memory_order_relaxed);
    }

    /** @brief Number of open↔closed session transitions detected. */
    [[nodiscard]] uint64_t session_transitions() const noexcept {
        return session_transitions_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise current stats to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex    mutex_;
    Config                config_;
    mutable bool          last_open_state_{false};
    mutable bool          first_check_{true};

    mutable std::atomic<uint64_t> signals_blocked_{0};
    mutable std::atomic<uint64_t> total_checks_{0};
    mutable std::atomic<uint64_t> session_transitions_{0};

    // -----------------------------------------------------------------------
    // Internal helpers (called under mutex)
    // -----------------------------------------------------------------------

    struct EtTime {
        int  year, month, day;        // Gregorian date in ET
        int  hour, minute, second;    // Time of day in ET
        int  weekday;                 // 0=Sun … 6=Sat
        bool is_dst;                  // true = EDT (UTC-4), false = EST (UTC-5)
    };

    [[nodiscard]] static EtTime to_et(std::chrono::system_clock::time_point tp) noexcept;
    [[nodiscard]] static bool   is_dst_active(int year, int month, int day,
                                              int hour_utc) noexcept;
    [[nodiscard]] bool check_open_locked(const EtTime& et) const noexcept;
    [[nodiscard]] bool is_holiday_locked(const EtTime& et) const noexcept;
    [[nodiscard]] static bool in_session(const EtTime& et,
                                         const Session& s) noexcept;
};

} // namespace llmquant
