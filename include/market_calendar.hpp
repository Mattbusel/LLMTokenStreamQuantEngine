#pragma once

/// @file include/market_calendar.hpp
/// @brief Market Calendar Integration — trading hours and economic event filter.
///
/// # Module: Market Calendar
///
/// ## Responsibility
/// Provide the trading engine with awareness of:
///
/// 1. **Trading sessions** — is the market currently open?  When is the next
///    open/close?  What session type (pre-market, regular, after-hours, closed)?
///
/// 2. **Economic event suppression** — disable signal emission for a
///    configurable window around high-impact events (FOMC, NFP, CPI, etc.)
///    to avoid trading into known volatility spikes.
///
/// ## Markets Supported
/// - NYSE / NASDAQ equities (Mon–Fri 09:30–16:00 ET, excluding holidays)
/// - CME futures (Mon–Fri 17:00–16:00 CT next day, ~23 h session)
/// - Crypto 24 × 7 (always open)
///
/// ## Economic Events
/// A minimal 2026 calendar is hardcoded.  The calendar can also load events
/// from a JSON file at runtime (path set via ``LLMQUANT_ECON_EVENTS`` env var
/// or passed to ``EconomicEventFilter::load_json``).
///
/// ## RiskManager Integration
/// ``RiskManager::evaluate()`` should call
/// ``MarketCalendar::should_suppress_signal()`` and block the signal if it
/// returns true.
///
/// ## Thread Safety
/// All public methods are const and thread-safe (no mutable state).

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llmquant {

// ─── Market identifiers ───────────────────────────────────────────────────────

enum class Market {
    NYSE_NASDAQ, ///< US equity exchanges (ET timezone)
    CME,         ///< CME futures (CT timezone)
    Crypto,      ///< 24/7 crypto markets
};

/// Human-readable string for a Market.
[[nodiscard]] const char* to_string(Market m) noexcept;

// ─── SessionType ──────────────────────────────────────────────────────────────

enum class SessionType {
    PreMarket,   ///< US equities before 09:30 ET
    Regular,     ///< Primary trading session
    AfterHours,  ///< US equities after 16:00 ET
    Closed,      ///< Market is closed (weekend, holiday, or overnight gap)
};

/// Human-readable string for a SessionType.
[[nodiscard]] const char* to_string(SessionType s) noexcept;

// ─── TradingSession ───────────────────────────────────────────────────────────

/// Snapshot of the current session state for a given market.
struct TradingSession {
    Market      market{Market::NYSE_NASDAQ};  ///< Which market this describes.
    SessionType session_type{SessionType::Closed}; ///< Current session classification.
    bool        is_open{false};              ///< True iff market is in regular session.
    uint64_t    next_open_ms{0};             ///< Unix ms until next regular open.
    uint64_t    next_close_ms{0};            ///< Unix ms until next regular close.

    /// Format a one-liner for logging.
    [[nodiscard]] std::string to_string() const;
};

// ─── MarketCalendar ───────────────────────────────────────────────────────────

/// Provides session queries for major trading markets.
///
/// Usage:
/// ```cpp
/// MarketCalendar cal;
/// auto session = cal.get_session(Market::NYSE_NASDAQ, now_ms);
/// if (!session.is_open) {
///     fmt::print("Market closed, next open in {} ms\n", session.next_open_ms);
/// }
/// bool should_trade = cal.is_regular_session(Market::NYSE_NASDAQ, now_ms);
/// ```
///
/// All times are expressed as Unix milliseconds (uint64_t).
class MarketCalendar {
public:
    MarketCalendar() = default;

    /// Return the session state for the given market at the given time.
    ///
    /// @param market   Market to query.
    /// @param now_ms   Current time as Unix milliseconds.
    /// @returns        TradingSession snapshot.
    [[nodiscard]] TradingSession
    get_session(Market market, uint64_t now_ms) const noexcept;

    /// Convenience: return true iff the market is in a regular trading session.
    ///
    /// Crypto is always considered open.
    [[nodiscard]] bool
    is_regular_session(Market market, uint64_t now_ms) const noexcept;

    /// Return true iff ``now_ms`` falls on a US market holiday (NYSE calendar).
    ///
    /// Covers all NYSE-observed holidays for 2025–2026.
    [[nodiscard]] static bool is_us_holiday(uint64_t now_ms) noexcept;

    /// Return true iff ``now_ms`` is a weekend (Saturday or Sunday UTC).
    [[nodiscard]] static bool is_weekend(uint64_t now_ms) noexcept;

private:
    /// Compute the next NYSE open/close timestamps from ``now_ms``.
    struct NextBoundaries {
        uint64_t next_open_ms{0};
        uint64_t next_close_ms{0};
    };
    [[nodiscard]] static NextBoundaries
    nyse_boundaries(uint64_t now_ms) noexcept;

    /// Get NYSE session type for a given UTC-millisecond timestamp.
    [[nodiscard]] static SessionType
    nyse_session_type(uint64_t now_ms) noexcept;
};

// ─── EconomicEvent ────────────────────────────────────────────────────────────

/// A single high-impact economic event that warrants signal suppression.
struct EconomicEvent {
    std::string name;            ///< Event name (e.g. "FOMC Meeting").
    uint64_t    timestamp_ms{0}; ///< Scheduled event time (Unix ms, UTC).
    uint64_t    pre_window_ms{30ULL * 60 * 1000};  ///< Suppress this many ms before.
    uint64_t    post_window_ms{30ULL * 60 * 1000}; ///< Suppress this many ms after.
    bool        high_impact{true}; ///< Only high-impact events suppress by default.

    /// Return true if ``now_ms`` falls within the suppression window.
    [[nodiscard]] bool is_suppressed(uint64_t now_ms) const noexcept;
};

// ─── EconomicEventFilter ──────────────────────────────────────────────────────

/// Suppresses signals around major economic events.
///
/// The filter is pre-loaded with hardcoded 2026 FOMC, NFP, and CPI dates.
/// Additional events can be loaded from a JSON file.
///
/// Usage:
/// ```cpp
/// EconomicEventFilter filter;
/// filter.load_defaults_2026();
///
/// if (filter.should_suppress(now_ms)) {
///     // skip signal emission
/// }
/// ```
class EconomicEventFilter {
public:
    EconomicEventFilter() = default;

    /// Populate the event list with all known 2026 FOMC, NFP, and CPI dates.
    ///
    /// This is idempotent: calling multiple times is safe.
    void load_defaults_2026();

    /// Attempt to load additional events from a JSON file.
    ///
    /// Expected JSON format:
    /// ```json
    /// [
    ///   {"name": "FOMC", "timestamp_ms": 1234567890000,
    ///    "pre_window_ms": 1800000, "post_window_ms": 1800000}
    /// ]
    /// ```
    ///
    /// @param json_path  Path to the JSON file.
    /// @returns          Number of events loaded, or -1 on error.
    int load_json(const std::string& json_path);

    /// Return true iff ``now_ms`` falls within any event's suppression window.
    ///
    /// @param now_ms   Current time as Unix milliseconds.
    /// @returns        true if signal emission should be suppressed.
    [[nodiscard]] bool should_suppress(uint64_t now_ms) const noexcept;

    /// Return the event that is currently suppressing signals (if any).
    ///
    /// @param now_ms  Current time as Unix milliseconds.
    /// @returns       Pointer to the active event, or nullptr if not suppressed.
    [[nodiscard]] const EconomicEvent*
    active_event(uint64_t now_ms) const noexcept;

    /// Add a single event to the filter list.
    void add_event(EconomicEvent event);

    /// Return all registered events.
    [[nodiscard]] const std::vector<EconomicEvent>& events() const noexcept {
        return events_;
    }

    /// Remove all events from the filter.
    void clear() noexcept { events_.clear(); }

private:
    std::vector<EconomicEvent> events_;
};

// ─── Integrated calendar query ─────────────────────────────────────────────────

/// Combined query: should the system suppress a signal at ``now_ms``?
///
/// Returns true if:
///   - The specified market is not in a regular session, OR
///   - An economic event is currently in its suppression window.
///
/// @param calendar  MarketCalendar instance.
/// @param filter    EconomicEventFilter instance.
/// @param market    Market to check session for.
/// @param now_ms    Current Unix milliseconds.
/// @param require_regular_session  If false, only economic events suppress.
[[nodiscard]] bool should_suppress_signal(
    const MarketCalendar&       calendar,
    const EconomicEventFilter&  filter,
    Market                      market,
    uint64_t                    now_ms,
    bool                        require_regular_session = true
) noexcept;

}  // namespace llmquant
