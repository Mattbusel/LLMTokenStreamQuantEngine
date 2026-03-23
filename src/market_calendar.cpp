/// @file src/market_calendar.cpp
/// @brief Implementation of MarketCalendar and EconomicEventFilter.
///
/// See include/market_calendar.hpp for full API documentation.
///
/// ## Time conventions
/// All times are expressed as Unix milliseconds (uint64_t, UTC).  NYSE/NASDAQ
/// trading hours are Eastern Time (ET = UTC−5 in winter, UTC−4 in summer).
/// This implementation uses a simplified static offset and does not perform a
/// full DST calculation; an ET_OFFSET_MS constant is used for approximate
/// correctness.  For production use, link against a tzdata library.

#include "market_calendar.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ─── Time helpers ─────────────────────────────────────────────────────────────

namespace {

/// Milliseconds in common time units.
static constexpr uint64_t MS_PER_SECOND = 1000ULL;
static constexpr uint64_t MS_PER_MINUTE = 60ULL * MS_PER_SECOND;
static constexpr uint64_t MS_PER_HOUR   = 60ULL * MS_PER_MINUTE;
static constexpr uint64_t MS_PER_DAY    = 24ULL * MS_PER_HOUR;

/// Approximate Eastern Time offset (UTC−5 hours, winter).
/// For DST (UTC−4), subtract MS_PER_HOUR less — simplified here.
static constexpr uint64_t ET_OFFSET_MS  = 5ULL * MS_PER_HOUR;

/// Convert Unix ms to "minutes since midnight ET" (approximate, no DST).
uint64_t et_minutes_since_midnight(uint64_t now_ms) noexcept {
    const uint64_t et_ms = (now_ms >= ET_OFFSET_MS)
                         ? (now_ms - ET_OFFSET_MS)
                         : 0;
    const uint64_t day_ms = et_ms % MS_PER_DAY;
    return day_ms / MS_PER_MINUTE;
}

/// Return the day-of-week for a Unix ms timestamp: 0=Sunday … 6=Saturday.
int day_of_week(uint64_t now_ms) noexcept {
    // Unix epoch (1970-01-01) was a Thursday = 4.
    const uint64_t days_since_epoch = now_ms / MS_PER_DAY;
    return static_cast<int>((days_since_epoch + 4) % 7);
}

/// NYSE regular session opens at 09:30 ET = 570 minutes from midnight ET.
static constexpr uint64_t NYSE_OPEN_MIN  = 570;
/// NYSE regular session closes at 16:00 ET = 960 minutes from midnight ET.
static constexpr uint64_t NYSE_CLOSE_MIN = 960;
/// Pre-market starts at 04:00 ET = 240 minutes.
static constexpr uint64_t NYSE_PRE_MIN   = 240;
/// After-hours ends at 20:00 ET = 1200 minutes.
static constexpr uint64_t NYSE_AH_MIN    = 1200;

}  // anonymous namespace

// ─── Market / SessionType strings ─────────────────────────────────────────────

const char* to_string(Market m) noexcept {
    switch (m) {
        case Market::NYSE_NASDAQ: return "NYSE_NASDAQ";
        case Market::CME:         return "CME";
        case Market::Crypto:      return "Crypto";
    }
    return "Unknown";
}

const char* to_string(SessionType s) noexcept {
    switch (s) {
        case SessionType::PreMarket:  return "PreMarket";
        case SessionType::Regular:    return "Regular";
        case SessionType::AfterHours: return "AfterHours";
        case SessionType::Closed:     return "Closed";
    }
    return "Unknown";
}

// ─── TradingSession ───────────────────────────────────────────────────────────

std::string TradingSession::to_string() const {
    std::ostringstream oss;
    oss << "TradingSession["
        << "market=" << llmquant::to_string(market)
        << " type=" << llmquant::to_string(session_type)
        << " is_open=" << (is_open ? "true" : "false")
        << " next_open_ms=" << next_open_ms
        << " next_close_ms=" << next_close_ms
        << "]";
    return oss.str();
}

// ─── MarketCalendar ───────────────────────────────────────────────────────────

bool MarketCalendar::is_weekend(uint64_t now_ms) noexcept {
    const int dow = day_of_week(now_ms);
    return (dow == 0 || dow == 6);  // Sunday=0, Saturday=6
}

bool MarketCalendar::is_us_holiday(uint64_t now_ms) noexcept {
    // Hardcoded NYSE holidays for 2025 and 2026.
    // Format: Unix milliseconds at 00:00:00 UTC on the holiday date.
    // We check if now_ms falls on the same UTC calendar day as any holiday.

    // Extract UTC date components from now_ms.
    const time_t t = static_cast<time_t>(now_ms / 1000ULL);
    struct tm gm;
#ifdef _WIN32
    gmtime_s(&gm, &t);
#else
    gmtime_r(&t, &gm);
#endif
    const int year  = gm.tm_year + 1900;
    const int month = gm.tm_mon + 1;   // 1-indexed
    const int day   = gm.tm_mday;

    // NYSE holidays 2025 (observed dates).
    if (year == 2025) {
        // New Year's Day
        if (month == 1  && day == 1)  return true;
        // MLK Day
        if (month == 1  && day == 20) return true;
        // Presidents' Day
        if (month == 2  && day == 17) return true;
        // Good Friday
        if (month == 4  && day == 18) return true;
        // Memorial Day
        if (month == 5  && day == 26) return true;
        // Juneteenth
        if (month == 6  && day == 19) return true;
        // Independence Day
        if (month == 7  && day == 4)  return true;
        // Labor Day
        if (month == 9  && day == 1)  return true;
        // Thanksgiving
        if (month == 11 && day == 27) return true;
        // Christmas
        if (month == 12 && day == 25) return true;
    }

    // NYSE holidays 2026 (observed dates).
    if (year == 2026) {
        // New Year's Day (observed Fri Jan 2 since Jan 1 is Thursday)
        if (month == 1  && day == 1)  return true;
        if (month == 1  && day == 2)  return true;
        // MLK Day
        if (month == 1  && day == 19) return true;
        // Presidents' Day
        if (month == 2  && day == 16) return true;
        // Good Friday
        if (month == 4  && day == 3)  return true;
        // Memorial Day
        if (month == 5  && day == 25) return true;
        // Juneteenth (observed Mon June 19)
        if (month == 6  && day == 19) return true;
        // Independence Day (observed Fri July 3 since July 4 is Saturday)
        if (month == 7  && day == 3)  return true;
        // Labor Day
        if (month == 9  && day == 7)  return true;
        // Thanksgiving
        if (month == 11 && day == 26) return true;
        // Christmas (observed Fri Dec 25)
        if (month == 12 && day == 25) return true;
    }

    return false;
}

SessionType MarketCalendar::nyse_session_type(uint64_t now_ms) noexcept {
    if (is_weekend(now_ms) || is_us_holiday(now_ms)) return SessionType::Closed;

    const uint64_t min = et_minutes_since_midnight(now_ms);
    if (min < NYSE_PRE_MIN)   return SessionType::Closed;
    if (min < NYSE_OPEN_MIN)  return SessionType::PreMarket;
    if (min < NYSE_CLOSE_MIN) return SessionType::Regular;
    if (min < NYSE_AH_MIN)    return SessionType::AfterHours;
    return SessionType::Closed;
}

MarketCalendar::NextBoundaries MarketCalendar::nyse_boundaries(
    uint64_t now_ms
) noexcept {
    // Compute the UTC ms for 09:30 ET today.
    // ET offset is 5 h (simplification, ignoring DST).
    const uint64_t day_start_utc = (now_ms / MS_PER_DAY) * MS_PER_DAY;
    const uint64_t open_utc  = day_start_utc + ET_OFFSET_MS + NYSE_OPEN_MIN  * MS_PER_MINUTE;
    const uint64_t close_utc = day_start_utc + ET_OFFSET_MS + NYSE_CLOSE_MIN * MS_PER_MINUTE;

    uint64_t next_open  = open_utc;
    uint64_t next_close = close_utc;

    // If past open, next open is tomorrow's (skip weekends/holidays).
    if (now_ms >= open_utc) {
        next_open += MS_PER_DAY;
        int tries = 0;
        while ((is_weekend(next_open) || is_us_holiday(next_open)) && tries++ < 7) {
            next_open += MS_PER_DAY;
        }
    }

    // If past close today, next close is tomorrow.
    if (now_ms >= close_utc) {
        next_close += MS_PER_DAY;
        int tries = 0;
        while ((is_weekend(next_close) || is_us_holiday(next_close)) && tries++ < 7) {
            next_close += MS_PER_DAY;
        }
    }

    // Return offsets from now.
    NextBoundaries nb;
    nb.next_open_ms  = (next_open  > now_ms) ? (next_open  - now_ms) : 0;
    nb.next_close_ms = (next_close > now_ms) ? (next_close - now_ms) : 0;
    return nb;
}

TradingSession MarketCalendar::get_session(
    Market market, uint64_t now_ms
) const noexcept {
    TradingSession s;
    s.market = market;

    switch (market) {
        case Market::Crypto:
            s.session_type  = SessionType::Regular;
            s.is_open       = true;
            s.next_open_ms  = 0;
            s.next_close_ms = 0;
            break;

        case Market::NYSE_NASDAQ: {
            s.session_type  = nyse_session_type(now_ms);
            s.is_open       = (s.session_type == SessionType::Regular);
            auto bounds     = nyse_boundaries(now_ms);
            s.next_open_ms  = bounds.next_open_ms;
            s.next_close_ms = bounds.next_close_ms;
            break;
        }

        case Market::CME: {
            // CME Globex: Mon–Fri 17:00 CT previous day to 16:00 CT (~23 h).
            // Simplified: treat as open when NYSE is open or after-hours.
            const SessionType nyse_st = nyse_session_type(now_ms);
            const bool cme_open =
                (nyse_st == SessionType::Regular ||
                 nyse_st == SessionType::AfterHours ||
                 nyse_st == SessionType::PreMarket);
            s.session_type  = cme_open ? SessionType::Regular : SessionType::Closed;
            s.is_open       = cme_open;
            auto bounds     = nyse_boundaries(now_ms);
            s.next_open_ms  = bounds.next_open_ms;
            s.next_close_ms = bounds.next_close_ms;
            break;
        }
    }
    return s;
}

bool MarketCalendar::is_regular_session(Market market, uint64_t now_ms) const noexcept {
    if (market == Market::Crypto) return true;
    return get_session(market, now_ms).is_open;
}

// ─── EconomicEvent ────────────────────────────────────────────────────────────

bool EconomicEvent::is_suppressed(uint64_t now_ms) const noexcept {
    if (timestamp_ms == 0) return false;
    const bool before = (timestamp_ms > now_ms) &&
                        ((timestamp_ms - now_ms) <= pre_window_ms);
    const bool after  = (now_ms >= timestamp_ms) &&
                        ((now_ms - timestamp_ms) <= post_window_ms);
    return before || after;
}

// ─── EconomicEventFilter ──────────────────────────────────────────────────────

void EconomicEventFilter::add_event(EconomicEvent event) {
    events_.push_back(std::move(event));
}

bool EconomicEventFilter::should_suppress(uint64_t now_ms) const noexcept {
    return active_event(now_ms) != nullptr;
}

const EconomicEvent* EconomicEventFilter::active_event(uint64_t now_ms) const noexcept {
    for (const auto& ev : events_) {
        if (ev.high_impact && ev.is_suppressed(now_ms)) {
            return &ev;
        }
    }
    return nullptr;
}

/// Helper: convert a "YYYY-MM-DD HH:MM UTC" string to Unix milliseconds.
/// Very minimal parser — production code should use a proper date library.
static uint64_t parse_datetime_ms(const char* s) noexcept {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0;
    if (std::sscanf(s, "%d-%d-%d %d:%d", &y, &mo, &d, &h, &mi) != 5) return 0;
    // Approximate: days from 1970-01-01.
    // Month lengths (non-leap-year).
    static const int month_days[13] = {
        0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    };
    int days = (y - 1970) * 365 + (y - 1969) / 4;  // rough leap-year correction
    for (int m = 1; m < mo; ++m) {
        days += month_days[m];
    }
    // Leap year Feb correction.
    if (mo > 2 && (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0))) ++days;
    days += d - 1;
    const uint64_t total_ms =
        static_cast<uint64_t>(days) * 24ULL * 3600ULL * 1000ULL +
        static_cast<uint64_t>(h)  * 3600ULL * 1000ULL +
        static_cast<uint64_t>(mi) * 60ULL   * 1000ULL;
    return total_ms;
}

void EconomicEventFilter::load_defaults_2026() {
    // 2026 FOMC meeting dates (decision day, announcement ~14:00 ET = 19:00 UTC).
    // Source: Federal Reserve calendar (approximate).
    static const char* fomc_2026[] = {
        "2026-01-28 19:00", "2026-03-18 18:00", "2026-05-06 18:00",
        "2026-06-17 18:00", "2026-07-29 18:00", "2026-09-16 18:00",
        "2026-11-04 19:00", "2026-12-16 19:00",
    };
    for (const auto* s : fomc_2026) {
        add_event(EconomicEvent{
            .name          = "FOMC Meeting",
            .timestamp_ms  = parse_datetime_ms(s),
            .pre_window_ms  = 30ULL * 60 * 1000,
            .post_window_ms = 30ULL * 60 * 1000,
            .high_impact   = true,
        });
    }

    // 2026 Non-Farm Payrolls (first Friday of each month, 08:30 ET = 13:30 UTC).
    static const char* nfp_2026[] = {
        "2026-01-09 13:30", "2026-02-06 13:30", "2026-03-06 13:30",
        "2026-04-03 13:30", "2026-05-08 13:30", "2026-06-05 13:30",
        "2026-07-10 13:30", "2026-08-07 13:30", "2026-09-04 13:30",
        "2026-10-02 13:30", "2026-11-06 13:30", "2026-12-04 13:30",
    };
    for (const auto* s : nfp_2026) {
        add_event(EconomicEvent{
            .name          = "Non-Farm Payrolls",
            .timestamp_ms  = parse_datetime_ms(s),
            .pre_window_ms  = 30ULL * 60 * 1000,
            .post_window_ms = 30ULL * 60 * 1000,
            .high_impact   = true,
        });
    }

    // 2026 CPI reports (mid-month, typically second or third Wednesday, 08:30 ET).
    static const char* cpi_2026[] = {
        "2026-01-14 13:30", "2026-02-11 13:30", "2026-03-11 13:30",
        "2026-04-15 13:30", "2026-05-13 13:30", "2026-06-10 13:30",
        "2026-07-15 13:30", "2026-08-12 13:30", "2026-09-09 13:30",
        "2026-10-14 13:30", "2026-11-12 13:30", "2026-12-09 13:30",
    };
    for (const auto* s : cpi_2026) {
        add_event(EconomicEvent{
            .name          = "CPI Report",
            .timestamp_ms  = parse_datetime_ms(s),
            .pre_window_ms  = 30ULL * 60 * 1000,
            .post_window_ms = 30ULL * 60 * 1000,
            .high_impact   = true,
        });
    }
}

int EconomicEventFilter::load_json(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) return -1;

    // Minimal JSON array parser — does not use a full JSON library to keep
    // dependencies lightweight.  Handles the documented format only.
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    int loaded = 0;
    std::size_t pos = 0;

    // Helper lambdas.
    auto skip_ws = [&]() {
        while (pos < content.size() && std::isspace((unsigned char)content[pos])) ++pos;
    };
    auto read_string = [&]() -> std::string {
        skip_ws();
        if (pos >= content.size() || content[pos] != '"') return {};
        ++pos;
        std::string s;
        while (pos < content.size() && content[pos] != '"') {
            s += content[pos++];
        }
        ++pos;  // closing '"'
        return s;
    };
    auto read_uint64 = [&]() -> uint64_t {
        skip_ws();
        uint64_t v = 0;
        while (pos < content.size() && std::isdigit((unsigned char)content[pos])) {
            v = v * 10 + static_cast<uint64_t>(content[pos++] - '0');
        }
        return v;
    };

    while (pos < content.size()) {
        skip_ws();
        if (pos >= content.size()) break;

        if (content[pos] == '{') {
            ++pos;
            EconomicEvent ev;
            ev.pre_window_ms  = 30ULL * 60 * 1000;
            ev.post_window_ms = 30ULL * 60 * 1000;
            ev.high_impact    = true;

            while (pos < content.size() && content[pos] != '}') {
                std::string key = read_string();
                skip_ws();
                if (pos < content.size() && content[pos] == ':') ++pos;
                skip_ws();

                if (key == "name") {
                    ev.name = read_string();
                } else if (key == "timestamp_ms") {
                    ev.timestamp_ms = read_uint64();
                } else if (key == "pre_window_ms") {
                    ev.pre_window_ms = read_uint64();
                } else if (key == "post_window_ms") {
                    ev.post_window_ms = read_uint64();
                } else {
                    // skip value
                    while (pos < content.size() &&
                           content[pos] != ',' && content[pos] != '}') ++pos;
                }
                skip_ws();
                if (pos < content.size() && content[pos] == ',') ++pos;
            }
            if (pos < content.size()) ++pos;  // skip '}'
            if (!ev.name.empty() && ev.timestamp_ms > 0) {
                add_event(std::move(ev));
                ++loaded;
            }
        } else {
            ++pos;
        }
    }
    return loaded;
}

// ─── should_suppress_signal ───────────────────────────────────────────────────

bool should_suppress_signal(
    const MarketCalendar&      calendar,
    const EconomicEventFilter& filter,
    Market                     market,
    uint64_t                   now_ms,
    bool                       require_regular_session
) noexcept {
    if (require_regular_session && !calendar.is_regular_session(market, now_ms)) {
        return true;
    }
    if (filter.should_suppress(now_ms)) {
        return true;
    }
    return false;
}

}  // namespace llmquant
