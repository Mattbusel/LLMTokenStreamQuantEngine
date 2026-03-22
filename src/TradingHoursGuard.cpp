#include "TradingHoursGuard.h"

#include <cstring>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TradingHoursGuard::TradingHoursGuard(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// DST calculation
// ---------------------------------------------------------------------------

// Returns true if US/Eastern DST is active at the given UTC date/time.
// DST starts: 02:00 ET (= 07:00 UTC) on 2nd Sunday of March  (EDT = UTC-4).
// DST ends:   02:00 ET (= 06:00 UTC) on 1st Sunday of November (EST = UTC-5).
bool TradingHoursGuard::is_dst_active(int year, int month, int day,
                                       int hour_utc) noexcept {
    if (month < 3 || month > 11) return false;
    if (month > 3 && month < 11) return true;

    // Find the Nth Sunday of the given month in the given year.
    // Uses Zeller's congruence to find day-of-week of the 1st.
    auto nth_sunday = [&](int m, int n) -> int {
        int y = year;
        int mo = m;
        if (mo < 3) { mo += 12; --y; }
        int k = y % 100;
        int j = y / 100;
        // h: 0=Sat,1=Sun,...,6=Fri
        int h = (1 + (13 * (mo + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7;
        // Convert to 0=Sun..6=Sat
        int dow_first = (h + 6) % 7;
        int first_sunday = (dow_first == 0) ? 1 : (8 - dow_first);
        return first_sunday + (n - 1) * 7;
    };

    if (month == 3) {
        int spring = nth_sunday(3, 2);
        if (day < spring) return false;
        if (day > spring) return true;
        return hour_utc >= 7; // 07:00 UTC = 02:00 EST
    }
    // month == 11
    int fall = nth_sunday(11, 1);
    if (day < fall) return true;
    if (day > fall) return false;
    return hour_utc < 6; // 06:00 UTC = 02:00 EDT
}

// ---------------------------------------------------------------------------
// UTC -> ET conversion
// ---------------------------------------------------------------------------

TradingHoursGuard::EtTime
TradingHoursGuard::to_et(std::chrono::system_clock::time_point tp) noexcept {
    using namespace std::chrono;
    time_t tt = system_clock::to_time_t(tp);
    struct tm utc_tm{};
#ifdef _WIN32
    gmtime_s(&utc_tm, &tt);
#else
    gmtime_r(&tt, &utc_tm);
#endif
    int year  = utc_tm.tm_year + 1900;
    int month = utc_tm.tm_mon + 1;
    int day   = utc_tm.tm_mday;
    int h_utc = utc_tm.tm_hour;

    bool dst = is_dst_active(year, month, day, h_utc);
    int offset_h = dst ? -4 : -5;  // EDT=UTC-4, EST=UTC-5

    time_t et_tt = tt + static_cast<time_t>(offset_h) * 3600;
    struct tm et_tm{};
#ifdef _WIN32
    gmtime_s(&et_tm, &et_tt);
#else
    gmtime_r(&et_tm, &et_tt);
#endif

    EtTime ret{};
    ret.year    = et_tm.tm_year + 1900;
    ret.month   = et_tm.tm_mon + 1;
    ret.day     = et_tm.tm_mday;
    ret.hour    = et_tm.tm_hour;
    ret.minute  = et_tm.tm_min;
    ret.second  = et_tm.tm_sec;
    ret.weekday = et_tm.tm_wday; // 0=Sun
    ret.is_dst  = dst;
    return ret;
}

// ---------------------------------------------------------------------------
// Holiday check
// ---------------------------------------------------------------------------

bool TradingHoursGuard::is_holiday_locked(const EtTime& et) const noexcept {
    char buf[11]{};
    buf[0] = static_cast<char>('0' + et.year / 1000);
    buf[1] = static_cast<char>('0' + (et.year / 100) % 10);
    buf[2] = static_cast<char>('0' + (et.year / 10) % 10);
    buf[3] = static_cast<char>('0' + et.year % 10);
    buf[4] = '-';
    buf[5] = static_cast<char>('0' + et.month / 10);
    buf[6] = static_cast<char>('0' + et.month % 10);
    buf[7] = '-';
    buf[8] = static_cast<char>('0' + et.day / 10);
    buf[9] = static_cast<char>('0' + et.day % 10);
    std::string date_str(buf, 10);
    for (const auto& h : config_.holiday_dates) {
        if (h == date_str) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Session window check
// ---------------------------------------------------------------------------

bool TradingHoursGuard::in_session(const EtTime& et,
                                    const Session& s) noexcept {
    int now_min  = et.hour * 60 + et.minute;
    int open_min = s.open_hour  * 60 + s.open_minute;
    int cls_min  = s.close_hour * 60 + s.close_minute;
    return now_min >= open_min && now_min < cls_min;
}

// ---------------------------------------------------------------------------
// Core open/closed logic (called under mutex)
// ---------------------------------------------------------------------------

bool TradingHoursGuard::check_open_locked(const EtTime& et) const noexcept {
    if (config_.block_weekends && (et.weekday == 0 || et.weekday == 6))
        return false;
    if (is_holiday_locked(et)) return false;
    if (in_session(et, config_.regular)) return true;
    if (config_.allow_pre_market  && in_session(et, config_.pre_market))  return true;
    if (config_.allow_post_market && in_session(et, config_.post_market)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Public query
// ---------------------------------------------------------------------------

bool TradingHoursGuard::is_market_open() const noexcept {
    total_checks_.fetch_add(1, std::memory_order_relaxed);

    auto tp   = std::chrono::system_clock::now();
    EtTime et = to_et(tp);

    bool open = false;
    std::function<void(bool)> cb;
    bool fire_cb = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        open = check_open_locked(et);
        if (first_check_) {
            last_open_state_ = open;
            first_check_     = false;
        } else if (open != last_open_state_) {
            last_open_state_ = open;
            session_transitions_.fetch_add(1, std::memory_order_relaxed);
            cb      = config_.on_session_change;
            fire_cb = true;
            spdlog::info("[trading_hours] market {} at {:02d}:{:02d} ET {}",
                         open ? "OPENED" : "CLOSED",
                         et.hour, et.minute,
                         et.is_dst ? "EDT" : "EST");
        }
    }
    if (fire_cb && cb) cb(open);
    return open;
}

bool TradingHoursGuard::should_block() const noexcept {
    bool open = is_market_open();
    if (!open) signals_blocked_.fetch_add(1, std::memory_order_relaxed);
    return !open;
}

// ---------------------------------------------------------------------------
// seconds_until_next_event — scan forward minute by minute (max 7 days)
// ---------------------------------------------------------------------------

int64_t TradingHoursGuard::seconds_until_next_event() const noexcept {
    using namespace std::chrono;
    auto now  = system_clock::now();
    EtTime et = to_et(now);
    bool open_now = check_open_locked(et);

    constexpr int MAX_MINUTES = 7 * 24 * 60;
    for (int i = 1; i <= MAX_MINUTES; ++i) {
        EtTime future = to_et(now + minutes(i));
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (check_open_locked(future) != open_now) {
                return static_cast<int64_t>(i) * 60;
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// current_et_time_str
// ---------------------------------------------------------------------------

std::string TradingHoursGuard::current_et_time_str() const {
    auto et = to_et(std::chrono::system_clock::now());
    std::ostringstream s;
    s << std::setfill('0')
      << std::setw(2) << et.hour   << ':'
      << std::setw(2) << et.minute << ':'
      << std::setw(2) << et.second
      << (et.is_dst ? " EDT " : " EST ")
      << et.year << '-'
      << std::setw(2) << et.month << '-'
      << std::setw(2) << et.day;
    return s.str();
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void TradingHoursGuard::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string TradingHoursGuard::to_stats_json() const {
    bool open = is_market_open();
    std::ostringstream s;
    s << "{\"is_open\":"              << (open ? "true" : "false")
      << ",\"et_time\":\""            << current_et_time_str() << "\""
      << ",\"total_checks\":"         << total_checks()
      << ",\"signals_blocked\":"      << signals_blocked()
      << ",\"session_transitions\":"  << session_transitions()
      << "}";
    return s.str();
}

} // namespace llmquant
