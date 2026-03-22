#include "CrossSessionMemory.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Minimal JSON helpers — avoids pulling in yaml-cpp / nlohmann for this tiny
// state file.
// ---------------------------------------------------------------------------

namespace {

// Extract a double value from a flat JSON key.
// Returns def if key not found.
double json_double(const std::string& j, const char* key, double def = 0.0) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return def;
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) ++pos;
    try { return std::stod(j.substr(pos)); } catch (...) { return def; }
}

uint64_t json_uint64(const std::string& j, const char* key, uint64_t def = 0) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return def;
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) ++pos;
    try { return static_cast<uint64_t>(std::stoull(j.substr(pos))); } catch (...) { return def; }
}

bool json_has_key(const std::string& j, const char* key) {
    return j.find(std::string("\"") + key + "\":") != std::string::npos;
}

} // namespace

// ---------------------------------------------------------------------------
// ISO-8601 timestamp
// ---------------------------------------------------------------------------

std::string CrossSessionMemory::make_iso8601_now() {
    auto now   = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32]{};
#ifdef _WIN32
    struct tm tm_val{};
    gmtime_s(&tm_val, &t);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_val);
#else
    struct tm tm_val{};
    gmtime_r(&t, &tm_val);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_val);
#endif
    return buf;
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

std::string CrossSessionMemory::to_json() const {
    std::ostringstream o;
    o << std::fixed;
    o.precision(10);
    o << "{\n"
      << "  \"session\":"       << session_       << ",\n"
      << "  \"saved_at\":\""    << last_save_time_ << "\",\n";
    if (has_kelly_) {
        o << "  \"kelly\":{\n"
          << "    \"total_outcomes\":"  << kelly_state_.total_outcomes  << ",\n"
          << "    \"win_count\":"       << kelly_state_.win_count        << ",\n"
          << "    \"win_rate_ema\":"    << kelly_state_.win_rate_ema    << ",\n"
          << "    \"avg_win_ema\":"     << kelly_state_.avg_win_ema     << ",\n"
          << "    \"avg_loss_ema\":"    << kelly_state_.avg_loss_ema    << "\n"
          << "  },\n";
    }
    if (has_drawdown_) {
        o << "  \"drawdown\":{\n"
          << "    \"cumulative_pnl\":" << drawdown_state_.cumulative_pnl << ",\n"
          << "    \"peak_pnl\":"       << drawdown_state_.peak_pnl       << "\n"
          << "  },\n";
    }
    o << "  \"_version\":\"1.0\"\n"
      << "}\n";
    return o.str();
}

// ---------------------------------------------------------------------------
// load / save
// ---------------------------------------------------------------------------

bool CrossSessionMemory::load() {
    std::ifstream f(cfg_.path);
    if (!f.is_open()) {
        return cfg_.ignore_missing;  // missing → ok if ignore_missing
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    if (!parse_json(json)) return false;
    loaded_ = true;
    if (cfg_.on_load) cfg_.on_load(cfg_.path, session_);
    return true;
}

bool CrossSessionMemory::parse_json(const std::string& j) {
    if (j.empty() || j[0] != '{') return false;

    session_        = json_uint64(j, "session", 0);

    has_kelly_ = json_has_key(j, "kelly");
    if (has_kelly_) {
        kelly_state_.total_outcomes = json_uint64(j, "total_outcomes", 0);
        kelly_state_.win_count      = json_uint64(j, "win_count",      0);
        kelly_state_.win_rate_ema   = json_double(j, "win_rate_ema",   0.5);
        kelly_state_.avg_win_ema    = json_double(j, "avg_win_ema",    0.01);
        kelly_state_.avg_loss_ema   = json_double(j, "avg_loss_ema",   0.01);
    }

    has_drawdown_ = json_has_key(j, "drawdown");
    if (has_drawdown_) {
        drawdown_state_.cumulative_pnl = json_double(j, "cumulative_pnl", 0.0);
        drawdown_state_.peak_pnl       = json_double(j, "peak_pnl",       0.0);
    }

    return true;
}

bool CrossSessionMemory::save() {
    ++session_;
    last_save_time_ = make_iso8601_now();

    std::string json = to_json();

    // Write to temp file then rename for atomic update.
    std::string tmp_path = cfg_.path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::trunc);
        if (!f.is_open()) return false;
        f << json;
        if (!f.good()) return false;
    }

#ifdef _WIN32
    // On Windows, rename fails if destination exists; remove first.
    std::remove(cfg_.path.c_str());
#endif
    if (std::rename(tmp_path.c_str(), cfg_.path.c_str()) != 0) return false;

    if (cfg_.on_save) cfg_.on_save(cfg_.path, session_);
    return true;
}

// ---------------------------------------------------------------------------
// Kelly state
// ---------------------------------------------------------------------------

bool CrossSessionMemory::restore_kelly(KellyState& state) const {
    if (!has_kelly_) return false;
    state = kelly_state_;
    return true;
}

void CrossSessionMemory::capture_kelly(const KellyState& state) {
    kelly_state_ = state;
    has_kelly_   = true;
}

// ---------------------------------------------------------------------------
// Drawdown state
// ---------------------------------------------------------------------------

bool CrossSessionMemory::restore_drawdown(DrawdownState& state) const {
    if (!has_drawdown_) return false;
    state = drawdown_state_;
    return true;
}

void CrossSessionMemory::capture_drawdown(const DrawdownState& state) {
    drawdown_state_ = state;
    has_drawdown_   = true;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string CrossSessionMemory::to_stats_json() const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"loaded\":%s,\"session\":%llu,\"has_kelly\":%s,"
        "\"has_drawdown\":%s,\"last_save\":\"%s\"}",
        loaded_        ? "true" : "false",
        static_cast<unsigned long long>(session_),
        has_kelly_     ? "true" : "false",
        has_drawdown_  ? "true" : "false",
        last_save_time_.c_str());
    return buf;
}

} // namespace llmquant
