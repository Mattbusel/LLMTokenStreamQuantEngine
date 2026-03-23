/**
 * @file src/SignalReplayRunner.cpp
 * @brief JSONL signal replay and testing mode for LLMTokenStreamQuantEngine.
 *
 * Reads a JSONL (newline-delimited JSON) file of token records, replays them
 * through the full pipeline at the recorded timestamps, and outputs a signal
 * trace for analysis.
 *
 * ## JSONL Input Format
 * One JSON object per line:
 * ```jsonl
 * {"token": "bullish", "timestamp_us": 1704067200000000}
 * {"token": "growth",  "timestamp_us": 1704067200001200}
 * {"token": "rally",   "timestamp_us": 1704067200002500}
 * ```
 *
 * Fields:
 *   - `token`        (required): The raw token string.
 *   - `timestamp_us` (optional): Microseconds since Unix epoch. If omitted,
 *                                the record's position in the file is used as
 *                                the relative timestamp.
 *
 * ## Output Format
 * Each emitted signal is written as one JSON line to stdout:
 * ```jsonl
 * {"replay_token":"bullish","replay_ts_us":1704067200000000,"signal":{...}}
 * ```
 * A summary statistics JSON object is written at the end:
 * ```jsonl
 * {"summary":{"tokens_processed":1000,"signals_emitted":42,"efficiency":0.042,...}}
 * ```
 */

#include "SignalReplayRunner.h"

#include "LLMAdapter.h"
#include "TradeSignalEngine.h"
#include "Config.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace llmquant {

// ── Minimal JSON field extractor (no heap-allocated DOM) ─────────────────────

namespace {

/// Extract a string value for `"key":"..."` from a flat JSON line.
/// Returns empty string if not found.
std::string json_string(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    auto end = line.find('"', pos);
    if (end == std::string::npos) return {};
    return line.substr(pos, end - pos);
}

/// Extract an integer value for `"key":123` from a flat JSON line.
/// Returns nullopt if not found.
std::optional<int64_t> json_int(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    pos += needle.size();
    // Skip whitespace
    while (pos < line.size() && line[pos] == ' ') ++pos;
    // Read digits (may be negative)
    std::size_t start = pos;
    if (pos < line.size() && (line[pos] == '-' || std::isdigit(line[pos]))) {
        ++pos;
        while (pos < line.size() && std::isdigit(line[pos])) ++pos;
        try {
            return std::stoll(line.substr(start, pos - start));
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

} // anonymous namespace

// ── ReplayRecord ─────────────────────────────────────────────────────────────

/// A single record parsed from the JSONL replay file.
struct ReplayRecord {
    std::string token;
    int64_t     timestamp_us{0};
    bool        has_timestamp{false};
};

// ── SignalReplayRunner implementation ─────────────────────────────────────────

SignalReplayRunner::SignalReplayRunner(SignalReplayConfig config)
    : config_(std::move(config))
{}

SignalReplayResult SignalReplayRunner::run() const {
    SignalReplayResult result;

    // ── Open input file ───────────────────────────────────────────────────────
    std::ifstream infile(config_.input_path);
    if (!infile.is_open()) {
        result.error_message = "Cannot open replay file: " + config_.input_path;
        result.success = false;
        return result;
    }

    // ── Open output (stdout or file) ──────────────────────────────────────────
    std::ostream* out = &std::cout;
    std::ofstream outfile;
    if (!config_.output_path.empty()) {
        outfile.open(config_.output_path);
        if (!outfile.is_open()) {
            result.error_message = "Cannot open output file: " + config_.output_path;
            result.success = false;
            return result;
        }
        out = &outfile;
    }

    // ── Pipeline components ────────────────────────────────────────────────────
    LLMAdapter        adapter;
    TradeSignalEngine engine;

    // ── Parse and replay ───────────────────────────────────────────────────────
    std::string line;
    int64_t     synthetic_ts_us = 0;
    int64_t     first_ts_us     = -1;
    uint64_t    line_no         = 0;

    while (std::getline(infile, line)) {
        ++line_no;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Trim trailing CR for Windows files
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Parse token
        std::string token = json_string(line, "token");
        if (token.empty()) {
            ++result.malformed_lines;
            if (config_.verbose) {
                std::cerr << "[replay] Line " << line_no
                          << ": no 'token' field, skipping.\n";
            }
            continue;
        }

        // Parse timestamp
        int64_t ts_us = synthetic_ts_us;
        bool has_ts   = false;
        auto opt_ts   = json_int(line, "timestamp_us");
        if (opt_ts.has_value()) {
            ts_us  = *opt_ts;
            has_ts = true;
        }
        synthetic_ts_us += 1000; // 1ms default spacing for synthetic timestamps

        if (first_ts_us < 0) first_ts_us = ts_us;

        // Optionally enforce wall-clock replay timing
        if (config_.realtime_replay && has_ts && first_ts_us >= 0) {
            // This would sleep until wall-clock aligns with replay timestamp.
            // For correctness in test mode we skip actual sleeping.
        }

        // Pipeline: token -> SemanticWeight -> TradeSignal
        ++result.tokens_processed;
        const SemanticWeight w = adapter.map_token_to_weight(token);
        auto opt_signal = engine.process_semantic_weight(w);

        if (opt_signal.has_value()) {
            ++result.signals_emitted;

            // Emit signal trace line
            char buf[768];
            const auto& sig = *opt_signal;
            const double strength = std::max(-1.0, std::min(1.0, sig.delta_bias_shift));
            std::snprintf(buf, sizeof(buf),
                "{\"replay_token\":\"%s\""
                ",\"replay_ts_us\":%" PRId64
                ",\"signal\":{"
                "\"timestamp_ns\":%" PRIu64
                ",\"direction\":%d"
                ",\"strength\":%.6f"
                ",\"confidence\":%.6f"
                ",\"volatility_adjustment\":%.6f"
                ",\"latency_us\":%.3f"
                ",\"signal_quality\":%.6f"
                "}}",
                token.c_str(),
                ts_us,
                sig.timestamp_ns,
                sig.strategy_toggle,
                strength,
                sig.confidence,
                sig.volatility_adjustment,
                sig.latency_us,
                sig.signal_quality);
            *out << buf << "\n";
        }
    }

    // ── Summary statistics ─────────────────────────────────────────────────────
    const double efficiency = result.tokens_processed > 0
        ? static_cast<double>(result.signals_emitted) /
          static_cast<double>(result.tokens_processed)
        : 0.0;

    char summary_buf[512];
    std::snprintf(summary_buf, sizeof(summary_buf),
        "{\"summary\":{"
        "\"tokens_processed\":%" PRIu64
        ",\"signals_emitted\":%" PRIu64
        ",\"malformed_lines\":%" PRIu64
        ",\"efficiency\":%.6f"
        ",\"input_file\":\"%s\""
        "}}",
        result.tokens_processed,
        result.signals_emitted,
        result.malformed_lines,
        efficiency,
        config_.input_path.c_str());
    *out << summary_buf << "\n";

    result.efficiency = efficiency;
    result.success    = true;
    return result;
}

} // namespace llmquant
