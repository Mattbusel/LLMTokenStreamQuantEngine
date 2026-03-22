#include <gtest/gtest.h>

#ifdef LLMQUANT_AUDIT_LOG_ENABLED

#include "SignalAuditLog.h"
#include "TradeSignalEngine.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

namespace {

// Build a minimal TradeSignal with all required fields populated.
llmquant::TradeSignal make_signal(double bias = 0.1,
                                  double conf  = 0.8,
                                  bool   with_ts = true)
{
    llmquant::TradeSignal s;
    s.delta_bias_shift      = bias;
    s.volatility_adjustment = 0.02;
    s.spread_modifier       = 0.001;
    s.confidence            = conf;
    s.signal_quality        = conf * std::abs(bias);
    s.latency_us            = 5.0;
    s.strategy_toggle       = (bias >= 0) ? 1 : -1;
    s.strategy_weight       = 0.5;
    if (with_ts) {
        s.timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        s.timestamp = std::chrono::high_resolution_clock::now();
    }
    return s;
}

// Read all text from a file.
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, ConstructDestruct) {
    const std::string path = "test_audit_construct.ndjson";
    std::remove(path.c_str());
    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath = path;
        llmquant::SignalAuditLog log(cfg);
        EXPECT_TRUE(log.is_running());
        EXPECT_EQ(log.records_enqueued(), 0u);
        EXPECT_EQ(log.records_written(),  0u);
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Basic write — one passed signal
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, WritePassedSignal) {
    const std::string path = "test_audit_passed.ndjson";
    std::remove(path.c_str());

    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath     = path;
        cfg.log_passed   = true;
        cfg.log_blocked  = false;
        llmquant::SignalAuditLog log(cfg);

        auto sig = make_signal(0.15, 0.9);
        log.log_signal(sig, true, "");

        // Give the writer thread time to flush.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }  // destructor joins writer thread

    std::string content = read_file(path);
    EXPECT_FALSE(content.empty()) << "Audit file should not be empty";
    EXPECT_NE(content.find("\"passed\":true"), std::string::npos);
    EXPECT_NE(content.find("\"reason\":\"\""),  std::string::npos);
    EXPECT_NE(content.find("\"bias\":"),        std::string::npos);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Basic write — one blocked signal
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, WriteBlockedSignal) {
    const std::string path = "test_audit_blocked.ndjson";
    std::remove(path.c_str());

    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath     = path;
        cfg.log_passed   = false;
        cfg.log_blocked  = true;
        llmquant::SignalAuditLog log(cfg);

        auto sig = make_signal(-0.3, 0.6);
        log.log_signal(sig, false, "magnitude");

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::string content = read_file(path);
    EXPECT_FALSE(content.empty());
    EXPECT_NE(content.find("\"passed\":false"),     std::string::npos);
    EXPECT_NE(content.find("\"reason\":\"magnitude\""), std::string::npos);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Filter: log_passed=false blocks passed signals
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, FilterPassed) {
    const std::string path = "test_audit_filter_passed.ndjson";
    std::remove(path.c_str());

    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath    = path;
        cfg.log_passed  = false;
        cfg.log_blocked = false;
        llmquant::SignalAuditLog log(cfg);

        log.log_signal(make_signal(0.1, 0.8), true,  "");
        log.log_signal(make_signal(0.2, 0.7), false, "rate");
        EXPECT_EQ(log.records_enqueued(), 0u);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // File should be absent or empty (never opened by the writer).
    std::string content = read_file(path);
    EXPECT_TRUE(content.empty()) << "No records should be written when both filters are off";
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Multiple signals — all lines are valid JSON with a newline separator
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, MultipleSignalsAllNewlines) {
    const std::string path = "test_audit_multi.ndjson";
    std::remove(path.c_str());

    constexpr int kCount = 10;
    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath = path;
        llmquant::SignalAuditLog log(cfg);

        for (int i = 0; i < kCount; ++i) {
            log.log_signal(make_signal(0.01 * i, 0.5 + 0.04 * i),
                           i % 2 == 0, i % 2 != 0 ? "rate" : "");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::string content = read_file(path);
    // Count newline-separated lines.
    int lines = 0;
    for (char c : content) if (c == '\n') ++lines;
    EXPECT_EQ(lines, kCount) << "Expected " << kCount << " NDJSON records";

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Counter accuracy
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, CounterAccuracy) {
    const std::string path = "test_audit_counters.ndjson";
    std::remove(path.c_str());

    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath = path;
        llmquant::SignalAuditLog log(cfg);

        for (int i = 0; i < 5; ++i)
            log.log_signal(make_signal(), true, "");

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        EXPECT_EQ(log.records_enqueued(), 5u);
        // Allow async write to complete.
    }

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// JSON escaping: special chars in reason don't break the file
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, JsonEscapingInReason) {
    const std::string path = "test_audit_escape.ndjson";
    std::remove(path.c_str());

    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath    = path;
        cfg.log_blocked = true;
        llmquant::SignalAuditLog log(cfg);

        // Reason string with JSON-special characters.
        log.log_signal(make_signal(), false, "gate\t\"name\"\nnewline");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::string content = read_file(path);
    // File should be non-empty.
    EXPECT_FALSE(content.empty());
    // Raw tab/newline/unescaped-quote must NOT appear inside the JSON string.
    EXPECT_EQ(content.find("\t"),  std::string::npos);
    // The backslash-escaped versions should appear.
    EXPECT_NE(content.find("\\t"),  std::string::npos);
    EXPECT_NE(content.find("\\n"),  std::string::npos);
    EXPECT_NE(content.find("\\\""), std::string::npos);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Rotation: write enough bytes to trigger one rotation
// ---------------------------------------------------------------------------

TEST(SignalAuditLog, RotationOccurs) {
    const std::string path     = "test_audit_rotate.ndjson";
    const std::string rotated  = path + ".1";
    std::remove(path.c_str());
    std::remove(rotated.c_str());

    {
        llmquant::SignalAuditLog::Config cfg;
        cfg.filepath      = path;
        cfg.rotate_bytes  = 512;   // tiny limit to force rotation quickly
        cfg.keep_files    = 1;
        llmquant::SignalAuditLog log(cfg);

        // Write enough records to exceed the 512-byte limit.
        for (int i = 0; i < 20; ++i)
            log.log_signal(make_signal(0.01 * i, 0.5), true, "");

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        EXPECT_GE(log.rotations(), 1);
    }

    // After rotation the rotated file should exist.
    std::ifstream f(rotated);
    EXPECT_TRUE(f.good()) << "Rotated file '" << rotated << "' should exist after rotation";

    std::remove(path.c_str());
    std::remove(rotated.c_str());
}

#else  // LLMQUANT_AUDIT_LOG_ENABLED not defined

TEST(SignalAuditLog, DisabledAtBuildTime) {
    // When the audit log feature is compiled out, just verify the test binary
    // links and this placeholder test passes.
    SUCCEED();
}

#endif  // LLMQUANT_AUDIT_LOG_ENABLED
