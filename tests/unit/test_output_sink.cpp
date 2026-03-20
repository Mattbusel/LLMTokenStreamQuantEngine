#include "gtest/gtest.h"
#include "OutputSinkImpl.h"
#include "TradeSignalEngine.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <sstream>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TradeSignal make_signal(double bias, double vol, uint64_t ts_ns = 1000) {
    TradeSignal s;
    s.timestamp_ns          = ts_ns;
    s.timestamp             = std::chrono::high_resolution_clock::now();
    s.delta_bias_shift      = bias;
    s.volatility_adjustment = vol;
    s.spread_modifier       = 0.01;
    s.confidence            = 0.75;
    s.latency_us            = 5.0;
    s.strategy_toggle       = 1;
    s.strategy_weight       = 0.8;
    return s;
}

// ---------------------------------------------------------------------------
// MemoryOutputSink
// ---------------------------------------------------------------------------

TEST(OutputSinkTest, test_memory_sink_emit_stores_signals) {
    MemoryOutputSink sink;
    EXPECT_TRUE(sink.get_signals().empty());

    sink.emit(make_signal(0.5, 0.3));
    EXPECT_EQ(sink.get_signals().size(), 1u);

    sink.emit(make_signal(-0.2, 0.7));
    EXPECT_EQ(sink.get_signals().size(), 2u);
}

TEST(OutputSinkTest, test_memory_sink_get_signals_returns_all_emitted) {
    MemoryOutputSink sink;

    sink.emit(make_signal( 0.9, 0.1, 100));
    sink.emit(make_signal(-0.4, 0.6, 200));
    sink.emit(make_signal( 0.1, 0.2, 300));

    const auto& signals = sink.get_signals();
    ASSERT_EQ(signals.size(), 3u);

    EXPECT_DOUBLE_EQ(signals[0].delta_bias_shift, 0.9);
    EXPECT_EQ(signals[0].timestamp_ns, 100u);

    EXPECT_DOUBLE_EQ(signals[1].delta_bias_shift, -0.4);
    EXPECT_EQ(signals[1].timestamp_ns, 200u);

    EXPECT_DOUBLE_EQ(signals[2].delta_bias_shift, 0.1);
    EXPECT_EQ(signals[2].timestamp_ns, 300u);
}

TEST(OutputSinkTest, test_memory_sink_clear_empties_buffer) {
    MemoryOutputSink sink;
    sink.emit(make_signal(1.0, 0.5));
    ASSERT_EQ(sink.get_signals().size(), 1u);

    sink.clear();
    EXPECT_TRUE(sink.get_signals().empty());
}

// ---------------------------------------------------------------------------
// CsvOutputSink
// ---------------------------------------------------------------------------

TEST(OutputSinkTest, test_csv_sink_emit_writes_to_file) {
    const std::string path = "/tmp/test_output_sink.csv";

    {
        CsvOutputSink sink(path);
        sink.emit(make_signal(0.7,  0.4, 999));
        sink.emit(make_signal(-0.3, 0.2, 1001));
        sink.flush();
    }

    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Header must be present.
    EXPECT_NE(content.find("timestamp_ns"), std::string::npos);
    // First signal's timestamp must be present.
    EXPECT_NE(content.find("999"),          std::string::npos);
    // Second signal's timestamp must be present.
    EXPECT_NE(content.find("1001"),         std::string::npos);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// JsonOutputSink
// ---------------------------------------------------------------------------

TEST(OutputSinkTest, test_json_sink_emit_writes_to_file) {
    const std::string path = "/tmp/test_output_sink.json";

    {
        JsonOutputSink sink(path);
        sink.emit(make_signal(0.5, 0.8, 42));
        sink.flush();
    }

    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Must be valid-looking JSON with expected keys.
    EXPECT_NE(content.find("{"),               std::string::npos);
    EXPECT_NE(content.find("timestamp_ns"),    std::string::npos);
    EXPECT_NE(content.find("delta_bias_shift"),std::string::npos);
    EXPECT_NE(content.find("42"),              std::string::npos);

    std::remove(path.c_str());
}

TEST(OutputSinkTest, test_csv_sink_nonexistent_directory_throws) {
    EXPECT_THROW(
        CsvOutputSink sink("/nonexistent_dir_xyz/out.csv"),
        std::runtime_error
    );
}

TEST(OutputSinkTest, test_json_sink_nonexistent_directory_throws) {
    EXPECT_THROW(
        JsonOutputSink sink("/nonexistent_dir_xyz/out.json"),
        std::runtime_error
    );
}

TEST(OutputSinkTest, test_csv_sink_nan_and_inf_replaced_with_zero) {
    const std::string path = "/tmp/test_output_sink_nan.csv";
    {
        CsvOutputSink sink(path);
        TradeSignal s = make_signal(0.0, 0.0);
        s.delta_bias_shift      = std::numeric_limits<double>::quiet_NaN();
        s.volatility_adjustment = std::numeric_limits<double>::infinity();
        s.latency_us            = -std::numeric_limits<double>::infinity();
        sink.emit(s);
        sink.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // NaN and Inf must have been replaced with 0 — no "nan" or "inf" in output.
    EXPECT_EQ(content.find("nan"), std::string::npos);
    EXPECT_EQ(content.find("inf"), std::string::npos);
    std::remove(path.c_str());
}

TEST(OutputSinkTest, test_json_sink_nan_and_inf_replaced_with_zero) {
    const std::string path = "/tmp/test_output_sink_nan.json";
    {
        JsonOutputSink sink(path);
        TradeSignal s = make_signal(0.0, 0.0);
        s.delta_bias_shift      = std::numeric_limits<double>::quiet_NaN();
        s.volatility_adjustment = std::numeric_limits<double>::infinity();
        sink.emit(s);
        sink.flush();
    }
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    EXPECT_EQ(content.find("nan"), std::string::npos);
    EXPECT_EQ(content.find("inf"), std::string::npos);
    std::remove(path.c_str());
}

TEST(OutputSinkTest, test_memory_sink_size_tracks_signal_count) {
    MemoryOutputSink sink;
    EXPECT_EQ(sink.size(), 0u);
    sink.emit(make_signal(0.1, 0.2));
    EXPECT_EQ(sink.size(), 1u);
    sink.emit(make_signal(0.3, 0.4));
    EXPECT_EQ(sink.size(), 2u);
    sink.clear();
    EXPECT_EQ(sink.size(), 0u);
}

TEST(OutputSinkTest, test_memory_sink_capacity_cap_drops_oldest) {
    MemoryOutputSink sink(3);  // max 3 signals
    EXPECT_EQ(sink.dropped_count(), 0u);

    sink.emit(make_signal(0.1, 0.1, 100));
    sink.emit(make_signal(0.2, 0.1, 200));
    sink.emit(make_signal(0.3, 0.1, 300));
    EXPECT_EQ(sink.size(), 3u);
    EXPECT_EQ(sink.dropped_count(), 0u);

    // 4th signal evicts the oldest (ts=100).
    sink.emit(make_signal(0.4, 0.1, 400));
    EXPECT_EQ(sink.size(), 3u);
    EXPECT_EQ(sink.dropped_count(), 1u);
    EXPECT_EQ(sink.get_signals()[0].timestamp_ns, 200u);
    EXPECT_EQ(sink.get_signals()[2].timestamp_ns, 400u);

    // 5th signal evicts ts=200.
    sink.emit(make_signal(0.5, 0.1, 500));
    EXPECT_EQ(sink.size(), 3u);
    EXPECT_EQ(sink.dropped_count(), 2u);
    EXPECT_EQ(sink.get_signals()[0].timestamp_ns, 300u);
}

TEST(OutputSinkTest, test_memory_sink_capacity_zero_is_unlimited) {
    MemoryOutputSink sink(0);  // 0 = unlimited
    for (int i = 0; i < 1000; ++i) {
        sink.emit(make_signal(0.1, 0.1, static_cast<uint64_t>(i)));
    }
    EXPECT_EQ(sink.size(), 1000u);
    EXPECT_EQ(sink.dropped_count(), 0u);
}

TEST(OutputSinkTest, test_memory_sink_clear_resets_dropped_count) {
    MemoryOutputSink sink(2);
    sink.emit(make_signal(0.1, 0.1, 1));
    sink.emit(make_signal(0.2, 0.1, 2));
    sink.emit(make_signal(0.3, 0.1, 3));  // drops ts=1
    EXPECT_EQ(sink.dropped_count(), 1u);
    sink.clear();
    EXPECT_EQ(sink.dropped_count(), 0u);
    EXPECT_EQ(sink.size(), 0u);
}

TEST(OutputSinkTest, test_csv_sink_emit_count_tracks_records_written) {
    const std::string path = "/tmp/test_csv_emit_count.csv";
    {
        CsvOutputSink sink(path);
        EXPECT_EQ(sink.emit_count(), 0u);
        sink.emit(make_signal(0.1, 0.2, 10));
        EXPECT_EQ(sink.emit_count(), 1u);
        sink.emit(make_signal(0.3, 0.4, 20));
        sink.emit(make_signal(0.5, 0.6, 30));
        EXPECT_EQ(sink.emit_count(), 3u);
    }
    std::remove(path.c_str());
}

TEST(OutputSinkTest, test_json_sink_emit_count_tracks_records_written) {
    const std::string path = "/tmp/test_json_emit_count.json";
    {
        JsonOutputSink sink(path);
        EXPECT_EQ(sink.emit_count(), 0u);
        for (int i = 0; i < 5; ++i) {
            sink.emit(make_signal(0.1 * i, 0.1, static_cast<uint64_t>(i)));
        }
        EXPECT_EQ(sink.emit_count(), 5u);
    }
    std::remove(path.c_str());
}

TEST(OutputSinkTest, test_memory_sink_emit_count_tracks_all_including_dropped) {
    MemoryOutputSink sink(2);  // capacity of 2
    TradeSignal sig;
    sig.delta_bias_shift = 0.1; sig.confidence = 0.8;
    sink.emit(sig); sink.emit(sig); sink.emit(sig);  // 3 emits, 1 dropped
    EXPECT_EQ(sink.emit_count(), 3u)
        << "emit_count must count all emits including dropped ones";
    EXPECT_EQ(sink.dropped_count(), 1u)
        << "dropped_count must reflect the one evicted signal";
    EXPECT_EQ(sink.size(), 2u)
        << "size must not exceed capacity";
}

TEST(OutputSinkTest, test_memory_sink_drop_rate_correct) {
    MemoryOutputSink sink(2);
    TradeSignal sig;
    sig.delta_bias_shift = 0.1; sig.confidence = 0.8;
    sink.emit(sig); sink.emit(sig); sink.emit(sig); sink.emit(sig);  // 4 emits, 2 dropped
    EXPECT_DOUBLE_EQ(sink.drop_rate(), 0.5)
        << "drop_rate must be 0.5 when half of emits are dropped";
}

TEST(OutputSinkTest, test_memory_sink_drop_rate_zero_no_drops) {
    MemoryOutputSink sink(0);  // unlimited
    TradeSignal sig;
    sig.delta_bias_shift = 0.1; sig.confidence = 0.8;
    sink.emit(sig); sink.emit(sig);
    EXPECT_DOUBLE_EQ(sink.drop_rate(), 0.0)
        << "drop_rate must be 0 when no signals are dropped";
}

TEST(OutputSinkTest, test_memory_sink_clear_resets_emit_count) {
    MemoryOutputSink sink(0);
    TradeSignal sig;
    sig.delta_bias_shift = 0.1; sig.confidence = 0.8;
    sink.emit(sig); sink.emit(sig);
    EXPECT_EQ(sink.emit_count(), 2u);
    sink.clear();
    EXPECT_EQ(sink.emit_count(), 0u)
        << "clear() must reset emit_count to 0";
    EXPECT_EQ(sink.size(), 0u)
        << "clear() must empty the signal buffer";
}

} // namespace
} // namespace llmquant
