#include "gtest/gtest.h"
#include "OutputSink.h"
#include "TradeSignalEngine.h"

#include <chrono>
#include <cstdio>
#include <fstream>
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

} // namespace
} // namespace llmquant
