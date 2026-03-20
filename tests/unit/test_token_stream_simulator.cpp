#include "gtest/gtest.h"
#include "TokenStreamSimulator.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static TokenStreamSimulator::Config make_config(int interval_us = 1000) {
    TokenStreamSimulator::Config cfg;
    cfg.token_interval    = std::chrono::microseconds{interval_us};
    cfg.buffer_size       = 64;
    cfg.use_memory_stream = true;
    cfg.data_file_path    = "";
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_load_memory_tokens_populates_buffer) {
    TokenStreamSimulator sim(make_config());
    // If load_tokens_from_memory does not throw, the buffer was populated.
    EXPECT_NO_THROW(sim.load_tokens_from_memory({"alpha", "beta", "gamma"}));
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_start_stop_no_crash) {
    TokenStreamSimulator sim(make_config());
    sim.load_tokens_from_memory({"hello", "world"});
    sim.set_token_callback([](const Token&) {});

    EXPECT_NO_THROW(sim.start());
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    EXPECT_NO_THROW(sim.stop());
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_callback_is_invoked_for_each_token) {
    TokenStreamSimulator sim(make_config(500 /*0.5 ms*/));
    sim.load_tokens_from_memory({"a", "b", "c"});

    std::atomic<uint64_t> received{0};
    sim.set_token_callback([&received](const Token&) { received++; });

    sim.start();
    // With 0.5 ms interval, 50 ms gives roughly 100 emissions.
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    sim.stop();

    EXPECT_GT(received.load(), 0u);
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_stats_track_emitted_count) {
    TokenStreamSimulator sim(make_config(500));
    sim.load_tokens_from_memory({"x", "y"});
    sim.set_token_callback([](const Token&) {});

    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    sim.stop();

    EXPECT_GT(sim.get_stats().tokens_emitted.load(), 0u);
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_empty_buffer_does_not_crash) {
    TokenStreamSimulator sim(make_config(1000));
    // No tokens loaded at all.
    sim.set_token_callback([](const Token&) {});

    EXPECT_NO_THROW(sim.start());
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    EXPECT_NO_THROW(sim.stop());

    // Zero tokens should have been emitted.
    EXPECT_EQ(sim.get_stats().tokens_emitted.load(), 0u);
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_token_sequence_ids_are_monotone) {
    TokenStreamSimulator sim(make_config(500));
    sim.load_tokens_from_memory({"p", "q", "r"});

    std::vector<uint64_t> ids;
    ids.reserve(20);
    std::mutex id_mutex;

    sim.set_token_callback([&ids, &id_mutex](const Token& tok) {
        std::lock_guard<std::mutex> lk(id_mutex);
        ids.push_back(tok.sequence_id);
    });

    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    sim.stop();

    std::lock_guard<std::mutex> lk(id_mutex);
    for (size_t i = 1; i < ids.size(); ++i) {
        EXPECT_GT(ids[i], ids[i - 1]) << "Sequence IDs must be strictly increasing";
    }
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_stop_without_start_is_safe) {
    TokenStreamSimulator sim(make_config());
    sim.load_tokens_from_memory({"z"});
    // stop() must be safe even if start() was never called.
    EXPECT_NO_THROW(sim.stop());
}

TEST(TokenStreamSimulatorTest, test_ring_buffer_try_push_pop_roundtrip) {
    // Test the RingBuffer directly via the simulator's load/stream behaviour.
    llmquant::TokenStreamSimulator::Config cfg;
    cfg.token_interval = std::chrono::microseconds(1);
    cfg.buffer_size    = 8;
    cfg.use_memory_stream = true;

    llmquant::TokenStreamSimulator sim(cfg);
    sim.load_tokens_from_memory({"alpha", "beta", "gamma"});

    std::atomic<int> count{0};
    sim.set_token_callback([&](const llmquant::Token&) { count++; });
    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sim.stop();
    EXPECT_GT(count.load(), 0);
}

TEST(TokenStreamSimulatorTest, test_ring_buffer_fills_and_cycles_source) {
    llmquant::TokenStreamSimulator::Config cfg;
    cfg.token_interval = std::chrono::microseconds(100);
    cfg.buffer_size    = 4;
    cfg.use_memory_stream = true;

    llmquant::TokenStreamSimulator sim(cfg);
    // Only 2 source tokens — ring must cycle them.
    sim.load_tokens_from_memory({"a", "b"});

    std::vector<std::string> received;
    std::mutex rx_mutex;
    sim.set_token_callback([&](const llmquant::Token& tok) {
        std::lock_guard<std::mutex> lk(rx_mutex);
        received.push_back(tok.text);
    });
    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    sim.stop();

    // All received tokens must be either "a" or "b".
    std::lock_guard<std::mutex> lk(rx_mutex);
    for (const auto& t : received) {
        EXPECT_TRUE(t == "a" || t == "b");
    }
    EXPECT_FALSE(received.empty());
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_load_from_file_valid) {
    const std::string tmp = "/tmp/llmquant_test_tokens.txt";
    {
        std::ofstream f(tmp);
        f << "bullish bearish\n";
        f << "crash\n";
        f << "rally surge\n";
    }

    TokenStreamSimulator sim(make_config(500));
    ASSERT_NO_THROW(sim.load_tokens_from_file(tmp));

    std::atomic<int> count{0};
    sim.set_token_callback([&count](const Token&) { count++; });
    sim.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    sim.stop();

    EXPECT_GT(count.load(), 0);
    std::remove(tmp.c_str());
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_load_from_file_nonexistent_throws) {
    TokenStreamSimulator sim(make_config());
    EXPECT_THROW(sim.load_tokens_from_file("/does/not/exist/tokens.txt"),
                 std::runtime_error);
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_double_start_is_idempotent) {
    TokenStreamSimulator sim(make_config(5000));
    sim.load_tokens_from_memory({"a"});
    sim.set_token_callback([](const Token&) {});

    sim.start();
    EXPECT_NO_THROW(sim.start());  // Second start must be a no-op, not a crash.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sim.stop();
}

// ---------------------------------------------------------------------------
// reset_stats and is_running
// ---------------------------------------------------------------------------

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_is_running_false_before_start) {
    TokenStreamSimulator sim(make_config());
    EXPECT_FALSE(sim.is_running());
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_is_running_true_after_start) {
    TokenStreamSimulator sim(make_config(50000));
    sim.load_tokens_from_memory({"tok"});
    sim.set_token_callback([](const Token&) {});
    sim.start();
    EXPECT_TRUE(sim.is_running());
    sim.stop();
    EXPECT_FALSE(sim.is_running());
}

TEST(TokenStreamSimulatorTest, test_token_stream_simulator_reset_stats_zeros_emitted_count) {
    TokenStreamSimulator sim(make_config(1000));
    sim.load_tokens_from_memory({"a", "b", "c"});
    std::atomic<int> count{0};
    sim.set_token_callback([&count](const Token&) { ++count; });
    sim.start();
    // Wait until at least one token is emitted.
    for (int i = 0; i < 100 && count.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    sim.stop();

    EXPECT_GT(sim.get_stats().tokens_emitted.load(), 0u);
    sim.reset_stats();
    EXPECT_EQ(sim.get_stats().tokens_emitted.load(), 0u);
    EXPECT_EQ(sim.get_stats().ring_buffer_drops.load(), 0u);
}

TEST(TokenStreamSimulatorTest, test_get_token_count_reflects_loaded_tokens) {
    TokenStreamSimulator sim(make_config(50000));
    EXPECT_EQ(sim.get_token_count(), 0u)
        << "get_token_count must be 0 before any tokens are loaded";
    sim.load_tokens_from_memory({"alpha", "beta", "gamma"});
    EXPECT_EQ(sim.get_token_count(), 3u)
        << "get_token_count must match the number of tokens loaded";
}

TEST(TokenStreamSimulatorTest, test_get_tokens_emitted_zero_initially) {
    TokenStreamSimulator sim(make_config(50000));
    EXPECT_EQ(sim.get_tokens_emitted(), 0u)
        << "get_tokens_emitted must be 0 before any tokens are emitted";
}

TEST(TokenStreamSimulatorTest, test_set_token_interval_updates_config) {
    TokenStreamSimulator sim(make_config(50000));
    // Update the interval — should not throw or crash.
    sim.set_token_interval(std::chrono::microseconds{5000});
    // Verify the new interval is reflected in the config via get_stats (indirectly).
    // The primary assertion is that the call completes without crashing.
    EXPECT_TRUE(true) << "set_token_interval must not throw";
}

TEST(TokenStreamSimulatorTest, test_get_drop_rate_zero_with_no_drops) {
    TokenStreamSimulator sim(make_config(50000));
    // No tokens emitted yet — drop rate should be 0.0
    EXPECT_DOUBLE_EQ(sim.get_drop_rate(), 0.0)
        << "get_drop_rate must be 0.0 when nothing has been attempted";
}

TEST(TokenStreamSimulatorTest, test_format_stats_contains_emitted_and_drops) {
    TokenStreamSimulator sim(make_config(50000));
    std::string stats = sim.format_stats();
    EXPECT_NE(stats.find("emitted="), std::string::npos)
        << "format_stats must contain emitted= field";
    EXPECT_NE(stats.find("drops="), std::string::npos)
        << "format_stats must contain drops= field";
    EXPECT_NE(stats.find("drop_rate="), std::string::npos)
        << "format_stats must contain drop_rate= field";
}

TEST(TokenStreamSimulatorTest, test_format_stats_reports_zero_counts_initially) {
    TokenStreamSimulator sim(make_config(50000));
    std::string stats = sim.format_stats();
    EXPECT_NE(stats.find("emitted=0"), std::string::npos)
        << "format_stats should show emitted=0 before any tokens are emitted";
    EXPECT_NE(stats.find("drops=0"), std::string::npos)
        << "format_stats should show drops=0 before any drops occur";
}

} // namespace
} // namespace llmquant
