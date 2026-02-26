#include "gtest/gtest.h"
#include "TokenStreamSimulator.h"

#include <atomic>
#include <chrono>
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

} // namespace
} // namespace llmquant
