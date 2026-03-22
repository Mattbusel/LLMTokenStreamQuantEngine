#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
#include "SignalReplayBuffer.h"
using namespace llmquant;

TEST(SignalReplayBufferTest, EmptyBuffer) {
    SignalReplayBuffer buf;
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.total_pushed(), 0u);
    EXPECT_TRUE(buf.snapshot().empty());
}

TEST(SignalReplayBufferTest, PushAndSnapshot) {
    SignalReplayBuffer buf;
    TradeSignal s{};
    s.delta_bias_shift = 0.5;
    s.confidence       = 0.8;
    buf.push(s);
    EXPECT_EQ(buf.size(), 1u);
    EXPECT_EQ(buf.total_pushed(), 1u);
    auto snap = buf.snapshot();
    ASSERT_EQ(snap.size(), 1u);
    EXPECT_DOUBLE_EQ(snap[0].delta_bias_shift, 0.5);
}

TEST(SignalReplayBufferTest, RingOverwrite) {
    SignalReplayBuffer::Config cfg;
    cfg.capacity = 3;
    SignalReplayBuffer buf(cfg);
    for (int i = 0; i < 5; ++i) {
        TradeSignal s{};
        s.delta_bias_shift = static_cast<double>(i);
        buf.push(s);
    }
    EXPECT_EQ(buf.size(), 3u);
    EXPECT_EQ(buf.total_pushed(), 5u);
}

TEST(SignalReplayBufferTest, ReplayCallback) {
    SignalReplayBuffer::Config cfg;
    cfg.capacity = 16;
    SignalReplayBuffer buf(cfg);
    for (int i = 0; i < 4; ++i) {
        TradeSignal s{};
        s.delta_bias_shift = static_cast<double>(i);
        buf.push(s);
    }
    int count = 0;
    buf.replay([&](const TradeSignal&) { ++count; });
    EXPECT_EQ(count, 4);
}

TEST(SignalReplayBufferTest, OnPushCallback) {
    SignalReplayBuffer::Config cfg;
    int fired = 0;
    cfg.on_push = [&](const TradeSignal&) { ++fired; };
    SignalReplayBuffer buf(cfg);
    TradeSignal s{};
    buf.push(s);
    buf.push(s);
    EXPECT_EQ(fired, 2);
}

TEST(SignalReplayBufferTest, Reset) {
    SignalReplayBuffer buf;
    TradeSignal s{};
    buf.push(s);
    buf.reset();
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.total_pushed(), 0u);
}

TEST(SignalReplayBufferTest, StatsJson) {
    SignalReplayBuffer buf;
    TradeSignal s{};
    s.delta_bias_shift = 0.3;
    s.confidence       = 0.7;
    buf.push(s);
    std::string json = buf.to_stats_json();
    EXPECT_NE(json.find("capacity"), std::string::npos);
    EXPECT_NE(json.find("total_pushed"), std::string::npos);
}

#else
TEST(SignalReplayBufferTest, Disabled) { SUCCEED(); }
#endif
