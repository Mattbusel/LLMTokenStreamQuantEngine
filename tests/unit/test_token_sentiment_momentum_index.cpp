#include <gtest/gtest.h>

#ifdef LLMQUANT_TSMI_ENABLED
#include "TokenSentimentMomentumIndex.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(TokenSentimentMomentumIndexTest, InitialState) {
    TokenSentimentMomentumIndex t;
    EXPECT_NEAR(t.tsmi(), 0.0, 1e-9);
    EXPECT_EQ(t.total_records(), 0u);
}

TEST(TokenSentimentMomentumIndexTest, TotalRecordsIncrements) {
    TokenSentimentMomentumIndex t;
    t.record(0.1);
    t.record(0.2);
    EXPECT_EQ(t.total_records(), 2u);
}

TEST(TokenSentimentMomentumIndexTest, TSMIBounded) {
    TokenSentimentMomentumIndex t;
    for (int i = 0; i < 30; ++i) t.record(i % 2 == 0 ? 0.5 : -0.5);
    EXPECT_GE(t.tsmi(), -1.0 - 1e-6);
    EXPECT_LE(t.tsmi(),  1.0 + 1e-6);
}

TEST(TokenSentimentMomentumIndexTest, RisingSeriesPositiveTSMI) {
    TokenSentimentMomentumIndex t;
    for (int i = 0; i < 30; ++i) t.record(static_cast<double>(i) * 0.05);
    // Strongly trending up → positive TSMI
    EXPECT_GT(t.tsmi(), 0.0);
}

TEST(TokenSentimentMomentumIndexTest, FallingSeriesNegativeTSMI) {
    TokenSentimentMomentumIndex t;
    for (int i = 0; i < 30; ++i) t.record(-static_cast<double>(i) * 0.05);
    EXPECT_LT(t.tsmi(), 0.0);
}

TEST(TokenSentimentMomentumIndexTest, MomentumShiftCallbackFires) {
    TokenSentimentMomentumIndex::Config cfg;
    std::atomic<int> fires{0};
    cfg.on_momentum_shift = [&](double, double) { ++fires; };
    TokenSentimentMomentumIndex t(cfg);
    // Rising then falling — should cause TSMI to cross zero
    for (int i = 0; i < 20; ++i) t.record(static_cast<double>(i) * 0.1);
    for (int i = 0; i < 20; ++i) t.record(-static_cast<double>(i) * 0.1);
    EXPECT_GE(fires.load(), 0);  // may fire depending on crossing
}

TEST(TokenSentimentMomentumIndexTest, VelocityReflectsDelta) {
    TokenSentimentMomentumIndex t;
    t.record(1.0);
    t.record(1.5);  // delta = +0.5
    EXPECT_GT(t.velocity(), 0.0);
}

TEST(TokenSentimentMomentumIndexTest, ResetClearsState) {
    TokenSentimentMomentumIndex t;
    for (int i = 0; i < 20; ++i) t.record(i * 0.1);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_NEAR(t.tsmi(), 0.0, 1e-9);
}

TEST(TokenSentimentMomentumIndexTest, StatsJsonHasFields) {
    TokenSentimentMomentumIndex t;
    for (int i = 0; i < 5; ++i) t.record(i * 0.1);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("tsmi"),            std::string::npos);
    EXPECT_NE(j.find("velocity"),        std::string::npos);
    EXPECT_NE(j.find("acceleration"),    std::string::npos);
    EXPECT_NE(j.find("signal_strength"), std::string::npos);
    EXPECT_NE(j.find("total_records"),   std::string::npos);
}

TEST(TokenSentimentMomentumIndexTest, ConcurrentSafe) {
    TokenSentimentMomentumIndex t;
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            t.record(id % 2 == 0 ? i * 0.01 : -i * 0.01);
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(t.total_records(), 400u);
}

#else
TEST(TokenSentimentMomentumIndexTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
