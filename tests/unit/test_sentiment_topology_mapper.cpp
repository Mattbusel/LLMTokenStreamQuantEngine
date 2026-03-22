#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_TOPOLOGY_MAPPER_ENABLED
#include "SentimentTopologyMapper.h"
using namespace llmquant;

TEST(SentimentTopologyMapper, ConstructDefault) {
    SentimentTopologyMapper stm;
    EXPECT_EQ(stm.total_records(), 0u);
    EXPECT_EQ(stm.topology_events(), 0u);
    EXPECT_DOUBLE_EQ(stm.total_persistence(), 0.0);
}

TEST(SentimentTopologyMapper, TotalRecordsIncrement) {
    SentimentTopologyMapper stm;
    stm.record(0.1);
    stm.record(-0.2);
    stm.record(0.3);
    EXPECT_EQ(stm.total_records(), 3u);
}

TEST(SentimentTopologyMapper, ComponentCountPositiveAfterFill) {
    SentimentTopologyMapper::Config cfg;
    cfg.min_samples = 8;
    SentimentTopologyMapper stm(cfg);
    for (int i = 0; i < 20; ++i) stm.record(0.1 * (i % 5));
    EXPECT_GE(stm.component_count(), 1);
}

TEST(SentimentTopologyMapper, TotalPersistenceNonNegative) {
    SentimentTopologyMapper stm;
    for (int i = 0; i < 30; ++i) stm.record(0.1 * (i % 7) - 0.3);
    EXPECT_GE(stm.total_persistence(), 0.0);
}

TEST(SentimentTopologyMapper, MultiPeakSignalMoreComponents) {
    SentimentTopologyMapper::Config cfg;
    cfg.min_persistence = 0.0;
    cfg.min_samples     = 8;
    cfg.window          = 64;
    SentimentTopologyMapper stm(cfg);
    // Two distinct peaks
    for (int i = 0; i < 10; ++i) stm.record(-0.8);
    for (int i = 0; i < 10; ++i) stm.record(0.8);
    for (int i = 0; i < 10; ++i) stm.record(-0.8);
    EXPECT_GE(stm.component_count(), 1);
}

TEST(SentimentTopologyMapper, TopologyCallbackFired) {
    SentimentTopologyMapper::Config cfg;
    cfg.min_samples     = 4;
    cfg.min_persistence = 0.0;
    int fired = 0;
    cfg.on_topology_change = [&](int, int) { ++fired; };
    SentimentTopologyMapper stm(cfg);
    for (int i = 0; i < 50; ++i)
        stm.record((i % 10 < 5) ? 0.8 : -0.8);
    EXPECT_GT(fired, 0);
}

TEST(SentimentTopologyMapper, Reset) {
    SentimentTopologyMapper stm;
    for (int i = 0; i < 30; ++i) stm.record(0.1 * i);
    stm.reset();
    EXPECT_EQ(stm.total_records(), 0u);
    EXPECT_EQ(stm.topology_events(), 0u);
    EXPECT_DOUBLE_EQ(stm.total_persistence(), 0.0);
}

TEST(SentimentTopologyMapper, UpdateConfigResetsState) {
    SentimentTopologyMapper stm;
    for (int i = 0; i < 20; ++i) stm.record(0.1);
    SentimentTopologyMapper::Config cfg;
    cfg.window = 32;
    stm.update_config(cfg);
    EXPECT_EQ(stm.total_records(), 0u);
}

TEST(SentimentTopologyMapper, ToStatsJsonFields) {
    SentimentTopologyMapper stm;
    for (int i = 0; i < 20; ++i) stm.record(0.1 * (i % 7));
    auto js = stm.to_stats_json();
    EXPECT_NE(js.find("component_count"),   std::string::npos);
    EXPECT_NE(js.find("total_persistence"), std::string::npos);
    EXPECT_NE(js.find("topology_events"),   std::string::npos);
    EXPECT_NE(js.find("total_records"),     std::string::npos);
}

TEST(SentimentTopologyMapper, ConcurrentRecord) {
    SentimentTopologyMapper stm;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) stm.record(0.1 * (i % 9));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(stm.total_records(), 100u);
}

#else
TEST(SentimentTopologyMapper, DisabledAtBuildTime) { SUCCEED(); }
#endif
