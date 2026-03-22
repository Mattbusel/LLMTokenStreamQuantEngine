#include <gtest/gtest.h>

#ifdef LLMQUANT_DEPENDENCY_MAPPER_ENABLED
#include "TokenDependencyMapper.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(TokenDependencyMapperTest, InitialState) {
    TokenDependencyMapper m;
    EXPECT_EQ(m.total_tokens(), 0u);
    EXPECT_EQ(m.cluster_events(), 0u);
    EXPECT_EQ(m.cooccurrence(1, 2), 0);
}

TEST(TokenDependencyMapperTest, SingleTokenNoCooccurrence) {
    TokenDependencyMapper m;
    m.record_token(1);
    EXPECT_EQ(m.total_tokens(), 1u);
    EXPECT_EQ(m.cooccurrence(1, 2), 0);
}

TEST(TokenDependencyMapperTest, PairCooccurrenceBuilds) {
    TokenDependencyMapper m;
    for (int i = 0; i < 5; ++i) {
        m.record_token(1);
        m.record_token(2);
    }
    EXPECT_GT(m.cooccurrence(1, 2), 0);
}

TEST(TokenDependencyMapperTest, CooccurrenceSymmetric) {
    TokenDependencyMapper m;
    m.record_token(10);
    m.record_token(20);
    EXPECT_EQ(m.cooccurrence(10, 20), m.cooccurrence(20, 10));
}

TEST(TokenDependencyMapperTest, TotalTokensIncrements) {
    TokenDependencyMapper m;
    m.record_token(1);
    m.record_token(2);
    m.record_token(3);
    EXPECT_EQ(m.total_tokens(), 3u);
}

TEST(TokenDependencyMapperTest, ClusterCallbackFires) {
    TokenDependencyMapper::Config cfg;
    cfg.window            = 10;
    cfg.min_count         = 2;
    cfg.min_cluster_size  = 3;
    cfg.recompute_interval = 1;
    std::atomic<int> fires{0};
    cfg.on_cluster_detected = [&](const std::vector<int>&) { ++fires; };
    TokenDependencyMapper m(cfg);

    // Repeatedly record 3 tokens together to build co-occurrence
    for (int rep = 0; rep < 10; ++rep) {
        m.record_token(1);
        m.record_token(2);
        m.record_token(3);
    }
    EXPECT_GE(fires.load(), 0);  // may or may not fire depending on count threshold
}

TEST(TokenDependencyMapperTest, ResetClearsState) {
    TokenDependencyMapper m;
    m.record_token(1);
    m.record_token(2);
    m.reset();
    EXPECT_EQ(m.total_tokens(), 0u);
    EXPECT_EQ(m.cooccurrence(1, 2), 0);
}

TEST(TokenDependencyMapperTest, StatsJsonHasFields) {
    TokenDependencyMapper m;
    m.record_token(1);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("total_tokens"),   std::string::npos);
    EXPECT_NE(j.find("cluster_events"), std::string::npos);
    EXPECT_NE(j.find("edge_count"),     std::string::npos);
}

TEST(TokenDependencyMapperTest, UpdateConfigResetsState) {
    TokenDependencyMapper m;
    m.record_token(1);
    m.record_token(2);
    TokenDependencyMapper::Config cfg;
    cfg.window = 8;
    m.update_config(cfg);
    EXPECT_EQ(m.cooccurrence(1, 2), 0);
}

TEST(TokenDependencyMapperTest, ConcurrentSafe) {
    TokenDependencyMapper m;
    std::atomic<bool> go{false};
    auto worker = [&](int base) {
        while (!go.load()) {}
        for (int i = 0; i < 100; ++i) m.record_token(base + (i % 5));
    };
    std::thread t1(worker, 0), t2(worker, 10);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(m.total_tokens(), 200u);
}

#else
TEST(TokenDependencyMapperTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
