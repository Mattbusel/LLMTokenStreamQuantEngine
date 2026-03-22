#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef LLMQUANT_SENTIMENT_GRAPH_ENABLED
#include "TokenSentimentGraphBuilder.h"
using namespace llmquant;

TEST(TokenSentimentGraphBuilder, ConstructDefault) {
    TokenSentimentGraphBuilder g;
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(TokenSentimentGraphBuilder, TotalRecordsIncrement) {
    TokenSentimentGraphBuilder g;
    g.record(0, 0.1);
    g.record(1, -0.2);
    g.record(2, 0.3);
    EXPECT_EQ(g.total_records(), 3u);
}

TEST(TokenSentimentGraphBuilder, InvalidTokenIgnored) {
    TokenSentimentGraphBuilder g;
    g.record(-1, 0.5);   // out of range — ignored
    g.record(256, 0.5);  // out of range — ignored
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(TokenSentimentGraphBuilder, EdgeWeightAccumulates) {
    TokenSentimentGraphBuilder::Config cfg;
    cfg.min_edge_count = 1;
    TokenSentimentGraphBuilder g(cfg);
    // token 0 → token 1 with positive bias
    for (int i = 0; i < 5; ++i) {
        g.record(0, 0.5);
        g.record(1, 0.5);
    }
    double w = g.edge_weight(0, 1);
    EXPECT_GT(w, 0.0);
}

TEST(TokenSentimentGraphBuilder, EdgeWeightBelowMinCountZero) {
    TokenSentimentGraphBuilder::Config cfg;
    cfg.min_edge_count = 10;
    TokenSentimentGraphBuilder g(cfg);
    g.record(0, 0.5);
    g.record(1, 0.5);
    // Only 1 observation, below min_edge_count=10
    EXPECT_DOUBLE_EQ(g.edge_weight(0, 1), 0.0);
}

TEST(TokenSentimentGraphBuilder, InDegreeWeightNonNegativeForHub) {
    TokenSentimentGraphBuilder::Config cfg;
    cfg.min_edge_count = 1;
    TokenSentimentGraphBuilder g(cfg);
    for (int src = 0; src < 5; ++src)
        for (int rep = 0; rep < 3; ++rep) {
            g.record(src, 0.5);
            g.record(10, 0.8);  // token 10 is the hub
        }
    EXPECT_GE(g.in_degree_weight(10), 0.0);
}

TEST(TokenSentimentGraphBuilder, HubCallbackFired) {
    TokenSentimentGraphBuilder::Config cfg;
    cfg.hub_threshold  = 0.0;  // fire for any in-degree weight
    cfg.min_edge_count = 1;
    bool fired = false;
    cfg.on_hub_detected = [&](int, double) { fired = true; };
    TokenSentimentGraphBuilder g(cfg);
    for (int i = 0; i < 10; ++i) {
        g.record(i % 3, 0.5);
        g.record(5, 0.8);  // token 5 gets positive in-degree weight
    }
    EXPECT_TRUE(fired);
}

TEST(TokenSentimentGraphBuilder, TopHubsWithinBounds) {
    TokenSentimentGraphBuilder g;
    for (int i = 0; i < 20; ++i) {
        g.record(i % 5, 0.1 * i);
    }
    auto hubs = g.top_hubs();
    for (int h : hubs) {
        EXPECT_GE(h, -1);
        EXPECT_LT(h, TokenSentimentGraphBuilder::kMaxTokens);
    }
}

TEST(TokenSentimentGraphBuilder, Reset) {
    TokenSentimentGraphBuilder g;
    for (int i = 0; i < 20; ++i) g.record(i % 5, 0.1);
    g.reset();
    EXPECT_EQ(g.total_records(), 0u);
    EXPECT_DOUBLE_EQ(g.edge_weight(0, 1), 0.0);
}

TEST(TokenSentimentGraphBuilder, UpdateConfigResetsState) {
    TokenSentimentGraphBuilder g;
    for (int i = 0; i < 20; ++i) g.record(i % 3, 0.1);
    TokenSentimentGraphBuilder::Config cfg;
    cfg.window = 100;
    g.update_config(cfg);
    EXPECT_EQ(g.total_records(), 0u);
}

TEST(TokenSentimentGraphBuilder, ToStatsJsonFields) {
    TokenSentimentGraphBuilder g;
    for (int i = 0; i < 20; ++i) g.record(i % 5, 0.1 * i);
    auto js = g.to_stats_json();
    EXPECT_NE(js.find("active_edges"),  std::string::npos);
    EXPECT_NE(js.find("total_records"), std::string::npos);
}

TEST(TokenSentimentGraphBuilder, ConcurrentRecord) {
    TokenSentimentGraphBuilder g;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 50; ++i) g.record(i % 8, 0.1 * (i % 5));
        });
    for (auto& th : threads) th.join();
    EXPECT_GE(g.total_records(), 100u);
}

#else
TEST(TokenSentimentGraphBuilder, DisabledAtBuildTime) { SUCCEED(); }
#endif
