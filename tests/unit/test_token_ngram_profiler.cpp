#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
#include "TokenNgramProfiler.h"
using namespace llmquant;

TEST(TokenNgramProfilerTest, EmptyState) {
    TokenNgramProfiler prof;
    EXPECT_EQ(prof.total_tokens(), 0u);
    EXPECT_EQ(prof.distinct_ngrams(), 0u);
    EXPECT_EQ(prof.hot_events(), 0u);
}

TEST(TokenNgramProfilerTest, BigramTracking) {
    TokenNgramProfiler prof;
    prof.push("buy");
    prof.push("signal");
    prof.push("buy");
    prof.push("signal");
    EXPECT_EQ(prof.frequency("buy signal"), 2u);
    EXPECT_EQ(prof.total_tokens(), 4u);
}

TEST(TokenNgramProfilerTest, TrigramTracking) {
    TokenNgramProfiler prof;
    prof.push("a");
    prof.push("b");
    prof.push("c");
    prof.push("a");
    prof.push("b");
    prof.push("c");
    EXPECT_GE(prof.frequency("a b c"), 2u);
}

TEST(TokenNgramProfilerTest, HotNgramCallback) {
    TokenNgramProfiler::Config cfg;
    cfg.hot_threshold = 3;
    int fired = 0;
    cfg.on_hot_ngram = [&](const std::string&, uint64_t, int) { ++fired; };
    TokenNgramProfiler prof(cfg);
    for (int i = 0; i < 5; ++i) {
        prof.push("buy");
        prof.push("now");
    }
    EXPECT_GE(fired, 1);
}

TEST(TokenNgramProfilerTest, TopByFrequency) {
    TokenNgramProfiler prof;
    for (int i = 0; i < 5; ++i) {
        prof.push("bullish");
        prof.push("trend");
    }
    for (int i = 0; i < 2; ++i) {
        prof.push("bearish");
        prof.push("momentum");
    }
    auto top = prof.top_by_frequency(5);
    EXPECT_FALSE(top.empty());
    EXPECT_GE(top[0].count, top.back().count);
}

TEST(TokenNgramProfilerTest, Reset) {
    TokenNgramProfiler prof;
    prof.push("x");
    prof.push("y");
    prof.reset();
    EXPECT_EQ(prof.total_tokens(), 0u);
    EXPECT_EQ(prof.distinct_ngrams(), 0u);
}

TEST(TokenNgramProfilerTest, StatsJson) {
    TokenNgramProfiler prof;
    prof.push("hello");
    prof.push("world");
    std::string json = prof.to_stats_json();
    EXPECT_NE(json.find("total_tokens"), std::string::npos);
    EXPECT_NE(json.find("distinct_ngrams"), std::string::npos);
}

TEST(TokenNgramProfilerTest, CapacityBound) {
    TokenNgramProfiler::Config cfg;
    cfg.max_ngrams = 5;
    TokenNgramProfiler prof(cfg);
    for (int i = 0; i < 20; ++i) {
        prof.push("tok" + std::to_string(i));
        prof.push("end");
    }
    EXPECT_LE(prof.distinct_ngrams(), 5u + 2u); // slight slack for eviction
}

#else
TEST(TokenNgramProfilerTest, Disabled) { SUCCEED(); }
#endif
