#include <gtest/gtest.h>

#ifdef LLMQUANT_MULTI_FEED_AGGREGATOR_ENABLED

#include "MultiFeedSignalAggregator.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// Convenience: build a two-feed config.
MultiFeedSignalAggregator::Config two_feed_cfg(double div_threshold = 0.5) {
    MultiFeedSignalAggregator::Config cfg;
    cfg.divergence_threshold = div_threshold;
    cfg.min_feeds_active     = 2;
    cfg.feeds = {{"feedA", 1.0, 0.5}, {"feedB", 1.0, 0.5}};
    return cfg;
}

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, DefaultState) {
    MultiFeedSignalAggregator agg;
    EXPECT_DOUBLE_EQ(agg.aggregated_bias(), 0.0);
    EXPECT_DOUBLE_EQ(agg.divergence(), 0.0);
    EXPECT_FALSE(agg.is_diverged());
    EXPECT_EQ(agg.total_records(), 0u);
    EXPECT_EQ(agg.divergence_events(), 0u);
}

// ---------------------------------------------------------------------------
// Unknown feed is silently ignored
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, UnknownFeedIgnored) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    agg.record("unknown", 10.0);
    EXPECT_EQ(agg.total_records(), 0u);  // unknown feed → not counted
    EXPECT_DOUBLE_EQ(agg.aggregated_bias(), 0.0);
}

// ---------------------------------------------------------------------------
// Below min_feeds_active → aggregated_bias stays 0
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, BelowMinFeedsAggZero) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    agg.record("feedA", 1.0);  // only one feed active
    EXPECT_DOUBLE_EQ(agg.aggregated_bias(), 0.0);
    EXPECT_EQ(agg.active_feed_count(), 1);
}

// ---------------------------------------------------------------------------
// Two feeds with same bias → low divergence, correct aggregate
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, AgreementLowDivergence) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    // Warm up both feeds.
    for (int i = 0; i < 20; ++i) {
        agg.record("feedA", 0.5);
        agg.record("feedB", 0.5);
    }
    EXPECT_NEAR(agg.aggregated_bias(), 0.5, 0.05);
    EXPECT_FALSE(agg.is_diverged());
}

// ---------------------------------------------------------------------------
// Opposite feeds → high divergence, callback fires
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, OppositesForceDivergence) {
    auto cfg = two_feed_cfg(0.3);
    std::atomic<int> fires{0};
    cfg.on_divergence = [&](double, const std::string&) { ++fires; };
    MultiFeedSignalAggregator agg(cfg);
    for (int i = 0; i < 15; ++i) {
        agg.record("feedA",  1.0);
        agg.record("feedB", -1.0);
    }
    EXPECT_TRUE(agg.is_diverged());
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(agg.divergence_events(), 0u);
}

// ---------------------------------------------------------------------------
// Convergence callback fires after feeds re-align
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, ConvergenceCallbackFires) {
    auto cfg = two_feed_cfg(0.3);
    cfg.clear_hysteresis = 0.8;
    std::atomic<int> convs{0};
    cfg.on_convergence = [&](double) { ++convs; };
    MultiFeedSignalAggregator agg(cfg);

    // Cause divergence.
    for (int i = 0; i < 15; ++i) {
        agg.record("feedA",  1.0);
        agg.record("feedB", -1.0);
    }
    EXPECT_TRUE(agg.is_diverged());

    // Re-align feeds.
    for (int i = 0; i < 30; ++i) {
        agg.record("feedA", 0.4);
        agg.record("feedB", 0.4);
    }
    EXPECT_GT(convs.load(), 0);
    EXPECT_FALSE(agg.is_diverged());
}

// ---------------------------------------------------------------------------
// Weighted feeds: heavier feed dominates aggregate
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, WeightedAggregation) {
    MultiFeedSignalAggregator::Config cfg;
    cfg.min_feeds_active = 2;
    cfg.feeds = {{"heavy", 4.0, 0.8}, {"light", 1.0, 0.8}};
    MultiFeedSignalAggregator agg(cfg);
    for (int i = 0; i < 20; ++i) {
        agg.record("heavy", 1.0);
        agg.record("light", 0.0);
    }
    // heavy weight=4, light weight=1 → agg ≈ (4×1 + 1×0)/5 = 0.8
    EXPECT_GT(agg.aggregated_bias(), 0.6);
}

// ---------------------------------------------------------------------------
// dominant_feed returns the highest-|ema| feed
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, DominantFeedCorrect) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    for (int i = 0; i < 10; ++i) agg.record("feedA", 2.0);
    for (int i = 0; i < 10; ++i) agg.record("feedB", 0.1);
    EXPECT_EQ(agg.dominant_feed(), "feedA");
}

// ---------------------------------------------------------------------------
// reset() clears state
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, ResetClearsState) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    for (int i = 0; i < 10; ++i) {
        agg.record("feedA", 0.5);
        agg.record("feedB", 0.5);
    }
    agg.reset();
    EXPECT_EQ(agg.total_records(), 0u);
    EXPECT_DOUBLE_EQ(agg.aggregated_bias(), 0.0);
    EXPECT_FALSE(agg.is_diverged());
    EXPECT_EQ(agg.active_feed_count(), 0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, UpdateConfigResetsState) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    for (int i = 0; i < 10; ++i) {
        agg.record("feedA", 0.5);
        agg.record("feedB", 0.5);
    }
    auto cfg2 = two_feed_cfg(0.4);
    agg.update_config(cfg2);
    EXPECT_EQ(agg.total_records(), 0u);
    EXPECT_EQ(agg.active_feed_count(), 0);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected keys
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, ToStatsJsonContainsKeys) {
    MultiFeedSignalAggregator agg(two_feed_cfg());
    agg.record("feedA", 0.5);
    agg.record("feedB", 0.6);
    std::string j = agg.to_stats_json();
    EXPECT_NE(j.find("aggregated_bias"),   std::string::npos);
    EXPECT_NE(j.find("divergence"),        std::string::npos);
    EXPECT_NE(j.find("is_diverged"),       std::string::npos);
    EXPECT_NE(j.find("dominant_feed"),     std::string::npos);
    EXPECT_NE(j.find("total_records"),     std::string::npos);
    EXPECT_NE(j.find("divergence_events"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record() is thread-safe
// ---------------------------------------------------------------------------

TEST(MultiFeedSignalAggregator, ConcurrentRecordSafe) {
    auto cfg = two_feed_cfg();
    MultiFeedSignalAggregator agg(cfg);

    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        const char* feed = (id % 2 == 0) ? "feedA" : "feedB";
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            agg.record(feed, static_cast<double>(i % 10) * 0.1 - 0.5);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();

    // 2 feeds × 2 threads each = 4×250 = 1000, but unknown feeds are ignored.
    // feedA: 2 threads × 250 = 500; feedB: 2 threads × 250 = 500.
    EXPECT_EQ(agg.total_records(), 1000u);
    EXPECT_TRUE(std::isfinite(agg.aggregated_bias()));
}

} // namespace

#else

TEST(MultiFeedSignalAggregator, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_MULTI_FEED_AGGREGATOR_ENABLED
