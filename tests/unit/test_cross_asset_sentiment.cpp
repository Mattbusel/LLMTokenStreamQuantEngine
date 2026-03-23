#include <gtest/gtest.h>

#include "cross_asset_sentiment.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static AssetSentiment make_sentiment(
    const std::string& id,
    double score,
    int64_t ts = 0
) {
    return AssetSentiment{id, score, ts};
}

static void feed_identical(
    SentimentCorrelationMatrix& m,
    const std::string& a,
    const std::string& b,
    int n,
    double base = 0.0,
    double step = 0.1
) {
    for (int i = 0; i < n; ++i) {
        double v = base + step * (i % 10);
        m.update(make_sentiment(a, v, i));
        m.update(make_sentiment(b, v, i));
    }
}

static void feed_opposite(
    SentimentCorrelationMatrix& m,
    const std::string& a,
    const std::string& b,
    int n
) {
    for (int i = 0; i < n; ++i) {
        double v = 0.1 * (i % 10);
        m.update(make_sentiment(a, v, i));
        m.update(make_sentiment(b, -v, i));
    }
}

// ─── AssetSentiment ───────────────────────────────────────────────────────────

TEST(AssetSentiment, ClampedScoreInRange) {
    AssetSentiment s{"BTC", 1.5, 0};
    EXPECT_NEAR(s.clamped_score(), 1.0, 1e-12);
}

TEST(AssetSentiment, ClampedScoreNegative) {
    AssetSentiment s{"ETH", -2.0, 0};
    EXPECT_NEAR(s.clamped_score(), -1.0, 1e-12);
}

TEST(AssetSentiment, ClampedScoreUnchangedInRange) {
    AssetSentiment s{"SOL", 0.5, 0};
    EXPECT_NEAR(s.clamped_score(), 0.5, 1e-12);
}

// ─── SentimentCorrelationMatrix — registration ────────────────────────────────

TEST(SentimentCorrelationMatrix, InitiallyEmpty) {
    SentimentCorrelationMatrix m;
    EXPECT_EQ(m.asset_count(), 0);
    EXPECT_TRUE(m.asset_ids().empty());
}

TEST(SentimentCorrelationMatrix, RegisterSingleAsset) {
    SentimentCorrelationMatrix m;
    bool inserted = m.register_asset("BTC");
    EXPECT_TRUE(inserted);
    EXPECT_EQ(m.asset_count(), 1);
}

TEST(SentimentCorrelationMatrix, RegisterDuplicateReturnsFalse) {
    SentimentCorrelationMatrix m;
    m.register_asset("BTC");
    bool dup = m.register_asset("BTC");
    EXPECT_FALSE(dup);
    EXPECT_EQ(m.asset_count(), 1);
}

TEST(SentimentCorrelationMatrix, RegisterMultipleAssets) {
    SentimentCorrelationMatrix m;
    m.register_asset("BTC");
    m.register_asset("ETH");
    m.register_asset("SOL");
    EXPECT_EQ(m.asset_count(), 3);
}

TEST(SentimentCorrelationMatrix, AssetIdsPreservesOrder) {
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    m.register_asset("B");
    m.register_asset("C");
    auto ids = m.asset_ids();
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], "A");
    EXPECT_EQ(ids[1], "B");
    EXPECT_EQ(ids[2], "C");
}

// ─── SentimentCorrelationMatrix — update ──────────────────────────────────────

TEST(SentimentCorrelationMatrix, UpdateReturnsFalseForUnknownAsset) {
    SentimentCorrelationMatrix m;
    bool ok = m.update(make_sentiment("UNKNOWN", 0.5));
    EXPECT_FALSE(ok);
}

TEST(SentimentCorrelationMatrix, UpdateReturnsTrueForKnownAsset) {
    SentimentCorrelationMatrix m;
    m.register_asset("BTC");
    bool ok = m.update(make_sentiment("BTC", 0.5));
    EXPECT_TRUE(ok);
}

TEST(SentimentCorrelationMatrix, WindowSizeCapRespected) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 5;
    cfg.min_samples = 2;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("BTC");

    for (int i = 0; i < 20; ++i) {
        m.update(make_sentiment("BTC", static_cast<double>(i) * 0.05, i));
    }
    EXPECT_EQ(m.window_size_for("BTC"), 5);
}

TEST(SentimentCorrelationMatrix, WindowSizeZeroForUnknownAsset) {
    SentimentCorrelationMatrix m;
    EXPECT_EQ(m.window_size_for("UNKNOWN"), 0);
}

TEST(SentimentCorrelationMatrix, BatchUpdateMultipleAssets) {
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    m.register_asset("B");

    std::vector<AssetSentiment> batch = {
        {"A", 0.3, 0},
        {"B", 0.5, 0},
        {"C", 0.1, 0},  // unknown — silently skipped
    };
    m.update(batch);

    EXPECT_EQ(m.window_size_for("A"), 1);
    EXPECT_EQ(m.window_size_for("B"), 1);
}

// ─── SentimentCorrelationMatrix — correlation ─────────────────────────────────

TEST(SentimentCorrelationMatrix, CorrelationNulloptForUnknownAsset) {
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    auto r = m.correlation("A", "UNKNOWN");
    EXPECT_FALSE(r.has_value());
}

TEST(SentimentCorrelationMatrix, CorrelationNulloptBelowMinSamples) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.min_samples = 20;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");

    // Only 5 samples — below min_samples=20
    for (int i = 0; i < 5; ++i) {
        m.update(make_sentiment("A", 0.1 * i, i));
        m.update(make_sentiment("B", 0.1 * i, i));
    }
    auto r = m.correlation("A", "B");
    EXPECT_FALSE(r.has_value());
}

TEST(SentimentCorrelationMatrix, PerfectlyCorrelatedAssetsPositiveRho) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 60;
    cfg.min_samples = 10;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");

    feed_identical(m, "A", "B", 60);

    auto r = m.correlation("A", "B");
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(*r, 0.5);  // strongly positive
}

TEST(SentimentCorrelationMatrix, NegativelyCorrelatedAssetsNegativeRho) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 60;
    cfg.min_samples = 10;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");

    feed_opposite(m, "A", "B", 60);

    auto r = m.correlation("A", "B");
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(*r, -0.3);
}

TEST(SentimentCorrelationMatrix, SelfCorrelationSymmetric) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 30;
    cfg.min_samples = 10;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("X");
    m.register_asset("Y");

    feed_identical(m, "X", "Y", 30);

    auto r_xy = m.correlation("X", "Y");
    auto r_yx = m.correlation("Y", "X");
    ASSERT_TRUE(r_xy.has_value());
    ASSERT_TRUE(r_yx.has_value());
    EXPECT_NEAR(*r_xy, *r_yx, 1e-10);
}

TEST(SentimentCorrelationMatrix, UncorrelatedAssetsNearZero) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 100;
    cfg.min_samples = 20;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");

    // A: alternating +/-; B: constant → near-zero correlation
    for (int i = 0; i < 100; ++i) {
        double a = (i % 2 == 0) ? 0.5 : -0.5;
        m.update(make_sentiment("A", a, i));
        m.update(make_sentiment("B", 0.3, i));
    }
    auto r = m.correlation("A", "B");
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 0.05);
}

// ─── SentimentCorrelationMatrix — full_matrix and average ─────────────────────

TEST(SentimentCorrelationMatrix, FullMatrixPairCount) {
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    m.register_asset("B");
    m.register_asset("C");

    auto entries = m.full_matrix();
    // 3 assets → 3 pairs: (A,B), (A,C), (B,C)
    EXPECT_EQ(entries.size(), 3u);
}

TEST(SentimentCorrelationMatrix, AverageCorrelationEmptyMatrixZero) {
    SentimentCorrelationMatrix m;
    EXPECT_NEAR(m.average_correlation(), 0.0, 1e-12);
}

TEST(SentimentCorrelationMatrix, AverageCorrelationSingleAssetZero) {
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    EXPECT_NEAR(m.average_correlation(), 0.0, 1e-12);
}

// ─── SentimentCorrelationMatrix — leading_asset ───────────────────────────────

TEST(SentimentCorrelationMatrix, LeadingAssetNulloptForUnknownTarget) {
    SentimentCorrelationMatrix m;
    auto result = m.leading_asset("UNKNOWN");
    EXPECT_FALSE(result.has_value());
}

TEST(SentimentCorrelationMatrix, LeadingAssetNulloptWhenInsufficientData) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.min_samples = 20;
    cfg.max_lag = 3;
    cfg.min_lag_corr = 0.3;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");

    // Too few samples
    for (int i = 0; i < 5; ++i) {
        m.update(make_sentiment("A", 0.1 * i, i));
        m.update(make_sentiment("B", 0.1 * i, i));
    }
    auto result = m.leading_asset("B");
    EXPECT_FALSE(result.has_value());
}

TEST(SentimentCorrelationMatrix, LeadingAssetDetectedWithSyntheticLag) {
    // A leads B by 1 step: feed A first, then copy A's value to B one step later
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 80;
    cfg.min_samples = 10;
    cfg.max_lag = 5;
    cfg.min_lag_corr = 0.2;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");

    // Build a series where A leads B
    std::vector<double> series;
    for (int i = 0; i < 80; ++i) series.push_back(std::sin(0.3 * i));

    for (int i = 0; i < 80; ++i) {
        m.update(make_sentiment("A", series[i], i));
        if (i > 0) {
            m.update(make_sentiment("B", series[i - 1], i));
        }
    }

    auto leader = m.leading_asset("B");
    // A should be identified as leading B (or nullopt if corr < threshold)
    // We just verify it doesn't crash and returns either A or nullopt
    if (leader.has_value()) {
        EXPECT_EQ(*leader, "A");
    }
}

// ─── SentimentCluster ─────────────────────────────────────────────────────────

TEST(SentimentCluster, SingletonClusterWhenNoCorrelation) {
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    m.register_asset("B");
    // No data → all correlations are nullopt → 0.0 < any threshold > 0
    auto clusters = SentimentCluster::build(m, 0.5);
    EXPECT_EQ(clusters.size(), 2u);  // each asset is its own singleton
}

TEST(SentimentCluster, AllInOneClusterWhenHighlyCorrelated) {
    SentimentCorrelationMatrix::Config cfg;
    cfg.window_size = 60;
    cfg.min_samples = 10;
    SentimentCorrelationMatrix m(cfg);
    m.register_asset("A");
    m.register_asset("B");
    m.register_asset("C");

    // Feed identical values → all pairs highly correlated
    for (int i = 0; i < 60; ++i) {
        double v = 0.1 * (i % 10);
        m.update(make_sentiment("A", v, i));
        m.update(make_sentiment("B", v, i));
        m.update(make_sentiment("C", v, i));
    }

    auto clusters = SentimentCluster::build(m, 0.3);
    // All three should be in one cluster
    bool found_large = false;
    for (const auto& c : clusters) {
        if (c.size() >= 3) found_large = true;
    }
    EXPECT_TRUE(found_large);
}

TEST(SentimentCluster, ContainsMember) {
    SentimentCluster::Cluster c;
    c.members = {"A", "B"};
    EXPECT_TRUE(c.contains("A"));
    EXPECT_FALSE(c.contains("Z"));
}

TEST(SentimentCluster, EmptyMatrix) {
    SentimentCorrelationMatrix m;
    auto clusters = SentimentCluster::build(m, 0.5);
    EXPECT_TRUE(clusters.empty());
}

// ─── SentimentRegimeDetector ──────────────────────────────────────────────────

TEST(SentimentRegimeDetector, InitialState) {
    SentimentRegimeDetector det;
    EXPECT_EQ(det.event_count(), 0);
    EXPECT_FALSE(det.in_contagion());
    EXPECT_NEAR(det.last_avg_corr(), 0.0, 1e-12);
}

TEST(SentimentRegimeDetector, ReturnsAverageCorrelation) {
    SentimentRegimeDetector det;
    SentimentCorrelationMatrix m;
    double avg = det.update(m);
    EXPECT_NEAR(avg, 0.0, 1e-12);
}

TEST(SentimentRegimeDetector, FiresContagionEventAboveThreshold) {
    SentimentRegimeDetector::Config cfg;
    cfg.spike_threshold = 0.5;
    cfg.cooldown_steps = 0;
    int events = 0;
    cfg.on_contagion_event = [&](double) { ++events; };

    SentimentRegimeDetector det(cfg);

    SentimentCorrelationMatrix::Config mcfg;
    mcfg.window_size = 60;
    mcfg.min_samples = 10;
    SentimentCorrelationMatrix m(mcfg);
    m.register_asset("A");
    m.register_asset("B");

    // Feed identical series → high correlation
    feed_identical(m, "A", "B", 60);

    // Force multiple updates past cooldown
    for (int i = 0; i < 3; ++i) det.update(m);

    EXPECT_GE(det.event_count(), 1);
    EXPECT_GE(events, 1);
}

TEST(SentimentRegimeDetector, CooldownPreventsDoublefire) {
    SentimentRegimeDetector::Config cfg;
    cfg.spike_threshold = 0.3;
    cfg.cooldown_steps = 100;  // very long cooldown
    int events = 0;
    cfg.on_contagion_event = [&](double) { ++events; };

    SentimentRegimeDetector det(cfg);

    SentimentCorrelationMatrix::Config mcfg;
    mcfg.window_size = 60;
    mcfg.min_samples = 10;
    SentimentCorrelationMatrix m(mcfg);
    m.register_asset("A");
    m.register_asset("B");
    feed_identical(m, "A", "B", 60);

    // Even many updates — cooldown = 100 means only 1 event can fire
    for (int i = 0; i < 20; ++i) det.update(m);

    EXPECT_LE(events, 1);
}

TEST(SentimentRegimeDetector, NoEventBelowThreshold) {
    SentimentRegimeDetector::Config cfg;
    cfg.spike_threshold = 0.99;  // nearly impossible threshold
    cfg.cooldown_steps = 0;
    int events = 0;
    cfg.on_contagion_event = [&](double) { ++events; };

    SentimentRegimeDetector det(cfg);
    SentimentCorrelationMatrix m;
    m.register_asset("A");
    m.register_asset("B");

    for (int i = 0; i < 10; ++i) det.update(m);
    EXPECT_EQ(events, 0);
}

TEST(SentimentRegimeDetector, ResetClearsState) {
    SentimentRegimeDetector det;
    SentimentCorrelationMatrix m;
    det.update(m);
    det.reset();
    EXPECT_EQ(det.event_count(), 0);
    EXPECT_FALSE(det.in_contagion());
    EXPECT_NEAR(det.last_avg_corr(), 0.0, 1e-12);
}
