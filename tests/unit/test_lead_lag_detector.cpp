#include <gtest/gtest.h>

#ifdef LLMQUANT_LEAD_LAG_ENABLED

#include "LeadLagDetector.h"

#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Construction and defaults
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, DefaultConstructs) {
    LeadLagDetector det;
    EXPECT_EQ(det.total_x(), 0u);
    EXPECT_EQ(det.total_y(), 0u);
}

TEST(LeadLagDetector, ConfigConstructs) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 64;
    cfg.max_lag     = 8;
    LeadLagDetector det(cfg);
    auto c = det.get_config();
    EXPECT_EQ(c.max_lag, 8);
}

// ---------------------------------------------------------------------------
// Insufficient data returns empty correlogram
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, InsufficientDataReturnsEmpty) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 64;
    cfg.max_lag     = 8;
    LeadLagDetector det(cfg);

    // Push fewer than 2*max_lag+2 samples.
    for (int i = 0; i < 10; ++i) det.push(1.0, 1.0);

    auto c = det.compute();
    EXPECT_TRUE(c.lags.empty());
    EXPECT_EQ(c.optimal_lag, 0);
}

// ---------------------------------------------------------------------------
// Contemporaneous perfect correlation detected
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, ContemporaneousPerfectCorrelation) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 64;
    cfg.max_lag     = 4;
    LeadLagDetector det(cfg);

    // X and Y identical → lag=0 should have r≈1.
    for (int i = 0; i < 60; ++i) {
        double v = static_cast<double>(i % 7) - 3.0;
        det.push(v, v);
    }

    auto c = det.compute();
    EXPECT_EQ(c.optimal_lag, 0);
    EXPECT_NEAR(c.max_corr, 1.0, 0.05);
    EXPECT_FALSE(c.x_leads);
    EXPECT_FALSE(c.y_leads);
}

// ---------------------------------------------------------------------------
// X leads Y by positive lag detected
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, XLeadsYDetected) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 128;
    cfg.max_lag     = 8;
    LeadLagDetector det(cfg);

    // Create a signal where Y = X shifted right by 3 steps (X leads Y by 3).
    std::vector<double> base;
    for (int i = 0; i < 100; ++i)
        base.push_back(static_cast<double>(i % 11) - 5.0);

    // Push X at all time steps.  Push Y starting 3 steps late so Y[t] = X[t-3].
    const int kShift = 3;
    for (int i = 0; i < 100; ++i) {
        double x = base[i];
        double y = (i >= kShift) ? base[i - kShift] : 0.0;
        det.push(x, y);
    }

    auto c = det.compute();
    EXPECT_GT(c.optimal_lag, 0);   // X leads Y
    EXPECT_TRUE(c.x_leads);
    EXPECT_FALSE(c.y_leads);
    EXPECT_GT(std::abs(c.max_corr), 0.5);
}

// ---------------------------------------------------------------------------
// Y leads X: negative lag detected
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, YLeadsXDetected) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 128;
    cfg.max_lag     = 8;
    LeadLagDetector det(cfg);

    const int kShift = 3;
    std::vector<double> base;
    for (int i = 0; i < 100; ++i)
        base.push_back(static_cast<double>(i % 11) - 5.0);

    // Y leads X: X[t] = Y[t - kShift].
    for (int i = 0; i < 100; ++i) {
        double y = base[i];
        double x = (i >= kShift) ? base[i - kShift] : 0.0;
        det.push(x, y);
    }

    auto c = det.compute();
    EXPECT_LT(c.optimal_lag, 0);   // Y leads X
    EXPECT_TRUE(c.y_leads);
    EXPECT_FALSE(c.x_leads);
    EXPECT_GT(std::abs(c.max_corr), 0.5);
}

// ---------------------------------------------------------------------------
// push() increments both counters
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, PushIncrementsBothCounters) {
    LeadLagDetector det;
    det.push(1.0, 2.0);
    det.push(3.0, 4.0);
    EXPECT_EQ(det.total_x(), 2u);
    EXPECT_EQ(det.total_y(), 2u);
}

// ---------------------------------------------------------------------------
// push_x / push_y increment independently
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, PushXYIndependent) {
    LeadLagDetector det;
    det.push_x(1.0);
    det.push_x(2.0);
    det.push_y(3.0);
    EXPECT_EQ(det.total_x(), 2u);
    EXPECT_EQ(det.total_y(), 1u);
}

// ---------------------------------------------------------------------------
// reset() clears state
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, ResetClearsState) {
    LeadLagDetector det;
    for (int i = 0; i < 10; ++i) det.push(1.0, 2.0);
    det.reset();
    EXPECT_EQ(det.total_x(), 0u);
    EXPECT_EQ(det.total_y(), 0u);
    auto c = det.compute();
    EXPECT_TRUE(c.lags.empty());
}

// ---------------------------------------------------------------------------
// update_config resizes and clears
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, UpdateConfigClearsData) {
    LeadLagDetector det;
    for (int i = 0; i < 10; ++i) det.push(1.0, 2.0);

    LeadLagDetector::Config cfg2;
    cfg2.window_size = 64;
    cfg2.max_lag     = 4;
    det.update_config(cfg2);

    EXPECT_EQ(det.get_config().max_lag, 4);
    auto c = det.compute();
    EXPECT_TRUE(c.lags.empty());
}

// ---------------------------------------------------------------------------
// Correlogram covers the right number of lag entries
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, CorrelogramSizeCorrect) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 64;
    cfg.max_lag     = 5;
    LeadLagDetector det(cfg);

    for (int i = 0; i < 60; ++i)
        det.push(static_cast<double>(i), static_cast<double>(i));

    auto c = det.compute();
    // Expect lags from -5 to +5 = 11 entries.
    EXPECT_EQ(static_cast<int>(c.lags.size()), 2 * 5 + 1);
}

// ---------------------------------------------------------------------------
// Describe returns non-empty string
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, DescribeReturnsString) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 64;
    cfg.max_lag     = 4;
    LeadLagDetector det(cfg);

    for (int i = 0; i < 60; ++i)
        det.push(static_cast<double>(i), static_cast<double>(i));

    auto c = det.compute();
    EXPECT_FALSE(c.describe().empty());
}

// ---------------------------------------------------------------------------
// to_stats_json returns valid JSON-like string
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, ToStatsJsonContainsFields) {
    LeadLagDetector det;
    for (int i = 0; i < 40; ++i)
        det.push(static_cast<double>(i), static_cast<double>(i));

    std::string j = det.to_stats_json();
    EXPECT_NE(j.find("optimal_lag"), std::string::npos);
    EXPECT_NE(j.find("max_corr"), std::string::npos);
    EXPECT_NE(j.find("x_leads"), std::string::npos);
    EXPECT_NE(j.find("total_x"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent push is safe
// ---------------------------------------------------------------------------

TEST(LeadLagDetector, ConcurrentPushSafe) {
    LeadLagDetector::Config cfg;
    cfg.window_size = 256;
    cfg.max_lag     = 4;
    LeadLagDetector det(cfg);

    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            det.push(static_cast<double>(i), static_cast<double>(i));
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(det.total_x(), 800u);
    EXPECT_EQ(det.total_y(), 800u);
}

} // namespace

#else

TEST(LeadLagDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_LEAD_LAG_ENABLED
