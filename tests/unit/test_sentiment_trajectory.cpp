#include "gtest/gtest.h"
#include "SentimentTrajectoryAnalyzer.h"

#include <cmath>
#include <thread>
#include <vector>

namespace llmquant {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static SentimentTrajectoryAnalyzer::Config make_config() {
    SentimentTrajectoryAnalyzer::Config cfg;
    cfg.window_size           = 10;
    cfg.trend_threshold       = 0.02;
    cfg.volatility_threshold  = 0.20;
    cfg.min_samples_for_trend = 3;
    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: New analyser returns Undefined before min_samples_for_trend.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_undefined_before_min_samples) {
    SentimentTrajectoryAnalyzer sa(make_config());
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Undefined)
        << "No samples — must return Undefined";
    sa.add_sample(0.1);
    sa.add_sample(0.2);
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Undefined)
        << "< min_samples_for_trend (3) — must still return Undefined";
}

// ---------------------------------------------------------------------------
// Test 2: Monotonically increasing sentiment classifies as Improving.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_improving_trend) {
    SentimentTrajectoryAnalyzer sa(make_config());
    for (int i = 0; i < 10; ++i)
        sa.add_sample(0.05 * static_cast<double>(i));  // 0.0 → 0.45, clear uptrend
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Improving)
        << "Monotonically rising scores must be classified as Improving";
    EXPECT_GT(sa.get_slope(), 0.0) << "Improving trend must have positive slope";
}

// ---------------------------------------------------------------------------
// Test 3: Monotonically decreasing sentiment classifies as Declining.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_declining_trend) {
    SentimentTrajectoryAnalyzer sa(make_config());
    for (int i = 0; i < 10; ++i)
        sa.add_sample(0.5 - 0.05 * static_cast<double>(i));  // 0.5 → 0.05, clear downtrend
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Declining)
        << "Monotonically falling scores must be classified as Declining";
    EXPECT_LT(sa.get_slope(), 0.0) << "Declining trend must have negative slope";
}

// ---------------------------------------------------------------------------
// Test 4: Flat (constant) sentiment classifies as Stable.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_stable_trend) {
    SentimentTrajectoryAnalyzer sa(make_config());
    for (int i = 0; i < 10; ++i)
        sa.add_sample(0.3);  // constant — slope exactly 0
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Stable)
        << "Constant scores must be classified as Stable";
    EXPECT_NEAR(sa.get_slope(), 0.0, 1e-9) << "Flat signal must have near-zero slope";
}

// ---------------------------------------------------------------------------
// Test 5: High-variance random-sign scores classify as Volatile.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_volatile_trend) {
    SentimentTrajectoryAnalyzer::Config cfg = make_config();
    cfg.volatility_threshold = 0.10;  // low threshold to make Volatile easy to trigger
    SentimentTrajectoryAnalyzer sa(cfg);
    // Alternating extreme values — residual std dev will be very high.
    for (int i = 0; i < 10; ++i)
        sa.add_sample((i % 2 == 0) ? 1.0 : -1.0);
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Volatile)
        << "Alternating ±1 scores must be classified as Volatile";
    EXPECT_GT(sa.get_residual_std(), cfg.volatility_threshold)
        << "Residual std dev must exceed volatility_threshold for Volatile classification";
}

// ---------------------------------------------------------------------------
// Test 6: R² is 1.0 for a perfectly linear signal.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_r_squared_perfect_linear) {
    SentimentTrajectoryAnalyzer sa(make_config());
    for (int i = 0; i < 10; ++i)
        sa.add_sample(0.1 * static_cast<double>(i));
    EXPECT_NEAR(sa.get_r_squared(), 1.0, 1e-6)
        << "Perfectly linear signal must have R²=1.0";
}

// ---------------------------------------------------------------------------
// Test 7: Sliding window evicts old samples (window_size=5, add 10 samples).
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_window_evicts_old_samples) {
    SentimentTrajectoryAnalyzer::Config cfg = make_config();
    cfg.window_size = 5;
    cfg.min_samples_for_trend = 3;
    SentimentTrajectoryAnalyzer sa(cfg);
    // Add 5 samples that look declining...
    for (int i = 0; i < 5; ++i) sa.add_sample(0.5 - 0.1 * static_cast<double>(i));
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Declining);
    // ...then overwrite with 5 strongly improving samples.
    for (int i = 0; i < 5; ++i) sa.add_sample(0.1 * static_cast<double>(i));
    // Now the window only sees the improving samples.
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Improving)
        << "After window overwrite, trend must reflect only recent samples";
    EXPECT_EQ(sa.get_sample_count(), 5)
        << "sample_count must not exceed window_size";
}

// ---------------------------------------------------------------------------
// Test 8: total_samples increments even when samples are evicted.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_total_samples_monotone) {
    SentimentTrajectoryAnalyzer::Config cfg = make_config();
    cfg.window_size = 3;
    SentimentTrajectoryAnalyzer sa(cfg);
    for (int i = 0; i < 10; ++i) sa.add_sample(0.1);
    EXPECT_EQ(sa.get_total_samples(), 10u)
        << "total_samples must count all add_sample() calls";
    EXPECT_EQ(sa.get_sample_count(), 3)
        << "sample_count must not exceed window_size";
}

// ---------------------------------------------------------------------------
// Test 9: reset() clears all state.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_reset_clears_state) {
    SentimentTrajectoryAnalyzer sa(make_config());
    for (int i = 0; i < 10; ++i) sa.add_sample(0.1 * static_cast<double>(i));
    EXPECT_NE(sa.get_trend(), SentimentTrend::Undefined);
    sa.reset();
    EXPECT_EQ(sa.get_trend(), SentimentTrend::Undefined)
        << "After reset(), trend must return Undefined (no samples)";
    EXPECT_EQ(sa.get_sample_count(), 0) << "sample_count must be 0 after reset()";
    EXPECT_EQ(sa.get_total_samples(), 0u) << "total_samples must be 0 after reset()";
    EXPECT_NEAR(sa.get_slope(), 0.0, 1e-9) << "slope must be 0.0 after reset()";
}

// ---------------------------------------------------------------------------
// Test 10: to_stats_json() produces valid JSON with required fields.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_to_stats_json_has_required_fields) {
    SentimentTrajectoryAnalyzer sa(make_config());
    for (int i = 0; i < 5; ++i) sa.add_sample(0.1 * static_cast<double>(i));
    std::string json = sa.to_stats_json();
    EXPECT_EQ(json.front(), '{') << "JSON must start with '{'";
    EXPECT_EQ(json.back(),  '}') << "JSON must end with '}'";
    EXPECT_NE(json.find("\"trend\""),        std::string::npos);
    EXPECT_NE(json.find("\"slope\""),        std::string::npos);
    EXPECT_NE(json.find("\"r_squared\""),    std::string::npos);
    EXPECT_NE(json.find("\"residual_std\""), std::string::npos);
    EXPECT_NE(json.find("\"mean\""),         std::string::npos);
    EXPECT_NE(json.find("\"sample_count\""), std::string::npos);
    EXPECT_NE(json.find("\"total_samples\""),std::string::npos);
}

// ---------------------------------------------------------------------------
// Test 11: trend_name() returns the correct string for each value.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_trend_name_strings) {
    EXPECT_STREQ(SentimentTrajectoryAnalyzer::trend_name(SentimentTrend::Improving), "Improving");
    EXPECT_STREQ(SentimentTrajectoryAnalyzer::trend_name(SentimentTrend::Declining), "Declining");
    EXPECT_STREQ(SentimentTrajectoryAnalyzer::trend_name(SentimentTrend::Stable),    "Stable");
    EXPECT_STREQ(SentimentTrajectoryAnalyzer::trend_name(SentimentTrend::Volatile),  "Volatile");
    EXPECT_STREQ(SentimentTrajectoryAnalyzer::trend_name(SentimentTrend::Undefined), "Undefined");
}

// ---------------------------------------------------------------------------
// Test 12: Thread-safe concurrent add_sample() does not crash.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_concurrent_add_sample_does_not_crash) {
    SentimentTrajectoryAnalyzer sa(make_config());
    constexpr int kThreads = 4;
    constexpr int kSamplesPerThread = 100;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&sa, t] {
            for (int i = 0; i < kSamplesPerThread; ++i) {
                sa.add_sample(static_cast<double>(t * 0.1 + i * 0.01));
                (void)sa.get_trend();
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(sa.get_total_samples(),
              static_cast<uint64_t>(kThreads * kSamplesPerThread))
        << "total_samples must equal kThreads * kSamplesPerThread";
}

// ---------------------------------------------------------------------------
// Test 13: get_mean() matches hand-computed average.
// ---------------------------------------------------------------------------
TEST(SentimentTrajectoryTest, test_mean_matches_expected) {
    SentimentTrajectoryAnalyzer sa(make_config());
    // Add values 0.1, 0.2, 0.3, 0.4, 0.5 — mean = 0.3
    for (int i = 1; i <= 5; ++i)
        sa.add_sample(static_cast<double>(i) * 0.1);
    EXPECT_NEAR(sa.get_mean(), 0.3, 1e-9)
        << "mean must equal (0.1+0.2+0.3+0.4+0.5)/5 = 0.3";
}

} // namespace
} // namespace llmquant
