#include <gtest/gtest.h>

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)

#include "SentimentMomentumFilter.h"

#include <cmath>
#include <vector>

using namespace llmquant;

namespace {

static TradeSignal make_signal(double bias) {
    TradeSignal s{};
    s.delta_bias_shift      = bias;
    s.confidence            = 0.8;
    s.volatility_adjustment = 0.01;
    s.signal_quality        = 0.9;
    return s;
}

// Push N identical samples to drive the analyzer to a known trend.
static void push_samples(SentimentMomentumFilter& f,
                         const std::vector<double>& vals) {
    for (double v : vals) f.record_sample(v);
}

// Build a rising sequence of N points.
static std::vector<double> rising(int n, double start = 0.0, double step = 0.03) {
    std::vector<double> v;
    for (int i = 0; i < n; ++i) v.push_back(start + i * step);
    return v;
}

static std::vector<double> falling(int n, double start = 0.5, double step = 0.03) {
    std::vector<double> v;
    for (int i = 0; i < n; ++i) v.push_back(start - i * step);
    return v;
}

static std::vector<double> flat(int n, double val = 0.3) {
    return std::vector<double>(static_cast<size_t>(n), val);
}

// ============================================================
// Construction
// ============================================================

TEST(SentimentMomentumFilter, DefaultConstructs) {
    SentimentMomentumFilter f;
    EXPECT_EQ(f.current_trend(), SentimentTrend::Undefined);
    auto s = f.get_stats();
    EXPECT_EQ(s.signals_in, 0u);
}

TEST(SentimentMomentumFilter, ConfigConstructs) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    SentimentMomentumFilter f(cfg);
    EXPECT_EQ(f.get_config().mode, SentimentMomentumFilter::Mode::Strict);
}

// ============================================================
// Disabled mode always passes
// ============================================================

TEST(SentimentMomentumFilter, DisabledModePassesAll) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Disabled;
    SentimentMomentumFilter f(cfg);

    auto out = f.filter_signal(make_signal(2.0));
    EXPECT_NEAR(out.delta_bias_shift, 2.0, 1e-9);

    auto s = f.get_stats();
    EXPECT_EQ(s.signals_passed, 1u);
    EXPECT_EQ(s.signals_blocked, 0u);
}

// ============================================================
// Undefined trend blocks in Strict/Relaxed modes
// ============================================================

TEST(SentimentMomentumFilter, UndefinedTrendBlocksBullishStrict) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    SentimentMomentumFilter f(cfg);
    // No samples → Undefined
    auto out = f.filter_signal(make_signal(1.0));
    EXPECT_NEAR(out.delta_bias_shift, 0.0, 1e-9);
    EXPECT_EQ(f.get_stats().signals_blocked, 1u);
}

TEST(SentimentMomentumFilter, UndefinedTrendBlocksBullishRelaxed) {
    SentimentMomentumFilter f;  // default = Relaxed
    auto out = f.filter_signal(make_signal(1.0));
    EXPECT_NEAR(out.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// Improving trend passes bullish, blocks bearish
// ============================================================

TEST(SentimentMomentumFilter, ImprovingTrendPassesBullish) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    cfg.analyzer.min_samples_for_trend = 5;
    SentimentMomentumFilter f(cfg);
    push_samples(f, rising(10));

    ASSERT_EQ(f.current_trend(), SentimentTrend::Improving);
    auto out = f.filter_signal(make_signal(1.5));
    EXPECT_NEAR(out.delta_bias_shift, 1.5, 1e-9);
}

TEST(SentimentMomentumFilter, ImprovingTrendBlocksBearishStrict) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    cfg.analyzer.min_samples_for_trend = 5;
    SentimentMomentumFilter f(cfg);
    push_samples(f, rising(10));

    ASSERT_EQ(f.current_trend(), SentimentTrend::Improving);
    auto out = f.filter_signal(make_signal(-1.5));
    EXPECT_NEAR(out.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// Declining trend passes bearish, blocks bullish
// ============================================================

TEST(SentimentMomentumFilter, DecliningTrendPassesBearishStrict) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    cfg.analyzer.min_samples_for_trend = 5;
    SentimentMomentumFilter f(cfg);
    push_samples(f, falling(10));

    ASSERT_EQ(f.current_trend(), SentimentTrend::Declining);
    auto out = f.filter_signal(make_signal(-1.5));
    EXPECT_NEAR(out.delta_bias_shift, -1.5, 1e-9);
}

TEST(SentimentMomentumFilter, DecliningTrendBlocksBullishStrict) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    cfg.analyzer.min_samples_for_trend = 5;
    SentimentMomentumFilter f(cfg);
    push_samples(f, falling(10));

    ASSERT_EQ(f.current_trend(), SentimentTrend::Declining);
    auto out = f.filter_signal(make_signal(1.5));
    EXPECT_NEAR(out.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// Stable trend: Relaxed passes both, Strict blocks both
// ============================================================

TEST(SentimentMomentumFilter, StableTrendPassesBullishRelaxed) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Relaxed;
    cfg.analyzer.min_samples_for_trend = 5;
    cfg.analyzer.trend_threshold = 0.1;  // high threshold → flat = Stable
    SentimentMomentumFilter f(cfg);
    push_samples(f, flat(10));

    ASSERT_EQ(f.current_trend(), SentimentTrend::Stable);
    auto out = f.filter_signal(make_signal(1.0));
    EXPECT_NEAR(out.delta_bias_shift, 1.0, 1e-9);
}

TEST(SentimentMomentumFilter, StableTrendBlocksBullishStrict) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Strict;
    cfg.analyzer.min_samples_for_trend = 5;
    cfg.analyzer.trend_threshold = 0.1;
    SentimentMomentumFilter f(cfg);
    push_samples(f, flat(10));

    ASSERT_EQ(f.current_trend(), SentimentTrend::Stable);
    auto out = f.filter_signal(make_signal(1.0));
    EXPECT_NEAR(out.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// Volatile trend is blocked in all directional modes
// ============================================================

TEST(SentimentMomentumFilter, VolatileTrendBlocksSignal) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Relaxed;
    cfg.analyzer.min_samples_for_trend = 5;
    cfg.analyzer.volatility_threshold  = 0.01;  // very tight → noisy = Volatile
    SentimentMomentumFilter f(cfg);
    // Alternating noisy signal to trigger Volatile.
    for (int i = 0; i < 10; ++i)
        f.record_sample((i % 2 == 0) ? 0.9 : 0.1);

    ASSERT_EQ(f.current_trend(), SentimentTrend::Volatile);
    auto out = f.filter_signal(make_signal(1.0));
    EXPECT_NEAR(out.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// scale_by_slope mode
// ============================================================

TEST(SentimentMomentumFilter, SlopeScalingAttenuatesBias) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode           = SentimentMomentumFilter::Mode::Relaxed;
    cfg.scale_by_slope = true;
    cfg.slope_scale    = 0.06;
    cfg.analyzer.min_samples_for_trend = 5;
    SentimentMomentumFilter f(cfg);
    // Build a moderate slope.
    push_samples(f, rising(10, 0.0, 0.03));  // slope ≈ 0.03

    ASSERT_EQ(f.current_trend(), SentimentTrend::Improving);
    double slope = f.current_slope();
    ASSERT_GT(slope, 0.0);

    auto raw = make_signal(2.0);
    auto out = f.filter_signal(raw);
    // Scale = |slope| / slope_scale ≈ 0.03/0.06 = 0.5
    double expected_scale = std::min(1.0, std::abs(slope) / cfg.slope_scale);
    EXPECT_NEAR(out.delta_bias_shift, raw.delta_bias_shift * expected_scale, 1e-6);
}

// ============================================================
// Statistics accumulation
// ============================================================

TEST(SentimentMomentumFilter, StatsCountBothPassedAndBlocked) {
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Disabled;
    SentimentMomentumFilter f(cfg);
    (void)f.filter_signal(make_signal(1.0));
    (void)f.filter_signal(make_signal(-1.0));
    auto s = f.get_stats();
    EXPECT_EQ(s.signals_in, 2u);
    EXPECT_EQ(s.signals_passed, 2u);
}

// ============================================================
// reset
// ============================================================

TEST(SentimentMomentumFilter, ResetClearsStatsAndTrend) {
    SentimentMomentumFilter::Config cfg;
    cfg.analyzer.min_samples_for_trend = 5;
    SentimentMomentumFilter f(cfg);
    push_samples(f, rising(10));
    (void)f.filter_signal(make_signal(1.0));
    f.reset();
    EXPECT_EQ(f.current_trend(), SentimentTrend::Undefined);
    EXPECT_EQ(f.get_stats().signals_in, 0u);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(SentimentMomentumFilter, ToStatsJsonContainsKeys) {
    SentimentMomentumFilter f;
    std::string json = f.to_stats_json();
    EXPECT_NE(json.find("mode"),             std::string::npos);
    EXPECT_NE(json.find("trend"),            std::string::npos);
    EXPECT_NE(json.find("signals_in"),       std::string::npos);
    EXPECT_NE(json.find("signals_blocked"),  std::string::npos);
    EXPECT_NE(json.find("pass_rate"),        std::string::npos);
}

// ============================================================
// update_config
// ============================================================

TEST(SentimentMomentumFilter, UpdateConfigChangesMode) {
    SentimentMomentumFilter f;
    SentimentMomentumFilter::Config cfg;
    cfg.mode = SentimentMomentumFilter::Mode::Disabled;
    f.update_config(cfg);
    EXPECT_EQ(f.get_config().mode, SentimentMomentumFilter::Mode::Disabled);
}

} // namespace

#else  // LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED && LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED

TEST(SentimentMomentumFilter, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED && LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED
