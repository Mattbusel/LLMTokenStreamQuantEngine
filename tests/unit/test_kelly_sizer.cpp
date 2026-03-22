#include "KellyPositionSizer.h"
#include "gtest/gtest.h"

#include <cmath>
#include <string>

using namespace llmquant;

namespace {

// Helper: create a default signal with a given bias.
static TradeSignal make_sig(double bias = 0.5) {
    TradeSignal s;
    s.delta_bias_shift      = bias;
    s.volatility_adjustment = 0.1;
    s.confidence            = 0.8;
    s.signal_quality        = 0.5;
    s.timestamp_ns          = 1;
    return s;
}

// ============================================================
// Construction
// ============================================================

TEST(KellyPositionSizer, DefaultConstructsOk) {
    EXPECT_NO_THROW(KellyPositionSizer sizer);
}

TEST(KellyPositionSizer, CustomConfigConstructsOk) {
    KellyPositionSizer::Config cfg;
    cfg.max_fraction  = 0.5;
    cfg.ema_alpha     = 0.2;
    cfg.min_history   = 5;
    EXPECT_NO_THROW(KellyPositionSizer sizer(cfg));
}

TEST(KellyPositionSizer, InvalidMaxFractionThrows) {
    KellyPositionSizer::Config cfg;
    cfg.max_fraction = 0.0;
    EXPECT_THROW(KellyPositionSizer sizer(cfg), std::invalid_argument);
}

// ============================================================
// Insufficient history passes signal through unchanged
// ============================================================

TEST(KellyPositionSizer, InsufficientHistoryPassesSignalUnchanged) {
    KellyPositionSizer::Config cfg;
    cfg.min_history = 10;
    KellyPositionSizer sizer(cfg);

    // Record fewer outcomes than min_history.
    for (int i = 0; i < 5; ++i)
        sizer.record_outcome(true, 0.01);

    auto sig = make_sig(0.4);
    auto sized = sizer.size_signal(sig);
    EXPECT_DOUBLE_EQ(sized.delta_bias_shift, sig.delta_bias_shift)
        << "Signal must be unchanged before min_history is reached";
}

// ============================================================
// Zero Kelly fraction returns zero delta_bias_shift
// ============================================================

TEST(KellyPositionSizer, NegativeEdgeReturnZeroBias) {
    KellyPositionSizer::Config cfg;
    cfg.min_history       = 1;
    cfg.prior_win_rate    = 0.5;
    cfg.prior_avg_win     = 0.001;
    cfg.prior_avg_loss    = 0.1;   // huge loss → negative edge
    KellyPositionSizer sizer(cfg);

    // Record one outcome to pass min_history.
    sizer.record_outcome(false, 0.1);

    auto sized = sizer.size_signal(make_sig(0.5));
    // Edge should be negative (or zero); signal bias should be zeroed.
    EXPECT_LE(sized.delta_bias_shift, 0.0)
        << "Negative-edge Kelly must zero the delta_bias_shift";
}

// ============================================================
// Positive edge scales the signal
// ============================================================

TEST(KellyPositionSizer, PositiveEdgeScalesBiasDown) {
    KellyPositionSizer::Config cfg;
    cfg.min_history    = 5;
    cfg.max_fraction   = 0.25;
    cfg.prior_win_rate = 0.5;
    cfg.prior_avg_win  = 0.02;
    cfg.prior_avg_loss = 0.01;
    KellyPositionSizer sizer(cfg);

    // Feed enough wins to build a clear positive edge.
    for (int i = 0; i < 20; ++i)
        sizer.record_outcome(true, 0.02);

    double original_bias = 0.8;
    auto sized = sizer.size_signal(make_sig(original_bias));

    // Sized bias must be <= original (Kelly fraction <= 1.0).
    EXPECT_LE(sized.delta_bias_shift, original_bias);
    // And positive (positive edge).
    EXPECT_GT(sized.delta_bias_shift, 0.0);
}

// ============================================================
// Non-bias fields are unchanged
// ============================================================

TEST(KellyPositionSizer, NonBiasFieldsPreserved) {
    KellyPositionSizer::Config cfg;
    cfg.min_history  = 1;
    cfg.max_fraction = 0.5;
    KellyPositionSizer sizer(cfg);
    sizer.record_outcome(true, 0.02);

    auto sig   = make_sig(0.5);
    auto sized = sizer.size_signal(sig);

    EXPECT_DOUBLE_EQ(sized.volatility_adjustment, sig.volatility_adjustment);
    EXPECT_DOUBLE_EQ(sized.confidence,            sig.confidence);
    EXPECT_DOUBLE_EQ(sized.signal_quality,        sig.signal_quality);
}

// ============================================================
// reset() clears outcome history
// ============================================================

TEST(KellyPositionSizer, ResetClearsHistory) {
    KellyPositionSizer::Config cfg;
    cfg.min_history = 5;
    KellyPositionSizer sizer(cfg);

    for (int i = 0; i < 10; ++i) sizer.record_outcome(true, 0.01);
    EXPECT_EQ(sizer.total_outcomes(), 10u);

    sizer.reset();
    EXPECT_EQ(sizer.total_outcomes(), 0u)
        << "total_outcomes must be zero after reset()";

    // After reset, insufficient history again — signal passes unchanged.
    auto sig   = make_sig(0.5);
    auto sized = sizer.size_signal(sig);
    EXPECT_DOUBLE_EQ(sized.delta_bias_shift, sig.delta_bias_shift);
}

// ============================================================
// Stats / JSON
// ============================================================

TEST(KellyPositionSizer, GetStatsReturnsConsistentValues) {
    KellyPositionSizer sizer;
    sizer.record_outcome(true,  0.02);
    sizer.record_outcome(false, 0.01);

    auto stats = sizer.get_stats();
    EXPECT_EQ(stats.total_wins,   1u);
    EXPECT_EQ(stats.total_losses, 1u);
    EXPECT_GE(stats.win_rate_ema, 0.0);
    EXPECT_LE(stats.win_rate_ema, 1.0);
}

TEST(KellyPositionSizer, ToStatsJsonIsValidJson) {
    KellyPositionSizer sizer;
    sizer.record_outcome(true, 0.02);
    std::string json = sizer.to_stats_json();
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(),  '}');
    EXPECT_NE(json.find("kelly_fraction"), std::string::npos);
    EXPECT_NE(json.find("win_rate"),       std::string::npos);
}

} // namespace
