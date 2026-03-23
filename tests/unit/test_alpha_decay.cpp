/// @file tests/unit/test_alpha_decay.cpp
/// @brief GTest unit tests for AlphaDecay and AlphaPortfolio — 25+ tests.

#include "alpha_decay.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

using namespace llmquant;

static constexpr double kEps    = 1e-9;
static constexpr double kRelEps = 1e-6;

// ---------------------------------------------------------------------------
// Helper factories
// ---------------------------------------------------------------------------

static AlphaSignal make_exp(double half_life_ms, double strength = 1.0,
                             int64_t generated_at = 0) {
    return AlphaSignal{strength, generated_at, DecayModel{Exponential{half_life_ms}}};
}

static AlphaSignal make_linear(double duration_ms, double strength = 1.0,
                                int64_t generated_at = 0) {
    return AlphaSignal{strength, generated_at, DecayModel{Linear{duration_ms}}};
}

static AlphaSignal make_power(double exp, double scale, double strength = 1.0,
                               int64_t generated_at = 0) {
    return AlphaSignal{strength, generated_at, DecayModel{PowerLaw{exp, scale}}};
}

static AlphaSignal make_step(std::vector<std::pair<int64_t, double>> levels,
                              double strength = 1.0, int64_t generated_at = 0) {
    return AlphaSignal{strength, generated_at, DecayModel{StepFunction{std::move(levels)}}};
}

// ---------------------------------------------------------------------------
// Exponential decay
// ---------------------------------------------------------------------------

TEST(AlphaDecayExponential, FullStrengthAtZero) {
    auto sig = make_exp(1000.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 0), 1.0, kEps);
}

TEST(AlphaDecayExponential, HalfStrengthAtHalfLife) {
    auto sig = make_exp(500.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 500), 0.5, kRelEps);
}

TEST(AlphaDecayExponential, QuarterStrengthAtTwoHalfLives) {
    auto sig = make_exp(500.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 1000), 0.25, kRelEps);
}

TEST(AlphaDecayExponential, ScaledByInitialStrength) {
    auto sig = make_exp(500.0, 0.8, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 500), 0.4, kRelEps);
}

TEST(AlphaDecayExponential, NegativeDtReturnsZero) {
    auto sig = make_exp(500.0, 1.0, 1000);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 500), 0.0, kEps);
}

TEST(AlphaDecayExponential, HalfLifeAnalytical) {
    auto sig = make_exp(750.0);
    EXPECT_NEAR(AlphaDecay::half_life_ms(sig), 750.0, kEps);
}

TEST(AlphaDecayExponential, ZeroHalfLifeReturnsZero) {
    auto sig = make_exp(0.0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 100), 0.0, kEps);
}

// ---------------------------------------------------------------------------
// Linear decay
// ---------------------------------------------------------------------------

TEST(AlphaDecayLinear, FullStrengthAtZero) {
    auto sig = make_linear(2000.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 0), 1.0, kEps);
}

TEST(AlphaDecayLinear, HalfStrengthAtHalfDuration) {
    auto sig = make_linear(2000.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 1000), 0.5, kEps);
}

TEST(AlphaDecayLinear, ZeroAtDuration) {
    auto sig = make_linear(2000.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 2000), 0.0, kEps);
}

TEST(AlphaDecayLinear, ClampedToZeroBeyondDuration) {
    auto sig = make_linear(1000.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 5000), 0.0, kEps);
}

TEST(AlphaDecayLinear, HalfLifeIsHalfDuration) {
    auto sig = make_linear(1000.0);
    EXPECT_NEAR(AlphaDecay::half_life_ms(sig), 500.0, kEps);
}

// ---------------------------------------------------------------------------
// Power-law decay
// ---------------------------------------------------------------------------

TEST(AlphaDecayPowerLaw, FullStrengthAtZero) {
    auto sig = make_power(2.0, 500.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 0), 1.0, kEps);
}

TEST(AlphaDecayPowerLaw, DecaysAsExpected) {
    // s(t) = (500 / (500 + 500))^2 = (0.5)^2 = 0.25
    auto sig = make_power(2.0, 500.0, 1.0, 0);
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 500), 0.25, kRelEps);
}

TEST(AlphaDecayPowerLaw, HalfLifeAnalytical) {
    // t_hl = scale * (2^(1/exp) - 1)
    double scale = 500.0, exp = 2.0;
    double expected_hl = scale * (std::pow(2.0, 1.0 / exp) - 1.0);
    auto sig = make_power(exp, scale);
    EXPECT_NEAR(AlphaDecay::half_life_ms(sig), expected_hl, kRelEps);
}

TEST(AlphaDecayPowerLaw, MonotonicallyDecreasing) {
    auto sig = make_power(1.5, 300.0, 1.0, 0);
    double prev = 1.0;
    for (int64_t t = 100; t <= 5000; t += 100) {
        double s = AlphaDecay::current_strength(sig, t);
        EXPECT_LT(s, prev) << "Not monotone at t=" << t;
        prev = s;
    }
}

// ---------------------------------------------------------------------------
// Step function decay
// ---------------------------------------------------------------------------

TEST(AlphaDecayStep, ReturnsFirstLevelBeforeAnyThreshold) {
    auto sig = make_step({{500, 0.8}, {1000, 0.4}, {2000, 0.0}}, 1.0, 0);
    // Δt = 100 < 500: first level's alpha = 0.8... but wait, the levels are thresholds.
    // At Δt=100, threshold 500 not yet reached → uses the alpha up to the first threshold.
    // By our implementation: iterate thresholds; if threshold_ms <= dt, use alpha.
    // At Δt=100: no threshold <= 100, so default = first alpha = 0.8.
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 100), 0.8, kEps);
}

TEST(AlphaDecayStep, AtFirstThreshold) {
    auto sig = make_step({{500, 0.8}, {1000, 0.4}, {2000, 0.0}}, 1.0, 0);
    // Δt=500: threshold 500 <= 500 → alpha=0.8. threshold 1000 > 500 → stop.
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 500), 0.8, kEps);
}

TEST(AlphaDecayStep, BetweenThresholds) {
    auto sig = make_step({{500, 0.8}, {1000, 0.4}, {2000, 0.0}}, 1.0, 0);
    // Δt=800: thresholds 500 and below → alpha=0.8. 1000 > 800 → stop.
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 800), 0.8, kEps);
}

TEST(AlphaDecayStep, AtSecondThreshold) {
    auto sig = make_step({{500, 0.8}, {1000, 0.4}, {2000, 0.0}}, 1.0, 0);
    // Δt=1000: thresholds 500 and 1000 both <= 1000 → alpha=0.4.
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 1000), 0.4, kEps);
}

TEST(AlphaDecayStep, BeyondLastThreshold) {
    auto sig = make_step({{500, 0.8}, {1000, 0.4}, {2000, 0.0}}, 1.0, 0);
    // Δt=3000 > 2000 → alpha=0.0.
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 3000), 0.0, kEps);
}

TEST(AlphaDecayStep, EmptyLevelsReturnsZero) {
    auto sig = make_step({});
    EXPECT_NEAR(AlphaDecay::current_strength(sig, 100), 0.0, kEps);
}

// ---------------------------------------------------------------------------
// AlphaPortfolio
// ---------------------------------------------------------------------------

TEST(AlphaPortfolio, AddAndQuerySingleSignal) {
    AlphaPortfolio port;
    port.add_signal("AAPL", make_exp(1000.0, 1.0, 0));
    double net = port.net_alpha("AAPL", 0);
    EXPECT_NEAR(net, 1.0, kEps);
}

TEST(AlphaPortfolio, MultipleSignalsSummed) {
    AlphaPortfolio port;
    port.add_signal("AAPL", make_exp(1000.0, 1.0, 0));  // strength=1.0
    port.add_signal("AAPL", make_linear(2000.0, 0.5, 0)); // strength=0.5
    double net = port.net_alpha("AAPL", 0);
    EXPECT_NEAR(net, 1.5, kEps);
}

TEST(AlphaPortfolio, UnknownSymbolReturnsZero) {
    AlphaPortfolio port;
    EXPECT_NEAR(port.net_alpha("MSFT", 0), 0.0, kEps);
}

TEST(AlphaPortfolio, PruneRemovesWeakSignals) {
    AlphaPortfolio port;
    // Linear signal fully decays at t=1000ms
    port.add_signal("AAPL", make_linear(1000.0, 1.0, 0));
    EXPECT_EQ(port.signal_count("AAPL"), 1u);
    port.prune(2000, 0.01);  // strength < 0.01 after full decay
    EXPECT_EQ(port.signal_count("AAPL"), 0u);
}

TEST(AlphaPortfolio, PruneKeepsStrongSignals) {
    AlphaPortfolio port;
    port.add_signal("AAPL", make_exp(10000.0, 1.0, 0));  // strong
    port.add_signal("AAPL", make_linear(1.0, 1.0, 0));   // decays immediately
    port.prune(100, 0.01);
    EXPECT_EQ(port.signal_count("AAPL"), 1u);  // only strong one remains
}

TEST(AlphaPortfolio, PruneRemovesEmptySymbols) {
    AlphaPortfolio port;
    port.add_signal("AAPL", make_linear(1.0, 1.0, 0));
    EXPECT_EQ(port.symbols().size(), 1u);
    port.prune(100, 0.01);
    EXPECT_EQ(port.symbols().size(), 0u);
}

TEST(AlphaPortfolio, MultipleSymbols) {
    AlphaPortfolio port;
    port.add_signal("AAPL", make_exp(1000.0, 0.8, 0));
    port.add_signal("MSFT", make_exp(1000.0, 0.6, 0));
    EXPECT_NEAR(port.net_alpha("AAPL", 0), 0.8, kEps);
    EXPECT_NEAR(port.net_alpha("MSFT", 0), 0.6, kEps);
}

TEST(AlphaPortfolio, Clear) {
    AlphaPortfolio port;
    port.add_signal("AAPL", make_exp(1000.0));
    port.add_signal("MSFT", make_exp(1000.0));
    port.clear();
    EXPECT_EQ(port.symbols().size(), 0u);
    EXPECT_NEAR(port.net_alpha("AAPL", 0), 0.0, kEps);
}

TEST(AlphaPortfolio, NetAlphaDecaysOverTime) {
    AlphaPortfolio port;
    port.add_signal("TSLA", make_exp(500.0, 1.0, 0));
    double t0 = port.net_alpha("TSLA", 0);
    double t1 = port.net_alpha("TSLA", 500);
    double t2 = port.net_alpha("TSLA", 1000);
    EXPECT_GT(t0, t1);
    EXPECT_GT(t1, t2);
}
