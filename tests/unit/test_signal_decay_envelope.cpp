#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED

#include "SignalDecayEnvelope.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(SignalDecayEnvelope, DefaultConstructsWithZeroBias) {
    SignalDecayEnvelope env;
    EXPECT_NEAR(env.raw_bias(),     0.0, 1e-12);
    EXPECT_NEAR(env.decayed_bias(), 0.0, 1e-12);
    EXPECT_EQ(env.total_reinforcements(), 0u);
}

// ============================================================
// reinforce
// ============================================================

TEST(SignalDecayEnvelope, ReinforceSetsRawBias) {
    SignalDecayEnvelope env;
    env.reinforce(0.5);
    EXPECT_NEAR(env.raw_bias(), 0.5, 1e-9);
    EXPECT_EQ(env.total_reinforcements(), 1u);
}

TEST(SignalDecayEnvelope, MultipleReinforcementsAccumulate) {
    SignalDecayEnvelope env;
    env.reinforce( 0.3);
    env.reinforce( 0.4);
    env.reinforce(-0.1);
    EXPECT_NEAR(env.raw_bias(), 0.6, 1e-9);
    EXPECT_EQ(env.total_reinforcements(), 3u);
}

// ============================================================
// Decay: bias decays toward zero over time
// ============================================================

TEST(SignalDecayEnvelope, BiasDecaysAfterReinforcement) {
    SignalDecayEnvelope::Config cfg;
    cfg.half_life_ms = 50.0;  // 50 ms half-life
    cfg.clamp = false;
    SignalDecayEnvelope env(cfg);

    env.reinforce(1.0);
    double immediate = env.decayed_bias();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    double after = env.decayed_bias();

    // After ~2 half-lives, bias should be ~1/4 of original.
    EXPECT_LT(after, immediate);
    EXPECT_NEAR(after, immediate * 0.25, 0.1);  // loose tolerance for timing
}

// ============================================================
// Decay reaches zero without reinforce
// ============================================================

TEST(SignalDecayEnvelope, ZeroBiasRemainsZero) {
    SignalDecayEnvelope::Config cfg;
    cfg.half_life_ms = 10.0;
    SignalDecayEnvelope env(cfg);
    // No reinforce — bias is 0 throughout.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_NEAR(env.decayed_bias(), 0.0, 1e-12);
}

// ============================================================
// Clamping
// ============================================================

TEST(SignalDecayEnvelope, ClampingLimitsRawBias) {
    SignalDecayEnvelope::Config cfg;
    cfg.clamp   = true;
    cfg.max_bias = 0.5;
    cfg.min_bias = -0.5;
    SignalDecayEnvelope env(cfg);

    env.reinforce(2.0);  // would be 2.0 without clamp
    EXPECT_NEAR(env.raw_bias(), 0.5, 1e-9);
}

// ============================================================
// ms_since_last_reinforce increases over time
// ============================================================

TEST(SignalDecayEnvelope, MsSinceLastReinforceMeasuresTime) {
    SignalDecayEnvelope env;
    env.reinforce(0.1);
    int64_t t0 = env.ms_since_last_reinforce();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    int64_t t1 = env.ms_since_last_reinforce();
    EXPECT_GE(t1, t0);
}

// ============================================================
// reset
// ============================================================

TEST(SignalDecayEnvelope, ResetClearsBias) {
    SignalDecayEnvelope env;
    env.reinforce(0.8);
    env.reset();
    EXPECT_NEAR(env.raw_bias(),     0.0, 1e-12);
    EXPECT_NEAR(env.decayed_bias(), 0.0, 1e-12);
}

// ============================================================
// Zero-cross callback
// ============================================================

TEST(SignalDecayEnvelope, ZeroCrossCallbackFiredOnSignChange) {
    SignalDecayEnvelope::Config cfg;
    cfg.clamp = false;
    cfg.half_life_ms = 1'000'000.0;  // effectively no decay
    SignalDecayEnvelope env(cfg);

    std::atomic<int> crosses{0};
    env.set_zero_cross_callback([&](double, double) { crosses.fetch_add(1); });

    env.reinforce( 0.5);
    (void)env.decayed_bias();  // establishes positive sign (last_reported_sign_ = +1)
    env.reinforce(-1.5);       // overwrite bias to -1.5 (negative), no reset needed
    (void)env.decayed_bias();  // triggers cross: sign changed from +1 to -1

    EXPECT_GE(crosses.load(), 1);
}

// ============================================================
// update_config
// ============================================================

TEST(SignalDecayEnvelope, UpdateConfigResetsState) {
    SignalDecayEnvelope env;
    env.reinforce(0.9);
    SignalDecayEnvelope::Config cfg;
    cfg.half_life_ms = 5.0;
    env.update_config(cfg);
    EXPECT_NEAR(env.raw_bias(), 0.0, 1e-12);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(SignalDecayEnvelope, ToStatsJsonContainsKeys) {
    SignalDecayEnvelope env;
    env.reinforce(0.3);
    std::string json = env.to_stats_json();
    EXPECT_NE(json.find("raw_bias"),              std::string::npos);
    EXPECT_NE(json.find("decayed_bias"),          std::string::npos);
    EXPECT_NE(json.find("age_ms"),                std::string::npos);
    EXPECT_NE(json.find("half_life_ms"),          std::string::npos);
    EXPECT_NE(json.find("total_reinforcements"),  std::string::npos);
}

} // namespace

#else  // LLMQUANT_SIGNAL_DECAY_ENABLED not defined

TEST(SignalDecayEnvelope, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SIGNAL_DECAY_ENABLED
