#include <gtest/gtest.h>

#ifdef LLMQUANT_INTENSITY_RAMP_ENABLED
#include "NarrativeIntensityRamp.h"
using namespace llmquant;

TEST(NarrativeIntensityRampTest, InitialState) {
    NarrativeIntensityRamp r;
    EXPECT_EQ(r.intensity(), 0.0);
    EXPECT_EQ(r.ramp(), 0.0);
    EXPECT_FALSE(r.is_surging());
    EXPECT_FALSE(r.is_fading());
    EXPECT_EQ(r.observation_count(), 0u);
}

TEST(NarrativeIntensityRampTest, ObservationCountIncrement) {
    NarrativeIntensityRamp r;
    r.record(0.5);
    r.record(0.3);
    EXPECT_EQ(r.observation_count(), 2u);
}

TEST(NarrativeIntensityRampTest, IntensityIsNonNegative) {
    NarrativeIntensityRamp r;
    for (int i = 0; i < 20; ++i) r.record(i % 2 == 0 ? 0.5 : -0.3);
    EXPECT_GE(r.intensity(), 0.0);
}

TEST(NarrativeIntensityRampTest, SurgeDetected) {
    NarrativeIntensityRamp::Config cfg;
    cfg.ema_alpha       = 0.5;  // fast convergence
    cfg.surge_threshold = 0.01;
    cfg.saturation_ramp = 0.05;
    NarrativeIntensityRamp r(cfg);
    r.record(0.0);
    r.record(0.9); // large jump → positive ramp
    EXPECT_TRUE(r.is_surging());
}

TEST(NarrativeIntensityRampTest, FadeDetected) {
    NarrativeIntensityRamp::Config cfg;
    cfg.ema_alpha       = 0.5;
    cfg.surge_threshold = 0.01;
    cfg.saturation_ramp = 0.05;
    NarrativeIntensityRamp r(cfg);
    r.record(0.9); // high intensity
    r.record(0.0); // drops to near zero → negative ramp
    EXPECT_TRUE(r.is_fading());
}

TEST(NarrativeIntensityRampTest, SurgeCallback) {
    NarrativeIntensityRamp::Config cfg;
    cfg.ema_alpha       = 0.5;
    cfg.surge_threshold = 0.01;
    int fires = 0;
    cfg.on_surge = [&](double) { ++fires; };
    NarrativeIntensityRamp r(cfg);
    r.record(0.0);
    r.record(0.9);
    EXPECT_GE(fires, 1);
}

TEST(NarrativeIntensityRampTest, RampScoreRange) {
    NarrativeIntensityRamp r;
    for (int i = 0; i < 30; ++i) r.record(i % 5 == 0 ? 0.8 : 0.1);
    EXPECT_GE(r.ramp_score(), -1.0);
    EXPECT_LE(r.ramp_score(), 1.0);
}

TEST(NarrativeIntensityRampTest, Reset) {
    NarrativeIntensityRamp r;
    for (int i = 0; i < 20; ++i) r.record(0.5);
    r.reset();
    EXPECT_EQ(r.observation_count(), 0u);
    EXPECT_EQ(r.intensity(), 0.0);
    EXPECT_FALSE(r.is_surging());
}

TEST(NarrativeIntensityRampTest, StatsJson) {
    NarrativeIntensityRamp r;
    r.record(0.5);
    std::string json = r.to_stats_json();
    EXPECT_NE(json.find("intensity"), std::string::npos);
    EXPECT_NE(json.find("ramp"), std::string::npos);
    EXPECT_NE(json.find("ramp_score"), std::string::npos);
}

#else
TEST(NarrativeIntensityRampTest, Disabled) { SUCCEED(); }
#endif
