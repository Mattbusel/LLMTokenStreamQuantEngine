#define LLMQUANT_COHERENCE_TRACKER_ENABLED
#include <gtest/gtest.h>
#include "NarrativeCoherenceTracker.h"

using namespace llmquant;

TEST(NarrativeCoherenceTracker, DefaultConstruction) {
    NarrativeCoherenceTracker t(NarrativeCoherenceTracker::Config{});
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.coherence(), 0.0);
}

TEST(NarrativeCoherenceTracker, AllPositiveHighCoherence) {
    NarrativeCoherenceTracker::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    cfg.high_threshold = 0.7;
    NarrativeCoherenceTracker t(cfg);
    for (int i = 0; i < 10; ++i) t.record(0.1);
    EXPECT_GT(t.coherence(), 0.7);
}

TEST(NarrativeCoherenceTracker, MixedLowCoherence) {
    NarrativeCoherenceTracker::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    NarrativeCoherenceTracker t(cfg);
    for (int i = 0; i < 10; ++i) t.record(i % 2 == 0 ? 0.1 : -0.1);
    // Should be around 0.5 — neither extreme
    EXPECT_LT(t.coherence(), 0.8);
}

TEST(NarrativeCoherenceTracker, HighCoherenceCallback) {
    NarrativeCoherenceTracker::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.high_threshold = 0.6;
    int high_count = 0;
    cfg.on_high_coherence = [&](double) { ++high_count; };
    NarrativeCoherenceTracker t(cfg);
    for (int i = 0; i < 5; ++i) t.record(0.1);
    EXPECT_GE(high_count, 1);
}

TEST(NarrativeCoherenceTracker, LowCoherenceCallback) {
    NarrativeCoherenceTracker::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.low_threshold = 0.6; // very high threshold so mixed → low
    int low_count = 0;
    cfg.on_low_coherence = [&](double) { ++low_count; };
    NarrativeCoherenceTracker t(cfg);
    // Start uniform, then alternate
    for (int i = 0; i < 3; ++i) t.record(0.1);
    for (int i = 0; i < 5; ++i) t.record(i % 2 == 0 ? 0.1 : -0.1);
    EXPECT_GE(low_count, 1);
}

TEST(NarrativeCoherenceTracker, Reset) {
    NarrativeCoherenceTracker t(NarrativeCoherenceTracker::Config{});
    for (int i = 0; i < 10; ++i) t.record(0.1);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_EQ(t.high_coherence_events(), 0u);
    EXPECT_EQ(t.low_coherence_events(), 0u);
}

TEST(NarrativeCoherenceTracker, ToStatsJson) {
    NarrativeCoherenceTracker t(NarrativeCoherenceTracker::Config{});
    for (int i = 0; i < 5; ++i) t.record(0.05);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("coherence"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
