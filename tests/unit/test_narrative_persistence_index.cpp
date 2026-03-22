#include <gtest/gtest.h>
#include "NarrativePersistenceIndex.h"

namespace llmquant::tests {

TEST(NarrativePersistenceIndex, DefaultConstruction) {
    NarrativePersistenceIndex p;
    EXPECT_EQ(p.total_records(), 0u);
    // Default PI is 0.5 (neutral)
    EXPECT_DOUBLE_EQ(p.persistence_index(), 0.5);
}

TEST(NarrativePersistenceIndex, HighPIForTrend) {
    NarrativePersistenceIndex::Config cfg;
    cfg.window = 20; cfg.min_samples = 8;
    cfg.persistent_threshold = 0.5;
    NarrativePersistenceIndex p(cfg);
    // Monotone uptrend → one long run → high PI
    for (int i = 0; i < 30; ++i) p.record(static_cast<double>(i));
    EXPECT_GT(p.persistence_index(), 0.5);
}

TEST(NarrativePersistenceIndex, LowPIForAlternating) {
    NarrativePersistenceIndex::Config cfg;
    cfg.window = 20; cfg.min_samples = 8;
    cfg.reverting_threshold = 0.5;
    NarrativePersistenceIndex p(cfg);
    // Alternating → many runs → low PI
    for (int i = 0; i < 30; ++i) p.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_LT(p.persistence_index(), 0.5);
}

TEST(NarrativePersistenceIndex, PersistentCallback) {
    NarrativePersistenceIndex::Config cfg;
    cfg.window = 10; cfg.min_samples = 5;
    cfg.persistent_threshold = 0.3; // easy to trigger
    int events = 0;
    cfg.on_persistent = [&](double) { ++events; };
    NarrativePersistenceIndex p(cfg);
    for (int i = 0; i < 30; ++i) p.record(static_cast<double>(i));
    EXPECT_GT(events, 0);
}

TEST(NarrativePersistenceIndex, RevertingCallback) {
    NarrativePersistenceIndex::Config cfg;
    cfg.window = 10; cfg.min_samples = 5;
    cfg.reverting_threshold = 0.9; // easy to trigger
    int events = 0;
    cfg.on_mean_reverting = [&](double) { ++events; };
    NarrativePersistenceIndex p(cfg);
    for (int i = 0; i < 30; ++i) p.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GT(events, 0);
}

TEST(NarrativePersistenceIndex, Reset) {
    NarrativePersistenceIndex p;
    for (int i = 0; i < 30; ++i) p.record(0.1);
    p.reset();
    EXPECT_EQ(p.total_records(), 0u);
    EXPECT_DOUBLE_EQ(p.persistence_index(), 0.5);
}

TEST(NarrativePersistenceIndex, ToStatsJson) {
    NarrativePersistenceIndex p;
    for (int i = 0; i < 20; ++i) p.record(0.05);
    auto json = p.to_stats_json();
    EXPECT_NE(json.find("persistence_index"), std::string::npos);
    EXPECT_NE(json.find("persistent_events"), std::string::npos);
}

} // namespace llmquant::tests
