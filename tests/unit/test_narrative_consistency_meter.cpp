#include <gtest/gtest.h>

#ifdef LLMQUANT_CONSISTENCY_METER_ENABLED

#include "NarrativeConsistencyMeter.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(NarrativeConsistencyMeter, DefaultState) {
    NarrativeConsistencyMeter m;
    EXPECT_NEAR(m.consistency_score(), 1.0, 1e-9);
    EXPECT_NEAR(m.aad(), 0.0, 1e-9);
    EXPECT_FALSE(m.is_inconsistent());
    EXPECT_EQ(m.total_records(), 0u);
}

TEST(NarrativeConsistencyMeter, TotalRecordsIncrements) {
    NarrativeConsistencyMeter m;
    m.record(0.1);
    m.record(0.2);
    m.record(0.3);
    EXPECT_EQ(m.total_records(), 3u);
}

TEST(NarrativeConsistencyMeter, SteadySignalHighScore) {
    NarrativeConsistencyMeter::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    cfg.normalizer  = 0.3;
    NarrativeConsistencyMeter m(cfg);
    for (int i = 0; i < 32; ++i) m.record(0.5);
    EXPECT_GT(m.consistency_score(), 0.9);
}

TEST(NarrativeConsistencyMeter, LargeJumpsLowScore) {
    NarrativeConsistencyMeter::Config cfg;
    cfg.window                  = 16;
    cfg.min_samples             = 4;
    cfg.normalizer              = 0.1;
    cfg.consistency_threshold   = 0.8;
    NarrativeConsistencyMeter m(cfg);
    for (int i = 0; i < 32; ++i)
        m.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_LT(m.consistency_score(), 0.5);
}

TEST(NarrativeConsistencyMeter, ScoreBounded) {
    NarrativeConsistencyMeter::Config cfg;
    cfg.window      = 16;
    cfg.min_samples = 4;
    NarrativeConsistencyMeter m(cfg);
    for (int i = 0; i < 64; ++i)
        m.record(static_cast<double>(i % 7) * 0.15 - 0.4);
    double s = m.consistency_score();
    EXPECT_GE(s, 0.0);
    EXPECT_LE(s, 1.0);
}

TEST(NarrativeConsistencyMeter, InconsistentCallbackFires) {
    NarrativeConsistencyMeter::Config cfg;
    cfg.window                = 8;
    cfg.min_samples           = 4;
    cfg.normalizer            = 0.05;
    cfg.consistency_threshold = 0.95;
    std::atomic<int> fires{0};
    cfg.on_inconsistent = [&](double) { ++fires; };
    NarrativeConsistencyMeter m(cfg);
    for (int i = 0; i < 32; ++i)
        m.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(m.inconsistency_events(), 0u);
}

TEST(NarrativeConsistencyMeter, RecoveredCallbackFires) {
    NarrativeConsistencyMeter::Config cfg;
    cfg.window                = 8;
    cfg.min_samples           = 4;
    cfg.normalizer            = 0.05;
    cfg.consistency_threshold = 0.9;
    cfg.clear_hysteresis      = 0.5;
    std::atomic<int> recovered{0};
    cfg.on_recovered = [&](double) { ++recovered; };
    NarrativeConsistencyMeter m(cfg);
    // Enter inconsistent state.
    for (int i = 0; i < 16; ++i) m.record(i % 2 == 0 ? 1.0 : -1.0);
    // Stabilize.
    for (int i = 0; i < 32; ++i) m.record(0.5);
    EXPECT_GE(recovered.load(), 1);
}

TEST(NarrativeConsistencyMeter, ResetClearsState) {
    NarrativeConsistencyMeter m;
    for (int i = 0; i < 16; ++i) m.record(static_cast<double>(i) * 0.1);
    m.reset();
    EXPECT_EQ(m.total_records(), 0u);
    EXPECT_NEAR(m.consistency_score(), 1.0, 1e-9);
    EXPECT_FALSE(m.is_inconsistent());
}

TEST(NarrativeConsistencyMeter, UpdateConfigResetsState) {
    NarrativeConsistencyMeter m;
    m.record(0.5);
    NarrativeConsistencyMeter::Config cfg2;
    cfg2.window = 8;
    m.update_config(cfg2);
    EXPECT_EQ(m.total_records(), 0u);
}

TEST(NarrativeConsistencyMeter, StatsJsonContainsFields) {
    NarrativeConsistencyMeter m;
    for (int i = 0; i < 10; ++i) m.record(static_cast<double>(i) * 0.1);
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("consistency_score"),    std::string::npos);
    EXPECT_NE(j.find("aad"),                  std::string::npos);
    EXPECT_NE(j.find("is_inconsistent"),      std::string::npos);
    EXPECT_NE(j.find("inconsistency_events"), std::string::npos);
    EXPECT_NE(j.find("total_records"),        std::string::npos);
}

TEST(NarrativeConsistencyMeter, ConcurrentRecordSafe) {
    NarrativeConsistencyMeter::Config cfg;
    cfg.window      = 32;
    cfg.min_samples = 8;
    NarrativeConsistencyMeter m(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            m.record(id % 2 == 0 ? 0.5 : static_cast<double>(i % 5) * 0.2 - 0.4);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(m.total_records(), 1000u);
}

}  // namespace

#else

TEST(NarrativeConsistencyMeter, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_CONSISTENCY_METER_ENABLED
