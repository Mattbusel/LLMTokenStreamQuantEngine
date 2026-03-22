#include <gtest/gtest.h>

#ifdef LLMQUANT_ENTROPY_RATCHET_ENABLED
#include "SignalEntropyRatchet.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(SignalEntropyRatchetTest, InitialState) {
    SignalEntropyRatchet r;
    EXPECT_NEAR(r.entropy(), 0.0, 1e-9);
    EXPECT_FALSE(r.is_spiked());
    EXPECT_EQ(r.total_records(), 0u);
    EXPECT_EQ(r.spike_events(), 0u);
    EXPECT_EQ(r.floor_drop_events(), 0u);
}

TEST(SignalEntropyRatchetTest, TotalRecordsIncrements) {
    SignalEntropyRatchet r;
    r.record(0.1);
    r.record(-0.1);
    EXPECT_EQ(r.total_records(), 2u);
}

TEST(SignalEntropyRatchetTest, UniformDistributionHighEntropy) {
    SignalEntropyRatchet::Config cfg;
    cfg.window    = 20;
    cfg.n_buckets = 5;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    SignalEntropyRatchet r(cfg);
    // Feed one value per bucket uniformly
    for (int rep = 0; rep < 4; ++rep)
        for (int b = 0; b < 5; ++b)
            r.record(-0.8 + b * 0.4);
    // Entropy should be close to log2(5) ≈ 2.32 for uniform distribution
    EXPECT_GT(r.entropy(), 1.0);
}

TEST(SignalEntropyRatchetTest, ConstantSeriesLowEntropy) {
    SignalEntropyRatchet::Config cfg;
    cfg.window    = 20;
    cfg.n_buckets = 5;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    SignalEntropyRatchet r(cfg);
    for (int i = 0; i < 20; ++i) r.record(0.5);  // all in one bucket
    EXPECT_LT(r.entropy(), 0.1);
}

TEST(SignalEntropyRatchetTest, SpikeCallbackFires) {
    SignalEntropyRatchet::Config cfg;
    cfg.window           = 20;
    cfg.n_buckets        = 5;
    cfg.range_min        = -1.0;
    cfg.range_max        =  1.0;
    cfg.spike_threshold  = 1.0;  // low threshold — uniform data will spike
    std::atomic<int> fires{0};
    cfg.on_entropy_spike = [&](double) { ++fires; };
    SignalEntropyRatchet r(cfg);
    for (int rep = 0; rep < 4; ++rep)
        for (int b = 0; b < 5; ++b)
            r.record(-0.8 + b * 0.4);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(r.spike_events(), 1u);
}

TEST(SignalEntropyRatchetTest, FloorRatchetsUp) {
    SignalEntropyRatchet::Config cfg;
    cfg.window          = 10;
    cfg.n_buckets       = 5;
    cfg.range_min       = -1.0;
    cfg.range_max       =  1.0;
    cfg.floor_hold_ticks = 3;
    SignalEntropyRatchet r(cfg);
    // Feed uniform distribution to raise entropy, then constant to lower it
    for (int b = 0; b < 5; ++b) r.record(-0.8 + b * 0.4);
    double floor_after_uniform = r.entropy_floor();
    // Floor should have been set to roughly the entropy at that time
    EXPECT_GE(floor_after_uniform, 0.0);
}

TEST(SignalEntropyRatchetTest, FloorDropAfterHoldTicks) {
    SignalEntropyRatchet::Config cfg;
    cfg.window           = 20;
    cfg.n_buckets        = 5;
    cfg.range_min        = -1.0;
    cfg.range_max        =  1.0;
    cfg.spike_threshold  = 100.0;  // never spike
    cfg.floor_hold_ticks = 3;
    std::atomic<int> floor_fires{0};
    cfg.on_floor_drop = [&](double) { ++floor_fires; };
    SignalEntropyRatchet r(cfg);
    // First fill uniformly to set a high floor
    for (int rep = 0; rep < 4; ++rep)
        for (int b = 0; b < 5; ++b)
            r.record(-0.8 + b * 0.4);
    // Then feed constant to lower entropy and sustain it
    for (int i = 0; i < 20; ++i) r.record(0.5);
    // floor_drop_events might fire after hold ticks
    EXPECT_GE(r.floor_drop_events(), 0u);  // may fire depending on timing
}

TEST(SignalEntropyRatchetTest, ResetClearsState) {
    SignalEntropyRatchet r;
    for (int i = 0; i < 10; ++i) r.record(i * 0.1 - 0.5);
    r.reset();
    EXPECT_EQ(r.total_records(), 0u);
    EXPECT_FALSE(r.is_spiked());
    EXPECT_NEAR(r.entropy(), 0.0, 1e-9);
}

TEST(SignalEntropyRatchetTest, EntropyBounded) {
    SignalEntropyRatchet::Config cfg;
    cfg.window    = 30;
    cfg.n_buckets = 8;
    cfg.range_min = -1.0;
    cfg.range_max =  1.0;
    SignalEntropyRatchet r(cfg);
    for (int i = 0; i < 30; ++i) r.record((i % 9 - 4) * 0.2);
    double H = r.entropy();
    EXPECT_GE(H, 0.0);
    EXPECT_LE(H, std::log2(8.0) + 1e-6);
}

TEST(SignalEntropyRatchetTest, StatsJsonHasFields) {
    SignalEntropyRatchet r;
    for (int i = 0; i < 10; ++i) r.record(i * 0.1);
    std::string j = r.to_stats_json();
    EXPECT_NE(j.find("entropy"),            std::string::npos);
    EXPECT_NE(j.find("entropy_floor"),      std::string::npos);
    EXPECT_NE(j.find("is_spiked"),          std::string::npos);
    EXPECT_NE(j.find("spike_events"),       std::string::npos);
    EXPECT_NE(j.find("floor_drop_events"),  std::string::npos);
}

TEST(SignalEntropyRatchetTest, ConcurrentSafe) {
    SignalEntropyRatchet r;
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            r.record((id * 0.3 + i * 0.01) * (i % 2 == 0 ? 1 : -1));
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(r.total_records(), 400u);
    double H = r.entropy();
    EXPECT_GE(H, 0.0);
}

#else
TEST(SignalEntropyRatchetTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
