#include <gtest/gtest.h>

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED

#include "BiasVelocityBreaker.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(BiasVelocityBreaker, DefaultState) {
    BiasVelocityBreaker b;
    EXPECT_NEAR(b.velocity(), 0.0, 1e-9);
    EXPECT_FALSE(b.is_tripped());
    EXPECT_EQ(b.trip_count(), 0u);
    EXPECT_EQ(b.total_records(), 0u);
}

TEST(BiasVelocityBreaker, TotalRecordsIncrements) {
    BiasVelocityBreaker b;
    b.record(0.1);
    b.record(0.2);
    EXPECT_EQ(b.total_records(), 2u);
}

TEST(BiasVelocityBreaker, SteadySignalNoTrip) {
    BiasVelocityBreaker::Config cfg;
    cfg.max_velocity = 0.15;
    BiasVelocityBreaker b(cfg);
    for (int i = 0; i < 20; ++i) b.record(0.5);
    EXPECT_FALSE(b.is_tripped());
    EXPECT_EQ(b.trip_count(), 0u);
}

TEST(BiasVelocityBreaker, LargeJumpTripsBreaker) {
    BiasVelocityBreaker::Config cfg;
    cfg.ema_alpha    = 1.0;  // instant update
    cfg.max_velocity = 0.1;
    BiasVelocityBreaker b(cfg);
    b.record(0.0);
    b.record(0.5);  // |Δ| = 0.5 > 0.1
    EXPECT_TRUE(b.is_tripped());
    EXPECT_GT(b.trip_count(), 0u);
}

TEST(BiasVelocityBreaker, TripCallbackFires) {
    BiasVelocityBreaker::Config cfg;
    cfg.ema_alpha    = 1.0;
    cfg.max_velocity = 0.1;
    std::atomic<int> fires{0};
    cfg.on_trip = [&](double) { ++fires; };
    BiasVelocityBreaker b(cfg);
    b.record(0.0);
    b.record(0.5);
    EXPECT_GE(fires.load(), 1);
}

TEST(BiasVelocityBreaker, ClearCallbackFires) {
    BiasVelocityBreaker::Config cfg;
    cfg.ema_alpha        = 0.9;
    cfg.max_velocity     = 0.05;
    cfg.clear_hysteresis = 0.5;
    std::atomic<int> cleared{0};
    cfg.on_clear = [&](double) { ++cleared; };
    BiasVelocityBreaker b(cfg);
    b.record(0.0);
    for (int i = 0; i < 5; ++i) b.record(0.5);   // trip
    for (int i = 0; i < 20; ++i) b.record(0.5);  // stabilize
    EXPECT_GE(cleared.load(), 1);
}

TEST(BiasVelocityBreaker, ReturnValueReflectsTrippedState) {
    BiasVelocityBreaker::Config cfg;
    cfg.ema_alpha    = 1.0;
    cfg.max_velocity = 0.1;
    BiasVelocityBreaker b(cfg);
    b.record(0.0);
    bool tripped = b.record(0.5);
    EXPECT_TRUE(tripped);
}

TEST(BiasVelocityBreaker, VelocityBounded) {
    BiasVelocityBreaker b;
    for (int i = 0; i < 100; ++i)
        b.record(i % 2 == 0 ? 1.0 : -1.0);
    EXPECT_GE(b.velocity(), 0.0);
}

TEST(BiasVelocityBreaker, ResetClearsState) {
    BiasVelocityBreaker b;
    b.record(0.0);
    b.record(1.0);
    b.reset();
    EXPECT_NEAR(b.velocity(), 0.0, 1e-9);
    EXPECT_FALSE(b.is_tripped());
    EXPECT_EQ(b.total_records(), 0u);
}

TEST(BiasVelocityBreaker, UpdateConfigResetsState) {
    BiasVelocityBreaker b;
    b.record(0.5);
    BiasVelocityBreaker::Config cfg2;
    cfg2.max_velocity = 0.2;
    b.update_config(cfg2);
    EXPECT_EQ(b.total_records(), 0u);
}

TEST(BiasVelocityBreaker, StatsJsonContainsFields) {
    BiasVelocityBreaker b;
    b.record(0.1);
    b.record(0.5);
    std::string j = b.to_stats_json();
    EXPECT_NE(j.find("velocity"),      std::string::npos);
    EXPECT_NE(j.find("is_tripped"),    std::string::npos);
    EXPECT_NE(j.find("trip_count"),    std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(BiasVelocityBreaker, ConcurrentRecordSafe) {
    BiasVelocityBreaker::Config cfg;
    cfg.max_velocity = 0.5;
    BiasVelocityBreaker b(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            b.record(id % 2 == 0 ? static_cast<double>(i) * 0.01 : -static_cast<double>(i) * 0.01);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();
    EXPECT_EQ(b.total_records(), 1000u);
}

}  // namespace

#else

TEST(BiasVelocityBreaker, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_VELOCITY_BREAKER_ENABLED
