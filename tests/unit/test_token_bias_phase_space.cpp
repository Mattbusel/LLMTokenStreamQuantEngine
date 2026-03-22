#include <gtest/gtest.h>

#ifdef LLMQUANT_PHASE_SPACE_ENABLED
#include "TokenBiasPhaseSpace.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(TokenBiasPhaseSpace, DefaultState) {
    TokenBiasPhaseSpace ps;
    EXPECT_EQ(ps.total_records(), 0u);
    EXPECT_EQ(ps.shift_events(), 0u);
    EXPECT_NEAR(ps.occupancy_entropy(), 0.0, 1e-9);
}

TEST(TokenBiasPhaseSpace, TotalRecordsIncrements) {
    TokenBiasPhaseSpace ps;
    for (int i = 0; i < 20; ++i) ps.record(static_cast<double>(i % 5) * 0.2 - 0.4);
    EXPECT_EQ(ps.total_records(), 20u);
}

TEST(TokenBiasPhaseSpace, EntropyNonNegativeAfterRecords) {
    TokenBiasPhaseSpace ps;
    for (int i = 0; i < 50; ++i)
        ps.record(static_cast<double>(i % 10) * 0.2 - 1.0);
    EXPECT_GE(ps.occupancy_entropy(), 0.0);
}

TEST(TokenBiasPhaseSpace, DominantCellInBounds) {
    TokenBiasPhaseSpace::Config cfg;
    cfg.window    = 32;
    cfg.grid_size = 4;
    TokenBiasPhaseSpace ps(cfg);
    for (int i = 0; i < 40; ++i)
        ps.record(static_cast<double>(i % 8) * 0.25 - 1.0);
    auto cell = ps.dominant_cell();
    EXPECT_GE(cell.row, 0);
    EXPECT_LT(cell.row, 4);
    EXPECT_GE(cell.col, 0);
    EXPECT_LT(cell.col, 4);
}

TEST(TokenBiasPhaseSpace, AttractorShiftCallbackFires) {
    TokenBiasPhaseSpace::Config cfg;
    cfg.window    = 16;
    cfg.grid_size = 4;
    std::atomic<int> fires{0};
    cfg.on_attractor_shift = [&](TokenBiasPhaseSpace::CellId, TokenBiasPhaseSpace::CellId) { ++fires; };
    TokenBiasPhaseSpace ps(cfg);
    for (int i = 0; i < 20; ++i) ps.record(0.8);
    for (int i = 0; i < 20; ++i) ps.record(-0.8);
    EXPECT_GE(fires.load(), 1);
    EXPECT_GE(ps.shift_events(), 1u);
}

TEST(TokenBiasPhaseSpace, EntropyHighForUniformSpread) {
    TokenBiasPhaseSpace::Config cfg;
    cfg.window    = 64;
    cfg.grid_size = 4;
    TokenBiasPhaseSpace ps(cfg);
    for (int i = 0; i < 128; ++i)
        ps.record(-1.0 + static_cast<double>(i % 20) * 0.1);
    EXPECT_GT(ps.occupancy_entropy(), 0.0);
}

TEST(TokenBiasPhaseSpace, ResetClearsState) {
    TokenBiasPhaseSpace ps;
    for (int i = 0; i < 30; ++i) ps.record(0.3);
    ps.reset();
    EXPECT_EQ(ps.total_records(), 0u);
    EXPECT_EQ(ps.shift_events(), 0u);
    EXPECT_NEAR(ps.occupancy_entropy(), 0.0, 1e-9);
}

TEST(TokenBiasPhaseSpace, UpdateConfigResetsState) {
    TokenBiasPhaseSpace ps;
    for (int i = 0; i < 30; ++i) ps.record(0.3);
    TokenBiasPhaseSpace::Config cfg2;
    cfg2.grid_size = 6;
    ps.update_config(cfg2);
    EXPECT_EQ(ps.total_records(), 0u);
}

TEST(TokenBiasPhaseSpace, StatsJsonContainsFields) {
    TokenBiasPhaseSpace ps;
    for (int i = 0; i < 20; ++i) ps.record(static_cast<double>(i % 5) * 0.3 - 0.6);
    std::string j = ps.to_stats_json();
    EXPECT_NE(j.find("entropy"),       std::string::npos);
    EXPECT_NE(j.find("shift_events"),  std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(TokenBiasPhaseSpace, ConcurrentRecordSafe) {
    TokenBiasPhaseSpace ps;
    std::atomic<bool> go{false};
    auto w0 = [&] { while (!go) {} for (int i = 0; i < 200; ++i) ps.record(static_cast<double>(i % 10) * 0.2 - 1.0); };
    auto w1 = [&] { while (!go) {} for (int i = 0; i < 200; ++i) ps.record(-0.5 + static_cast<double>(i % 5) * 0.25); };
    std::thread t0(w0), t1(w1);
    go = true;
    t0.join(); t1.join();
    EXPECT_EQ(ps.total_records(), 400u);
    EXPECT_GE(ps.occupancy_entropy(), 0.0);
}

#else
TEST(TokenBiasPhaseSpace, DisabledAtBuildTime) { SUCCEED(); }
#endif
