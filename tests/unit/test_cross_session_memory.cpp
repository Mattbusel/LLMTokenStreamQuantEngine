#include <gtest/gtest.h>

#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED

#include "CrossSessionMemory.h"

#include <cstdio>
#include <string>

using namespace llmquant;

namespace {

// Temporary file name for test isolation.
static const char* kTmpFile = "test_cross_session_memory.json.tmp";

struct TmpFileGuard {
    ~TmpFileGuard() { std::remove(kTmpFile); }
};

TEST(CrossSessionMemory, DefaultConstructs) {
    CrossSessionMemory mem;
    EXPECT_FALSE(mem.has_loaded_state());
    EXPECT_EQ(mem.session_number(), 0u);
}

TEST(CrossSessionMemory, LoadMissingFileSucceedsWithIgnoreMissing) {
    TmpFileGuard guard;
    CrossSessionMemory::Config cfg;
    cfg.path           = "definitely_does_not_exist_xyz.json";
    cfg.ignore_missing = true;
    CrossSessionMemory mem(cfg);
    EXPECT_TRUE(mem.load());
    EXPECT_FALSE(mem.has_loaded_state());
}

TEST(CrossSessionMemory, SaveThenLoadRestoresKellyState) {
    TmpFileGuard guard;

    // Save.
    {
        CrossSessionMemory::Config cfg;
        cfg.path = kTmpFile;
        CrossSessionMemory mem(cfg);
        CrossSessionMemory::KellyState ks;
        ks.total_outcomes = 42;
        ks.win_count      = 28;
        ks.win_rate_ema   = 0.67;
        ks.avg_win_ema    = 0.04;
        ks.avg_loss_ema   = 0.02;
        mem.capture_kelly(ks);
        EXPECT_TRUE(mem.save());
        EXPECT_EQ(mem.session_number(), 1u);
    }

    // Reload.
    {
        CrossSessionMemory::Config cfg;
        cfg.path = kTmpFile;
        CrossSessionMemory mem(cfg);
        EXPECT_TRUE(mem.load());
        EXPECT_TRUE(mem.has_loaded_state());
        CrossSessionMemory::KellyState ks;
        EXPECT_TRUE(mem.restore_kelly(ks));
        EXPECT_EQ(ks.total_outcomes, 42u);
        EXPECT_EQ(ks.win_count,      28u);
        EXPECT_NEAR(ks.win_rate_ema,  0.67, 1e-6);
        EXPECT_NEAR(ks.avg_win_ema,   0.04, 1e-6);
        EXPECT_NEAR(ks.avg_loss_ema,  0.02, 1e-6);
    }
}

TEST(CrossSessionMemory, SaveThenLoadRestoresDrawdownState) {
    TmpFileGuard guard;

    {
        CrossSessionMemory::Config cfg;
        cfg.path = kTmpFile;
        CrossSessionMemory mem(cfg);
        CrossSessionMemory::DrawdownState ds;
        ds.cumulative_pnl = 1.234;
        ds.peak_pnl       = 2.567;
        mem.capture_drawdown(ds);
        EXPECT_TRUE(mem.save());
    }

    {
        CrossSessionMemory::Config cfg;
        cfg.path = kTmpFile;
        CrossSessionMemory mem(cfg);
        EXPECT_TRUE(mem.load());
        CrossSessionMemory::DrawdownState ds;
        EXPECT_TRUE(mem.restore_drawdown(ds));
        EXPECT_NEAR(ds.cumulative_pnl, 1.234, 1e-6);
        EXPECT_NEAR(ds.peak_pnl,       2.567, 1e-6);
    }
}

TEST(CrossSessionMemory, SessionCounterIncrementsOnEachSave) {
    TmpFileGuard guard;
    CrossSessionMemory::Config cfg;
    cfg.path = kTmpFile;
    CrossSessionMemory mem(cfg);
    mem.save();
    EXPECT_EQ(mem.session_number(), 1u);
    mem.save();
    EXPECT_EQ(mem.session_number(), 2u);
}

TEST(CrossSessionMemory, RestoreWithoutKellyReturnsFalse) {
    CrossSessionMemory mem;
    CrossSessionMemory::KellyState ks;
    EXPECT_FALSE(mem.restore_kelly(ks));
}

TEST(CrossSessionMemory, RestoreWithoutDrawdownReturnsFalse) {
    CrossSessionMemory mem;
    CrossSessionMemory::DrawdownState ds;
    EXPECT_FALSE(mem.restore_drawdown(ds));
}

TEST(CrossSessionMemory, SaveCallbackFires) {
    TmpFileGuard guard;
    CrossSessionMemory::Config cfg;
    cfg.path = kTmpFile;
    int fired = 0;
    cfg.on_save = [&](const std::string&, uint64_t) { ++fired; };
    CrossSessionMemory mem(cfg);
    mem.save();
    EXPECT_EQ(fired, 1);
}

TEST(CrossSessionMemory, LoadCallbackFires) {
    TmpFileGuard guard;
    // First save so the file exists.
    {
        CrossSessionMemory::Config cfg;
        cfg.path = kTmpFile;
        CrossSessionMemory mem(cfg);
        mem.save();
    }
    int fired = 0;
    CrossSessionMemory::Config cfg;
    cfg.path    = kTmpFile;
    cfg.on_load = [&](const std::string&, uint64_t) { ++fired; };
    CrossSessionMemory mem(cfg);
    (void)mem.load();
    // on_load fires in load() when file is present.
    EXPECT_EQ(fired, 1);
}

TEST(CrossSessionMemory, ToStatsJsonContainsKeys) {
    CrossSessionMemory mem;
    std::string json = mem.to_stats_json();
    EXPECT_NE(json.find("loaded"),       std::string::npos);
    EXPECT_NE(json.find("session"),      std::string::npos);
    EXPECT_NE(json.find("has_kelly"),    std::string::npos);
    EXPECT_NE(json.find("has_drawdown"), std::string::npos);
}

} // namespace

#else

TEST(CrossSessionMemory, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
