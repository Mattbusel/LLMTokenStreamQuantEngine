#include <gtest/gtest.h>

#ifdef LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED

#include "ReinforcementSignalWeighter.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

ReinforcementSignalWeighter::Config two_arm_cfg() {
    ReinforcementSignalWeighter::Config cfg;
    cfg.arms = {{"alpha", 1.0}, {"beta", 1.0}};
    return cfg;
}

// -----------------------------------------------------------------------
// Default state
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, DefaultState) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    EXPECT_EQ(rw.total_updates(), 0u);
    EXPECT_EQ(rw.dominant_arm_changes(), 0u);
    EXPECT_FALSE(rw.dominant_arm().empty());
}

// -----------------------------------------------------------------------
// Unknown arm is ignored
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, UnknownArmIgnored) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    rw.update("unknown", 1.0);
    EXPECT_EQ(rw.total_updates(), 0u);
}

// -----------------------------------------------------------------------
// Total updates increments
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, TotalUpdatesIncrements) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    rw.update("alpha", 1.0);
    rw.update("beta",  0.5);
    EXPECT_EQ(rw.total_updates(), 2u);
}

// -----------------------------------------------------------------------
// High-reward arm dominates after many pulls
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, HighRewardArmDominates) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    // Feed alpha consistently high rewards, beta near zero
    for (int i = 0; i < 30; ++i) rw.update("alpha", 1.0);
    for (int i = 0; i < 30; ++i) rw.update("beta",  0.01);
    EXPECT_EQ(rw.dominant_arm(), "alpha");
    EXPECT_GT(rw.weight("alpha"), rw.weight("beta"));
}

// -----------------------------------------------------------------------
// Dominant arm callback fires on change
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, DominantArmChangeCbFires) {
    auto cfg = two_arm_cfg();
    std::atomic<int> fires{0};
    cfg.on_dominant_arm_change = [&](const std::string&) { ++fires; };
    ReinforcementSignalWeighter rw(cfg);

    // First lock alpha as dominant
    for (int i = 0; i < 20; ++i) rw.update("alpha", 1.0);
    for (int i = 0; i < 20; ++i) rw.update("beta",  0.0);

    // Swing to beta with high rewards
    for (int i = 0; i < 40; ++i) rw.update("beta", 2.0);

    EXPECT_GE(fires.load(), 1);
    EXPECT_GT(rw.dominant_arm_changes(), 0u);
}

// -----------------------------------------------------------------------
// Weights sum to 1
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, WeightsSumToOne) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    for (int i = 0; i < 10; ++i) {
        rw.update("alpha", 1.0);
        rw.update("beta",  0.5);
    }
    double sum = rw.weight("alpha") + rw.weight("beta");
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

// -----------------------------------------------------------------------
// weight() returns 0 for unknown arm
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, WeightZeroForUnknown) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    EXPECT_DOUBLE_EQ(rw.weight("nonexistent"), 0.0);
}

// -----------------------------------------------------------------------
// reset() clears state
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, ResetClearsState) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    for (int i = 0; i < 10; ++i) rw.update("alpha", 1.0);
    rw.reset();
    EXPECT_EQ(rw.total_updates(), 0u);
    EXPECT_EQ(rw.dominant_arm_changes(), 0u);
}

// -----------------------------------------------------------------------
// update_config resets state
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, UpdateConfigResetsState) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    for (int i = 0; i < 5; ++i) rw.update("alpha", 1.0);
    auto cfg2 = two_arm_cfg();
    cfg2.exploration_c = 2.0;
    rw.update_config(cfg2);
    EXPECT_EQ(rw.total_updates(), 0u);
}

// -----------------------------------------------------------------------
// to_stats_json contains expected keys
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, ToStatsJsonContainsKeys) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    rw.update("alpha", 0.5);
    rw.update("beta",  0.3);
    std::string j = rw.to_stats_json();
    EXPECT_NE(j.find("total_updates"),       std::string::npos);
    EXPECT_NE(j.find("dominant_arm"),        std::string::npos);
    EXPECT_NE(j.find("dominant_arm_changes"), std::string::npos);
    EXPECT_NE(j.find("arms"),                std::string::npos);
    EXPECT_NE(j.find("weight"),              std::string::npos);
}

// -----------------------------------------------------------------------
// Concurrent update() is thread-safe
// -----------------------------------------------------------------------

TEST(ReinforcementSignalWeighter, ConcurrentUpdateSafe) {
    ReinforcementSignalWeighter rw(two_arm_cfg());
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            const char* arm = (t % 2 == 0) ? "alpha" : "beta";
            for (int i = 0; i < 250; ++i)
                rw.update(arm, static_cast<double>(i) * 0.01);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(rw.total_updates(), 1000u);
    double w_a = rw.weight("alpha");
    double w_b = rw.weight("beta");
    EXPECT_TRUE(std::isfinite(w_a));
    EXPECT_TRUE(std::isfinite(w_b));
    EXPECT_NEAR(w_a + w_b, 1.0, 1e-6);
}

} // namespace

#else

TEST(ReinforcementSignalWeighter, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED
