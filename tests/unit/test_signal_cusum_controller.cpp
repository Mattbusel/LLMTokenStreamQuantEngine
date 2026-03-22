#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_CUSUM_ENABLED

#include "SignalCUSUMController.h"
#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(SignalCUSUMController, DefaultConstructs) {
    SignalCUSUMController c;
    EXPECT_DOUBLE_EQ(c.cusum_pos(), 0.0);
    EXPECT_DOUBLE_EQ(c.cusum_neg(), 0.0);
    EXPECT_FALSE(c.is_alarmed());
    EXPECT_EQ(c.upward_alarms(), 0u);
    EXPECT_EQ(c.downward_alarms(), 0u);
    EXPECT_EQ(c.total_updates(), 0u);
}

TEST(SignalCUSUMController, PositiveShiftTriggersUpwardAlarm) {
    SignalCUSUMController::Config cfg;
    cfg.target_mean    = 0.0;
    cfg.allowance      = 0.1;
    cfg.threshold      = 1.0;
    cfg.reset_on_alarm = false;
    std::atomic<int> fires{0};
    cfg.on_upward_shift = [&](double) { ++fires; };
    SignalCUSUMController c(cfg);
    // Feed 20 samples of +0.5 (above allowance of 0.1 → each adds 0.4)
    for (int i = 0; i < 10; ++i) c.update(0.5);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(c.upward_alarms(), 0u);
}

TEST(SignalCUSUMController, NegativeShiftTriggersDownwardAlarm) {
    SignalCUSUMController::Config cfg;
    cfg.target_mean    = 0.0;
    cfg.allowance      = 0.1;
    cfg.threshold      = 1.0;
    cfg.reset_on_alarm = false;
    std::atomic<int> fires{0};
    cfg.on_downward_shift = [&](double) { ++fires; };
    SignalCUSUMController c(cfg);
    for (int i = 0; i < 10; ++i) c.update(-0.5);
    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(c.downward_alarms(), 0u);
}

TEST(SignalCUSUMController, AccumulatorsResetAfterAlarm) {
    SignalCUSUMController::Config cfg;
    cfg.target_mean    = 0.0;
    cfg.allowance      = 0.1;
    cfg.threshold      = 1.0;
    cfg.reset_on_alarm = true;
    SignalCUSUMController c(cfg);
    for (int i = 0; i < 20; ++i) c.update(0.5);
    // After each alarm, accumulators are reset to 0.
    // Final value should be near 0 (just reset).
    EXPECT_LE(c.cusum_pos(), cfg.threshold);
}

TEST(SignalCUSUMController, NeutralInputKeepsAccumulatorsNearZero) {
    SignalCUSUMController c;  // target=0, allowance=0.1
    // Feed small values |x| < allowance — should never accumulate.
    for (int i = 0; i < 100; ++i) c.update(0.05);
    EXPECT_DOUBLE_EQ(c.cusum_pos(), 0.0);
    EXPECT_EQ(c.upward_alarms(), 0u);
}

TEST(SignalCUSUMController, ResetClearsState) {
    SignalCUSUMController c;
    for (int i = 0; i < 10; ++i) c.update(1.0);
    c.reset();
    EXPECT_DOUBLE_EQ(c.cusum_pos(), 0.0);
    EXPECT_DOUBLE_EQ(c.cusum_neg(), 0.0);
    EXPECT_EQ(c.total_updates(), 0u);
    EXPECT_EQ(c.upward_alarms(), 0u);
    EXPECT_FALSE(c.is_alarmed());
}

TEST(SignalCUSUMController, UpdateConfigResetsState) {
    SignalCUSUMController c;
    for (int i = 0; i < 5; ++i) c.update(0.5);
    SignalCUSUMController::Config cfg2;
    cfg2.threshold = 2.0;
    c.update_config(cfg2);
    EXPECT_DOUBLE_EQ(c.cusum_pos(), 0.0);
    EXPECT_EQ(c.total_updates(), 0u);
}

TEST(SignalCUSUMController, ToStatsJsonContainsKeys) {
    SignalCUSUMController c;
    c.update(0.3);
    std::string j = c.to_stats_json();
    EXPECT_NE(j.find("cusum_pos"),    std::string::npos);
    EXPECT_NE(j.find("cusum_neg"),    std::string::npos);
    EXPECT_NE(j.find("is_alarmed"),   std::string::npos);
    EXPECT_NE(j.find("up_alarms"),    std::string::npos);
}

TEST(SignalCUSUMController, TotalUpdatesIncrement) {
    SignalCUSUMController c;
    c.update(0.1); c.update(0.2); c.update(0.3);
    EXPECT_EQ(c.total_updates(), 3u);
}

TEST(SignalCUSUMController, ConcurrentUpdateSafe) {
    SignalCUSUMController::Config cfg;
    cfg.threshold = 1000.0;  // high threshold to avoid races on alarm
    SignalCUSUMController c(cfg);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 100; ++i) c.update(0.01);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(c.total_updates(), 400u);
}

} // namespace

#else

TEST(SignalCUSUMController, DisabledAtBuildTime) {
    SUCCEED();
}

#endif
