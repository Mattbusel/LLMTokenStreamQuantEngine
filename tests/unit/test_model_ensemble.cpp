#include <gtest/gtest.h>

#include "ModelEnsemble.h"

#include <atomic>
#include <cmath>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Helper: approximately-equal comparison for doubles
// ============================================================

static bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

// ============================================================
// Construction / defaults
// ============================================================

TEST(ModelEnsemble, DefaultConstructs) {
    ModelEnsemble ens;
    EXPECT_EQ(ens.channel_count(), 0u);
    EXPECT_EQ(ens.total_updates(), 0u);
    EXPECT_FALSE(ens.aggregate().has_value());
}

TEST(ModelEnsemble, NormalisationRangeGuard) {
    ModelEnsemble::Config cfg;
    cfg.normalisation_range = -1.0;  // invalid — should be reset to 2.0
    ModelEnsemble ens(cfg);
    // After the guard, aggregate with one channel should not produce NaN.
    ens.add_channel("a");
    ens.update_channel("a", 0.5);
    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_FALSE(std::isnan(res->agreement));
    EXPECT_FALSE(std::isinf(res->agreement));
}

// ============================================================
// Channel management
// ============================================================

TEST(ModelEnsemble, AddChannel) {
    ModelEnsemble ens;
    EXPECT_TRUE(ens.add_channel("gpt4"));
    EXPECT_EQ(ens.channel_count(), 1u);
}

TEST(ModelEnsemble, AddDuplicateChannelFails) {
    ModelEnsemble ens;
    EXPECT_TRUE(ens.add_channel("gpt4"));
    EXPECT_FALSE(ens.add_channel("gpt4"));
    EXPECT_EQ(ens.channel_count(), 1u);
}

TEST(ModelEnsemble, AddUpToMaxChannels) {
    ModelEnsemble ens;
    for (std::size_t i = 0; i < ModelEnsemble::kMaxChannels; ++i) {
        EXPECT_TRUE(ens.add_channel("ch" + std::to_string(i)));
    }
    EXPECT_EQ(ens.channel_count(), ModelEnsemble::kMaxChannels);
    // One more should fail.
    EXPECT_FALSE(ens.add_channel("overflow"));
    EXPECT_EQ(ens.channel_count(), ModelEnsemble::kMaxChannels);
}

TEST(ModelEnsemble, RemoveChannel) {
    ModelEnsemble ens;
    EXPECT_TRUE(ens.add_channel("a"));
    EXPECT_TRUE(ens.add_channel("b"));
    EXPECT_TRUE(ens.remove_channel("a"));
    EXPECT_EQ(ens.channel_count(), 1u);
    // "b" must still be findable.
    EXPECT_GE(ens.find_channel("b"), 0);
}

TEST(ModelEnsemble, RemoveNonExistentChannelFails) {
    ModelEnsemble ens;
    EXPECT_FALSE(ens.remove_channel("ghost"));
}

TEST(ModelEnsemble, FindChannelReturnsCorrectSlot) {
    ModelEnsemble ens;
    ens.add_channel("x");
    ens.add_channel("y");
    EXPECT_GE(ens.find_channel("x"), 0);
    EXPECT_GE(ens.find_channel("y"), 0);
    EXPECT_NE(ens.find_channel("x"), ens.find_channel("y"));
    EXPECT_EQ(ens.find_channel("z"), -1);
}

// ============================================================
// Update channel
// ============================================================

TEST(ModelEnsemble, UpdateChannelByName) {
    ModelEnsemble ens;
    ens.add_channel("m1");
    EXPECT_TRUE(ens.update_channel("m1", 0.7));
    EXPECT_EQ(ens.total_updates(), 1u);
}

TEST(ModelEnsemble, UpdateChannelBySlot) {
    ModelEnsemble ens;
    ens.add_channel("m1");
    int slot = ens.find_channel("m1");
    ASSERT_GE(slot, 0);
    EXPECT_TRUE(ens.update_channel_by_slot(slot, 0.3));
    EXPECT_EQ(ens.total_updates(), 1u);
}

TEST(ModelEnsemble, UpdateUnknownChannelReturnsFalse) {
    ModelEnsemble ens;
    EXPECT_FALSE(ens.update_channel("unknown", 0.5));
    EXPECT_EQ(ens.total_updates(), 0u);
}

TEST(ModelEnsemble, UpdateInvalidSlotReturnsFalse) {
    ModelEnsemble ens;
    EXPECT_FALSE(ens.update_channel_by_slot(99, 0.5));
    EXPECT_FALSE(ens.update_channel_by_slot(-1, 0.5));
}

// ============================================================
// Aggregate — single channel
// ============================================================

TEST(ModelEnsemble, SingleChannelAgreementIsOne) {
    ModelEnsemble ens;
    ens.add_channel("solo");
    ens.update_channel("solo", 0.6);
    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_NEAR(res->mean_bias, 0.6, 1e-9);
    EXPECT_NEAR(res->agreement, 1.0, 1e-9);  // single channel => MAD=0 => agreement=1
    EXPECT_NEAR(res->final_signal, 0.6, 1e-9);
    EXPECT_EQ(res->channels_active, 1u);
}

// ============================================================
// Aggregate — two channels, perfect agreement
// ============================================================

TEST(ModelEnsemble, TwoChannelsPerfectAgreement) {
    ModelEnsemble ens;
    ens.add_channel("a");
    ens.add_channel("b");
    ens.update_channel("a", 0.5);
    ens.update_channel("b", 0.5);

    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_NEAR(res->mean_bias,    0.5, 1e-9);
    EXPECT_NEAR(res->agreement,    1.0, 1e-9);
    EXPECT_NEAR(res->final_signal, 0.5, 1e-9);
    EXPECT_EQ(res->channels_active, 2u);
}

// ============================================================
// Aggregate — two channels, maximum disagreement
// ============================================================

TEST(ModelEnsemble, TwoChannelsMaxDisagreement) {
    // Biases: +1.0, -1.0 with normalisation_range = 2.0
    // mean = 0.0, MAD = mean(|1-0|, |-1-0|) = 1.0
    // agreement = 1 - 1.0/2.0 = 0.5
    ModelEnsemble::Config cfg;
    cfg.normalisation_range = 2.0;
    ModelEnsemble ens(cfg);
    ens.add_channel("bull");
    ens.add_channel("bear");
    ens.update_channel("bull",  1.0);
    ens.update_channel("bear", -1.0);

    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_NEAR(res->mean_bias,    0.0, 1e-9);
    EXPECT_NEAR(res->agreement,    0.5, 1e-9);
    EXPECT_NEAR(res->final_signal, 0.0, 1e-9);
}

// ============================================================
// Aggregate — agreement clamped to [0, 1]
// ============================================================

TEST(ModelEnsemble, AgreementClampedToZeroWhenBeyondRange) {
    // Biases: +2.0, -2.0 with normalisation_range = 2.0
    // MAD = 2.0, 1 - 2.0/2.0 = 0.0 (clamps at 0)
    ModelEnsemble::Config cfg;
    cfg.normalisation_range = 2.0;
    ModelEnsemble ens(cfg);
    ens.add_channel("a");
    ens.add_channel("b");
    ens.update_channel("a",  2.0);
    ens.update_channel("b", -2.0);

    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_GE(res->agreement, 0.0);
    EXPECT_LE(res->agreement, 1.0);
}

// ============================================================
// Aggregate — four channels, partial spread
// ============================================================

TEST(ModelEnsemble, FourChannelsPartialSpread) {
    // biases: 0.8, 0.6, 0.4, 0.2   mean = 0.5
    // MAD = mean(0.3, 0.1, 0.1, 0.3) = 0.2
    // agreement = 1 - 0.2/2.0 = 0.9
    // final = 0.5 * 0.9 = 0.45
    ModelEnsemble::Config cfg;
    cfg.normalisation_range = 2.0;
    ModelEnsemble ens(cfg);
    ens.add_channel("m1");
    ens.add_channel("m2");
    ens.add_channel("m3");
    ens.add_channel("m4");
    ens.update_channel("m1", 0.8);
    ens.update_channel("m2", 0.6);
    ens.update_channel("m3", 0.4);
    ens.update_channel("m4", 0.2);

    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_NEAR(res->mean_bias,    0.5,  1e-9);
    EXPECT_NEAR(res->agreement,    0.9,  1e-9);
    EXPECT_NEAR(res->final_signal, 0.45, 1e-9);
    EXPECT_EQ(res->channels_active, 4u);
}

// ============================================================
// min_channels guard
// ============================================================

TEST(ModelEnsemble, ReturnsNulloptWhenBelowMinChannels) {
    ModelEnsemble::Config cfg;
    cfg.min_channels = 3;
    ModelEnsemble ens(cfg);
    ens.add_channel("a");
    ens.add_channel("b");
    ens.add_channel("c");
    ens.update_channel("a", 0.5);
    ens.update_channel("b", 0.3);
    // Only 2 active, min_channels=3 => nullopt.
    EXPECT_FALSE(ens.aggregate().has_value());

    ens.update_channel("c", 0.1);
    // Now 3 active => result.
    EXPECT_TRUE(ens.aggregate().has_value());
}

// ============================================================
// on_aggregate callback
// ============================================================

TEST(ModelEnsemble, OnAggregateCallbackFired) {
    std::atomic<int> fires{0};
    ModelEnsemble::Config cfg;
    cfg.on_aggregate = [&](const ModelEnsemble::AggregateResult&) { ++fires; };
    ModelEnsemble ens(cfg);
    ens.add_channel("x");
    ens.update_channel("x", 0.5);
    ens.aggregate();
    EXPECT_EQ(fires.load(), 1);
    ens.aggregate();
    EXPECT_EQ(fires.load(), 2);
}

// ============================================================
// reset
// ============================================================

TEST(ModelEnsemble, ResetClearsActiveBiases) {
    ModelEnsemble ens;
    ens.add_channel("a");
    ens.update_channel("a", 0.9);
    EXPECT_TRUE(ens.aggregate().has_value());

    ens.reset();
    // After reset, the channel is still registered but no longer active.
    EXPECT_EQ(ens.channel_count(), 1u);
    EXPECT_FALSE(ens.aggregate().has_value());
}

// ============================================================
// update_config
// ============================================================

TEST(ModelEnsemble, UpdateConfigChangesNormalisationRange) {
    ModelEnsemble ens;
    ens.add_channel("a");
    ens.add_channel("b");
    ens.update_channel("a", 1.0);
    ens.update_channel("b", 0.0);
    // With default range=2.0: MAD=0.5, agreement=1-0.5/2.0=0.75
    auto r1 = ens.aggregate();
    ASSERT_TRUE(r1.has_value());
    EXPECT_NEAR(r1->agreement, 0.75, 1e-9);

    // Change range to 1.0: agreement=1-0.5/1.0=0.5
    ModelEnsemble::Config cfg2;
    cfg2.normalisation_range = 1.0;
    ens.update_config(cfg2);
    auto r2 = ens.aggregate();
    ASSERT_TRUE(r2.has_value());
    EXPECT_NEAR(r2->agreement, 0.5, 1e-9);
}

TEST(ModelEnsemble, UpdateConfigIgnoresInvalidRange) {
    ModelEnsemble ens;
    ModelEnsemble::Config bad_cfg;
    bad_cfg.normalisation_range = 0.0;
    ens.update_config(bad_cfg);
    // Should not crash; internal range stays at its last valid value.
    ens.add_channel("x");
    ens.update_channel("x", 0.4);
    EXPECT_TRUE(ens.aggregate().has_value());
}

// ============================================================
// Thread safety: concurrent updates on 8 channels
// ============================================================

TEST(ModelEnsemble, ConcurrentUpdatesDontCrash) {
    ModelEnsemble ens;
    const int kThreads = 8;
    for (int i = 0; i < kThreads; ++i) {
        ens.add_channel("ch" + std::to_string(i));
    }

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([&ens, i]() {
            const std::string name = "ch" + std::to_string(i);
            for (int j = 0; j < 1000; ++j) {
                ens.update_channel(name, static_cast<double>(j % 3 - 1));
            }
        });
    }
    for (auto& t : workers) t.join();

    auto res = ens.aggregate();
    EXPECT_TRUE(res.has_value());
    EXPECT_GE(res->agreement, 0.0);
    EXPECT_LE(res->agreement, 1.0);
    EXPECT_FALSE(std::isnan(res->final_signal));
    EXPECT_EQ(ens.total_updates(), static_cast<uint64_t>(kThreads * 1000));
}

// ============================================================
// Long channel name truncation (must not overflow)
// ============================================================

TEST(ModelEnsemble, LongChannelNameTruncated) {
    ModelEnsemble ens;
    std::string long_name(100, 'x');
    EXPECT_TRUE(ens.add_channel(long_name));
    // Should be findable by its truncated form.
    EXPECT_GE(ens.find_channel(long_name), 0);
}

// ============================================================
// All 8 channels, full ensemble correctness
// ============================================================

TEST(ModelEnsemble, FullEightChannelEnsemble) {
    ModelEnsemble::Config cfg;
    cfg.normalisation_range = 2.0;
    ModelEnsemble ens(cfg);
    for (int i = 0; i < 8; ++i) {
        ens.add_channel("p" + std::to_string(i));
    }
    // All agree on +0.4: mean=0.4, MAD=0, agreement=1, final=0.4
    for (int i = 0; i < 8; ++i) {
        ens.update_channel("p" + std::to_string(i), 0.4);
    }
    auto res = ens.aggregate();
    ASSERT_TRUE(res.has_value());
    EXPECT_NEAR(res->mean_bias,    0.4, 1e-9);
    EXPECT_NEAR(res->agreement,    1.0, 1e-9);
    EXPECT_NEAR(res->final_signal, 0.4, 1e-9);
    EXPECT_EQ(res->channels_active, 8u);
}

} // namespace
