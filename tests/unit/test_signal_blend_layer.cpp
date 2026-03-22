#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_BLEND_ENABLED

#include "SignalBlendLayer.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

static TradeSignal make_signal(double bias, double confidence = 0.8,
                                double vol = 0.01, double spread = 0.001)
{
    TradeSignal s{};
    s.delta_bias_shift      = bias;
    s.confidence            = confidence;
    s.volatility_adjustment = vol;
    s.spread_modifier       = spread;
    s.signal_quality        = confidence;
    s.timestamp_ns          = 1'000'000;
    return s;
}

// ============================================================
// Construction
// ============================================================

TEST(SignalBlendLayer, DefaultConstructsOk) {
    SignalBlendLayer layer;
    EXPECT_EQ(layer.blends_emitted(), 0u);
    EXPECT_EQ(layer.total_pushes(),   0u);
}

// ============================================================
// Source registration
// ============================================================

TEST(SignalBlendLayer, RegisterSourceReflectsInStats) {
    SignalBlendLayer layer;
    layer.register_source("reddit_wsb", 1.0);
    layer.register_source("reddit_stocks", 0.5);
    auto stats = layer.get_source_stats();
    EXPECT_EQ(stats.size(), 2u);
}

TEST(SignalBlendLayer, RegisterSameSourceTwiceIsNoop) {
    SignalBlendLayer layer;
    layer.register_source("src_a");
    layer.register_source("src_a");
    EXPECT_EQ(layer.get_source_stats().size(), 1u);
}

TEST(SignalBlendLayer, PushFromUnregisteredSourceReturnsFalse) {
    SignalBlendLayer layer;
    auto sig = make_signal(1.0);
    EXPECT_FALSE(layer.push_signal("no_such_source", sig));
}

// ============================================================
// Quorum: min_sources=1 (pass-through)
// ============================================================

TEST(SignalBlendLayer, QuorumOneEmitsOnFirstPush) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 1;
    cfg.mode = SignalBlendLayer::BlendMode::Equal;
    SignalBlendLayer layer(cfg);
    layer.register_source("src");

    std::atomic<int> cb_count{0};
    layer.set_blend_callback([&](const TradeSignal&) { ++cb_count; });

    auto sig = make_signal(2.0);
    bool emitted = layer.push_signal("src", sig);
    EXPECT_TRUE(emitted);
    EXPECT_EQ(cb_count.load(), 1);
    EXPECT_EQ(layer.blends_emitted(), 1u);
}

// ============================================================
// Quorum: min_sources=2
// ============================================================

TEST(SignalBlendLayer, QuorumTwoRequiresBothSources) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 2;
    cfg.mode = SignalBlendLayer::BlendMode::Equal;
    SignalBlendLayer layer(cfg);
    layer.register_source("src_a");
    layer.register_source("src_b");

    std::atomic<int> cb_count{0};
    layer.set_blend_callback([&](const TradeSignal&) { ++cb_count; });

    // First source only → no blend yet.
    EXPECT_FALSE(layer.push_signal("src_a", make_signal(1.0)));
    EXPECT_EQ(cb_count.load(), 0);

    // Second source meets quorum → blend emitted.
    EXPECT_TRUE(layer.push_signal("src_b", make_signal(3.0)));
    EXPECT_EQ(cb_count.load(), 1);
}

// ============================================================
// Equal blend arithmetic
// ============================================================

TEST(SignalBlendLayer, EqualBlendAveragesBias) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 2;
    cfg.mode = SignalBlendLayer::BlendMode::Equal;
    SignalBlendLayer layer(cfg);
    layer.register_source("a");
    layer.register_source("b");

    TradeSignal blended{};
    layer.set_blend_callback([&](const TradeSignal& s) { blended = s; });

    layer.push_signal("a", make_signal(4.0));
    layer.push_signal("b", make_signal(2.0));

    EXPECT_NEAR(blended.delta_bias_shift, 3.0, 1e-9);  // (4+2)/2
}

// ============================================================
// ConfidenceWeighted blend
// ============================================================

TEST(SignalBlendLayer, ConfidenceWeightedBlend) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 2;
    cfg.mode = SignalBlendLayer::BlendMode::ConfidenceWeighted;
    SignalBlendLayer layer(cfg);
    layer.register_source("high_conf");
    layer.register_source("low_conf");

    TradeSignal blended{};
    layer.set_blend_callback([&](const TradeSignal& s) { blended = s; });

    // high_conf: bias=10, confidence=0.8
    // low_conf:  bias=0,  confidence=0.2
    // expected blend: (10*0.8 + 0*0.2) / (0.8+0.2) = 8.0
    layer.push_signal("high_conf", make_signal(10.0, 0.8));
    layer.push_signal("low_conf",  make_signal(0.0,  0.2));

    EXPECT_NEAR(blended.delta_bias_shift, 8.0, 1e-9);
}

// ============================================================
// CustomWeighted blend
// ============================================================

TEST(SignalBlendLayer, CustomWeightedBlend) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 2;
    cfg.mode = SignalBlendLayer::BlendMode::CustomWeighted;
    SignalBlendLayer layer(cfg);
    layer.register_source("wsb",    3.0);
    layer.register_source("stocks", 1.0);

    TradeSignal blended{};
    layer.set_blend_callback([&](const TradeSignal& s) { blended = s; });

    // wsb: bias=8, weight=3; stocks: bias=4, weight=1
    // expected: (8*3 + 4*1)/(3+1) = 28/4 = 7.0
    layer.push_signal("wsb",    make_signal(8.0));
    layer.push_signal("stocks", make_signal(4.0));

    EXPECT_NEAR(blended.delta_bias_shift, 7.0, 1e-9);
}

TEST(SignalBlendLayer, SetSourceWeightUpdatesBlend) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 1;
    cfg.mode = SignalBlendLayer::BlendMode::CustomWeighted;
    SignalBlendLayer layer(cfg);
    layer.register_source("src", 1.0);
    layer.set_source_weight("src", 0.5);

    // With only 1 source, blend = signal (weight normalises to 1).
    TradeSignal blended{};
    layer.set_blend_callback([&](const TradeSignal& s) { blended = s; });
    layer.push_signal("src", make_signal(6.0));
    EXPECT_NEAR(blended.delta_bias_shift, 6.0, 1e-9);
}

// ============================================================
// reset_pending_after_blend
// ============================================================

TEST(SignalBlendLayer, ResetPendingFlagsMeansEachBlendNeedsFreshUpdate) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources              = 2;
    cfg.mode                     = SignalBlendLayer::BlendMode::Equal;
    cfg.reset_pending_after_blend = true;
    SignalBlendLayer layer(cfg);
    layer.register_source("a");
    layer.register_source("b");

    std::atomic<int> cb_count{0};
    layer.set_blend_callback([&](const TradeSignal&) { ++cb_count; });

    layer.push_signal("a", make_signal(1.0));
    layer.push_signal("b", make_signal(1.0));
    EXPECT_EQ(cb_count.load(), 1);

    // After blend, flags are reset.  Single source doesn't trigger another.
    layer.push_signal("a", make_signal(2.0));
    EXPECT_EQ(cb_count.load(), 1);

    // Second source → second blend.
    layer.push_signal("b", make_signal(2.0));
    EXPECT_EQ(cb_count.load(), 2);
}

// ============================================================
// force_blend
// ============================================================

TEST(SignalBlendLayer, ForceBlendWorksWithSingleSource) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 3;  // high quorum never met
    cfg.mode = SignalBlendLayer::BlendMode::Equal;
    SignalBlendLayer layer(cfg);
    layer.register_source("a");

    layer.push_signal("a", make_signal(5.0));

    TradeSignal blended = layer.force_blend();
    EXPECT_NEAR(blended.delta_bias_shift, 5.0, 1e-9);
}

TEST(SignalBlendLayer, ForceBlendOnEmptyLayerReturnsZero) {
    SignalBlendLayer layer;
    auto blended = layer.force_blend();
    EXPECT_NEAR(blended.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// reset
// ============================================================

TEST(SignalBlendLayer, ResetClearsSourceHistory) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 1;
    SignalBlendLayer layer(cfg);
    layer.register_source("src");
    layer.push_signal("src", make_signal(1.0));
    layer.reset();

    // After reset, force_blend should return zero (no pushes since reset).
    auto blended = layer.force_blend();
    EXPECT_NEAR(blended.delta_bias_shift, 0.0, 1e-9);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(SignalBlendLayer, ToStatsJsonContainsKeys) {
    SignalBlendLayer layer;
    layer.register_source("src");
    std::string json = layer.to_stats_json();
    EXPECT_NE(json.find("blends_emitted"), std::string::npos);
    EXPECT_NE(json.find("total_pushes"),   std::string::npos);
    EXPECT_NE(json.find("source_count"),   std::string::npos);
}

// ============================================================
// update_config
// ============================================================

TEST(SignalBlendLayer, UpdateConfigChangesMinSources) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 3;
    SignalBlendLayer layer(cfg);
    layer.register_source("a");

    SignalBlendLayer::Config cfg2;
    cfg2.min_sources = 1;
    layer.update_config(cfg2);

    std::atomic<int> cb_count{0};
    layer.set_blend_callback([&](const TradeSignal&) { ++cb_count; });
    layer.push_signal("a", make_signal(1.0));
    EXPECT_EQ(cb_count.load(), 1);
}

// ============================================================
// Thread safety
// ============================================================

TEST(SignalBlendLayer, ConcurrentPushesAreSafe) {
    SignalBlendLayer::Config cfg;
    cfg.min_sources = 1;
    cfg.mode = SignalBlendLayer::BlendMode::Equal;
    SignalBlendLayer layer(cfg);
    layer.register_source("src_a");
    layer.register_source("src_b");
    layer.register_source("src_c");

    std::atomic<bool> start{false};
    std::atomic<int>  blends{0};
    layer.set_blend_callback([&](const TradeSignal&) { blends.fetch_add(1); });

    auto pusher = [&](const std::string& name) {
        while (!start.load()) {}
        for (int i = 0; i < 100; ++i)
            layer.push_signal(name, make_signal(static_cast<double>(i)));
    };

    std::vector<std::thread> threads;
    threads.emplace_back(pusher, "src_a");
    threads.emplace_back(pusher, "src_b");
    threads.emplace_back(pusher, "src_c");
    start.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(layer.total_pushes(), 300u);
    EXPECT_GT(blends.load(), 0);
}

} // namespace

#else  // LLMQUANT_SIGNAL_BLEND_ENABLED not defined

TEST(SignalBlendLayer, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SIGNAL_BLEND_ENABLED
