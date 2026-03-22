#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED

#include "SignalEnsembleLayer.h"

#include <cmath>
#include <numeric>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(SignalEnsembleLayer, DefaultConstructsEmpty) {
    SignalEnsembleLayer ens;
    EXPECT_EQ(ens.source_count(), 0);
    EXPECT_EQ(ens.total_outcomes(), 0u);
    EXPECT_NEAR(ens.ensemble_output(), 0.0, 1e-12);
}

// ============================================================
// register_source
// ============================================================

TEST(SignalEnsembleLayer, RegisterSourceReturnsSequentialIds) {
    SignalEnsembleLayer ens;
    EXPECT_EQ(ens.register_source("a"), 0);
    EXPECT_EQ(ens.register_source("b"), 1);
    EXPECT_EQ(ens.register_source("c"), 2);
    EXPECT_EQ(ens.source_count(), 3);
}

// ============================================================
// Uniform initial weights sum to 1
// ============================================================

TEST(SignalEnsembleLayer, InitialWeightsSumToOne) {
    SignalEnsembleLayer ens;
    ens.register_source("x");
    ens.register_source("y");
    ens.register_source("z");
    auto w = ens.weights();
    double sum = 0.0;
    for (double wi : w) sum += wi;
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

// ============================================================
// ensemble_output is weighted sum
// ============================================================

TEST(SignalEnsembleLayer, EnsembleOutputIsWeightedSum) {
    SignalEnsembleLayer ens;
    int a = ens.register_source("a");
    int b = ens.register_source("b");

    ens.update_source(a, 1.0);
    ens.update_source(b, 3.0);

    // Equal weights (0.5 each) → output = 0.5*1 + 0.5*3 = 2.0
    EXPECT_NEAR(ens.ensemble_output(), 2.0, 1e-9);
}

// ============================================================
// update_source with out-of-range ID is a no-op
// ============================================================

TEST(SignalEnsembleLayer, OutOfRangeUpdateIsNoOp) {
    SignalEnsembleLayer ens;
    ens.register_source("a");
    ens.update_source(99, 999.0);
    EXPECT_NEAR(ens.ensemble_output(), 0.0, 1e-12);
}

// ============================================================
// SGD updates weights
// ============================================================

TEST(SignalEnsembleLayer, RecordOutcomeUpdatesWeights) {
    SignalEnsembleLayer::Config cfg;
    cfg.learning_rate = 0.1;
    SignalEnsembleLayer ens(cfg);
    int a = ens.register_source("a");
    int b = ens.register_source("b");

    ens.update_source(a, 1.0);
    ens.update_source(b, 0.0);

    double w0_before = ens.weight(a);
    ens.record_outcome(1.0);  // target = 1.0
    double w0_after = ens.weight(a);

    // Weights changed.
    EXPECT_NE(w0_before, w0_after);
    EXPECT_EQ(ens.total_outcomes(), 1u);
}

// ============================================================
// Weights stay in [0, 1] after many updates
// ============================================================

TEST(SignalEnsembleLayer, WeightsBoundedAfterManyUpdates) {
    SignalEnsembleLayer::Config cfg;
    cfg.learning_rate = 0.05;
    SignalEnsembleLayer ens(cfg);
    int a = ens.register_source("a");
    int b = ens.register_source("b");

    for (int i = 0; i < 100; ++i) {
        ens.update_source(a,  1.0);
        ens.update_source(b, -1.0);
        ens.record_outcome(i % 2 == 0 ? 1.0 : -1.0);
    }

    auto w = ens.weights();
    double sum = 0.0;
    for (double wi : w) {
        EXPECT_GE(wi, 0.0);
        EXPECT_LE(wi, 1.0);
        sum += wi;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

// ============================================================
// weight(id) accessor
// ============================================================

TEST(SignalEnsembleLayer, WeightAccessorOutOfRange) {
    SignalEnsembleLayer ens;
    EXPECT_NEAR(ens.weight(0), 0.0, 1e-12);
    EXPECT_NEAR(ens.weight(-1), 0.0, 1e-12);
}

// ============================================================
// on_weight_update callback
// ============================================================

TEST(SignalEnsembleLayer, WeightUpdateCallbackFires) {
    SignalEnsembleLayer::Config cfg;
    cfg.learning_rate = 0.1;
    int cb_count = 0;
    cfg.on_weight_update = [&](const std::vector<double>&) { ++cb_count; };
    SignalEnsembleLayer ens(cfg);
    ens.register_source("x");
    ens.update_source(0, 1.0);
    ens.record_outcome(1.0);
    EXPECT_GE(cb_count, 1);
}

// ============================================================
// reset restores uniform weights
// ============================================================

TEST(SignalEnsembleLayer, ResetRestoresUniformWeights) {
    SignalEnsembleLayer::Config cfg;
    cfg.learning_rate = 0.3;
    SignalEnsembleLayer ens(cfg);
    int a = ens.register_source("a");
    int b = ens.register_source("b");

    ens.update_source(a, 1.0);
    ens.update_source(b, 0.0);
    for (int i = 0; i < 50; ++i) ens.record_outcome(1.0);

    ens.reset();
    EXPECT_NEAR(ens.weight(a), 0.5, 1e-9);
    EXPECT_NEAR(ens.weight(b), 0.5, 1e-9);
    EXPECT_EQ(ens.total_outcomes(), 0u);
}

// ============================================================
// update_config doesn't reset weights
// ============================================================

TEST(SignalEnsembleLayer, UpdateConfigKeepsWeights) {
    SignalEnsembleLayer::Config cfg;
    cfg.learning_rate = 0.3;
    SignalEnsembleLayer ens(cfg);
    int a = ens.register_source("a");
    int b = ens.register_source("b");

    ens.update_source(a, 1.0);
    ens.update_source(b, 0.0);
    for (int i = 0; i < 20; ++i) ens.record_outcome(1.0);

    double wa = ens.weight(a);
    double wb = ens.weight(b);

    SignalEnsembleLayer::Config cfg2;
    cfg2.learning_rate = 0.001;
    ens.update_config(cfg2);

    // Weights should be preserved.
    EXPECT_NEAR(ens.weight(a), wa, 1e-9);
    EXPECT_NEAR(ens.weight(b), wb, 1e-9);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(SignalEnsembleLayer, ToStatsJsonContainsKeys) {
    SignalEnsembleLayer ens;
    ens.register_source("regime");
    ens.update_source(0, 0.7);
    std::string json = ens.to_stats_json();
    EXPECT_NE(json.find("source_count"),    std::string::npos);
    EXPECT_NE(json.find("total_outcomes"),  std::string::npos);
    EXPECT_NE(json.find("sources"),         std::string::npos);
    EXPECT_NE(json.find("regime"),          std::string::npos);
    EXPECT_NE(json.find("weight"),          std::string::npos);
}

} // namespace

#else  // LLMQUANT_SIGNAL_ENSEMBLE_ENABLED not defined

TEST(SignalEnsembleLayer, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
