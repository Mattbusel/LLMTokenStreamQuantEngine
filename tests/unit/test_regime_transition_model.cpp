#include <gtest/gtest.h>

#if defined(LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED) && defined(LLMQUANT_REGIME_DETECTOR_ENABLED)

#include "RegimeTransitionModel.h"

#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;
using Regime = RegimeDetector::Regime;

namespace {

// ---------------------------------------------------------------------------
// Construction and defaults
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, DefaultConstructs) {
    RegimeTransitionModel model;
    EXPECT_EQ(model.total_transitions(), 0u);
}

// ---------------------------------------------------------------------------
// record_transition increments counter
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, RecordIncrements) {
    RegimeTransitionModel model;
    model.record_transition(Regime::Neutral, Regime::Bull);
    EXPECT_EQ(model.total_transitions(), 1u);
    model.record_transition(Regime::Bull, Regime::Bear);
    EXPECT_EQ(model.total_transitions(), 2u);
}

// ---------------------------------------------------------------------------
// predict_next with no data returns Neutral, prob=0
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, PredictNextNoDataReturnsNeutral) {
    RegimeTransitionModel model;
    auto pred = model.predict_next(Regime::Bull);
    EXPECT_NEAR(pred.prob, 0.0, 1e-9);
    EXPECT_FALSE(pred.reliable);
}

// ---------------------------------------------------------------------------
// predict_next returns dominant transition
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, PredictNextReturnsDominant) {
    RegimeTransitionModel model;
    model.min_transitions_for_reliability = 3;

    // Bull → Bear 3 times, Bull → Volatile 1 time.
    for (int i = 0; i < 3; ++i) model.record_transition(Regime::Bull, Regime::Bear);
    model.record_transition(Regime::Bull, Regime::Volatile);

    auto pred = model.predict_next(Regime::Bull);
    EXPECT_EQ(pred.regime, Regime::Bear);
    EXPECT_NEAR(pred.prob, 0.75, 1e-6);
    EXPECT_TRUE(pred.reliable);
}

// ---------------------------------------------------------------------------
// transition_probabilities rows sum to 1 for non-zero rows
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, ProbMatrixRowsSumToOne) {
    RegimeTransitionModel model;
    model.record_transition(Regime::Neutral, Regime::Bull);
    model.record_transition(Regime::Neutral, Regime::Bear);
    model.record_transition(Regime::Neutral, Regime::Volatile);
    model.record_transition(Regime::Bull, Regime::RiskOff);

    auto P = model.transition_probabilities();

    // Neutral row: 3 transitions → each = 1/3 → sum = 1.
    double sum_neutral = 0;
    for (double v : P[static_cast<int>(Regime::Neutral)]) sum_neutral += v;
    EXPECT_NEAR(sum_neutral, 1.0, 1e-9);

    // Bull row: 1 transition.
    double sum_bull = 0;
    for (double v : P[static_cast<int>(Regime::Bull)]) sum_bull += v;
    EXPECT_NEAR(sum_bull, 1.0, 1e-9);

    // Bear row: never transitioned → sum = 0.
    double sum_bear = 0;
    for (double v : P[static_cast<int>(Regime::Bear)]) sum_bear += v;
    EXPECT_NEAR(sum_bear, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// steady_state returns uniform for insufficient data
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, SteadyStateUniformForNoData) {
    RegimeTransitionModel model;
    auto ss = model.steady_state();
    for (double v : ss) EXPECT_NEAR(v, 0.2, 1e-9);
}

// ---------------------------------------------------------------------------
// steady_state converges for ergodic chain
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, SteadyStateConvergesForErgodicChain) {
    RegimeTransitionModel model;
    model.min_transitions_for_reliability = 1;

    // Build an ergodic chain: cycle Neutral → Bull → Bear → Neutral (many times).
    for (int i = 0; i < 30; ++i) {
        model.record_transition(Regime::Neutral, Regime::Bull);
        model.record_transition(Regime::Bull,    Regime::Bear);
        model.record_transition(Regime::Bear,    Regime::Neutral);
    }

    auto ss = model.steady_state();

    // Sum must be ≈ 1.
    double sum = 0;
    for (double v : ss) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-6);

    // The three active regimes should each have ~1/3 probability.
    EXPECT_NEAR(ss[static_cast<int>(Regime::Neutral)], 1.0/3, 0.05);
    EXPECT_NEAR(ss[static_cast<int>(Regime::Bull)],    1.0/3, 0.05);
    EXPECT_NEAR(ss[static_cast<int>(Regime::Bear)],    1.0/3, 0.05);
}

// ---------------------------------------------------------------------------
// reset clears all counts
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, ResetClearsState) {
    RegimeTransitionModel model;
    for (int i = 0; i < 5; ++i)
        model.record_transition(Regime::Bull, Regime::Bear);
    model.reset();
    EXPECT_EQ(model.total_transitions(), 0u);
    auto pred = model.predict_next(Regime::Bull);
    EXPECT_NEAR(pred.prob, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, ToStatsJsonContainsFields) {
    RegimeTransitionModel model;
    model.record_transition(Regime::Neutral, Regime::Bull);
    std::string j = model.to_stats_json();
    EXPECT_NE(j.find("total_transitions"), std::string::npos);
    EXPECT_NE(j.find("steady_state"), std::string::npos);
    EXPECT_NE(j.find("transition_matrix"), std::string::npos);
    EXPECT_NE(j.find("Bull"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Self-transition recorded correctly
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, SelfTransitionRecorded) {
    RegimeTransitionModel model;
    model.record_transition(Regime::Volatile, Regime::Volatile);
    EXPECT_EQ(model.total_transitions(), 1u);
    auto pred = model.predict_next(Regime::Volatile);
    EXPECT_EQ(pred.regime, Regime::Volatile);
    EXPECT_NEAR(pred.prob, 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Concurrent record_transition is safe
// ---------------------------------------------------------------------------

TEST(RegimeTransitionModel, ConcurrentRecordSafe) {
    RegimeTransitionModel model;
    std::atomic<bool> go{false};

    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 100; ++i) {
            model.record_transition(Regime::Bull, Regime::Bear);
            model.record_transition(Regime::Bear, Regime::Neutral);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(model.total_transitions(), 800u);
    auto P = model.transition_probabilities();
    // Bull → Bear should dominate.
    double bull_to_bear = P[static_cast<int>(Regime::Bull)]
                           [static_cast<int>(Regime::Bear)];
    EXPECT_NEAR(bull_to_bear, 1.0, 1e-6);
}

} // namespace

#else

TEST(RegimeTransitionModel, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED && LLMQUANT_REGIME_DETECTOR_ENABLED
