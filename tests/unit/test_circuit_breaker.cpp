#include <gtest/gtest.h>

#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED

#include "PipelineCircuitBreaker.h"

#include <chrono>
#include <thread>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Feed N blocked signals into the breaker.
static void feed_blocked(PipelineCircuitBreaker& cb, int n) {
    for (int i = 0; i < n; ++i) cb.record_signal(/*blocked=*/true);
}

// Feed N passing signals into the breaker.
static void feed_passed(PipelineCircuitBreaker& cb, int n) {
    for (int i = 0; i < n; ++i) cb.record_signal(/*blocked=*/false);
}

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, InitialStateClosed) {
    PipelineCircuitBreaker cb;
    EXPECT_EQ(cb.state(), PipelineCircuitBreaker::State::Closed);
    EXPECT_FALSE(cb.is_open());
    EXPECT_FALSE(cb.is_half_open());
    EXPECT_EQ(cb.trips(), 0u);
    EXPECT_EQ(cb.recoveries(), 0u);
    EXPECT_DOUBLE_EQ(cb.block_rate(), 0.0);
}

// ---------------------------------------------------------------------------
// EMA update — rate rises with repeated blocked signals
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, EmaRisesWithBlockedSignals) {
    PipelineCircuitBreaker::Config cfg;
    cfg.ema_alpha = 0.5;
    cfg.open_duration_s = 9999;  // don't actually trip
    PipelineCircuitBreaker cb(cfg);

    feed_blocked(cb, 10);
    EXPECT_GT(cb.block_rate(), 0.9) << "EMA should be near 1.0 after 10 consecutive blocked signals";
}

TEST(CircuitBreaker, EmaFallsWithPassedSignals) {
    PipelineCircuitBreaker::Config cfg;
    cfg.ema_alpha = 0.5;
    cfg.open_duration_s = 9999;
    PipelineCircuitBreaker cb(cfg);

    feed_blocked(cb, 10);
    feed_passed(cb, 20);
    EXPECT_LT(cb.block_rate(), 0.1) << "EMA should be near 0.0 after many passing signals";
}

// ---------------------------------------------------------------------------
// Circuit trips to OPEN after sustained high block rate
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, TripsAfterSustainedHighRate) {
    PipelineCircuitBreaker::Config cfg;
    cfg.open_threshold   = 0.80;
    cfg.close_threshold  = 0.40;
    cfg.open_duration_s  = 0;   // trip immediately (no sustain window)
    cfg.ema_alpha        = 1.0; // no smoothing — instant response
    PipelineCircuitBreaker cb(cfg);

    // First blocked signal sets rate=1.0 > 0.80 and timer starts.
    // With open_duration_s=0 the very first call should trip.
    feed_blocked(cb, 2);

    EXPECT_EQ(cb.state(), PipelineCircuitBreaker::State::Open);
    EXPECT_TRUE(cb.is_open());
    EXPECT_GE(cb.trips(), 1u);
}

// ---------------------------------------------------------------------------
// state_name returns the right string
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, StateNameClosed) {
    PipelineCircuitBreaker cb;
    EXPECT_EQ(cb.state_name(), "closed");
}

TEST(CircuitBreaker, StateNameOpen) {
    PipelineCircuitBreaker::Config cfg;
    cfg.open_threshold  = 0.80;
    cfg.open_duration_s = 0;
    cfg.ema_alpha       = 1.0;
    PipelineCircuitBreaker cb(cfg);
    feed_blocked(cb, 2);
    EXPECT_EQ(cb.state_name(), "open");
}

// ---------------------------------------------------------------------------
// force_close() overrides state
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, ForceCloseResets) {
    PipelineCircuitBreaker::Config cfg;
    cfg.open_threshold  = 0.80;
    cfg.open_duration_s = 0;
    cfg.ema_alpha       = 1.0;
    PipelineCircuitBreaker cb(cfg);
    feed_blocked(cb, 2);
    ASSERT_EQ(cb.state(), PipelineCircuitBreaker::State::Open);

    cb.force_close();
    EXPECT_EQ(cb.state(), PipelineCircuitBreaker::State::Closed);
}

// ---------------------------------------------------------------------------
// reset_stats() zeroes counters without changing state
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, ResetStatsZeroesCounters) {
    PipelineCircuitBreaker::Config cfg;
    cfg.open_threshold  = 0.80;
    cfg.open_duration_s = 0;
    cfg.ema_alpha       = 1.0;
    PipelineCircuitBreaker cb(cfg);
    feed_blocked(cb, 2);
    ASSERT_GE(cb.trips(), 1u);

    cb.reset_stats();
    EXPECT_EQ(cb.trips(), 0u);
    EXPECT_EQ(cb.recoveries(), 0u);
    EXPECT_DOUBLE_EQ(cb.block_rate(), 0.0);
    // State is unchanged.
    EXPECT_EQ(cb.state(), PipelineCircuitBreaker::State::Open);
}

// ---------------------------------------------------------------------------
// StateChangeCallback fires on OPEN transition
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, StateChangeCallbackFired) {
    PipelineCircuitBreaker::Config cfg;
    cfg.open_threshold  = 0.80;
    cfg.open_duration_s = 0;
    cfg.ema_alpha       = 1.0;
    PipelineCircuitBreaker cb(cfg);

    int callbacks = 0;
    PipelineCircuitBreaker::State last_state = PipelineCircuitBreaker::State::Closed;
    cb.set_state_change_callback([&](PipelineCircuitBreaker::State s, double) {
        callbacks++;
        last_state = s;
    });

    feed_blocked(cb, 2);
    EXPECT_GE(callbacks, 1);
    EXPECT_EQ(last_state, PipelineCircuitBreaker::State::Open);
}

// ---------------------------------------------------------------------------
// Closed circuit does not suppress signals
// ---------------------------------------------------------------------------

TEST(CircuitBreaker, ClosedCircuitDoesNotBlock) {
    PipelineCircuitBreaker cb;
    EXPECT_FALSE(cb.is_open());
    // Simulate logic: signal passes risk manager, circuit is closed → emit
    bool emit = !cb.is_open();
    EXPECT_TRUE(emit);
}

#else  // LLMQUANT_CIRCUIT_BREAKER_ENABLED not defined

TEST(CircuitBreaker, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_CIRCUIT_BREAKER_ENABLED
