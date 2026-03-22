#include <gtest/gtest.h>

#ifdef LLMQUANT_BLIND_SPOT_ENABLED

#include "SignalBlindSpotDetector.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction — initial state
// ============================================================

TEST(SignalBlindSpotDetector, DefaultStateIsClean) {
    SignalBlindSpotDetector det;
    EXPECT_EQ(det.total_outcomes(), 0u);
    EXPECT_EQ(det.detection_events(), 0u);
    EXPECT_EQ(det.blind_spot_count(), 0);
    for (int s = 0; s < SignalBlindSpotDetector::kSlots; ++s)
        EXPECT_FALSE(det.is_blind_spot(s));
}

// ============================================================
// Out-of-range slot is silently ignored
// ============================================================

TEST(SignalBlindSpotDetector, OutOfRangeSlotIgnored) {
    SignalBlindSpotDetector det;
    det.record_outcome(-1, true);
    det.record_outcome(SignalBlindSpotDetector::kSlots, false);
    EXPECT_EQ(det.total_outcomes(), 0u);
}

// ============================================================
// Win-rate default (neutral 0.5) before samples
// ============================================================

TEST(SignalBlindSpotDetector, WinRateDefaultBeforeSamples) {
    SignalBlindSpotDetector det;
    EXPECT_DOUBLE_EQ(det.win_rate(0), 0.5);
}

// ============================================================
// Blind spot requires min_samples before triggering
// ============================================================

TEST(SignalBlindSpotDetector, NotBlindWithFewSamples) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples          = 20;
    cfg.blind_spot_threshold = 0.4;
    cfg.forgetting           = 1.0;  // no forgetting for predictable counts
    SignalBlindSpotDetector det(cfg);

    // Feed 5 losses — not enough samples yet.
    for (int i = 0; i < 5; ++i) det.record_outcome(3, false);
    EXPECT_FALSE(det.is_blind_spot(3));
}

// ============================================================
// Enough losses flags the slot as a blind spot
// ============================================================

TEST(SignalBlindSpotDetector, BlindSpotDetectedAfterMinSamples) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples          = 5;
    cfg.blind_spot_threshold = 0.4;
    cfg.forgetting           = 0.9;
    SignalBlindSpotDetector det(cfg);

    // All losses → win-rate ≈ 0 → blind spot
    for (int i = 0; i < 30; ++i) det.record_outcome(7, false);

    EXPECT_TRUE(det.is_blind_spot(7));
    EXPECT_GE(det.blind_spot_count(), 1);
}

// ============================================================
// Slot recovers when wins arrive
// ============================================================

TEST(SignalBlindSpotDetector, SlotRecoverAfterWins) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples          = 5;
    cfg.blind_spot_threshold = 0.3;
    cfg.forgetting           = 0.8;  // faster forgetting for test speed
    SignalBlindSpotDetector det(cfg);

    // Make slot 2 a blind spot.
    for (int i = 0; i < 20; ++i) det.record_outcome(2, false);
    EXPECT_TRUE(det.is_blind_spot(2));

    // Flood with wins; forgetting factor 0.8 means old losses decay fast.
    for (int i = 0; i < 80; ++i) det.record_outcome(2, true);
    EXPECT_FALSE(det.is_blind_spot(2));
}

// ============================================================
// on_blind_spot_found callback fires exactly once per slot
// ============================================================

TEST(SignalBlindSpotDetector, CallbackFiresOnce) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples          = 3;
    cfg.blind_spot_threshold = 0.4;
    cfg.forgetting           = 0.9;
    std::atomic<int> fires{0};
    cfg.on_blind_spot_found = [&](int, double) { ++fires; };
    SignalBlindSpotDetector det(cfg);

    // Drive slot 5 into blind spot.
    for (int i = 0; i < 20; ++i) det.record_outcome(5, false);

    EXPECT_EQ(fires.load(), 1);
    EXPECT_EQ(det.detection_events(), 1u);
}

// ============================================================
// Multiple slots can be blind simultaneously
// ============================================================

TEST(SignalBlindSpotDetector, MultipleBlindSlots) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples          = 5;
    cfg.blind_spot_threshold = 0.4;
    cfg.forgetting           = 0.9;
    SignalBlindSpotDetector det(cfg);

    for (int s : {1, 2, 3}) {
        for (int i = 0; i < 30; ++i) det.record_outcome(s, false);
    }

    EXPECT_GE(det.blind_spot_count(), 3);
}

// ============================================================
// reset() clears all state
// ============================================================

TEST(SignalBlindSpotDetector, ResetClearsState) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples = 5;
    cfg.forgetting  = 0.9;
    SignalBlindSpotDetector det(cfg);

    for (int i = 0; i < 30; ++i) det.record_outcome(0, false);
    EXPECT_TRUE(det.is_blind_spot(0));

    det.reset();
    EXPECT_FALSE(det.is_blind_spot(0));
    EXPECT_EQ(det.blind_spot_count(), 0);
}

// ============================================================
// update_config resets state
// ============================================================

TEST(SignalBlindSpotDetector, UpdateConfigResetsState) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples          = 5;
    cfg.blind_spot_threshold = 0.4;
    cfg.forgetting           = 0.9;
    SignalBlindSpotDetector det(cfg);

    for (int i = 0; i < 30; ++i) det.record_outcome(10, false);
    EXPECT_TRUE(det.is_blind_spot(10));

    SignalBlindSpotDetector::Config cfg2;
    det.update_config(cfg2);
    EXPECT_FALSE(det.is_blind_spot(10));
}

// ============================================================
// sample_count reflects EMA effective count
// ============================================================

TEST(SignalBlindSpotDetector, SampleCountNonZeroAfterRecords) {
    SignalBlindSpotDetector det;
    for (int i = 0; i < 20; ++i) det.record_outcome(0, true);
    EXPECT_GT(det.sample_count(0), 0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(SignalBlindSpotDetector, ToStatsJsonContainsKeys) {
    SignalBlindSpotDetector det;
    det.record_outcome(0, true);
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("blind_spot_count"),  std::string::npos);
    EXPECT_NE(json.find("total_outcomes"),    std::string::npos);
    EXPECT_NE(json.find("detection_events"),  std::string::npos);
    EXPECT_NE(json.find("slots"),             std::string::npos);
}

// ============================================================
// Concurrent record_outcome() is thread-safe
// ============================================================

TEST(SignalBlindSpotDetector, ConcurrentRecordSafe) {
    SignalBlindSpotDetector::Config cfg;
    cfg.min_samples = 5;
    cfg.forgetting  = 0.95;
    SignalBlindSpotDetector det(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 200; ++i)
                det.record_outcome(t % SignalBlindSpotDetector::kSlots, i % 2 == 0);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(det.total_outcomes(), 800u);
    // Should not crash; blind_spot_count is a valid non-negative int.
    EXPECT_GE(det.blind_spot_count(), 0);
}

} // namespace

#else  // LLMQUANT_BLIND_SPOT_ENABLED not defined

TEST(SignalBlindSpotDetector, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_BLIND_SPOT_ENABLED
