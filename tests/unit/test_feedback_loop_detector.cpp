#include <gtest/gtest.h>

#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED

#include "FeedbackLoopDetector.h"

#include <cmath>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(FeedbackLoopDetector, DefaultConstructsWithZeroScore) {
    FeedbackLoopDetector det;
    EXPECT_NEAR(det.feedback_score(), 0.0, 1e-12);
    EXPECT_FALSE(det.feedback_detected());
    EXPECT_EQ(det.total_activity_records(), 0u);
    EXPECT_EQ(det.total_sentiment_records(), 0u);
}

// ============================================================
// Returns 0 before min_samples
// ============================================================

TEST(FeedbackLoopDetector, ZeroScoreBeforeMinSamples) {
    FeedbackLoopDetector::Config cfg;
    cfg.min_samples = 20;
    FeedbackLoopDetector det(cfg);

    for (int i = 0; i < 5; ++i) {
        det.record_own_activity(1.0);
        det.record_sentiment(1.0);
    }
    EXPECT_NEAR(det.feedback_score(), 0.0, 1e-12);
    EXPECT_FALSE(det.feedback_detected());
}

// ============================================================
// Perfectly correlated signal detects feedback
// ============================================================

TEST(FeedbackLoopDetector, PerfectCorrelationDetected) {
    FeedbackLoopDetector::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 20;
    cfg.max_lag     = 3;
    cfg.threshold   = 0.5;
    FeedbackLoopDetector det(cfg);

    // Activity at t, sentiment at t+1 = activity (perfect lag-1 feedback).
    std::vector<double> activity(50);
    for (int i = 0; i < 50; ++i) activity[static_cast<std::size_t>(i)] = (i % 5 == 0) ? 2.0 : 0.0;

    for (int i = 0; i < 50; ++i) {
        det.record_own_activity(activity[static_cast<std::size_t>(i)]);
        // Sentiment follows activity with a lag of 1.
        det.record_sentiment(i > 0 ? activity[static_cast<std::size_t>(i - 1)] : 0.0);
    }

    EXPECT_GT(det.feedback_score(), 0.0);
}

// ============================================================
// Uncorrelated signals give low feedback score
// ============================================================

TEST(FeedbackLoopDetector, UncorrelatedSignalsLowScore) {
    FeedbackLoopDetector::Config cfg;
    cfg.window_size = 60;
    cfg.min_samples = 30;
    cfg.max_lag     = 5;
    cfg.threshold   = 0.8;
    FeedbackLoopDetector det(cfg);

    // Activity = constant 1.0, sentiment = alternating +/-1 (uncorrelated).
    for (int i = 0; i < 60; ++i) {
        det.record_own_activity(1.0);
        det.record_sentiment(i % 2 == 0 ? 1.0 : -1.0);
    }

    // Cross-correlation should be near zero for constant vs alternating.
    EXPECT_LT(std::abs(det.feedback_score()), 0.5);
}

// ============================================================
// feedback_events increments on callback
// ============================================================

TEST(FeedbackLoopDetector, FeedbackCallbackFiresOnDetection) {
    FeedbackLoopDetector::Config cfg;
    cfg.window_size = 30;
    cfg.min_samples = 15;
    cfg.max_lag     = 2;
    cfg.threshold   = 0.3;  // low threshold for easy triggering
    int fires = 0;
    cfg.on_feedback = [&](double, int) { ++fires; };
    FeedbackLoopDetector det(cfg);

    // Feed perfectly correlated activity and sentiment with lag 1.
    for (int i = 0; i < 30; ++i) {
        double act  = (i % 3 == 0) ? 1.0 : 0.0;
        double sent = (i > 0 && (i - 1) % 3 == 0) ? 1.0 : 0.0;
        det.record_own_activity(act);
        det.record_sentiment(sent);
    }

    EXPECT_GE(fires, 1);
}

// ============================================================
// peak_lag is in [1, max_lag]
// ============================================================

TEST(FeedbackLoopDetector, PeakLagInRange) {
    FeedbackLoopDetector::Config cfg;
    cfg.window_size = 40;
    cfg.min_samples = 20;
    cfg.max_lag     = 5;
    FeedbackLoopDetector det(cfg);

    for (int i = 0; i < 40; ++i) {
        det.record_own_activity(static_cast<double>(i % 7));
        det.record_sentiment(static_cast<double>((i + 2) % 7));
    }

    int lag = det.peak_lag();
    EXPECT_GE(lag, 0);
    EXPECT_LE(lag, cfg.max_lag);
}

// ============================================================
// reset clears state
// ============================================================

TEST(FeedbackLoopDetector, ResetClearsState) {
    FeedbackLoopDetector::Config cfg;
    cfg.window_size = 30;
    cfg.min_samples = 15;
    cfg.max_lag     = 3;
    cfg.threshold   = 0.3;
    FeedbackLoopDetector det(cfg);

    for (int i = 0; i < 30; ++i) {
        det.record_own_activity(static_cast<double>(i));
        det.record_sentiment(static_cast<double>(i));
    }

    det.reset();
    EXPECT_NEAR(det.feedback_score(), 0.0, 1e-12);
    EXPECT_FALSE(det.feedback_detected());
    EXPECT_EQ(det.total_activity_records(), 30u);  // counters not reset
}

// ============================================================
// update_config resets state
// ============================================================

TEST(FeedbackLoopDetector, UpdateConfigResetsState) {
    FeedbackLoopDetector det;
    for (int i = 0; i < 30; ++i) {
        det.record_own_activity(1.0);
        det.record_sentiment(1.0);
    }
    FeedbackLoopDetector::Config cfg2;
    cfg2.window_size = 20;
    det.update_config(cfg2);
    EXPECT_NEAR(det.feedback_score(), 0.0, 1e-12);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(FeedbackLoopDetector, ToStatsJsonContainsKeys) {
    FeedbackLoopDetector det;
    det.record_own_activity(1.0);
    det.record_sentiment(1.0);
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("feedback_score"),       std::string::npos);
    EXPECT_NE(json.find("peak_lag"),             std::string::npos);
    EXPECT_NE(json.find("feedback_detected"),    std::string::npos);
    EXPECT_NE(json.find("total_activity"),       std::string::npos);
    EXPECT_NE(json.find("total_sentiment"),      std::string::npos);
    EXPECT_NE(json.find("feedback_events"),      std::string::npos);
}

} // namespace

#else  // LLMQUANT_FEEDBACK_LOOP_ENABLED not defined

TEST(FeedbackLoopDetector, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_FEEDBACK_LOOP_ENABLED
