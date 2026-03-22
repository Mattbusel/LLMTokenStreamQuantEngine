#include <gtest/gtest.h>

#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED

#include "SentimentDivergenceDetector.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(SentimentDivergenceDetector, DefaultConstructs) {
    SentimentDivergenceDetector det;
    EXPECT_NEAR(det.divergence(), 0.0, 1e-9);
    EXPECT_FALSE(det.is_diverged());
    EXPECT_EQ(det.source_count(), 0u);
    EXPECT_EQ(det.divergence_events(), 0u);
}

// ============================================================
// register_source adds sources
// ============================================================

TEST(SentimentDivergenceDetector, RegisterSourceAddsCount) {
    SentimentDivergenceDetector det;
    det.register_source("reddit");
    det.register_source("twitter");
    EXPECT_EQ(det.source_count(), 2u);
}

// ============================================================
// Duplicate registration is idempotent
// ============================================================

TEST(SentimentDivergenceDetector, DuplicateRegistrationIgnored) {
    SentimentDivergenceDetector det;
    det.register_source("reddit");
    det.register_source("reddit");
    EXPECT_EQ(det.source_count(), 1u);
}

// ============================================================
// Agreeing sources produce low divergence
// ============================================================

TEST(SentimentDivergenceDetector, AgreingSourcesLowDivergence) {
    SentimentDivergenceDetector::Config cfg;
    cfg.ema_alpha      = 0.5;
    cfg.min_observations = 1;
    SentimentDivergenceDetector det(cfg);
    det.register_source("reddit");
    det.register_source("twitter");

    for (int i = 0; i < 20; ++i) {
        det.record("reddit",  0.5);
        det.record("twitter", 0.5);
    }

    EXPECT_LT(det.divergence(), 0.2);
    EXPECT_FALSE(det.is_diverged());
}

// ============================================================
// Disagreeing sources produce high divergence
// ============================================================

TEST(SentimentDivergenceDetector, DisagreeingSourcesHighDivergence) {
    SentimentDivergenceDetector::Config cfg;
    cfg.ema_alpha          = 0.8;
    cfg.divergence_threshold = 0.4;
    cfg.min_observations   = 1;
    SentimentDivergenceDetector det(cfg);
    det.register_source("reddit");
    det.register_source("twitter");

    for (int i = 0; i < 30; ++i) {
        det.record("reddit",   1.0);   // very bullish
        det.record("twitter", -1.0);   // very bearish
    }

    EXPECT_GT(det.divergence(), 0.4);
    EXPECT_TRUE(det.is_diverged());
}

// ============================================================
// on_divergence callback fires
// ============================================================

TEST(SentimentDivergenceDetector, DivergenceCallbackFires) {
    SentimentDivergenceDetector::Config cfg;
    cfg.ema_alpha          = 0.9;
    cfg.divergence_threshold = 0.3;
    cfg.min_observations   = 1;
    std::atomic<int> fires{0};
    cfg.on_divergence = [&](double, const std::string&, const std::string&) {
        ++fires;
    };
    SentimentDivergenceDetector det(cfg);
    det.register_source("a");
    det.register_source("b");

    for (int i = 0; i < 20; ++i) {
        det.record("a",  1.0);
        det.record("b", -1.0);
    }

    EXPECT_GT(fires.load(), 0);
    EXPECT_GT(det.divergence_events(), 0u);
}

// ============================================================
// on_recovery fires when sources converge
// ============================================================

TEST(SentimentDivergenceDetector, RecoveryCallbackFires) {
    SentimentDivergenceDetector::Config cfg;
    cfg.ema_alpha             = 0.8;
    cfg.divergence_threshold  = 0.3;
    cfg.recovery_hysteresis   = 0.5;
    cfg.min_observations      = 1;
    std::atomic<int> recoveries{0};
    cfg.on_recovery = [&](double) { ++recoveries; };
    SentimentDivergenceDetector det(cfg);
    det.register_source("a");
    det.register_source("b");

    // Create divergence.
    for (int i = 0; i < 15; ++i) { det.record("a", 1.0); det.record("b", -1.0); }
    EXPECT_TRUE(det.is_diverged());

    // Converge.
    for (int i = 0; i < 30; ++i) { det.record("a", 0.0); det.record("b", 0.0); }

    EXPECT_GT(recoveries.load(), 0);
    EXPECT_GT(det.recovery_events(), 0u);
}

// ============================================================
// source_ema returns -2.0 for unknown source
// ============================================================

TEST(SentimentDivergenceDetector, UnknownSourceEmaReturnsSentinel) {
    SentimentDivergenceDetector det;
    EXPECT_NEAR(det.source_ema("nonexistent"), -2.0, 1e-9);
}

// ============================================================
// reset() clears EMAs
// ============================================================

TEST(SentimentDivergenceDetector, ResetClearsEmas) {
    SentimentDivergenceDetector::Config cfg;
    cfg.ema_alpha = 0.5;
    cfg.min_observations = 1;
    SentimentDivergenceDetector det(cfg);
    det.register_source("a");
    det.register_source("b");
    for (int i = 0; i < 20; ++i) { det.record("a", 1.0); det.record("b", -1.0); }
    EXPECT_GT(det.divergence(), 0.0);

    det.reset();
    EXPECT_NEAR(det.divergence(), 0.0, 1e-9);
    EXPECT_FALSE(det.is_diverged());
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(SentimentDivergenceDetector, ToStatsJsonContainsKeys) {
    SentimentDivergenceDetector det;
    det.register_source("reddit");
    det.record("reddit", 0.3);
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("divergence"),        std::string::npos);
    EXPECT_NE(json.find("is_diverged"),       std::string::npos);
    EXPECT_NE(json.find("sources"),           std::string::npos);
    EXPECT_NE(json.find("divergence_events"), std::string::npos);
}

// ============================================================
// record for unregistered source is silently ignored
// ============================================================

TEST(SentimentDivergenceDetector, UnknownSourceRecordIsNoop) {
    SentimentDivergenceDetector det;
    det.register_source("a");
    // No crash, no state change.
    det.record("unknown", 999.0);
    EXPECT_NEAR(det.divergence(), 0.0, 1e-9);
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(SentimentDivergenceDetector, ConcurrentRecordSafe) {
    SentimentDivergenceDetector::Config cfg;
    cfg.ema_alpha        = 0.1;
    cfg.min_observations = 1;
    SentimentDivergenceDetector det(cfg);
    det.register_source("s0");
    det.register_source("s1");
    det.register_source("s2");
    det.register_source("s3");

    std::vector<std::thread> threads;
    const char* src[] = {"s0", "s1", "s2", "s3"};
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                det.record(src[t], static_cast<double>(t % 2 == 0 ? 1 : -1) * 0.4);
        });
    }
    for (auto& th : threads) th.join();

    double div = det.divergence();
    EXPECT_GE(div, 0.0);
    EXPECT_LE(div, 2.0);
}

} // namespace

#else

TEST(SentimentDivergenceDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
