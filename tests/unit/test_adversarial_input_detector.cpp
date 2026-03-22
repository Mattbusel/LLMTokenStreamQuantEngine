#include <gtest/gtest.h>

#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED

#include "AdversarialInputDetector.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(AdversarialInputDetector, DefaultConstructs) {
    AdversarialInputDetector det;
    EXPECT_FALSE(det.is_armed());
    EXPECT_EQ(det.anomaly_count(), 0u);
    EXPECT_EQ(det.total_tokens(), 0u);
}

// ============================================================
// Normal tokens do not arm the detector
// ============================================================

TEST(AdversarialInputDetector, NormalTokensDoNotArm) {
    AdversarialInputDetector::Config cfg;
    cfg.anomaly_threshold  = 4.0;
    cfg.min_warmup_tokens  = 5;
    cfg.max_repeat_fraction = 0.8;
    AdversarialInputDetector det(cfg);

    const char* words[] = {"bull", "bear", "rally", "fear", "momentum"};
    for (int i = 0; i < 30; ++i)
        det.inspect(words[i % 5], 0.1 + 0.01 * (i % 5));

    EXPECT_FALSE(det.is_armed());
}

// ============================================================
// Weight anomaly fires on extreme token weight
// ============================================================

TEST(AdversarialInputDetector, WeightAnomalyFires) {
    AdversarialInputDetector::Config cfg;
    cfg.anomaly_threshold = 3.0;
    cfg.min_warmup_tokens = 10;
    std::atomic<int> fires{0};
    cfg.on_anomaly = [&](AdversarialInputDetector::AnomalyKind,
                         const std::string&, double) { ++fires; };
    AdversarialInputDetector det(cfg);

    // Warm up with tiny weights.
    for (int i = 0; i < 20; ++i)
        det.inspect("normal", 0.01);

    // Inject a weight far beyond the learned mean.
    bool armed = det.inspect("attack", 1000.0);
    EXPECT_TRUE(armed);
    EXPECT_TRUE(det.is_armed());
    EXPECT_GT(fires.load(), 0);
}

// ============================================================
// Repetition attack fires when single token dominates window
// ============================================================

TEST(AdversarialInputDetector, RepetitionAttackFires) {
    AdversarialInputDetector::Config cfg;
    cfg.repeat_window       = 10;
    cfg.max_repeat_fraction = 0.6;
    cfg.min_warmup_tokens   = 100;  // disable weight anomaly
    std::atomic<int> rep_fires{0};
    cfg.on_anomaly = [&](AdversarialInputDetector::AnomalyKind kind,
                         const std::string&, double) {
        if (kind == AdversarialInputDetector::AnomalyKind::RepetitionAttack)
            ++rep_fires;
    };
    AdversarialInputDetector det(cfg);

    // Flood the window with a single token.
    for (int i = 0; i < 15; ++i)
        det.inspect("ATTACK", 0.01);

    EXPECT_TRUE(det.is_armed());
    EXPECT_GT(rep_fires.load(), 0);
}

// ============================================================
// Vocabulary inflation fires on sudden large new vocabulary
// ============================================================

TEST(AdversarialInputDetector, VocabularyInflationFires) {
    AdversarialInputDetector::Config cfg;
    cfg.vocab_window      = 10;
    cfg.max_novel_fraction = 0.7;
    cfg.min_warmup_tokens  = 100;  // disable weight anomaly
    cfg.repeat_window      = 100;  // disable repetition
    std::atomic<int> vocab_fires{0};
    cfg.on_anomaly = [&](AdversarialInputDetector::AnomalyKind kind,
                         const std::string&, double) {
        if (kind == AdversarialInputDetector::AnomalyKind::VocabularyInflation)
            ++vocab_fires;
    };
    AdversarialInputDetector det(cfg);

    // Flood with completely novel tokens.
    for (int i = 0; i < 20; ++i)
        det.inspect("novel_token_" + std::to_string(i), 0.01);

    EXPECT_GT(vocab_fires.load(), 0);
}

// ============================================================
// reset() disarms the detector
// ============================================================

TEST(AdversarialInputDetector, ResetDisarms) {
    AdversarialInputDetector::Config cfg;
    cfg.anomaly_threshold = 2.0;
    cfg.min_warmup_tokens = 5;
    AdversarialInputDetector det(cfg);

    for (int i = 0; i < 10; ++i) det.inspect("norm", 0.01);
    det.inspect("extreme", 999.0);
    EXPECT_TRUE(det.is_armed());

    det.reset();
    EXPECT_FALSE(det.is_armed());
    EXPECT_EQ(det.anomaly_count(), 0u);
}

// ============================================================
// total_tokens increments
// ============================================================

TEST(AdversarialInputDetector, TotalTokensIncrements) {
    AdversarialInputDetector det;
    det.inspect("a", 0.1);
    det.inspect("b", 0.2);
    det.inspect("c", 0.3);
    EXPECT_EQ(det.total_tokens(), 3u);
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(AdversarialInputDetector, ToStatsJsonContainsKeys) {
    AdversarialInputDetector det;
    det.inspect("bull", 0.5);
    std::string json = det.to_stats_json();
    EXPECT_NE(json.find("is_armed"),      std::string::npos);
    EXPECT_NE(json.find("anomaly_count"), std::string::npos);
    EXPECT_NE(json.find("weight_mean"),   std::string::npos);
    EXPECT_NE(json.find("weight_std"),    std::string::npos);
}

// ============================================================
// Concurrent inspect() is thread-safe
// ============================================================

TEST(AdversarialInputDetector, ConcurrentInspectSafe) {
    AdversarialInputDetector::Config cfg;
    cfg.anomaly_threshold  = 100.0;  // very high to avoid spurious arms
    cfg.min_warmup_tokens  = 1000;
    cfg.max_repeat_fraction = 1.0;
    cfg.max_novel_fraction  = 1.0;
    AdversarialInputDetector det(cfg);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                det.inspect("tok" + std::to_string(t), 0.1 * t);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(det.total_tokens(), 200u);
}

} // namespace

#else

TEST(AdversarialInputDetector, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_ADVERSARIAL_DETECT_ENABLED
