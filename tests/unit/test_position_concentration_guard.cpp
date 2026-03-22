#include <gtest/gtest.h>

#ifdef LLMQUANT_POSITION_CONCENTRATION_ENABLED

#include "PositionConcentrationGuard.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, DefaultState) {
    PositionConcentrationGuard pcg;
    EXPECT_EQ(pcg.total_signals(), 0u);
    EXPECT_EQ(pcg.concentration_events(), 0u);
    EXPECT_FALSE(pcg.is_concentrated());
    EXPECT_DOUBLE_EQ(pcg.hhi(), 0.0);
    EXPECT_EQ(pcg.distinct_themes(), 0);
}

// ---------------------------------------------------------------------------
// Records increment total and distinct themes
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, RecordsIncrementCounters) {
    PositionConcentrationGuard pcg;
    pcg.record_signal("macro", 0.5);
    pcg.record_signal("earnings", 0.3);
    pcg.record_signal("macro", 0.2);
    EXPECT_EQ(pcg.total_signals(), 3u);
    EXPECT_EQ(pcg.distinct_themes(), 2);
}

// ---------------------------------------------------------------------------
// Single theme → HHI = 1
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, SingleThemeHHIIsOne) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size            = 8;
    cfg.concentration_threshold = 0.9;
    PositionConcentrationGuard pcg(cfg);
    for (int i = 0; i < 5; ++i) pcg.record_signal("only", 1.0);
    EXPECT_NEAR(pcg.hhi(), 1.0, 0.001);
}

// ---------------------------------------------------------------------------
// Equal themes → HHI = 1/K
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, EqualThemesHHI) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size = 32;
    PositionConcentrationGuard pcg(cfg);
    // 4 themes, each gets the same total bias.
    for (int i = 0; i < 10; ++i) {
        pcg.record_signal("a", 1.0);
        pcg.record_signal("b", 1.0);
        pcg.record_signal("c", 1.0);
        pcg.record_signal("d", 1.0);
    }
    // HHI should be ≈ 0.25 (= 1/4) for equal shares.
    EXPECT_NEAR(pcg.hhi(), 0.25, 0.05);
}

// ---------------------------------------------------------------------------
// Concentration callback fires when HHI > threshold
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, ConcentrationCallbackFires) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size            = 16;
    cfg.concentration_threshold = 0.5;
    std::atomic<int> fires{0};
    cfg.on_concentrated = [&](double, const std::string&) { ++fires; };
    PositionConcentrationGuard pcg(cfg);

    // Feed predominantly one theme to force HHI > 0.5.
    for (int i = 0; i < 10; ++i) pcg.record_signal("dominant", 10.0);
    pcg.record_signal("minor", 0.1);

    EXPECT_GT(fires.load(), 0);
    EXPECT_TRUE(pcg.is_concentrated());
    EXPECT_GT(pcg.concentration_events(), 0u);
}

// ---------------------------------------------------------------------------
// Diversification callback fires after concentration clears
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, DiversificationCallbackFires) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size            = 8;
    cfg.concentration_threshold = 0.5;
    cfg.clear_hysteresis        = 0.5;
    std::atomic<int> divs{0};
    cfg.on_diversified = [&](double) { ++divs; };
    PositionConcentrationGuard pcg(cfg);

    // Concentrate.
    for (int i = 0; i < 6; ++i) pcg.record_signal("dom", 10.0);
    pcg.record_signal("minor", 0.01);
    // Flush "dom"'s window: record 8 zeros so all 10.0 entries are evicted.
    for (int i = 0; i < 8; ++i) pcg.record_signal("dom", 0.0);
    // Now diversify with 5 equal themes (HHI = 5×(1/5)² = 0.2 < threshold×hysteresis).
    for (int i = 0; i < 20; ++i) {
        pcg.record_signal("a", 1.0);
        pcg.record_signal("b", 1.0);
        pcg.record_signal("c", 1.0);
        pcg.record_signal("d", 1.0);
        pcg.record_signal("e", 1.0);
    }
    EXPECT_GT(divs.load(), 0);
}

// ---------------------------------------------------------------------------
// dominant_theme is the largest contributor
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, DominantThemeCorrect) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size = 32;
    PositionConcentrationGuard pcg(cfg);
    for (int i = 0; i < 10; ++i) pcg.record_signal("big", 5.0);
    for (int i = 0; i < 10; ++i) pcg.record_signal("small", 1.0);
    EXPECT_EQ(pcg.dominant_theme(), "big");
    EXPECT_GT(pcg.dominant_share(), 0.5);
}

// ---------------------------------------------------------------------------
// theme_shares sum to 1
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, ThemeSharesSumToOne) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size = 16;
    PositionConcentrationGuard pcg(cfg);
    for (int i = 0; i < 5; ++i) pcg.record_signal("x", 1.0);
    for (int i = 0; i < 5; ++i) pcg.record_signal("y", 2.0);
    for (int i = 0; i < 5; ++i) pcg.record_signal("z", 3.0);
    auto shares = pcg.theme_shares();
    double s = 0.0;
    for (auto& [name, share] : shares) s += share;
    EXPECT_NEAR(s, 1.0, 0.001);
    // Check sorted descending.
    for (size_t i = 1; i < shares.size(); ++i)
        EXPECT_GE(shares[i-1].second, shares[i].second);
}

// ---------------------------------------------------------------------------
// Reset clears all state
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, ResetClearsState) {
    PositionConcentrationGuard pcg;
    for (int i = 0; i < 10; ++i) pcg.record_signal("t", 1.0);
    pcg.reset();
    EXPECT_EQ(pcg.total_signals(), 0u);
    EXPECT_EQ(pcg.distinct_themes(), 0);
    EXPECT_FALSE(pcg.is_concentrated());
    EXPECT_DOUBLE_EQ(pcg.hhi(), 0.0);
}

// ---------------------------------------------------------------------------
// update_config resets state
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, UpdateConfigResetsState) {
    PositionConcentrationGuard pcg;
    for (int i = 0; i < 10; ++i) pcg.record_signal("t", 1.0);
    PositionConcentrationGuard::Config cfg2;
    cfg2.window_size = 8;
    pcg.update_config(cfg2);
    EXPECT_EQ(pcg.total_signals(), 0u);
    EXPECT_EQ(pcg.distinct_themes(), 0);
}

// ---------------------------------------------------------------------------
// to_stats_json contains expected fields
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, ToStatsJsonContainsFields) {
    PositionConcentrationGuard pcg;
    pcg.record_signal("t", 1.0);
    std::string j = pcg.to_stats_json();
    EXPECT_NE(j.find("hhi"),                  std::string::npos);
    EXPECT_NE(j.find("is_concentrated"),       std::string::npos);
    EXPECT_NE(j.find("dominant_theme"),        std::string::npos);
    EXPECT_NE(j.find("concentration_events"),  std::string::npos);
    EXPECT_NE(j.find("total_signals"),         std::string::npos);
}

// ---------------------------------------------------------------------------
// Concurrent record_signal() is thread-safe
// ---------------------------------------------------------------------------

TEST(PositionConcentrationGuard, ConcurrentRecordSafe) {
    PositionConcentrationGuard::Config cfg;
    cfg.window_size = 32;
    PositionConcentrationGuard pcg(cfg);

    std::atomic<bool> go{false};
    auto worker = [&](int id) {
        std::string theme = (id % 3 == 0) ? "a" : (id % 3 == 1) ? "b" : "c";
        while (!go.load()) {}
        for (int i = 0; i < 250; ++i)
            pcg.record_signal(theme, static_cast<double>(i % 5) * 0.2);
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(worker, t);
    go.store(true);
    for (auto& t : threads) t.join();

    EXPECT_EQ(pcg.total_signals(), 1000u);
    double h = pcg.hhi();
    EXPECT_GE(h, 0.0);
    EXPECT_LE(h, 1.0 + 1e-9);
}

}  // namespace

#else

TEST(PositionConcentrationGuard, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_POSITION_CONCENTRATION_ENABLED
