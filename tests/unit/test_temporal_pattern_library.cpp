#include <gtest/gtest.h>

#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED

#include "TemporalPatternLibrary.h"

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(TemporalPatternLibrary, DefaultConstructsEmpty) {
    TemporalPatternLibrary lib;
    EXPECT_EQ(lib.pattern_count(), 0);
    EXPECT_EQ(lib.total_tokens(), 0u);
    EXPECT_EQ(lib.total_matches(), 0u);
    EXPECT_EQ(lib.pending_matches(), 0);
}

// ============================================================
// register_pattern
// ============================================================

TEST(TemporalPatternLibrary, RegisterIncrementsCount) {
    TemporalPatternLibrary lib;
    lib.register_pattern("beat", {"earnings", "beat"}, 0.5);
    lib.register_pattern("miss", {"earnings", "miss"}, -0.5);
    EXPECT_EQ(lib.pattern_count(), 2);
}

// ============================================================
// Single-token pattern matches immediately
// ============================================================

TEST(TemporalPatternLibrary, SingleTokenPatternMatches) {
    TemporalPatternLibrary lib;
    lib.register_pattern("buy", {"buy"}, 1.0);

    bool matched = lib.push_token("buy");
    EXPECT_TRUE(matched);
    EXPECT_EQ(lib.pending_matches(), 1);
    EXPECT_EQ(lib.total_matches(), 1u);
}

// ============================================================
// Multi-token pattern requires full sequence
// ============================================================

TEST(TemporalPatternLibrary, MultiTokenPatternRequiresFullSequence) {
    TemporalPatternLibrary lib;
    lib.register_pattern("strong_buy", {"strong", "buy"}, 2.0);

    bool m1 = lib.push_token("strong");
    bool m2 = lib.push_token("buy");

    EXPECT_FALSE(m1);
    EXPECT_TRUE(m2);
    EXPECT_EQ(lib.pending_matches(), 1);
}

// ============================================================
// Unmatched tokens produce no match
// ============================================================

TEST(TemporalPatternLibrary, UnmatchedTokenNoMatch) {
    TemporalPatternLibrary lib;
    lib.register_pattern("beat", {"earnings", "beat"}, 0.5);

    lib.push_token("hello");
    lib.push_token("world");
    EXPECT_EQ(lib.pending_matches(), 0);
}

// ============================================================
// Partial match abandoned on mismatch
// ============================================================

TEST(TemporalPatternLibrary, PartialMatchAbandonedOnMismatch) {
    TemporalPatternLibrary lib;
    lib.register_pattern("beat", {"earnings", "beat"}, 0.5);

    lib.push_token("earnings");
    lib.push_token("miss");  // not "beat" — abandon partial
    EXPECT_EQ(lib.pending_matches(), 0);
}

// ============================================================
// pop_match returns correct boost and name
// ============================================================

TEST(TemporalPatternLibrary, PopMatchReturnsCorrectFields) {
    TemporalPatternLibrary lib;
    lib.register_pattern("beat", {"earnings", "beat"}, 0.8);

    lib.push_token("earnings");
    lib.push_token("beat");

    TemporalPatternLibrary::Match m{};
    EXPECT_TRUE(lib.pop_match(m));
    EXPECT_EQ(m.name, "beat");
    EXPECT_NEAR(m.boost, 0.8, 1e-9);
    EXPECT_FALSE(lib.pop_match(m));  // queue empty now
}

// ============================================================
// on_match callback fires
// ============================================================

TEST(TemporalPatternLibrary, OnMatchCallbackFires) {
    TemporalPatternLibrary::Config cfg;
    int fires = 0;
    double last_boost = 0.0;
    cfg.on_match = [&](const std::string&, double b) {
        ++fires;
        last_boost = b;
    };
    TemporalPatternLibrary lib(cfg);
    lib.register_pattern("sell", {"strong", "sell"}, -1.5);

    lib.push_token("strong");
    lib.push_token("sell");

    EXPECT_EQ(fires, 1);
    EXPECT_NEAR(last_boost, -1.5, 1e-9);
}

// ============================================================
// Overlapping patterns both fire
// ============================================================

TEST(TemporalPatternLibrary, OverlappingPatternsBothFire) {
    TemporalPatternLibrary lib;
    lib.register_pattern("buy",       {"buy"},        0.5);
    lib.register_pattern("strong_buy", {"strong", "buy"}, 1.5);

    lib.push_token("strong");
    lib.push_token("buy");  // "buy" alone should also fire

    // "buy" single-token pattern fires, AND "strong buy" two-token fires.
    EXPECT_EQ(lib.pending_matches(), 2);
}

// ============================================================
// reset clears pending and active states
// ============================================================

TEST(TemporalPatternLibrary, ResetClearsPendingAndActive) {
    TemporalPatternLibrary lib;
    lib.register_pattern("beat", {"earnings", "beat"}, 0.5);

    lib.push_token("earnings");  // partial match in progress
    lib.reset();

    EXPECT_EQ(lib.pending_matches(), 0);
    // Patterns should still be registered after reset().
    EXPECT_EQ(lib.pattern_count(), 1);
}

// ============================================================
// clear removes all patterns
// ============================================================

TEST(TemporalPatternLibrary, ClearRemovesAllPatterns) {
    TemporalPatternLibrary lib;
    lib.register_pattern("beat", {"earnings", "beat"}, 0.5);
    lib.clear();
    EXPECT_EQ(lib.pattern_count(), 0);
}

// ============================================================
// remove_pattern removes one pattern
// ============================================================

TEST(TemporalPatternLibrary, RemovePatternReducesCount) {
    TemporalPatternLibrary lib;
    lib.register_pattern("a", {"a"}, 1.0);
    lib.register_pattern("b", {"b"}, 2.0);
    lib.remove_pattern("a");
    EXPECT_EQ(lib.pattern_count(), 1);

    // "a" should no longer match.
    lib.push_token("a");
    EXPECT_EQ(lib.pending_matches(), 0);
}

// ============================================================
// to_stats_json contains required keys
// ============================================================

TEST(TemporalPatternLibrary, ToStatsJsonContainsKeys) {
    TemporalPatternLibrary lib;
    lib.register_pattern("x", {"x"}, 1.0);
    lib.push_token("x");
    std::string json = lib.to_stats_json();
    EXPECT_NE(json.find("pattern_count"),  std::string::npos);
    EXPECT_NE(json.find("pending_matches"), std::string::npos);
    EXPECT_NE(json.find("total_tokens"),   std::string::npos);
    EXPECT_NE(json.find("total_matches"),  std::string::npos);
}

} // namespace

#else  // LLMQUANT_TEMPORAL_PATTERN_ENABLED not defined

TEST(TemporalPatternLibrary, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_TEMPORAL_PATTERN_ENABLED
