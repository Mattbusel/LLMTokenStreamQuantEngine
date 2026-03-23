#include "gtest/gtest.h"
#include "TokenRouter.h"

using namespace llmquant;

// ---------------------------------------------------------------------------
// TokenRouter unit tests
// ---------------------------------------------------------------------------

TEST(TokenRouterTest, test_empty_router_routes_to_default) {
    TokenRouter router;
    EXPECT_EQ(router.route("hello"), "default");
}

TEST(TokenRouterTest, test_add_rule_increments_rule_count) {
    TokenRouter router;
    EXPECT_EQ(router.rule_count(), 0u);
    router.add_rule({"bull", 1, "bullish_proc"});
    EXPECT_EQ(router.rule_count(), 1u);
}

TEST(TokenRouterTest, test_exact_prefix_match_routes_correctly) {
    TokenRouter router;
    router.add_rule({"bull", 1, "bull_proc"});
    EXPECT_EQ(router.route("bullish"), "bull_proc");
    EXPECT_EQ(router.route("bull"),    "bull_proc");
}

TEST(TokenRouterTest, test_non_matching_token_goes_to_default) {
    TokenRouter router;
    router.add_rule({"bull", 1, "bull_proc"});
    EXPECT_EQ(router.route("bearish"), "default");
}

TEST(TokenRouterTest, test_priority_lower_wins) {
    TokenRouter router;
    // Priority 2 added first.
    router.add_rule({"bull", 2, "low_priority_proc"});
    // Priority 1 added second but should win.
    router.add_rule({"bull", 1, "high_priority_proc"});
    EXPECT_EQ(router.route("bullish"), "high_priority_proc");
}

TEST(TokenRouterTest, test_multiple_rules_first_prefix_match_wins) {
    TokenRouter router;
    router.add_rule({"b",    1, "b_proc"});
    router.add_rule({"bull", 2, "bull_proc"});
    // Priority 1 has shorter prefix "b" but higher priority — should match.
    EXPECT_EQ(router.route("bullish"), "b_proc");
}

TEST(TokenRouterTest, test_exact_token_equal_to_pattern_matches) {
    TokenRouter router;
    router.add_rule({"hello", 1, "greeting_proc"});
    EXPECT_EQ(router.route("hello"), "greeting_proc");
}

TEST(TokenRouterTest, test_token_shorter_than_pattern_does_not_match) {
    TokenRouter router;
    router.add_rule({"bullish", 1, "bull_proc"});
    // "bull" is shorter than "bullish" prefix → no match.
    EXPECT_EQ(router.route("bull"), "default");
}

TEST(TokenRouterTest, test_route_batch_returns_correct_size) {
    TokenRouter router;
    router.add_rule({"buy", 1, "buy_proc"});
    auto results = router.route_batch({"buy_order", "sell_order", "buy_limit"});
    EXPECT_EQ(results.size(), 3u);
}

TEST(TokenRouterTest, test_route_batch_maps_tokens_correctly) {
    TokenRouter router;
    router.add_rule({"buy", 1, "buy_proc"});
    auto results = router.route_batch({"buy_order", "sell_order"});
    EXPECT_EQ(results[0].first,  "buy_order");
    EXPECT_EQ(results[0].second, "buy_proc");
    EXPECT_EQ(results[1].first,  "sell_order");
    EXPECT_EQ(results[1].second, "default");
}

TEST(TokenRouterTest, test_route_batch_empty_input) {
    TokenRouter router;
    auto results = router.route_batch({});
    EXPECT_TRUE(results.empty());
}

TEST(TokenRouterTest, test_stats_contains_total_routed) {
    TokenRouter router;
    router.add_rule({"a", 1, "proc_a"});
    router.route("alpha");
    router.route("beta");
    auto s = router.stats();
    EXPECT_NE(s.find("total_routed"), std::string::npos);
}

TEST(TokenRouterTest, test_stats_mentions_rule_count) {
    TokenRouter router;
    router.add_rule({"x", 1, "x_proc"});
    auto s = router.stats();
    EXPECT_NE(s.find("rules"), std::string::npos);
}

TEST(TokenRouterTest, test_multiple_routes_accumulate_stats) {
    TokenRouter router;
    router.add_rule({"pos", 1, "positive"});
    for (int i = 0; i < 5; ++i) router.route("positive_signal");
    for (int i = 0; i < 3; ++i) router.route("negative_signal");
    auto s = router.stats();
    // "positive" should appear in stats.
    EXPECT_NE(s.find("positive"), std::string::npos);
}

TEST(TokenRouterTest, test_empty_pattern_matches_everything) {
    TokenRouter router;
    router.add_rule({"", 1, "catch_all"});
    EXPECT_EQ(router.route("anything"), "catch_all");
    EXPECT_EQ(router.route(""),         "catch_all");
}
