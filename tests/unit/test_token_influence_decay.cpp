#include <gtest/gtest.h>
#include "TokenInfluenceDecay.h"

using namespace llmquant;

TEST(TokenInfluenceDecay, DefaultConstruction) {
    TokenInfluenceDecay t(TokenInfluenceDecay::Config{});
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.pos_influence(), 0.0);
    EXPECT_DOUBLE_EQ(t.neg_influence(), 0.0);
    EXPECT_DOUBLE_EQ(t.net_influence(), 0.0);
}

TEST(TokenInfluenceDecay, PositiveBiasBuildsPos) {
    TokenInfluenceDecay::Config cfg;
    cfg.min_samples = 2;
    TokenInfluenceDecay t(cfg);
    for (int i = 0; i < 10; ++i) t.record(0.1);
    EXPECT_GT(t.pos_influence(), 0.0);
    EXPECT_GT(t.net_influence(), 0.0);
}

TEST(TokenInfluenceDecay, NegativeBiasBuildsNeg) {
    TokenInfluenceDecay::Config cfg;
    cfg.min_samples = 2;
    TokenInfluenceDecay t(cfg);
    for (int i = 0; i < 10; ++i) t.record(-0.1);
    EXPECT_GT(t.neg_influence(), 0.0);
    EXPECT_LT(t.net_influence(), 0.0);
}

TEST(TokenInfluenceDecay, PositiveDominanceCallback) {
    TokenInfluenceDecay::Config cfg;
    cfg.dominance_threshold = 0.01;
    cfg.min_samples = 2;
    int pos_dom = 0;
    cfg.on_positive_dominant = [&](double) { ++pos_dom; };
    TokenInfluenceDecay t(cfg);
    for (int i = 0; i < 15; ++i) t.record(0.1);
    EXPECT_GE(pos_dom, 1);
}

TEST(TokenInfluenceDecay, NegativeDominanceCallback) {
    TokenInfluenceDecay::Config cfg;
    cfg.dominance_threshold = 0.01;
    cfg.min_samples = 2;
    int neg_dom = 0;
    cfg.on_negative_dominant = [&](double) { ++neg_dom; };
    TokenInfluenceDecay t(cfg);
    for (int i = 0; i < 15; ++i) t.record(-0.1);
    EXPECT_GE(neg_dom, 1);
}

TEST(TokenInfluenceDecay, DominanceFlagsCorrect) {
    TokenInfluenceDecay::Config cfg;
    cfg.dominance_threshold = 0.01;
    cfg.min_samples = 2;
    TokenInfluenceDecay t(cfg);
    for (int i = 0; i < 15; ++i) t.record(0.1);
    EXPECT_TRUE(t.is_positive_dominant());
    EXPECT_FALSE(t.is_negative_dominant());
}

TEST(TokenInfluenceDecay, Reset) {
    TokenInfluenceDecay t(TokenInfluenceDecay::Config{});
    for (int i = 0; i < 10; ++i) t.record(0.1);
    t.reset();
    EXPECT_EQ(t.total_records(), 0u);
    EXPECT_DOUBLE_EQ(t.pos_influence(), 0.0);
    EXPECT_DOUBLE_EQ(t.neg_influence(), 0.0);
    EXPECT_DOUBLE_EQ(t.net_influence(), 0.0);
    EXPECT_EQ(t.positive_dominant_events(), 0u);
    EXPECT_EQ(t.negative_dominant_events(), 0u);
}

TEST(TokenInfluenceDecay, ToStatsJson) {
    TokenInfluenceDecay t(TokenInfluenceDecay::Config{});
    for (int i = 0; i < 5; ++i) t.record(0.05);
    std::string j = t.to_stats_json();
    EXPECT_NE(j.find("pos_influence"), std::string::npos);
    EXPECT_NE(j.find("neg_influence"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(TokenInfluenceDecay, UpdateConfig) {
    TokenInfluenceDecay::Config cfg;
    TokenInfluenceDecay t(cfg);
    cfg.alpha_pos = 0.5;
    cfg.alpha_neg = 0.5;
    t.update_config(cfg);
    t.record(0.1);
    EXPECT_EQ(t.total_records(), 1u);
}
