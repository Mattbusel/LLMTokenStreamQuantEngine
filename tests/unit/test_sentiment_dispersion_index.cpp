#include <gtest/gtest.h>

#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
#include "SentimentDispersionIndex.h"
using namespace llmquant;

TEST(SentimentDispersionIndexTest, InitialZero) {
    SentimentDispersionIndex sdi;
    EXPECT_DOUBLE_EQ(sdi.sdi(), 0.0);
    EXPECT_EQ(sdi.total_records(), 0u);
}

TEST(SentimentDispersionIndexTest, CoherentSignalsLowSdi) {
    SentimentDispersionIndex sdi;
    // All three signals equal → CV = 0.
    for (int i = 0; i < 20; ++i)
        sdi.record(0.5, 0.5, 0.5);
    EXPECT_LT(sdi.sdi(), 0.1);
}

TEST(SentimentDispersionIndexTest, IncoherentSignalsHighSdi) {
    SentimentDispersionIndex sdi;
    // Extreme disagreement: bias=1, vol=0, conf=0.5.
    for (int i = 0; i < 50; ++i)
        sdi.record(1.0, 0.0, 0.5);
    EXPECT_GT(sdi.sdi(), 0.0);
}

TEST(SentimentDispersionIndexTest, IsCoherentFlag) {
    SentimentDispersionIndex::Config cfg;
    cfg.alpha         = 1.0; // instantaneous update
    cfg.low_threshold = 0.5;
    SentimentDispersionIndex sdi(cfg);
    sdi.record(0.5, 0.5, 0.5); // CV = 0 < 0.5
    EXPECT_TRUE(sdi.is_coherent());
}

TEST(SentimentDispersionIndexTest, IsDispersedFlag) {
    SentimentDispersionIndex::Config cfg;
    cfg.alpha          = 1.0;
    cfg.high_threshold = 0.3;
    SentimentDispersionIndex sdi(cfg);
    sdi.record(1.0, 0.0, 0.5); // high CV
    EXPECT_TRUE(sdi.is_dispersed());
}

TEST(SentimentDispersionIndexTest, HighDispersionCallback) {
    SentimentDispersionIndex::Config cfg;
    cfg.high_threshold = 0.3;
    cfg.alpha          = 1.0;
    int fired = 0;
    cfg.on_high_dispersion = [&](double) { ++fired; };
    SentimentDispersionIndex sdi(cfg);
    // First record: seeded, prev_high_ = false → should fire if CV > 0.3.
    sdi.record(1.0, 0.0, 0.0);
    sdi.record(0.5, 0.5, 0.5); // exit high
    sdi.record(1.0, 0.0, 0.0); // re-enter high
    EXPECT_GE(fired, 1);
}

TEST(SentimentDispersionIndexTest, LowDispersionCallback) {
    SentimentDispersionIndex::Config cfg;
    cfg.low_threshold = 0.5;
    cfg.alpha         = 1.0;
    int fired = 0;
    cfg.on_low_dispersion = [&](double) { ++fired; };
    SentimentDispersionIndex sdi(cfg);
    sdi.record(0.5, 0.5, 0.5); // CV ~= 0 < 0.5 → coherent (initial was also low, no transition)
    sdi.record(1.0, 0.0, 0.0); // exit coherent
    sdi.record(0.5, 0.5, 0.5); // re-enter coherent → fire
    EXPECT_GE(fired, 1);
}

TEST(SentimentDispersionIndexTest, TotalRecords) {
    SentimentDispersionIndex sdi;
    for (int i = 0; i < 7; ++i)
        sdi.record(0.3, 0.3, 0.3);
    EXPECT_EQ(sdi.total_records(), 7u);
}

TEST(SentimentDispersionIndexTest, Reset) {
    SentimentDispersionIndex sdi;
    sdi.record(0.5, 0.5, 0.5);
    sdi.reset();
    EXPECT_DOUBLE_EQ(sdi.sdi(), 0.0);
    EXPECT_EQ(sdi.total_records(), 0u);
}

TEST(SentimentDispersionIndexTest, StatsJson) {
    SentimentDispersionIndex sdi;
    sdi.record(0.5, 0.3, 0.8);
    std::string json = sdi.to_stats_json();
    EXPECT_NE(json.find("sdi"), std::string::npos);
    EXPECT_NE(json.find("is_dispersed"), std::string::npos);
    EXPECT_NE(json.find("total_records"), std::string::npos);
}

#else
TEST(SentimentDispersionIndexTest, Disabled) { SUCCEED(); }
#endif
