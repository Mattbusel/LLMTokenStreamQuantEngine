#include <gtest/gtest.h>
#include "TokenBiasHistogram.h"

using namespace llmquant;

TEST(TokenBiasHistogram, DefaultConstruction) {
    TokenBiasHistogram h(TokenBiasHistogram::Config{});
    EXPECT_EQ(h.total_records(), 0u);
}

TEST(TokenBiasHistogram, BinCountAccumulates) {
    TokenBiasHistogram::Config cfg;
    cfg.n_bins = 4;
    cfg.lo = -1.0;
    cfg.hi =  1.0;
    cfg.min_samples = 2;
    TokenBiasHistogram h(cfg);
    h.record(0.0);
    h.record(0.1);
    uint64_t total = 0;
    for (int i = 0; i < 4; ++i) total += h.bin_count(i);
    EXPECT_EQ(total, 2u);
}

TEST(TokenBiasHistogram, PeakBinInRange) {
    TokenBiasHistogram::Config cfg;
    cfg.n_bins = 5;
    cfg.lo = -1.0;
    cfg.hi =  1.0;
    cfg.min_samples = 3;
    TokenBiasHistogram h(cfg);
    for (int i = 0; i < 5; ++i) h.record(0.05);
    int pb = h.peak_bin();
    EXPECT_GE(pb, 0);
    EXPECT_LT(pb, 5);
}

TEST(TokenBiasHistogram, EntropyPositive) {
    TokenBiasHistogram::Config cfg;
    cfg.n_bins = 4;
    cfg.lo = -1.0;
    cfg.hi =  1.0;
    TokenBiasHistogram h(cfg);
    for (int i = 0; i < 8; ++i) h.record(-0.5 + 0.25 * i);
    EXPECT_GT(h.entropy(), 0.0);
}

TEST(TokenBiasHistogram, ModeChangeCallback) {
    TokenBiasHistogram::Config cfg;
    cfg.n_bins = 4;
    cfg.lo = -1.0;
    cfg.hi =  1.0;
    cfg.min_samples = 2;
    int mode_changes = 0;
    cfg.on_mode_change = [&](int, int) { ++mode_changes; };
    TokenBiasHistogram h(cfg);
    // Alternate between extreme bins to force mode changes
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 3; ++j) h.record(i % 2 == 0 ? -0.9 : 0.9);
    }
    EXPECT_GE(mode_changes, 1);
}

TEST(TokenBiasHistogram, Reset) {
    TokenBiasHistogram h(TokenBiasHistogram::Config{});
    for (int i = 0; i < 5; ++i) h.record(0.1);
    h.reset();
    EXPECT_EQ(h.total_records(), 0u);
    EXPECT_EQ(h.mode_changes(), 0u);
    for (int b = 0; b < 10; ++b) EXPECT_EQ(h.bin_count(b), 0u);
}

TEST(TokenBiasHistogram, ToStatsJson) {
    TokenBiasHistogram h(TokenBiasHistogram::Config{});
    for (int i = 0; i < 5; ++i) h.record(0.05);
    std::string j = h.to_stats_json();
    EXPECT_NE(j.find("peak_bin"), std::string::npos);
    EXPECT_NE(j.find("entropy"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
