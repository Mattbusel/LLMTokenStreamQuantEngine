#include <gtest/gtest.h>
#include "BiasConcentrationRisk.h"

using namespace llmquant;

TEST(BiasConcentrationRisk, DefaultConstruction) {
    BiasConcentrationRisk b(BiasConcentrationRisk::Config{});
    EXPECT_EQ(b.total_records(), 0u);
    EXPECT_DOUBLE_EQ(b.hhi(), 0.0);
}

TEST(BiasConcentrationRisk, HhiInUnitRange) {
    BiasConcentrationRisk::Config cfg;
    cfg.min_samples = 3;
    BiasConcentrationRisk b(cfg);
    for (int i = 0; i < 10; ++i) b.record(0.05 * (i + 1));
    double h = b.hhi();
    EXPECT_GT(h, 0.0);
    EXPECT_LE(h, 1.0);
}

TEST(BiasConcentrationRisk, UniformBiasLowHhi) {
    BiasConcentrationRisk::Config cfg;
    cfg.window = 10;
    cfg.min_samples = 5;
    BiasConcentrationRisk b(cfg);
    for (int i = 0; i < 10; ++i) b.record(0.1);
    // All equal shares → HHI = 1/n
    EXPECT_NEAR(b.hhi(), 1.0 / 10.0, 1e-6);
}

TEST(BiasConcentrationRisk, ConcentratedCallback) {
    BiasConcentrationRisk::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.concentration_threshold = 0.2;
    int conc_count = 0;
    cfg.on_concentrated = [&](double) { ++conc_count; };
    BiasConcentrationRisk b(cfg);
    // One large value dominates → high HHI
    for (int i = 0; i < 5; ++i) b.record(i == 0 ? 1.0 : 0.001);
    EXPECT_GE(conc_count, 1);
}

TEST(BiasConcentrationRisk, DispersedCallback) {
    BiasConcentrationRisk::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    cfg.concentration_threshold = 0.9;
    int disp_count = 0;
    cfg.on_dispersed = [&](double) { ++disp_count; };
    BiasConcentrationRisk b(cfg);
    // Start concentrated, then uniform
    for (int i = 0; i < 3; ++i) b.record(i == 0 ? 1.0 : 0.001);
    for (int i = 0; i < 5; ++i) b.record(0.1);
    EXPECT_GE(disp_count, 1);
}

TEST(BiasConcentrationRisk, Reset) {
    BiasConcentrationRisk b(BiasConcentrationRisk::Config{});
    for (int i = 0; i < 10; ++i) b.record(0.05);
    b.reset();
    EXPECT_EQ(b.total_records(), 0u);
    EXPECT_DOUBLE_EQ(b.hhi(), 0.0);
    EXPECT_EQ(b.concentration_events(), 0u);
    EXPECT_EQ(b.dispersed_events(), 0u);
}

TEST(BiasConcentrationRisk, ToStatsJson) {
    BiasConcentrationRisk b(BiasConcentrationRisk::Config{});
    for (int i = 0; i < 5; ++i) b.record(0.1);
    std::string j = b.to_stats_json();
    EXPECT_NE(j.find("hhi"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}

TEST(BiasConcentrationRisk, NegativeBiasUsesAbsValue) {
    BiasConcentrationRisk::Config cfg;
    cfg.window = 4;
    cfg.min_samples = 2;
    BiasConcentrationRisk b(cfg);
    b.record(-0.5);
    b.record(0.5);
    b.record(-0.5);
    b.record(0.5);
    // |bias| all equal → HHI = 1/4
    EXPECT_NEAR(b.hhi(), 0.25, 1e-6);
}
