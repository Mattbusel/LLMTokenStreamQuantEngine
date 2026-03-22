#include <gtest/gtest.h>

#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
#include "ExecutionQualityMonitor.h"
using namespace llmquant;

TEST(ExecutionQualityMonitorTest, EmptyStats) {
    ExecutionQualityMonitor mon;
    auto s = mon.compute_stats();
    EXPECT_EQ(s.sample_count, 0u);
    EXPECT_DOUBLE_EQ(s.mean_latency_us, 0.0);
}

TEST(ExecutionQualityMonitorTest, RecordAndStats) {
    ExecutionQualityMonitor mon;
    mon.record(1, 1000.0, 2.0);
    mon.record(2, 2000.0, 4.0);
    auto s = mon.compute_stats();
    EXPECT_EQ(s.sample_count, 2u);
    EXPECT_NEAR(s.mean_latency_us, 1500.0, 1.0);
    EXPECT_NEAR(s.mean_slippage_bps, 3.0, 0.1);
}

TEST(ExecutionQualityMonitorTest, SlaBreachCallback) {
    ExecutionQualityMonitor::Config cfg;
    cfg.latency_sla_us   = 100.0;
    cfg.slippage_sla_bps = 1.0;
    int breaches = 0;
    cfg.on_sla_breach = [&](const ExecutionQualityMonitor::FillRecord&, bool, bool) {
        ++breaches;
    };
    ExecutionQualityMonitor mon(cfg);
    mon.record(1, 200.0, 0.5);  // latency breach
    mon.record(2, 50.0,  5.0);  // slippage breach
    EXPECT_EQ(breaches, 2);
}

TEST(ExecutionQualityMonitorTest, TotalFillsCounter) {
    ExecutionQualityMonitor mon;
    mon.record(1, 100.0, 0.0);
    mon.record(2, 200.0, 0.0);
    mon.record(3, 300.0, 0.0);
    EXPECT_EQ(mon.total_fills(), 3u);
}

TEST(ExecutionQualityMonitorTest, SlaBreachCount) {
    ExecutionQualityMonitor::Config cfg;
    cfg.latency_sla_us = 1.0;
    ExecutionQualityMonitor mon(cfg);
    mon.record(1, 1000.0, 0.0);
    mon.record(2, 1000.0, 0.0);
    EXPECT_EQ(mon.sla_breach_count(), 2u);
}

TEST(ExecutionQualityMonitorTest, Reset) {
    ExecutionQualityMonitor mon;
    mon.record(1, 500.0, 1.0);
    mon.reset();
    EXPECT_EQ(mon.total_fills(), 0u);
    EXPECT_EQ(mon.sla_breach_count(), 0u);
    auto s = mon.compute_stats();
    EXPECT_EQ(s.sample_count, 0u);
}

TEST(ExecutionQualityMonitorTest, PercentileOrdering) {
    ExecutionQualityMonitor::Config cfg;
    cfg.window_size = 100;
    ExecutionQualityMonitor mon(cfg);
    for (int i = 1; i <= 100; ++i)
        mon.record(i, static_cast<double>(i) * 10.0, 0.0);
    auto s = mon.compute_stats();
    EXPECT_LE(s.p50_latency_us, s.p95_latency_us);
    EXPECT_LE(s.p95_latency_us, s.p99_latency_us);
}

TEST(ExecutionQualityMonitorTest, StatsJson) {
    ExecutionQualityMonitor mon;
    mon.record(1, 300.0, 2.0);
    std::string json = mon.to_stats_json();
    EXPECT_NE(json.find("mean_lat_us"), std::string::npos);
    EXPECT_NE(json.find("sla_breaches"), std::string::npos);
}

#else
TEST(ExecutionQualityMonitorTest, Disabled) { SUCCEED(); }
#endif
