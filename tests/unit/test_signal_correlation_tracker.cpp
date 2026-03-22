#include <gtest/gtest.h>

#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED

#include "SignalCorrelationTracker.h"

#include <cmath>
#include <string>

using namespace llmquant;

namespace {

// Feed N identical values to both sources so they are perfectly correlated.
static void feed_identical(SignalCorrelationTracker& t, int n) {
    for (int i = 0; i < n; ++i) {
        double v = static_cast<double>(i % 10) / 10.0;
        t.record("A", v);
        t.record("B", v);
    }
}

// ============================================================
// Construction
// ============================================================

TEST(SignalCorrelationTracker, ConstructsOk) {
    SignalCorrelationTracker t;
    EXPECT_EQ(t.total_samples(), 0u);
    EXPECT_EQ(t.divergence_events(), 0u);
    EXPECT_EQ(t.convergence_events(), 0u);
}

// ============================================================
// Record and sample_count
// ============================================================

TEST(SignalCorrelationTracker, RecordIncreasesSampleCount) {
    SignalCorrelationTracker t;
    t.record("src1", 0.5);
    t.record("src1", 0.7);
    EXPECT_EQ(t.sample_count("src1"), 2);
    EXPECT_EQ(t.total_samples(), 2u);
}

TEST(SignalCorrelationTracker, UnknownSourceReturnsZeroCount) {
    SignalCorrelationTracker t;
    EXPECT_EQ(t.sample_count("nope"), 0);
}

// ============================================================
// Window eviction
// ============================================================

TEST(SignalCorrelationTracker, WindowSizeCapEnforced) {
    SignalCorrelationTracker::Config cfg;
    cfg.window_size = 5;
    cfg.min_samples = 2;
    SignalCorrelationTracker t(cfg);
    for (int i = 0; i < 20; ++i) t.record("X", static_cast<double>(i));
    EXPECT_EQ(t.sample_count("X"), 5);
}

// ============================================================
// get_correlation — NaN before min_samples
// ============================================================

TEST(SignalCorrelationTracker, CorrelationNanBeforeMinSamples) {
    SignalCorrelationTracker::Config cfg;
    cfg.min_samples = 10;
    SignalCorrelationTracker t(cfg);
    t.record("A", 1.0);
    t.record("B", 1.0);
    double r = t.get_correlation("A", "B");
    EXPECT_TRUE(std::isnan(r));
}

TEST(SignalCorrelationTracker, CorrelationNanForUnknownSource) {
    SignalCorrelationTracker t;
    double r = t.get_correlation("A", "B");
    EXPECT_TRUE(std::isnan(r));
}

// ============================================================
// Perfect positive correlation
// ============================================================

TEST(SignalCorrelationTracker, PerfectPositiveCorrelation) {
    SignalCorrelationTracker::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 20;
    cfg.converge_threshold = 0.95;
    cfg.diverge_threshold = -0.5;
    SignalCorrelationTracker t(cfg);

    for (int i = 0; i < 30; ++i) {
        double v = static_cast<double>(i);
        t.record("A", v);
        t.record("B", v);
    }
    double r = t.get_correlation("A", "B");
    ASSERT_FALSE(std::isnan(r));
    EXPECT_NEAR(r, 1.0, 1e-9);
}

// ============================================================
// Perfect negative correlation
// ============================================================

TEST(SignalCorrelationTracker, PerfectNegativeCorrelation) {
    SignalCorrelationTracker::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 20;
    cfg.diverge_threshold = -0.95;
    cfg.converge_threshold = 0.9;
    SignalCorrelationTracker t(cfg);

    for (int i = 0; i < 30; ++i) {
        t.record("A",  static_cast<double>(i));
        t.record("B", -static_cast<double>(i));
    }
    double r = t.get_correlation("A", "B");
    ASSERT_FALSE(std::isnan(r));
    EXPECT_NEAR(r, -1.0, 1e-9);
}

// ============================================================
// Zero correlation (constant series)
// ============================================================

TEST(SignalCorrelationTracker, ZeroVarianceReturnsNan) {
    SignalCorrelationTracker::Config cfg;
    cfg.min_samples = 5;
    SignalCorrelationTracker t(cfg);
    for (int i = 0; i < 10; ++i) {
        t.record("A", 1.0);  // constant
        t.record("B", static_cast<double>(i));
    }
    double r = t.get_correlation("A", "B");
    EXPECT_TRUE(std::isnan(r));  // zero variance in A => NaN
}

// ============================================================
// Convergence callback fires
// ============================================================

TEST(SignalCorrelationTracker, ConvergenceCallbackFires) {
    SignalCorrelationTracker::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 10;
    cfg.converge_threshold = 0.9;
    cfg.diverge_threshold  = -0.9;
    SignalCorrelationTracker t(cfg);

    int cb_count = 0;
    t.set_convergence_callback([&](const std::string&, const std::string&, double) {
        ++cb_count;
    });

    // Feed perfectly correlated data — should trigger convergence.
    for (int i = 0; i < 15; ++i) {
        double v = static_cast<double>(i);
        t.record("A", v);
        t.record("B", v);
    }
    EXPECT_GT(cb_count, 0);
}

// ============================================================
// Divergence callback fires
// ============================================================

TEST(SignalCorrelationTracker, DivergenceCallbackFires) {
    SignalCorrelationTracker::Config cfg;
    cfg.window_size = 50;
    cfg.min_samples = 10;
    cfg.converge_threshold = 0.9;
    cfg.diverge_threshold  = -0.9;
    SignalCorrelationTracker t(cfg);

    int cb_count = 0;
    t.set_divergence_callback([&](const std::string&, const std::string&, double) {
        ++cb_count;
    });

    // Feed anti-correlated data — should trigger divergence.
    for (int i = 0; i < 15; ++i) {
        t.record("A",  static_cast<double>(i));
        t.record("B", -static_cast<double>(i));
    }
    EXPECT_GT(cb_count, 0);
}

// ============================================================
// source_names
// ============================================================

TEST(SignalCorrelationTracker, SourceNamesReturnsAllSources) {
    SignalCorrelationTracker t;
    t.record("alpha", 0.1);
    t.record("beta",  0.2);
    t.record("gamma", 0.3);
    auto names = t.source_names();
    EXPECT_EQ(names.size(), 3u);
}

// ============================================================
// update_config
// ============================================================

TEST(SignalCorrelationTracker, UpdateConfigDoesNotCrash) {
    SignalCorrelationTracker t;
    SignalCorrelationTracker::Config cfg2;
    cfg2.window_size = 200;
    cfg2.min_samples = 50;
    t.update_config(cfg2);
    feed_identical(t, 5);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(SignalCorrelationTracker, StatsJsonContainsRequiredKeys) {
    SignalCorrelationTracker t;
    std::string js = t.to_stats_json();
    EXPECT_NE(js.find("total_samples"), std::string::npos);
    EXPECT_NE(js.find("divergence_events"), std::string::npos);
    EXPECT_NE(js.find("convergence_events"), std::string::npos);
    EXPECT_NE(js.find("sources"), std::string::npos);
    EXPECT_NE(js.find("correlations"), std::string::npos);
}

TEST(SignalCorrelationTracker, StatsJsonContainsCorrelationValue) {
    SignalCorrelationTracker::Config cfg;
    cfg.min_samples = 5;
    SignalCorrelationTracker t(cfg);
    for (int i = 0; i < 10; ++i) {
        t.record("A", static_cast<double>(i));
        t.record("B", static_cast<double>(i));
    }
    std::string js = t.to_stats_json();
    EXPECT_NE(js.find("A<>B"), std::string::npos);
}

} // namespace

#else  // LLMQUANT_SIGNAL_CORRELATION_ENABLED not defined

TEST(SignalCorrelationTracker, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_SIGNAL_CORRELATION_ENABLED
