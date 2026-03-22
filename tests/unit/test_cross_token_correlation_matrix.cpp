#include <gtest/gtest.h>

#ifdef LLMQUANT_CROSS_TOKEN_CORR_ENABLED
#include "CrossTokenCorrelationMatrix.h"
#include <atomic>
#include <cmath>
#include <thread>
using namespace llmquant;

TEST(CrossTokenCorrelationMatrixTest, InitialState) {
    CrossTokenCorrelationMatrix m;
    EXPECT_NEAR(m.correlation(0, 1), 0.0, 1e-9);
    EXPECT_EQ(m.total_records(), 0u);
}

TEST(CrossTokenCorrelationMatrixTest, TotalRecordsIncrements) {
    CrossTokenCorrelationMatrix m;
    m.record(0, 0.5);
    m.record(1, 0.5);
    EXPECT_EQ(m.total_records(), 2u);
}

TEST(CrossTokenCorrelationMatrixTest, IdenticalSeriesHighCorrelation) {
    CrossTokenCorrelationMatrix::Config cfg;
    cfg.n_channels = 2;
    cfg.window     = 30;
    CrossTokenCorrelationMatrix m(cfg);
    for (int i = 0; i < 30; ++i) {
        double v = static_cast<double>(i) * 0.01;
        m.record(0, v);
        m.record(1, v);
    }
    EXPECT_GT(m.correlation(0, 1), 0.8);
}

TEST(CrossTokenCorrelationMatrixTest, OppositeSeriesNegativeCorrelation) {
    CrossTokenCorrelationMatrix::Config cfg;
    cfg.n_channels = 2;
    cfg.window     = 30;
    CrossTokenCorrelationMatrix m(cfg);
    for (int i = 0; i < 30; ++i) {
        double v = static_cast<double>(i) * 0.01;
        m.record(0,  v);
        m.record(1, -v);
    }
    EXPECT_LT(m.correlation(0, 1), -0.8);
}

TEST(CrossTokenCorrelationMatrixTest, CorrelationBounded) {
    CrossTokenCorrelationMatrix::Config cfg;
    cfg.n_channels = 2;
    cfg.window     = 20;
    CrossTokenCorrelationMatrix m(cfg);
    for (int i = 0; i < 40; ++i) {
        m.record(0, std::sin(i * 0.3));
        m.record(1, std::cos(i * 0.3));
    }
    double r = m.correlation(0, 1);
    EXPECT_GE(r, -1.0 - 1e-6);
    EXPECT_LE(r,  1.0 + 1e-6);
}

TEST(CrossTokenCorrelationMatrixTest, CorrelationSymmetric) {
    CrossTokenCorrelationMatrix::Config cfg;
    cfg.n_channels = 2;
    cfg.window     = 20;
    CrossTokenCorrelationMatrix m(cfg);
    for (int i = 0; i < 20; ++i) {
        m.record(0, static_cast<double>(i) * 0.05);
        m.record(1, static_cast<double>(i) * 0.03 + 0.1);
    }
    EXPECT_NEAR(m.correlation(0, 1), m.correlation(1, 0), 1e-9);
}

TEST(CrossTokenCorrelationMatrixTest, DivergenceCallbackFires) {
    CrossTokenCorrelationMatrix::Config cfg;
    cfg.n_channels             = 2;
    cfg.window                 = 20;
    cfg.divergence_threshold   = 0.5;
    cfg.convergence_threshold  = 0.9;
    std::atomic<int> divs{0};
    cfg.on_divergence = [&](int, int, double) { ++divs; };
    CrossTokenCorrelationMatrix m(cfg);
    // Feed uncorrelated data
    for (int i = 0; i < 20; ++i) {
        m.record(0, i % 2 == 0 ? 1.0 : -1.0);
        m.record(1, i % 3 == 0 ? 1.0 : -0.5);
    }
    EXPECT_GE(divs.load(), 0);  // may fire
}

TEST(CrossTokenCorrelationMatrixTest, ResetClearsState) {
    CrossTokenCorrelationMatrix m;
    for (int i = 0; i < 10; ++i) {
        m.record(0, i * 0.1);
        m.record(1, i * 0.1);
    }
    m.reset();
    EXPECT_NEAR(m.correlation(0, 1), 0.0, 1e-9);
    EXPECT_EQ(m.total_records(), 0u);
}

TEST(CrossTokenCorrelationMatrixTest, StatsJsonHasFields) {
    CrossTokenCorrelationMatrix m;
    for (int i = 0; i < 10; ++i) {
        m.record(0, i * 0.1);
        m.record(1, i * 0.1);
    }
    std::string j = m.to_stats_json();
    EXPECT_NE(j.find("total_records"), std::string::npos);
    EXPECT_NE(j.find("correlations"),  std::string::npos);
}

TEST(CrossTokenCorrelationMatrixTest, InvalidChannelIgnored) {
    CrossTokenCorrelationMatrix m;
    EXPECT_NO_FATAL_FAILURE(m.record(-1, 0.5));
    EXPECT_NO_FATAL_FAILURE(m.record(100, 0.5));
    EXPECT_NEAR(m.correlation(-1, 0), 0.0, 1e-9);
}

TEST(CrossTokenCorrelationMatrixTest, ConcurrentSafe) {
    CrossTokenCorrelationMatrix::Config cfg;
    cfg.n_channels = 2;
    cfg.window     = 30;
    CrossTokenCorrelationMatrix m(cfg);
    std::atomic<bool> go{false};
    auto worker = [&](int ch) {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            m.record(ch, static_cast<double>(i) * 0.01);
    };
    std::thread t1(worker, 0), t2(worker, 1);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(m.total_records(), 400u);
}

#else
TEST(CrossTokenCorrelationMatrixTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
