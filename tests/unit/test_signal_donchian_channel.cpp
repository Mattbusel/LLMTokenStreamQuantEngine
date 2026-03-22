#include <gtest/gtest.h>
#include "SignalDonchianChannel.h"

using namespace llmquant;

TEST(SignalDonchianChannel, DefaultConstruction) {
    SignalDonchianChannel d(SignalDonchianChannel::Config{});
    EXPECT_EQ(d.total_records(), 0u);
}

TEST(SignalDonchianChannel, UpperGeqLower) {
    SignalDonchianChannel::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    SignalDonchianChannel d(cfg);
    for (int i = 0; i < 8; ++i) d.record(0.01 * (i % 5 - 2));
    EXPECT_GE(d.upper_band(), d.lower_band());
}

TEST(SignalDonchianChannel, MidlineIsMean) {
    SignalDonchianChannel::Config cfg;
    cfg.window = 4;
    cfg.min_samples = 2;
    SignalDonchianChannel d(cfg);
    for (int i = 0; i < 4; ++i) d.record(i == 0 ? -1.0 : 1.0);
    EXPECT_NEAR(d.midline(), (d.upper_band() + d.lower_band()) / 2.0, 1e-6);
}

TEST(SignalDonchianChannel, UpperBreakoutCallback) {
    SignalDonchianChannel::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    int ub_count = 0;
    cfg.on_upper_breakout = [&](double, double) { ++ub_count; };
    SignalDonchianChannel d(cfg);
    for (int i = 0; i < 8; ++i) d.record(0.1 * (i + 1));
    EXPECT_GE(ub_count, 1);
}

TEST(SignalDonchianChannel, LowerBreakoutCallback) {
    SignalDonchianChannel::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    int lb_count = 0;
    cfg.on_lower_breakout = [&](double, double) { ++lb_count; };
    SignalDonchianChannel d(cfg);
    for (int i = 7; i >= 0; --i) d.record(-0.1 * (i + 1));
    EXPECT_GE(lb_count, 1);
}

TEST(SignalDonchianChannel, AtHighWhenMaxValue) {
    SignalDonchianChannel::Config cfg;
    cfg.window = 5;
    cfg.min_samples = 3;
    SignalDonchianChannel d(cfg);
    for (int i = 0; i < 4; ++i) d.record(0.01 * i);
    d.record(1.0); // clear high
    EXPECT_TRUE(d.is_at_high());
}

TEST(SignalDonchianChannel, Reset) {
    SignalDonchianChannel d(SignalDonchianChannel::Config{});
    for (int i = 0; i < 10; ++i) d.record(0.05);
    d.reset();
    EXPECT_EQ(d.total_records(), 0u);
    EXPECT_EQ(d.upper_breakouts(), 0u);
    EXPECT_EQ(d.lower_breakouts(), 0u);
}

TEST(SignalDonchianChannel, ToStatsJson) {
    SignalDonchianChannel d(SignalDonchianChannel::Config{});
    for (int i = 0; i < 5; ++i) d.record(0.05);
    std::string j = d.to_stats_json();
    EXPECT_NE(j.find("upper_band"), std::string::npos);
    EXPECT_NE(j.find("total_records"), std::string::npos);
}
