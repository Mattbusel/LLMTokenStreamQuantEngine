#include <gtest/gtest.h>

#ifdef LLMQUANT_RUN_LENGTH_ENABLED
#include "BiasRunLengthEncoder.h"
using namespace llmquant;

TEST(BiasRunLengthEncoderTest, InitialState) {
    BiasRunLengthEncoder enc;
    EXPECT_EQ(enc.current_run(), 0);
    EXPECT_EQ(enc.current_direction(), 0);
    EXPECT_EQ(enc.max_run(), 0);
    EXPECT_EQ(enc.run_count(), 0u);
    EXPECT_EQ(enc.observation_count(), 0u);
}

TEST(BiasRunLengthEncoderTest, ObservationCountIncrement) {
    BiasRunLengthEncoder enc;
    enc.record(0.5);
    enc.record(-0.3);
    EXPECT_EQ(enc.observation_count(), 2u);
}

TEST(BiasRunLengthEncoderTest, BullishRunBuilds) {
    BiasRunLengthEncoder enc;
    for (int i = 0; i < 5; ++i) enc.record(0.5);
    EXPECT_EQ(enc.current_run(), 5);
    EXPECT_EQ(enc.current_direction(), 1);
}

TEST(BiasRunLengthEncoderTest, BearishRunBuilds) {
    BiasRunLengthEncoder enc;
    for (int i = 0; i < 4; ++i) enc.record(-0.3);
    EXPECT_EQ(enc.current_run(), 4);
    EXPECT_EQ(enc.current_direction(), -1);
}

TEST(BiasRunLengthEncoderTest, RunCompletesOnFlip) {
    BiasRunLengthEncoder enc;
    for (int i = 0; i < 5; ++i) enc.record(0.5);
    enc.record(-0.1); // flip — finalizes run of 5
    EXPECT_EQ(enc.run_count(), 1u);
    EXPECT_EQ(enc.max_run(), 5);
    EXPECT_EQ(enc.current_run(), 1);
    EXPECT_EQ(enc.current_direction(), -1);
}

TEST(BiasRunLengthEncoderTest, MaxRunTracked) {
    BiasRunLengthEncoder enc;
    for (int i = 0; i < 10; ++i) enc.record(0.5);
    enc.record(-0.1); // finalize run of 10
    for (int i = 0; i < 3; ++i) enc.record(0.5);
    enc.record(-0.1); // finalize run of 3
    EXPECT_EQ(enc.max_run(), 10);
}

TEST(BiasRunLengthEncoderTest, LongRunCallback) {
    BiasRunLengthEncoder::Config cfg;
    cfg.long_run_threshold = 5;
    int fires = 0;
    cfg.on_long_run = [&](int, int) { ++fires; };
    BiasRunLengthEncoder enc(cfg);
    for (int i = 0; i < 8; ++i) enc.record(0.5);
    enc.record(-0.1); // finalize run of 8 >= 5 → fire
    EXPECT_GE(fires, 1);
}

TEST(BiasRunLengthEncoderTest, AvgRunSmoothed) {
    BiasRunLengthEncoder enc;
    // Alternate: run of 3 positive, then flip
    for (int rep = 0; rep < 5; ++rep) {
        for (int i = 0; i < 3; ++i) enc.record(0.5);
        enc.record(-0.3);
        for (int i = 0; i < 3; ++i) enc.record(-0.3);
        enc.record(0.5);
    }
    EXPECT_GT(enc.avg_run(), 0.0);
}

TEST(BiasRunLengthEncoderTest, Reset) {
    BiasRunLengthEncoder enc;
    for (int i = 0; i < 10; ++i) enc.record(0.5);
    enc.reset();
    EXPECT_EQ(enc.current_run(), 0);
    EXPECT_EQ(enc.max_run(), 0);
    EXPECT_EQ(enc.observation_count(), 0u);
}

TEST(BiasRunLengthEncoderTest, StatsJson) {
    BiasRunLengthEncoder enc;
    enc.record(0.5);
    std::string json = enc.to_stats_json();
    EXPECT_NE(json.find("current_run"), std::string::npos);
    EXPECT_NE(json.find("max_run"), std::string::npos);
}

#else
TEST(BiasRunLengthEncoderTest, Disabled) { SUCCEED(); }
#endif
