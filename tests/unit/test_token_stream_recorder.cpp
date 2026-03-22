#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_RECORDER_ENABLED

#include "TokenStreamRecorder.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

static const char* kTmpFile  = "test_recorder_tmp.bin";
static const char* kTmpFile2 = "test_recorder_tmp2.bin";

struct TmpGuard {
    ~TmpGuard() {
        std::remove(kTmpFile);
        std::remove(kTmpFile2);
        std::remove((std::string(kTmpFile) + ".1").c_str());
    }
};

// ============================================================
// Default construction — recorder starts closed
// ============================================================

TEST(TokenStreamRecorder, DefaultConstructsClosedWhenNoPath) {
    TmpGuard g;
    TokenStreamRecorder rec;
    EXPECT_FALSE(rec.is_open());
    EXPECT_EQ(rec.records_written(), 0u);
    EXPECT_EQ(rec.bytes_written(),   0u);
    EXPECT_EQ(rec.error_count(),     0u);
}

// ============================================================
// open() opens a file
// ============================================================

TEST(TokenStreamRecorder, OpenCreatesFile) {
    TmpGuard g;
    TokenStreamRecorder rec;
    EXPECT_TRUE(rec.open(kTmpFile));
    EXPECT_TRUE(rec.is_open());
    rec.close();
}

// ============================================================
// record() writes records
// ============================================================

TEST(TokenStreamRecorder, RecordWritesEntries) {
    TmpGuard g;
    TokenStreamRecorder rec;
    rec.open(kTmpFile);
    rec.record("bullish", 0.7);
    rec.record("bearish", -0.5);
    EXPECT_EQ(rec.records_written(), 2u);
    EXPECT_GT(rec.bytes_written(), 0u);
    rec.close();
}

// ============================================================
// record() is silent no-op when file is closed
// ============================================================

TEST(TokenStreamRecorder, RecordNoOpWhenClosed) {
    TmpGuard g;
    TokenStreamRecorder rec;
    // No file opened.
    EXPECT_TRUE(rec.record("token", 0.1));  // should silently succeed
    EXPECT_EQ(rec.records_written(), 0u);
}

// ============================================================
// close() + re-open appends to file
// ============================================================

TEST(TokenStreamRecorder, CloseAndReopenAppends) {
    TmpGuard g;
    TokenStreamRecorder rec;
    rec.open(kTmpFile);
    rec.record("a", 0.1);
    rec.close();
    rec.open(kTmpFile);  // append mode
    rec.record("b", 0.2);
    EXPECT_EQ(rec.records_written(), 1u);  // second session wrote 1 record
    rec.close();
}

// ============================================================
// rotate() switches to new file
// ============================================================

TEST(TokenStreamRecorder, RotateSwitchesFile) {
    TmpGuard g;
    std::atomic<int> rotate_calls{0};
    TokenStreamRecorder::Config cfg;
    cfg.on_rotate = [&](const std::string&, const std::string&, uint64_t) {
        ++rotate_calls;
    };
    TokenStreamRecorder rec(cfg);
    rec.open(kTmpFile);
    rec.record("x", 0.5);
    EXPECT_TRUE(rec.rotate(kTmpFile2));
    EXPECT_EQ(rotate_calls.load(), 1);
    rec.record("y", 0.3);
    EXPECT_EQ(rec.records_written(), 1u);  // count reset after rotate
    rec.close();
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(TokenStreamRecorder, ToStatsJsonContainsKeys) {
    TmpGuard g;
    TokenStreamRecorder rec;
    rec.open(kTmpFile);
    rec.record("bull", 0.8);
    std::string json = rec.to_stats_json();
    EXPECT_NE(json.find("is_open"),     std::string::npos);
    EXPECT_NE(json.find("records"),     std::string::npos);
    EXPECT_NE(json.find("bytes"),       std::string::npos);
    EXPECT_NE(json.find("error_count"), std::string::npos);
    rec.close();
}

// ============================================================
// Concurrent record() is thread-safe
// ============================================================

TEST(TokenStreamRecorder, ConcurrentRecordSafe) {
    TmpGuard g;
    TokenStreamRecorder rec;
    rec.open(kTmpFile);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i)
                rec.record("tok" + std::to_string(t),
                            static_cast<double>(i) * 0.01);
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(rec.records_written(), 200u);
    rec.close();
}

} // namespace

#else

TEST(TokenStreamRecorder, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_TOKEN_RECORDER_ENABLED
