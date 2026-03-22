#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_REPLAY_ENABLED

#include "TokenReplayRecorder.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace llmquant;

namespace {

static std::string tmp_path(const char* name) {
    return std::string("tmp_replay_") + name + ".bin";
}

// ============================================================
// Construction
// ============================================================

TEST(TokenReplayRecorder, ConstructsOk) {
    TokenReplayRecorder rec(tmp_path("construct"));
    EXPECT_FALSE(rec.is_recording());
    EXPECT_EQ(rec.frames_written(), 0u);
}

// ============================================================
// start_recording / stop_recording lifecycle
// ============================================================

TEST(TokenReplayRecorder, StartRecordingReturnsTrue) {
    std::string path = tmp_path("start");
    TokenReplayRecorder rec(path);
    bool ok = rec.start_recording();
    rec.close();
    std::remove(path.c_str());
    EXPECT_TRUE(ok);
}

TEST(TokenReplayRecorder, IsRecordingAfterStart) {
    std::string path = tmp_path("isrec");
    TokenReplayRecorder rec(path);
    (void)rec.start_recording();
    EXPECT_TRUE(rec.is_recording());
    rec.close();
    std::remove(path.c_str());
}

TEST(TokenReplayRecorder, NotRecordingAfterStop) {
    std::string path = tmp_path("stop");
    TokenReplayRecorder rec(path);
    (void)rec.start_recording();
    rec.stop_recording();
    EXPECT_FALSE(rec.is_recording());
    rec.close();
    std::remove(path.c_str());
}

TEST(TokenReplayRecorder, DoubleStartReturnsFalse) {
    std::string path = tmp_path("doublestart");
    TokenReplayRecorder rec(path);
    (void)rec.start_recording();
    EXPECT_FALSE(rec.start_recording());
    rec.close();
    std::remove(path.c_str());
}

TEST(TokenReplayRecorder, RecordNoOpBeforeStart) {
    TokenReplayRecorder rec(tmp_path("noop"));
    rec.record("hello", 0.5);
    EXPECT_EQ(rec.frames_written(), 0u);
}

// ============================================================
// record() and frames_written
// ============================================================

TEST(TokenReplayRecorder, FramesWrittenCountsRecords) {
    std::string path = tmp_path("count");
    TokenReplayRecorder rec(path);
    (void)rec.start_recording();
    rec.record("bullish", 0.8);
    rec.record("crash",  -0.9);
    rec.record("rally",   0.6);
    rec.close();
    std::remove(path.c_str());
    EXPECT_EQ(rec.frames_written(), 3u);
}

// ============================================================
// load_recording round-trip
// ============================================================

TEST(TokenReplayRecorder, LoadRecordingRoundTrip) {
    std::string path = tmp_path("roundtrip");
    {
        TokenReplayRecorder rec(path);
        (void)rec.start_recording();
        rec.record("bullish",  0.85, 1'000'000'000ULL);
        rec.record("crash",   -0.90, 2'000'000'000ULL);
        rec.record("rally",    0.60, 3'000'000'000ULL);
        rec.close();
    }

    std::vector<TokenReplayRecorder::TokenEvent> events;
    bool ok = TokenReplayRecorder::load_recording(path, events);
    std::remove(path.c_str());

    ASSERT_TRUE(ok);
    ASSERT_EQ(events.size(), 3u);

    EXPECT_EQ(events[0].token, "bullish");
    EXPECT_NEAR(events[0].sentiment_score, 0.85, 1e-9);
    EXPECT_EQ(events[0].timestamp_ns, 1'000'000'000ULL);

    EXPECT_EQ(events[1].token, "crash");
    EXPECT_NEAR(events[1].sentiment_score, -0.90, 1e-9);

    EXPECT_EQ(events[2].token, "rally");
    EXPECT_NEAR(events[2].sentiment_score, 0.60, 1e-9);
}

TEST(TokenReplayRecorder, LoadRecordingMissingFileReturnsFalse) {
    std::vector<TokenReplayRecorder::TokenEvent> events;
    EXPECT_FALSE(TokenReplayRecorder::load_recording("no_such_file.bin", events));
}

TEST(TokenReplayRecorder, EmptyTokenRecordsAndLoadsOk) {
    std::string path = tmp_path("emptytoken");
    {
        TokenReplayRecorder rec(path);
        (void)rec.start_recording();
        rec.record("", 0.0, 1ULL);
        rec.close();
    }
    std::vector<TokenReplayRecorder::TokenEvent> events;
    EXPECT_TRUE(TokenReplayRecorder::load_recording(path, events));
    std::remove(path.c_str());
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].token, "");
}

// ============================================================
// read_file_header
// ============================================================

TEST(TokenReplayRecorder, ReadFileHeaderReturnsCorrectMagic) {
    std::string path = tmp_path("header");
    {
        TokenReplayRecorder rec(path);
        (void)rec.start_recording();
        rec.record("test", 0.1, 999ULL);
        rec.close();
    }
    TokenReplayRecorder::FileHeader hdr{};
    bool ok = TokenReplayRecorder::read_file_header(path, hdr);
    std::remove(path.c_str());

    ASSERT_TRUE(ok);
    EXPECT_EQ(hdr.magic,   TokenReplayRecorder::kFileMagic);
    EXPECT_EQ(hdr.version, TokenReplayRecorder::kFileVersion);
    EXPECT_EQ(hdr.frame_count, 1u);
}

TEST(TokenReplayRecorder, ReadFileHeaderMissingFileReturnsFalse) {
    TokenReplayRecorder::FileHeader hdr{};
    EXPECT_FALSE(TokenReplayRecorder::read_file_header("no_such_file.bin", hdr));
}

// ============================================================
// replay
// ============================================================

TEST(TokenReplayRecorder, ReplayInvokesCallbackForEachFrame) {
    std::string path = tmp_path("replay");
    {
        TokenReplayRecorder rec(path);
        (void)rec.start_recording();
        rec.record("a", 0.1, 100ULL);
        rec.record("b", 0.2, 200ULL);
        rec.record("c", 0.3, 300ULL);
        rec.close();
    }

    std::vector<std::string> seen_tokens;
    int64_t count = TokenReplayRecorder::replay(
        path,
        [&](const TokenReplayRecorder::TokenEvent& ev, uint64_t /*idx*/) {
            seen_tokens.push_back(ev.token);
        },
        0.0  // no sleep
    );
    std::remove(path.c_str());

    EXPECT_EQ(count, 3LL);
    ASSERT_EQ(seen_tokens.size(), 3u);
    EXPECT_EQ(seen_tokens[0], "a");
    EXPECT_EQ(seen_tokens[1], "b");
    EXPECT_EQ(seen_tokens[2], "c");
}

TEST(TokenReplayRecorder, ReplayMissingFileReturnsMinusOne) {
    int64_t count = TokenReplayRecorder::replay("no_such_file.bin",
                                                [](auto&, auto){}, 0.0);
    EXPECT_EQ(count, -1LL);
}

TEST(TokenReplayRecorder, ReplayPassesFrameIndexInOrder) {
    std::string path = tmp_path("replayidx");
    {
        TokenReplayRecorder rec(path);
        (void)rec.start_recording();
        for (int i = 0; i < 5; ++i)
            rec.record("tok", static_cast<double>(i), static_cast<uint64_t>(i + 1));
        rec.close();
    }
    std::vector<uint64_t> indices;
    (void)TokenReplayRecorder::replay(path,
                                     [&](auto&, uint64_t idx) { indices.push_back(idx); },
                                     0.0);
    std::remove(path.c_str());
    ASSERT_EQ(indices.size(), 5u);
    for (uint64_t i = 0; i < 5; ++i) EXPECT_EQ(indices[i], i);
}

// ============================================================
// Destructor closes file gracefully
// ============================================================

TEST(TokenReplayRecorder, DestructorClosesGracefully) {
    std::string path = tmp_path("dtor");
    {
        TokenReplayRecorder rec(path);
        (void)rec.start_recording();
        rec.record("token", 0.5);
        // destructor called here
    }
    // File should be readable after destruction.
    std::vector<TokenReplayRecorder::TokenEvent> events;
    bool ok = TokenReplayRecorder::load_recording(path, events);
    std::remove(path.c_str());
    EXPECT_TRUE(ok);
    EXPECT_EQ(events.size(), 1u);
}

} // namespace

#else  // LLMQUANT_TOKEN_REPLAY_ENABLED not defined

TEST(TokenReplayRecorder, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_TOKEN_REPLAY_ENABLED
