// test_InferenceSession.cpp — GTest unit tests for InferenceSession.

#include "InferenceSession.h"
#include <gtest/gtest.h>

using namespace llmquant;

// ─── Construction tests ───────────────────────────────────────────────────────

TEST(InferenceSessionTest, InitialStateIsCreated) {
    InferenceSession sess("sess-001", "gpt4", GenerationConfig{});
    EXPECT_EQ(sess.state(), SessionState::Created);
}

TEST(InferenceSessionTest, SessionIdAndModelIdStored) {
    InferenceSession sess("my-session", "my-model");
    EXPECT_EQ(sess.session_id(), "my-session");
    EXPECT_EQ(sess.model_id(),   "my-model");
}

TEST(InferenceSessionTest, InitialTokenCountIsZero) {
    InferenceSession sess("s", "m");
    EXPECT_EQ(sess.token_count(), 0u);
}

// ─── State transition tests ───────────────────────────────────────────────────

TEST(InferenceSessionTest, StartTransitionsToRunning) {
    InferenceSession sess("s", "m");
    sess.start();
    EXPECT_EQ(sess.state(), SessionState::Running);
}

TEST(InferenceSessionTest, StartTwiceIsIdempotent) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.start();  // should not throw
    EXPECT_EQ(sess.state(), SessionState::Running);
}

TEST(InferenceSessionTest, PauseFromRunning) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.pause();
    EXPECT_EQ(sess.state(), SessionState::Paused);
}

TEST(InferenceSessionTest, ResumeFromPaused) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.pause();
    sess.resume();
    EXPECT_EQ(sess.state(), SessionState::Running);
}

TEST(InferenceSessionTest, CompleteFromRunning) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.complete();
    EXPECT_EQ(sess.state(), SessionState::Completed);
}

TEST(InferenceSessionTest, FailFromRunning) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.fail("OOM error");
    EXPECT_EQ(sess.state(), SessionState::Failed);
    EXPECT_EQ(sess.failure_reason(), "OOM error");
}

TEST(InferenceSessionTest, FailureReasonEmptyOnSuccess) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.complete();
    EXPECT_TRUE(sess.failure_reason().empty());
}

TEST(InferenceSessionTest, StartAfterCompletedThrows) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.complete();
    EXPECT_THROW(sess.start(), std::logic_error);
}

// ─── Token management tests ───────────────────────────────────────────────────

TEST(InferenceSessionTest, AppendTokenIncreasesCount) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.append_token("Hello");
    sess.append_token(" ");
    sess.append_token("World");
    EXPECT_EQ(sess.token_count(), 3u);
}

TEST(InferenceSessionTest, AppendTokenIgnoredInCreatedState) {
    InferenceSession sess("s", "m");
    sess.append_token("ignored");
    EXPECT_EQ(sess.token_count(), 0u);
}

TEST(InferenceSessionTest, OutputTokensMatchAppended) {
    InferenceSession sess("s", "m");
    sess.start();
    sess.append_token("foo");
    sess.append_token("bar");
    const auto& tokens = sess.output_tokens();
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "foo");
    EXPECT_EQ(tokens[1], "bar");
}

// ─── Timing tests ─────────────────────────────────────────────────────────────

TEST(InferenceSessionTest, ElapsedMsIsTokenCountTimes10) {
    InferenceSession sess("s", "m");
    sess.start();
    for (int i = 0; i < 5; ++i) sess.append_token("t");
    EXPECT_EQ(sess.elapsed_ms(), 50u);
}

TEST(InferenceSessionTest, ThroughputTpsIs100) {
    InferenceSession sess("s", "m");
    sess.start();
    // 10 tokens * 10ms = 100ms => 10/0.1 = 100 tps
    for (int i = 0; i < 10; ++i) sess.append_token("t");
    EXPECT_NEAR(sess.throughput_tps(), 100.0, 1e-6);
}

TEST(InferenceSessionTest, ThroughputZeroWhenNoTokens) {
    InferenceSession sess("s", "m");
    EXPECT_EQ(sess.throughput_tps(), 0.0);
}

// ─── Summary test ─────────────────────────────────────────────────────────────

TEST(InferenceSessionTest, SummaryContainsKeyFields) {
    InferenceSession sess("test-session", "gpt-turbo");
    sess.start();
    sess.append_token("hi");
    std::string s = sess.summary();
    EXPECT_NE(s.find("test-session"), std::string::npos);
    EXPECT_NE(s.find("gpt-turbo"),    std::string::npos);
    EXPECT_NE(s.find("Running"),       std::string::npos);
}
