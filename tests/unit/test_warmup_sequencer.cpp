#include <gtest/gtest.h>

#ifdef LLMQUANT_WARMUP_SEQUENCER_ENABLED

#include "WarmupSequencer.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(WarmupSequencer, ConstructsOk) {
    WarmupSequencer w;
    EXPECT_FALSE(w.is_complete());
    EXPECT_EQ(w.tokens_injected(), 0u);
}

// ============================================================
// run — empty token list
// ============================================================

TEST(WarmupSequencer, RunWithNoTokensCompletesImmediately) {
    WarmupSequencer w;
    bool cb_fired = false;
    w.run([](const std::string&, double) {});
    EXPECT_TRUE(w.is_complete());
    EXPECT_EQ(w.tokens_injected(), 0u);
    (void)cb_fired;
}

TEST(WarmupSequencer, OnCompleteCallbackFires) {
    WarmupSequencer::Config cfg;
    bool fired = false;
    cfg.on_complete = [&] { fired = true; };
    WarmupSequencer w(cfg);
    w.run([](const std::string&, double) {});
    EXPECT_TRUE(fired);
}

// ============================================================
// run — token delivery
// ============================================================

TEST(WarmupSequencer, TokensDeliveredInOrder) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"bullish", 0.7}, {"crash", -0.8}, {"rally", 0.5}};
    cfg.repeat_count = 1;
    WarmupSequencer w(cfg);

    std::vector<std::string> received;
    w.run([&](const std::string& tok, double) {
        received.push_back(tok);
    });

    ASSERT_EQ(received.size(), 3u);
    EXPECT_EQ(received[0], "bullish");
    EXPECT_EQ(received[1], "crash");
    EXPECT_EQ(received[2], "rally");
}

TEST(WarmupSequencer, SentimentScoresDelivered) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"tok", 0.42}};
    WarmupSequencer w(cfg);

    double received_score = 0.0;
    w.run([&](const std::string&, double s) { received_score = s; });
    EXPECT_NEAR(received_score, 0.42, 1e-9);
}

// ============================================================
// repeat_count
// ============================================================

TEST(WarmupSequencer, RepeatCountMultipliesTokens) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"tok", 0.1}};
    cfg.repeat_count = 7;
    WarmupSequencer w(cfg);

    int count = 0;
    w.run([&](const std::string&, double) { ++count; });
    EXPECT_EQ(count, 7);
    EXPECT_EQ(w.tokens_injected(), 7u);
}

TEST(WarmupSequencer, ZeroRepeatCountTreatedAsOne) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"tok", 0.0}};
    cfg.repeat_count = 0;  // treated as 1
    WarmupSequencer w(cfg);

    int count = 0;
    w.run([&](const std::string&, double) { ++count; });
    EXPECT_EQ(count, 1);
}

// ============================================================
// is_complete and reset
// ============================================================

TEST(WarmupSequencer, IsCompleteAfterRun) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"x", 0.0}};
    WarmupSequencer w(cfg);
    w.run([](const std::string&, double) {});
    EXPECT_TRUE(w.is_complete());
}

TEST(WarmupSequencer, ResetClearsCompletionState) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"x", 0.0}};
    WarmupSequencer w(cfg);
    w.run([](const std::string&, double) {});
    EXPECT_TRUE(w.is_complete());
    w.reset();
    EXPECT_FALSE(w.is_complete());
    EXPECT_EQ(w.tokens_injected(), 0u);
}

TEST(WarmupSequencer, RunAgainAfterReset) {
    WarmupSequencer::Config cfg;
    cfg.synthetic_tokens = {{"a", 0.1}, {"b", 0.2}};
    WarmupSequencer w(cfg);

    w.run([](const std::string&, double) {});
    w.reset();
    int count = 0;
    w.run([&](const std::string&, double) { ++count; });
    EXPECT_EQ(count, 2);
    EXPECT_EQ(w.tokens_injected(), 2u);
}

// ============================================================
// load_from_file
// ============================================================

static std::string tmp_warmup_path(const char* name) {
    return std::string("tmp_warmup_") + name + ".tsv";
}

TEST(WarmupSequencer, LoadFromFileReadsTokensAndScores) {
    std::string path = tmp_warmup_path("basic");
    {
        std::ofstream f(path);
        f << "bullish\t0.8\n";
        f << "crash\t-0.9\n";
        f << "# comment line\n";
        f << "\n";
        f << "neutral\t0.0\n";
    }

    WarmupSequencer w;
    bool ok = w.load_from_file(path);
    std::remove(path.c_str());

    ASSERT_TRUE(ok);

    std::vector<std::string> toks;
    std::vector<double>      scores;
    w.run([&](const std::string& t, double s) {
        toks.push_back(t);
        scores.push_back(s);
    });

    ASSERT_EQ(toks.size(), 3u);
    EXPECT_EQ(toks[0], "bullish");
    EXPECT_NEAR(scores[0],  0.8, 1e-9);
    EXPECT_EQ(toks[1], "crash");
    EXPECT_NEAR(scores[1], -0.9, 1e-9);
    EXPECT_EQ(toks[2], "neutral");
    EXPECT_NEAR(scores[2],  0.0, 1e-9);
}

TEST(WarmupSequencer, LoadFromFileMissingFileReturnsFalse) {
    WarmupSequencer w;
    EXPECT_FALSE(w.load_from_file("no_such_file_warmup.tsv"));
}

TEST(WarmupSequencer, LoadFromFileNoScoreDefaultsToZero) {
    std::string path = tmp_warmup_path("noscore");
    {
        std::ofstream f(path);
        f << "token_only\n";
    }
    WarmupSequencer w;
    (void)w.load_from_file(path);
    std::remove(path.c_str());

    double score = 99.0;
    w.run([&](const std::string&, double s) { score = s; });
    EXPECT_NEAR(score, 0.0, 1e-9);
}

// ============================================================
// to_stats_json
// ============================================================

TEST(WarmupSequencer, StatsJsonContainsRequiredKeys) {
    WarmupSequencer w;
    std::string js = w.to_stats_json();
    EXPECT_NE(js.find("is_complete"),      std::string::npos);
    EXPECT_NE(js.find("tokens_injected"),  std::string::npos);
    EXPECT_NE(js.find("token_types"),      std::string::npos);
    EXPECT_NE(js.find("repeat_count"),     std::string::npos);
}

} // namespace

#else  // LLMQUANT_WARMUP_SEQUENCER_ENABLED not defined

TEST(WarmupSequencer, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_WARMUP_SEQUENCER_ENABLED
