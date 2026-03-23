#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_WINDOW_SUMMARISER_ENABLED

#include "TokenWindowSummariser.hpp"

#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// Helper: build a SemanticWeight with directional_bias = b, confidence = c,
// volatility = v (sentient_score left at default).
static SemanticWeight make_weight(double bias, double conf = 0.7, double vol = 0.1)
{
    SemanticWeight w;
    w.directional_bias  = bias;
    w.confidence_score  = conf;
    w.volatility_score  = vol;
    w.sentiment_score   = bias;   // mirror for convenience
    return w;
}

// ============================================================
// Construction
// ============================================================

TEST(TokenWindowSummariser, DefaultConstructs)
{
    TokenWindowSummariser s;
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.window_size(), 20u);
    EXPECT_FLOAT_EQ(s.decay(), 0.95f);
    EXPECT_FALSE(s.is_ready());
}

TEST(TokenWindowSummariser, CustomParams)
{
    TokenWindowSummariser s(10, 0.8f);
    EXPECT_EQ(s.window_size(), 10u);
    EXPECT_FLOAT_EQ(s.decay(), 0.8f);
}

TEST(TokenWindowSummariser, InvalidWindowSizeThrows)
{
    EXPECT_THROW(TokenWindowSummariser(0, 0.9f), std::invalid_argument);
}

TEST(TokenWindowSummariser, InvalidDecayThrows)
{
    EXPECT_THROW(TokenWindowSummariser(10, 0.0f), std::invalid_argument);
    EXPECT_THROW(TokenWindowSummariser(10, 1.1f), std::invalid_argument);
}

// ============================================================
// is_ready / size
// ============================================================

TEST(TokenWindowSummariser, NotReadyBeforeWindowFull)
{
    TokenWindowSummariser s(5, 1.0f);
    for (int i = 0; i < 4; ++i) {
        s.push(make_weight(0.1));
        EXPECT_FALSE(s.is_ready());
    }
}

TEST(TokenWindowSummariser, ReadyWhenWindowFull)
{
    TokenWindowSummariser s(5, 1.0f);
    for (int i = 0; i < 5; ++i)
        s.push(make_weight(0.1));
    EXPECT_TRUE(s.is_ready());
    EXPECT_EQ(s.size(), 5u);
}

TEST(TokenWindowSummariser, SizeCapAtWindowSize)
{
    TokenWindowSummariser s(4, 1.0f);
    for (int i = 0; i < 10; ++i)
        s.push(make_weight(static_cast<double>(i) * 0.1));
    EXPECT_EQ(s.size(), 4u);   // eviction keeps size bounded
    EXPECT_TRUE(s.is_ready());
}

// ============================================================
// net_bias
// ============================================================

TEST(TokenWindowSummariser, NetBiasNoDecay)
{
    // decay=1.0 means no ageing; mean of pushed biases.
    TokenWindowSummariser s(4, 1.0f);
    s.push(make_weight(0.2));
    s.push(make_weight(0.4));
    s.push(make_weight(0.6));
    s.push(make_weight(0.8));
    auto sum = s.summarise();
    // mean of {0.2, 0.4, 0.6, 0.8} = 0.5
    EXPECT_NEAR(sum.net_bias, 0.5f, 1e-4f);
}

TEST(TokenWindowSummariser, NetBiasDecayReducesOldEntries)
{
    // With decay < 1 older entries are attenuated; the latest entry has the
    // greatest weight.  Push alternating +1 / -1 — with strong decay (0.1)
    // the last push dominates.
    TokenWindowSummariser s(4, 0.1f);
    s.push(make_weight(-1.0));  // aged 3 times: -1 * 0.1^3 = -0.001
    s.push(make_weight(-1.0));  // aged 2 times: -1 * 0.1^2 = -0.01
    s.push(make_weight(-1.0));  // aged 1 time:  -1 * 0.1   = -0.1
    s.push(make_weight(+1.0));  // latest, no ageing: +1.0
    auto sum = s.summarise();
    // Mean of {-0.001, -0.01, -0.1, +1.0} / 4  ≈ positive
    EXPECT_GT(sum.net_bias, 0.0f);
}

// ============================================================
// confidence
// ============================================================

TEST(TokenWindowSummariser, ConfidenceMeanCorrect)
{
    TokenWindowSummariser s(3, 1.0f);
    s.push(make_weight(0.0, 0.5));
    s.push(make_weight(0.0, 0.7));
    s.push(make_weight(0.0, 0.9));
    auto sum = s.summarise();
    EXPECT_NEAR(sum.confidence, (0.5f + 0.7f + 0.9f) / 3.0f, 1e-4f);
}

// ============================================================
// volatility_est
// ============================================================

TEST(TokenWindowSummariser, VolatilityZeroForConstantBias)
{
    TokenWindowSummariser s(5, 1.0f);
    for (int i = 0; i < 5; ++i)
        s.push(make_weight(0.5));
    auto sum = s.summarise();
    EXPECT_NEAR(sum.volatility_est, 0.0f, 1e-5f);
}

TEST(TokenWindowSummariser, VolatilityNonZeroForMixedBias)
{
    TokenWindowSummariser s(4, 1.0f);
    s.push(make_weight(+1.0));
    s.push(make_weight(-1.0));
    s.push(make_weight(+1.0));
    s.push(make_weight(-1.0));
    auto sum = s.summarise();
    EXPECT_GT(sum.volatility_est, 0.0f);
}

TEST(TokenWindowSummariser, VolatilityZeroForSingleEntry)
{
    TokenWindowSummariser s(1, 1.0f);
    s.push(make_weight(0.5));
    auto sum = s.summarise();
    EXPECT_FLOAT_EQ(sum.volatility_est, 0.0f);
}

// ============================================================
// token_count
// ============================================================

TEST(TokenWindowSummariser, SummaryTokenCountMatchesSize)
{
    TokenWindowSummariser s(8, 1.0f);
    s.push(make_weight(0.1));
    s.push(make_weight(0.2));
    auto sum = s.summarise();
    EXPECT_EQ(sum.token_count, 2u);
}

// ============================================================
// reset()
// ============================================================

TEST(TokenWindowSummariser, ResetClearsWindow)
{
    TokenWindowSummariser s(4, 1.0f);
    for (int i = 0; i < 4; ++i)
        s.push(make_weight(0.5));
    EXPECT_TRUE(s.is_ready());
    s.reset();
    EXPECT_FALSE(s.is_ready());
    EXPECT_EQ(s.size(), 0u);
    auto sum = s.summarise();
    EXPECT_FLOAT_EQ(sum.net_bias,   0.0f);
    EXPECT_EQ(sum.token_count,      0u);
}

// ============================================================
// to_stats_json()
// ============================================================

TEST(TokenWindowSummariser, ToStatsJsonContainsKeys)
{
    TokenWindowSummariser s(4, 1.0f);
    s.push(make_weight(0.3, 0.6));
    std::string json = s.to_stats_json();
    EXPECT_NE(json.find("net_bias"),       std::string::npos);
    EXPECT_NE(json.find("confidence"),     std::string::npos);
    EXPECT_NE(json.find("volatility_est"), std::string::npos);
    EXPECT_NE(json.find("token_count"),    std::string::npos);
    EXPECT_NE(json.find("window_size"),    std::string::npos);
    EXPECT_NE(json.find("is_ready"),       std::string::npos);
}

// ============================================================
// Concurrent push is thread-safe
// ============================================================

TEST(TokenWindowSummariser, ConcurrentPushSafe)
{
    TokenWindowSummariser s(50, 0.99f);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 25; ++i)
                s.push(make_weight(static_cast<double>(t) * 0.1 + i * 0.01));
        });
    }
    for (auto& th : threads) th.join();
    // Window caps at 50; size must not exceed that.
    EXPECT_LE(s.size(), 50u);
    // summarise() must not crash.
    auto sum = s.summarise();
    EXPECT_GE(sum.token_count, 0u);
}

} // namespace

#else

TEST(TokenWindowSummariser, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_TOKEN_WINDOW_SUMMARISER_ENABLED
