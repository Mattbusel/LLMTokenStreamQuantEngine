#include "PromptEncoder.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, int> simple_vocab() {
    return {
        {"hello", 1},
        {"world", 2},
        {"foo",   3},
        {"bar",   4},
        {"the",   5},
        {"cat",   6},
        {"sat",   7},
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(PromptEncoderTest, ConstructsWithVocab) {
    EXPECT_NO_THROW(PromptEncoder enc(simple_vocab()));
}

TEST(PromptEncoderTest, TokenToIdKnownToken) {
    PromptEncoder enc(simple_vocab());
    EXPECT_EQ(enc.token_to_id("hello"), 1);
    EXPECT_EQ(enc.token_to_id("world"), 2);
}

TEST(PromptEncoderTest, TokenToIdUnknownReturnsZero) {
    PromptEncoder enc(simple_vocab());
    EXPECT_EQ(enc.token_to_id("nonexistent_xyz"), PromptEncoder::UNK_ID);
}

TEST(PromptEncoderTest, SplitBpeWhitespace) {
    PromptEncoder enc(simple_vocab());
    auto toks = enc.split_bpe("hello world");
    ASSERT_EQ(toks.size(), 2u);
    EXPECT_EQ(toks[0], "hello");
    EXPECT_EQ(toks[1], "world");
}

TEST(PromptEncoderTest, SplitBpePunctuation) {
    PromptEncoder enc(simple_vocab());
    auto toks = enc.split_bpe("hello,world");
    ASSERT_EQ(toks.size(), 3u);
    EXPECT_EQ(toks[1], ",");
}

TEST(PromptEncoderTest, SplitBpeLowercases) {
    PromptEncoder enc(simple_vocab());
    auto toks = enc.split_bpe("Hello World");
    ASSERT_EQ(toks.size(), 2u);
    EXPECT_EQ(toks[0], "hello");
    EXPECT_EQ(toks[1], "world");
}

TEST(PromptEncoderTest, EncodeBasic) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello world");
    EXPECT_EQ(ep.length, 2u);
    EXPECT_EQ(ep.token_ids[0], 1);
    EXPECT_EQ(ep.token_ids[1], 2);
}

TEST(PromptEncoderTest, EncodeAttentionMaskAllOnes) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello world foo");
    for (int mask : ep.attention_mask) {
        EXPECT_EQ(mask, 1);
    }
}

TEST(PromptEncoderTest, EncodePositionIdsSequential) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello world foo");
    for (size_t i = 0; i < ep.position_ids.size(); ++i) {
        EXPECT_EQ(ep.position_ids[i], static_cast<int>(i));
    }
}

TEST(PromptEncoderTest, EncodeTruncatesToMaxLength) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello world foo bar", 2);
    EXPECT_EQ(ep.length, 2u);
    EXPECT_EQ(ep.token_ids.size(), 2u);
}

TEST(PromptEncoderTest, DecodeRoundTrip) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello world");
    auto decoded = enc.decode(ep.token_ids);
    EXPECT_EQ(decoded, "hello world");
}

TEST(PromptEncoderTest, PadToLengthExtends) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello");
    auto padded = enc.pad_to_length(ep, 5, 0);
    EXPECT_EQ(padded.length, 5u);
    EXPECT_EQ(padded.token_ids[4], 0);
    EXPECT_EQ(padded.attention_mask[4], 0);
}

TEST(PromptEncoderTest, TruncateShortensPrompt) {
    PromptEncoder enc(simple_vocab());
    auto ep = enc.encode("hello world foo bar");
    auto trunc = enc.truncate(ep, 2);
    EXPECT_EQ(trunc.length, 2u);
}

TEST(PromptEncoderTest, BatchEncodeAllSameLength) {
    PromptEncoder enc(simple_vocab());
    std::vector<std::string> prompts = {"hello world", "foo", "bar the cat"};
    auto batch = enc.batch_encode(prompts, 8);
    ASSERT_EQ(batch.size(), 3u);
    for (const auto& ep : batch) {
        EXPECT_EQ(static_cast<int>(ep.token_ids.size()), 8);
    }
}

TEST(PromptEncoderTest, BuildVocabRespectsMaxSize) {
    std::vector<std::string> corpus = {
        "the quick brown fox",
        "the fox jumps over the lazy dog",
    };
    auto vocab = PromptEncoder::build_vocab(corpus, 5);
    // At most 4 tokens (id 0 reserved for UNK, so ids 1..4).
    EXPECT_LE(static_cast<int>(vocab.size()), 4);
}

TEST(PromptEncoderTest, BuildVocabFrequentTokenGetsLowerId) {
    std::vector<std::string> corpus = {
        "the the the the fox fox fox",
    };
    auto vocab = PromptEncoder::build_vocab(corpus, 100);
    // "the" appears 4 times, "fox" 3 — "the" should have id 1.
    EXPECT_EQ(vocab.at("the"), 1);
}

TEST(PromptEncoderTest, UnknownTokensDecodedAsEmpty) {
    PromptEncoder enc({{"hi", 1}});
    auto decoded = enc.decode({99, 100});
    // Unknown ids produce empty string (no tokens found).
    EXPECT_TRUE(decoded.empty());
}
