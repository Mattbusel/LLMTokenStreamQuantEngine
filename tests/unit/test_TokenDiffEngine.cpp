// test_TokenDiffEngine.cpp — 15 GTest tests for TokenDiffEngine.
#include "gtest/gtest.h"
#include "TokenDiffEngine.h"

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> vec(std::initializer_list<const char*> il) {
    std::vector<std::string> v;
    for (const char* s : il) v.emplace_back(s);
    return v;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(TokenDiffEngineTest, EditDistanceIdenticalSequences) {
    TokenDiffEngine eng;
    auto a = vec({"hello", "world"});
    EXPECT_EQ(eng.edit_distance(a, a), 0u);
}

TEST(TokenDiffEngineTest, EditDistanceEmptySequences) {
    TokenDiffEngine eng;
    EXPECT_EQ(eng.edit_distance({}, {}), 0u);
}

TEST(TokenDiffEngineTest, EditDistanceOneEmptyOneFull) {
    TokenDiffEngine eng;
    auto b = vec({"a", "b", "c"});
    EXPECT_EQ(eng.edit_distance({}, b), 3u);
    EXPECT_EQ(eng.edit_distance(b, {}), 3u);
}

TEST(TokenDiffEngineTest, EditDistanceSingleSubstitution) {
    TokenDiffEngine eng;
    auto a = vec({"the", "cat"});
    auto b = vec({"the", "dog"});
    EXPECT_EQ(eng.edit_distance(a, b), 1u);
}

TEST(TokenDiffEngineTest, EditDistanceCompletelyDifferent) {
    TokenDiffEngine eng;
    auto a = vec({"x", "y"});
    auto b = vec({"a", "b"});
    EXPECT_EQ(eng.edit_distance(a, b), 2u);
}

TEST(TokenDiffEngineTest, SimilarityIdentical) {
    TokenDiffEngine eng;
    auto a = vec({"foo", "bar"});
    EXPECT_DOUBLE_EQ(eng.similarity(a, a), 1.0);
}

TEST(TokenDiffEngineTest, SimilarityBothEmpty) {
    TokenDiffEngine eng;
    EXPECT_DOUBLE_EQ(eng.similarity({}, {}), 1.0);
}

TEST(TokenDiffEngineTest, SimilarityCompletelyDifferent) {
    TokenDiffEngine eng;
    auto a = vec({"a", "b"});
    auto b = vec({"c", "d"});
    EXPECT_DOUBLE_EQ(eng.similarity(a, b), 0.0);
}

TEST(TokenDiffEngineTest, DiffIdenticalSequencesAllKeep) {
    TokenDiffEngine eng;
    auto a = vec({"the", "cat", "sat"});
    auto diffs = eng.diff(a, a);
    for (const auto& d : diffs) {
        EXPECT_EQ(d.op, DiffOp::KEEP);
    }
    EXPECT_EQ(diffs.size(), a.size());
}

TEST(TokenDiffEngineTest, DiffEmptyToNonEmpty) {
    TokenDiffEngine eng;
    auto b = vec({"foo", "bar"});
    auto diffs = eng.diff({}, b);
    for (const auto& d : diffs) {
        EXPECT_EQ(d.op, DiffOp::ADD);
    }
    EXPECT_EQ(diffs.size(), b.size());
}

TEST(TokenDiffEngineTest, DiffNonEmptyToEmpty) {
    TokenDiffEngine eng;
    auto a = vec({"alpha", "beta"});
    auto diffs = eng.diff(a, {});
    for (const auto& d : diffs) {
        EXPECT_EQ(d.op, DiffOp::DELETE);
    }
    EXPECT_EQ(diffs.size(), a.size());
}

TEST(TokenDiffEngineTest, DiffSingleSubstitution) {
    TokenDiffEngine eng;
    auto a = vec({"the", "cat"});
    auto b = vec({"the", "dog"});
    auto diffs = eng.diff(a, b);
    // "the" should be KEEPed; "cat" deleted and "dog" added (or substituted as delete+add).
    bool found_keep = false;
    for (const auto& d : diffs) {
        if (d.op == DiffOp::KEEP && d.token == "the") found_keep = true;
    }
    EXPECT_TRUE(found_keep);
}

TEST(TokenDiffEngineTest, ApplyDiffReconstructsTarget) {
    TokenDiffEngine eng;
    auto a = vec({"alpha", "beta", "gamma"});
    auto b = vec({"alpha", "delta", "gamma", "epsilon"});
    auto diffs = eng.diff(a, b);
    auto reconstructed = eng.apply_diff(a, diffs);
    EXPECT_EQ(reconstructed, b);
}

TEST(TokenDiffEngineTest, ApplyDiffIdentityReconstructsBase) {
    TokenDiffEngine eng;
    auto a = vec({"x", "y", "z"});
    auto diffs = eng.diff(a, a);
    auto reconstructed = eng.apply_diff(a, diffs);
    EXPECT_EQ(reconstructed, a);
}

TEST(TokenDiffEngineTest, EditDistanceLongerSequences) {
    TokenDiffEngine eng;
    // Inserting one token into the middle.
    auto a = vec({"a", "b",     "c", "d"});
    auto b = vec({"a", "b", "X", "c", "d"});
    EXPECT_EQ(eng.edit_distance(a, b), 1u);
    EXPECT_GT(eng.similarity(a, b), 0.7);
}

}  // namespace
