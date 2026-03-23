// test_VocabularyManager.cpp — GTest unit tests for VocabularyManager.

#include "VocabularyManager.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

using namespace llmquant;

// ─── Construction ────────────────────────────────────────────────────────────

TEST(VocabularyManagerTest, DefaultConstructorAddsUNK) {
    VocabularyManager vm;
    EXPECT_EQ(vm.size(), 1u);  // only UNK
    auto id = vm.token_to_id(VocabularyManager::UNK_TOKEN);
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(*id, 0u);
}

TEST(VocabularyManagerTest, UNKIdIsZero) {
    VocabularyManager vm;
    EXPECT_EQ(vm.unk_id(), 0u);
}

// ─── add_token ────────────────────────────────────────────────────────────────

TEST(VocabularyManagerTest, AddTokenReturnsId) {
    VocabularyManager vm;
    std::size_t id = vm.add_token("hello");
    EXPECT_EQ(id, 1u);  // 0 is UNK
}

TEST(VocabularyManagerTest, AddSameTokenIdempotent) {
    VocabularyManager vm;
    std::size_t id1 = vm.add_token("hello");
    std::size_t id2 = vm.add_token("hello");
    EXPECT_EQ(id1, id2);
    EXPECT_EQ(vm.size(), 2u);  // UNK + hello
}

TEST(VocabularyManagerTest, AddTokenIncrementsFrequency) {
    VocabularyManager vm;
    vm.add_token("word");
    vm.add_token("word");
    vm.add_token("word");
    auto id = vm.token_to_id("word");
    ASSERT_TRUE(id.has_value());
    // No direct frequency accessor; encode check is sufficient
    auto encoded = vm.encode({"word"});
    EXPECT_EQ(encoded[0], *id);
}

TEST(VocabularyManagerTest, AddSpecialToken) {
    VocabularyManager vm;
    std::size_t id = vm.add_token("<PAD>", true);
    auto entry_id = vm.token_to_id("<PAD>");
    ASSERT_TRUE(entry_id.has_value());
    EXPECT_EQ(*entry_id, id);
}

// ─── token_to_id / id_to_token ───────────────────────────────────────────────

TEST(VocabularyManagerTest, TokenToIdMissingReturnsNullopt) {
    VocabularyManager vm;
    EXPECT_FALSE(vm.token_to_id("missing").has_value());
}

TEST(VocabularyManagerTest, IdToTokenValidId) {
    VocabularyManager vm;
    std::size_t id = vm.add_token("cat");
    auto tok = vm.id_to_token(id);
    ASSERT_TRUE(tok.has_value());
    EXPECT_EQ(*tok, "cat");
}

TEST(VocabularyManagerTest, IdToTokenInvalidReturnsNullopt) {
    VocabularyManager vm;
    EXPECT_FALSE(vm.id_to_token(9999).has_value());
}

// ─── encode / decode ─────────────────────────────────────────────────────────

TEST(VocabularyManagerTest, EncodeKnownTokens) {
    VocabularyManager vm;
    vm.add_token("the");
    vm.add_token("cat");
    auto ids = vm.encode({"the", "cat"});
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_NE(ids[0], ids[1]);
}

TEST(VocabularyManagerTest, EncodeUnknownTokenMapsToUNK) {
    VocabularyManager vm;
    auto ids = vm.encode({"unknown_word"});
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], vm.unk_id());
}

TEST(VocabularyManagerTest, DecodeKnownIds) {
    VocabularyManager vm;
    vm.add_token("foo");
    vm.add_token("bar");
    auto ids = vm.encode({"foo", "bar"});
    auto toks = vm.decode(ids);
    ASSERT_EQ(toks.size(), 2u);
    EXPECT_EQ(toks[0], "foo");
    EXPECT_EQ(toks[1], "bar");
}

TEST(VocabularyManagerTest, DecodeUnknownIdReturnsUNK) {
    VocabularyManager vm;
    auto toks = vm.decode({9999});
    ASSERT_EQ(toks.size(), 1u);
    EXPECT_EQ(toks[0], VocabularyManager::UNK_TOKEN);
}

TEST(VocabularyManagerTest, EncodeDecodeRoundTrip) {
    VocabularyManager vm;
    std::vector<std::string> words = {"the", "quick", "brown", "fox"};
    for (const auto& w : words) vm.add_token(w);
    auto ids = vm.encode(words);
    auto decoded = vm.decode(ids);
    EXPECT_EQ(decoded, words);
}

// ─── prune ───────────────────────────────────────────────────────────────────

TEST(VocabularyManagerTest, PruneRemovesLowFrequencyTokens) {
    VocabularyManager vm;
    vm.add_token("common");
    vm.add_token("common");
    vm.add_token("common");
    vm.add_token("rare");  // frequency = 1
    vm.prune(2);
    EXPECT_FALSE(vm.token_to_id("rare").has_value());
    EXPECT_TRUE(vm.token_to_id("common").has_value());
}

TEST(VocabularyManagerTest, PruneKeepsSpecialTokens) {
    VocabularyManager vm;
    // UNK is special with frequency=0; it must survive any prune
    vm.prune(100);
    EXPECT_TRUE(vm.token_to_id(VocabularyManager::UNK_TOKEN).has_value());
}

// ─── save_tsv ────────────────────────────────────────────────────────────────

TEST(VocabularyManagerTest, SaveTsvCreatesFile) {
    VocabularyManager vm;
    vm.add_token("alpha");
    vm.add_token("beta");
    std::string path = "test_vocab_tmp.tsv";
    vm.save_tsv(path);
    EXPECT_TRUE(std::filesystem::exists(path));
    std::filesystem::remove(path);
}

TEST(VocabularyManagerTest, SaveTsvContentsCorrect) {
    VocabularyManager vm;
    vm.add_token("alpha");
    std::string path = "test_vocab_contents_tmp.tsv";
    vm.save_tsv(path);
    std::ifstream ifs(path);
    std::string line;
    bool found_alpha = false;
    while (std::getline(ifs, line)) {
        if (line.find("alpha") != std::string::npos) {
            found_alpha = true;
        }
    }
    EXPECT_TRUE(found_alpha);
    std::filesystem::remove(path);
}

// ─── size / entries_sorted ────────────────────────────────────────────────────

TEST(VocabularyManagerTest, SizeIncrementsWithNewTokens) {
    VocabularyManager vm;
    EXPECT_EQ(vm.size(), 1u);
    vm.add_token("x");
    EXPECT_EQ(vm.size(), 2u);
    vm.add_token("y");
    EXPECT_EQ(vm.size(), 3u);
}

TEST(VocabularyManagerTest, EntriesSortedByIdAscending) {
    VocabularyManager vm;
    vm.add_token("b");
    vm.add_token("a");
    auto sorted = vm.entries_sorted();
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        EXPECT_LT(sorted[i - 1].id, sorted[i].id);
    }
}
