#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// A single node in the prefix trie.
struct TrieNode {
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    int token_id{-1};   ///< -1 if not a terminal node
    bool is_end{false};
};

/// Prefix trie for BPE vocabulary and constrained decoding.
class TokenTrie {
public:
    TokenTrie();

    /// Insert a word with its vocabulary token id.
    void insert(const std::string& word, int token_id);

    /// Search for a word; returns its token_id, or -1 if not found.
    int search(const std::string& word) const;

    /// Returns true if any word in the trie starts with prefix.
    bool starts_with(const std::string& prefix) const;

    /// Length of the longest vocabulary word that starts at position start in text.
    std::size_t common_prefix_length(const std::string& text, std::size_t start) const;

    /// Greedy longest-match tokenization of text into token ids.
    std::vector<int> tokenize(const std::string& text) const;

    /// All token ids that are valid continuations of the given prefix.
    std::vector<int> constrained_next_tokens(const std::string& prefix) const;

    /// Number of terminal nodes (distinct vocabulary entries).
    std::size_t size() const;

    /// Batch insert from a vocabulary list of (word, token_id) pairs.
    void load_vocabulary(const std::vector<std::pair<std::string, int>>& vocab);

private:
    std::unique_ptr<TrieNode> root_;
    std::size_t size_{0};

    /// Traverse to the node for prefix; returns nullptr if not reachable.
    const TrieNode* find_node(const std::string& prefix) const;

    /// Collect all token ids reachable from a given node.
    void collect_tokens(const TrieNode* node, std::vector<int>& out) const;
};
