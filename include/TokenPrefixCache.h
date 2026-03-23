#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Result of a trie prefix lookup.
 */
struct CacheHit {
    /// Number of tokens matched from the beginning of the query sequence.
    size_t matched_length{0};
    /// Pointer to the cached KV block, or nullptr on a miss.
    void*  kv_block_ptr{nullptr};

    bool is_hit() const noexcept { return kv_block_ptr != nullptr; }
};

/**
 * @brief Aggregate cache statistics.
 */
struct CacheStats {
    size_t  node_count{0};
    size_t  hit_count{0};
    size_t  miss_count{0};
    double  hit_rate{0.0};
    size_t  memory_estimate_bytes{0};
};

/**
 * @brief Trie-based KV cache prefix sharing for token sequences.
 *
 * Stores cached key-value blocks along the edges of a trie keyed by
 * token ID sequences.  ``lookup`` finds the longest prefix match; ``insert``
 * stores a block at the given path; ``evict_lru`` prunes the least-recently-
 * used leaf nodes when the node count exceeds a budget.
 *
 * ## Thread safety
 * Not thread-safe.  Callers must provide external synchronisation if the
 * cache is shared across threads.
 */
class TokenPrefixCache {
public:
    TokenPrefixCache();
    ~TokenPrefixCache();

    // Non-copyable, moveable.
    TokenPrefixCache(const TokenPrefixCache&)            = delete;
    TokenPrefixCache& operator=(const TokenPrefixCache&) = delete;
    TokenPrefixCache(TokenPrefixCache&&)                 = default;
    TokenPrefixCache& operator=(TokenPrefixCache&&)      = default;

    /**
     * @brief Look up the longest cached prefix of ``token_ids``.
     *
     * @param token_ids  Input token sequence.
     * @return CacheHit with matched_length and kv_block_ptr (nullptr if miss).
     */
    CacheHit lookup(const std::vector<int>& token_ids);

    /**
     * @brief Insert a KV block at the position described by ``token_ids``.
     *
     * If the path does not exist it is created; if a block is already stored
     * at that node it is overwritten.
     *
     * @param token_ids  Token sequence identifying the node.
     * @param kv_block   Pointer to the KV block to cache (ownership NOT transferred).
     */
    void insert(const std::vector<int>& token_ids, void* kv_block);

    /**
     * @brief Evict LRU leaf nodes until the total node count is <= max_nodes.
     *
     * Only leaf nodes (no children) are eligible for eviction.
     *
     * @param max_nodes  Target maximum node count after eviction.
     */
    void evict_lru(size_t max_nodes);

    /**
     * @brief Return aggregate cache statistics.
     */
    CacheStats stats() const;

private:
    struct Node {
        std::unordered_map<int, std::unique_ptr<Node>> children;
        void*    kv_block{nullptr};
        uint64_t last_access_time{0};  ///< Logical clock tick of last access.
        size_t   hit_count{0};

        bool is_leaf() const noexcept { return children.empty(); }
    };

    std::unique_ptr<Node> root_;
    uint64_t              clock_{0};   ///< Logical clock, incremented on each access.
    size_t                node_count_; ///< Total nodes (including root).
    mutable size_t        hit_count_;
    mutable size_t        miss_count_;

    /// Recursively collect (last_access_time, node*) pairs for all leaf nodes.
    void collect_leaves(Node* node, std::vector<std::pair<uint64_t, Node*>>& leaves);

    /// Recursively count nodes.
    size_t count_nodes(const Node* node) const;

    /// Recursively prune children that have no kv_block and no children.
    void prune_empty(Node* node);
};

} // namespace llmquant
