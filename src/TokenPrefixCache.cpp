#include "TokenPrefixCache.h"

#include <algorithm>
#include <cassert>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

TokenPrefixCache::TokenPrefixCache()
    : root_(std::make_unique<Node>()),
      clock_(0),
      node_count_(1),  // root counts as one node
      hit_count_(0),
      miss_count_(0)
{}

TokenPrefixCache::~TokenPrefixCache() = default;

// ---------------------------------------------------------------------------
// lookup
// ---------------------------------------------------------------------------

CacheHit TokenPrefixCache::lookup(const std::vector<int>& token_ids)
{
    ++clock_;

    Node*  current        = root_.get();
    size_t matched_length = 0;
    void*  best_block     = nullptr;

    for (int token : token_ids) {
        auto it = current->children.find(token);
        if (it == current->children.end()) {
            break;  // no further match
        }
        current = it->second.get();
        current->last_access_time = clock_;
        ++matched_length;

        if (current->kv_block != nullptr) {
            best_block = current->kv_block;
            ++current->hit_count;
        }
    }

    if (best_block != nullptr) {
        ++hit_count_;
    } else {
        ++miss_count_;
    }

    return CacheHit{matched_length, best_block};
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------

void TokenPrefixCache::insert(const std::vector<int>& token_ids, void* kv_block)
{
    ++clock_;
    Node* current = root_.get();
    current->last_access_time = clock_;

    for (int token : token_ids) {
        auto it = current->children.find(token);
        if (it == current->children.end()) {
            auto [ins_it, ok] = current->children.emplace(token, std::make_unique<Node>());
            (void)ok;
            ++node_count_;
            current = ins_it->second.get();
        } else {
            current = it->second.get();
        }
        current->last_access_time = clock_;
    }

    current->kv_block = kv_block;
}

// ---------------------------------------------------------------------------
// evict_lru
// ---------------------------------------------------------------------------

void TokenPrefixCache::evict_lru(size_t max_nodes)
{
    while (node_count_ > max_nodes) {
        // Collect all leaf nodes with their access times.
        std::vector<std::pair<uint64_t, Node*>> leaves;
        collect_leaves(root_.get(), leaves);

        if (leaves.empty()) break;

        // Sort by last_access_time ascending (oldest first).
        std::sort(leaves.begin(), leaves.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Evict oldest leaf: find its parent and remove it.
        // We do a second pass on the trie to find and detach the node.
        Node* victim = leaves.front().second;
        if (victim == root_.get()) break;  // never evict root

        // Walk trie to find and remove the victim leaf.
        bool removed = false;
        std::function<bool(Node*)> remove_leaf = [&](Node* node) -> bool {
            for (auto it = node->children.begin(); it != node->children.end(); ++it) {
                if (it->second.get() == victim) {
                    node->children.erase(it);
                    --node_count_;
                    removed = true;
                    return true;
                }
                if (remove_leaf(it->second.get())) return true;
            }
            return false;
        };
        remove_leaf(root_.get());

        if (!removed) break;  // safety: avoid infinite loop

        // Prune any newly-created empty non-root nodes.
        prune_empty(root_.get());
    }
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

CacheStats TokenPrefixCache::stats() const
{
    CacheStats s;
    s.node_count  = node_count_;
    s.hit_count   = hit_count_;
    s.miss_count  = miss_count_;
    s.hit_rate    = (hit_count_ + miss_count_) > 0
                    ? static_cast<double>(hit_count_) / static_cast<double>(hit_count_ + miss_count_)
                    : 0.0;
    // Rough memory estimate: each node has a fixed overhead + map entries.
    s.memory_estimate_bytes =
        node_count_ * (sizeof(Node) + 64);  // 64 bytes map overhead per node
    return s;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void TokenPrefixCache::collect_leaves(
    Node*                                      node,
    std::vector<std::pair<uint64_t, Node*>>&  leaves)
{
    if (node->is_leaf() && node != root_.get()) {
        leaves.emplace_back(node->last_access_time, node);
        return;
    }
    for (auto& [token, child] : node->children) {
        collect_leaves(child.get(), leaves);
    }
}

size_t TokenPrefixCache::count_nodes(const Node* node) const
{
    size_t count = 1;
    for (const auto& [token, child] : node->children) {
        count += count_nodes(child.get());
    }
    return count;
}

void TokenPrefixCache::prune_empty(Node* node)
{
    // Iteratively remove children that have no block and no children.
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = node->children.begin(); it != node->children.end(); ) {
            Node* child = it->second.get();
            prune_empty(child);  // recurse first
            if (child->is_leaf() && child->kv_block == nullptr) {
                it = node->children.erase(it);
                --node_count_;
                changed = true;
            } else {
                ++it;
            }
        }
    }
}

} // namespace llmquant
