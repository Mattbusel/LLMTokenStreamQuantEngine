#include "PatternMatcher.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <mutex>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PatternMatcher::PatternMatcher(Config cfg)
    : cfg_(std::move(cfg)) {
    // Always allocate node 0 as the root.
    trie_.push_back(TrieNode{});
}

// ---------------------------------------------------------------------------
// lowercase helper
// ---------------------------------------------------------------------------

std::string PatternMatcher::lowercase(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out += static_cast<char>(std::tolower(c));
    }
    return out;
}

// ---------------------------------------------------------------------------
// add_pattern
// ---------------------------------------------------------------------------

void PatternMatcher::add_pattern(const Pattern& pattern) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (pattern.tokens.empty()) return;
    insert_pattern_locked(pattern);
}

void PatternMatcher::insert_pattern_locked(const Pattern& pattern) {
    size_t node = 0;  // start at root

    for (const auto& raw_token : pattern.tokens) {
        std::string tok = cfg_.lowercase_tokens ? lowercase(raw_token) : raw_token;

        auto it = trie_[node].children.find(tok);
        if (it == trie_[node].children.end()) {
            // Create new child node.
            size_t child_idx = trie_.size();
            trie_.push_back(TrieNode{});
            trie_[node].children[tok] = child_idx;
            node = child_idx;
        } else {
            node = it->second;
        }
    }

    // Mark terminal and store multiplier + label.
    trie_[node].is_terminal = true;
    trie_[node].multiplier  = pattern.multiplier;
    trie_[node].label       = pattern.label;
}

// ---------------------------------------------------------------------------
// remove_pattern_by_label
// ---------------------------------------------------------------------------

size_t PatternMatcher::remove_pattern_by_label(const std::string& label) {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t removed = 0;
    for (auto& node : trie_) {
        if (node.is_terminal && node.label == label) {
            node.is_terminal = false;
            node.multiplier  = 1.0;
            node.label.clear();
            ++removed;
        }
    }
    return removed;
}

// ---------------------------------------------------------------------------
// feed
// ---------------------------------------------------------------------------

std::vector<std::pair<double, std::string>> PatternMatcher::feed(const std::string& token) {
    std::lock_guard<std::mutex> lk(mtx_);

    std::string tok = cfg_.lowercase_tokens ? lowercase(token) : token;
    ++total_tokens_fed_;

    std::vector<std::pair<double, std::string>> matches;

    // 1. Spawn a new cursor at the root for this token (allows matches starting here).
    //    We will immediately attempt to advance it below.
    cursors_.push_back(Cursor{0, 0});

    // 2. Advance all cursors through their trie edges for `tok`.
    std::vector<Cursor> next_cursors;
    next_cursors.reserve(cursors_.size());

    for (const auto& cursor : cursors_) {
        const TrieNode& node = trie_[cursor.node_index];
        auto it = node.children.find(tok);
        if (it == node.children.end()) {
            // No edge for this token — cursor dies.
            continue;
        }

        size_t child_idx       = it->second;
        const TrieNode& child  = trie_[child_idx];

        if (child.is_terminal) {
            // Pattern completed!
            matches.push_back({child.multiplier, child.label});
            ++total_matches_;
            // Don't keep this cursor alive — it completed.
        } else {
            // Cursor advances.
            next_cursors.push_back(Cursor{child_idx, cursor.depth + 1});
        }
    }

    // 3. Prune oldest cursors if we've exceeded the limit.
    if (next_cursors.size() > cfg_.max_active_cursors) {
        // Keep only the most recently added cursors (highest depth / youngest).
        size_t excess = next_cursors.size() - cfg_.max_active_cursors;
        next_cursors.erase(next_cursors.begin(),
                           next_cursors.begin() + static_cast<std::ptrdiff_t>(excess));
    }

    cursors_ = std::move(next_cursors);

    return matches;
}

// ---------------------------------------------------------------------------
// reset_cursors / clear
// ---------------------------------------------------------------------------

void PatternMatcher::reset_cursors() {
    std::lock_guard<std::mutex> lk(mtx_);
    cursors_.clear();
}

void PatternMatcher::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    trie_.clear();
    trie_.push_back(TrieNode{});  // reinstate root
    cursors_.clear();
    total_matches_    = 0;
    total_tokens_fed_ = 0;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

size_t PatternMatcher::pattern_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t count = 0;
    for (const auto& node : trie_) {
        if (node.is_terminal) ++count;
    }
    return count;
}

size_t PatternMatcher::active_cursor_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return cursors_.size();
}

uint64_t PatternMatcher::total_matches() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return total_matches_;
}

uint64_t PatternMatcher::total_tokens_fed() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return total_tokens_fed_;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string PatternMatcher::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    size_t pat_count = 0;
    for (const auto& node : trie_) {
        if (node.is_terminal) ++pat_count;
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"pattern_count\":%zu,\"active_cursors\":%zu,"
        "\"total_tokens_fed\":%llu,\"total_matches\":%llu,"
        "\"max_active_cursors\":%zu}",
        pat_count,
        cursors_.size(),
        static_cast<unsigned long long>(total_tokens_fed_),
        static_cast<unsigned long long>(total_matches_),
        cfg_.max_active_cursors);
    return buf;
}

} // namespace llmquant
