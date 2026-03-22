#include "TemporalPatternLibrary.h"

#include <algorithm>
#include <sstream>

namespace llmquant {

TemporalPatternLibrary::TemporalPatternLibrary(Config config)
    : config_(std::move(config)) {
    // Reserve node 0 as the root.
    trie_.push_back(TrieNode{});
}

// ---------------------------------------------------------------------------
// register_pattern
// ---------------------------------------------------------------------------
void TemporalPatternLibrary::register_pattern(
        const std::string& name,
        const std::vector<std::string>& tokens,
        double boost) {
    if (tokens.empty()) return;

    std::lock_guard<std::mutex> lk(mutex_);

    // Remove any existing pattern with this name.
    registry_.erase(name);

    int node = 0;
    int depth = 0;
    for (const auto& tok : tokens) {
        if (depth >= config_.max_pattern_length) break;
        auto it = trie_[static_cast<std::size_t>(node)].children.find(tok);
        if (it == trie_[static_cast<std::size_t>(node)].children.end()) {
            int new_id = next_id_++;
            trie_.push_back(TrieNode{});
            trie_[static_cast<std::size_t>(node)].children[tok] = new_id;
            node = new_id;
        } else {
            node = it->second;
        }
        ++depth;
    }

    trie_[static_cast<std::size_t>(node)].is_terminal = true;
    trie_[static_cast<std::size_t>(node)].boost        = boost;
    trie_[static_cast<std::size_t>(node)].name         = name;
    registry_[name] = node;
}

void TemporalPatternLibrary::remove_pattern(const std::string& name) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = registry_.find(name);
    if (it == registry_.end()) return;
    int node = it->second;
    trie_[static_cast<std::size_t>(node)].is_terminal = false;
    registry_.erase(it);
}

// ---------------------------------------------------------------------------
// push_token
// ---------------------------------------------------------------------------
bool TemporalPatternLibrary::push_token(const std::string& token) {
    total_tokens_.fetch_add(1, std::memory_order_relaxed);

    bool matched = false;
    std::vector<Match> new_matches;
    MatchCallback cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Always start a new root-level search from this token.
        active_.push_back({0, 0});

        std::vector<ActiveState> next_active;
        next_active.reserve(active_.size());

        for (auto& state : active_) {
            auto& node = trie_[static_cast<std::size_t>(state.node_idx)];
            auto  it   = node.children.find(token);
            if (it != node.children.end()) {
                int child = it->second;
                auto& child_node = trie_[static_cast<std::size_t>(child)];
                if (child_node.is_terminal) {
                    // Match found!
                    Match m{child_node.name, child_node.boost};
                    pending_.push_back(m);
                    new_matches.push_back(std::move(m));
                    total_matches_.fetch_add(1, std::memory_order_relaxed);
                    matched = true;
                } else {
                    // Advance this partial match.
                    next_active.push_back({child, state.depth + 1});
                }
            }
            // Non-matching states are dropped.
        }

        active_ = std::move(next_active);

        // Trim active states that are too long (shouldn't happen, but safety).
        while (static_cast<int>(active_.size()) > config_.max_pattern_length * 4)
            active_.erase(active_.begin());

        cb = config_.on_match;
    }

    // Fire callback outside the lock.
    if (cb) {
        for (const auto& m : new_matches)
            cb(m.name, m.boost);
    }

    return matched;
}

// ---------------------------------------------------------------------------
// pop_match
// ---------------------------------------------------------------------------
bool TemporalPatternLibrary::pop_match(Match& out) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (pending_.empty()) return false;
    out = pending_.front();
    pending_.erase(pending_.begin());
    return true;
}

int TemporalPatternLibrary::pending_matches() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(pending_.size());
}

int TemporalPatternLibrary::pattern_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(registry_.size());
}

void TemporalPatternLibrary::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    active_.clear();
    pending_.clear();
}

void TemporalPatternLibrary::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    trie_.clear();
    trie_.push_back(TrieNode{});
    next_id_ = 1;
    active_.clear();
    pending_.clear();
    registry_.clear();
}

void TemporalPatternLibrary::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    active_.clear();
    pending_.clear();
}

std::string TemporalPatternLibrary::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << "{"
       << "\"pattern_count\":"   << registry_.size() << ","
       << "\"pending_matches\":" << pending_.size() << ","
       << "\"active_states\":"   << active_.size() << ","
       << "\"total_tokens\":"    << total_tokens_.load(std::memory_order_relaxed) << ","
       << "\"total_matches\":"   << total_matches_.load(std::memory_order_relaxed)
       << "}";
    return os.str();
}

} // namespace llmquant
