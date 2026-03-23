#include "TokenTrie.h"

TokenTrie::TokenTrie() : root_(std::make_unique<TrieNode>()), size_(0) {}

void TokenTrie::insert(const std::string& word, int token_id) {
    TrieNode* cur = root_.get();
    for (char ch : word) {
        auto it = cur->children.find(ch);
        if (it == cur->children.end()) {
            cur->children[ch] = std::make_unique<TrieNode>();
            it = cur->children.find(ch);
        }
        cur = it->second.get();
    }
    if (!cur->is_end) {
        ++size_;
    }
    cur->is_end = true;
    cur->token_id = token_id;
}

int TokenTrie::search(const std::string& word) const {
    const TrieNode* node = find_node(word);
    if (node && node->is_end) {
        return node->token_id;
    }
    return -1;
}

bool TokenTrie::starts_with(const std::string& prefix) const {
    return find_node(prefix) != nullptr;
}

std::size_t TokenTrie::common_prefix_length(const std::string& text, std::size_t start) const {
    const TrieNode* cur = root_.get();
    std::size_t last_end = 0;
    std::size_t depth = 0;
    for (std::size_t i = start; i < text.size(); ++i) {
        char ch = text[i];
        auto it = cur->children.find(ch);
        if (it == cur->children.end()) {
            break;
        }
        cur = it->second.get();
        ++depth;
        if (cur->is_end) {
            last_end = depth;
        }
    }
    return last_end;
}

std::vector<int> TokenTrie::tokenize(const std::string& text) const {
    std::vector<int> result;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t len = common_prefix_length(text, pos);
        if (len == 0) {
            // Unknown character: skip
            ++pos;
            continue;
        }
        std::string token = text.substr(pos, len);
        int tid = search(token);
        result.push_back(tid);
        pos += len;
    }
    return result;
}

std::vector<int> TokenTrie::constrained_next_tokens(const std::string& prefix) const {
    const TrieNode* node = find_node(prefix);
    std::vector<int> out;
    if (!node) {
        return out;
    }
    // Collect all terminal token ids reachable from this node
    collect_tokens(node, out);
    return out;
}

std::size_t TokenTrie::size() const {
    return size_;
}

void TokenTrie::load_vocabulary(const std::vector<std::pair<std::string, int>>& vocab) {
    for (const auto& [word, tid] : vocab) {
        insert(word, tid);
    }
}

const TrieNode* TokenTrie::find_node(const std::string& prefix) const {
    const TrieNode* cur = root_.get();
    for (char ch : prefix) {
        auto it = cur->children.find(ch);
        if (it == cur->children.end()) {
            return nullptr;
        }
        cur = it->second.get();
    }
    return cur;
}

void TokenTrie::collect_tokens(const TrieNode* node, std::vector<int>& out) const {
    if (!node) return;
    if (node->is_end) {
        out.push_back(node->token_id);
    }
    for (const auto& [ch, child] : node->children) {
        collect_tokens(child.get(), out);
    }
}
