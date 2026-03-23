#include "TokenFilter.h"
#include <algorithm>
#include <cmath>

TokenFilter::TokenFilter()
    : max_repetition_(0)
    , consecutive_count_(0)
    , last_token_(-1)
{}

void TokenFilter::add_banned_token(int token_id) {
    banned_tokens_.insert(token_id);
}

void TokenFilter::add_soft_banned_token(int token_id, float penalty) {
    soft_banned_[token_id] = -std::abs(penalty);
}

void TokenFilter::add_bias(int token_id, float bias) {
    biases_[token_id] = bias;
}

void TokenFilter::set_max_repetition(int max_same_consecutive) {
    max_repetition_ = max_same_consecutive;
}

int TokenFilter::count_consecutive(int token_id, const std::vector<int>& history) const {
    int count = 0;
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (*it == token_id) {
            ++count;
        } else {
            break;
        }
    }
    return count;
}

FilterResult TokenFilter::filter(int token_id, const std::vector<int>& history) {
    // Hard ban check
    if (banned_tokens_.count(token_id) > 0) {
        return FilterResult{true, -1, "hard_banned"};
    }

    // Repetition check
    if (max_repetition_ > 0) {
        int consec = count_consecutive(token_id, history);
        if (consec >= max_repetition_) {
            return FilterResult{true, -1, "max_repetition_exceeded"};
        }
    }

    // Passed all filters
    return FilterResult{false, token_id, ""};
}

void TokenFilter::apply_biases(std::vector<float>& logits) const {
    for (const auto& [token_id, bias] : biases_) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < logits.size()) {
            logits[static_cast<size_t>(token_id)] += bias;
        }
    }
    for (const auto& [token_id, penalty] : soft_banned_) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < logits.size()) {
            logits[static_cast<size_t>(token_id)] += penalty;
        }
    }
}

void TokenFilter::apply_repetition_penalty(std::vector<float>& logits,
                                            const std::vector<int>& history,
                                            float penalty) const {
    if (penalty <= 1.0f) return;
    for (int tid : history) {
        if (tid >= 0 && static_cast<size_t>(tid) < logits.size()) {
            float& l = logits[static_cast<size_t>(tid)];
            if (l > 0.0f) {
                l /= penalty;
            } else {
                l *= penalty;
            }
        }
    }
}

bool TokenFilter::is_banned(int token_id) const {
    return banned_tokens_.count(token_id) > 0;
}

void TokenFilter::clear_history_dependent_state() {
    consecutive_count_ = 0;
    last_token_ = -1;
}
