/// @file AttentionMaskBuilder.cpp
/// @brief Implementation of AttentionMaskBuilder.

#ifdef LLMQUANT_ATTENTION_MASK_ENABLED

#include "AttentionMaskBuilder.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static AttentionMask make_mask(int seq_len, bool initial_value) {
    if (seq_len <= 0) {
        throw std::invalid_argument("seq_len must be > 0");
    }
    AttentionMask m;
    m.seq_len = seq_len;
    m.mask.assign(seq_len, std::vector<bool>(seq_len, initial_value));
    return m;
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

AttentionMask AttentionMaskBuilder::build_causal(int seq_len) {
    auto m = make_mask(seq_len, false);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j <= i; ++j) {
            m.mask[i][j] = true;
        }
    }
    return m;
}

AttentionMask AttentionMaskBuilder::build_full(int seq_len) {
    return make_mask(seq_len, true);
}

AttentionMask AttentionMaskBuilder::build_sliding_window(int seq_len, int window_size) {
    if (window_size <= 0) {
        throw std::invalid_argument("window_size must be > 0");
    }
    auto m = make_mask(seq_len, false);
    for (int i = 0; i < seq_len; ++i) {
        int j_start = std::max(0, i - window_size + 1);
        for (int j = j_start; j <= i; ++j) {
            m.mask[i][j] = true;
        }
    }
    return m;
}

AttentionMask AttentionMaskBuilder::build_prefix_lm(int seq_len, int prefix_len) {
    if (prefix_len < 0 || prefix_len > seq_len) {
        throw std::invalid_argument("prefix_len must be in [0, seq_len]");
    }
    auto m = make_mask(seq_len, false);
    for (int i = 0; i < seq_len; ++i) {
        if (i < prefix_len) {
            // Prefix position: bidirectional within the prefix.
            for (int j = 0; j < prefix_len; ++j) {
                m.mask[i][j] = true;
            }
        } else {
            // Non-prefix position: causal attention.
            for (int j = 0; j <= i; ++j) {
                m.mask[i][j] = true;
            }
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// apply_mask
// ---------------------------------------------------------------------------

std::vector<float> AttentionMaskBuilder::apply_mask(
    const std::vector<float>& attention_scores,
    const AttentionMask& mask,
    float mask_value)
{
    const int n = mask.seq_len;
    if (static_cast<int>(attention_scores.size()) != n * n) {
        throw std::invalid_argument(
            "attention_scores.size() must equal seq_len * seq_len");
    }

    std::vector<float> result = attention_scores;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!mask.mask[i][j]) {
                result[i * n + j] = mask_value;
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// visualize
// ---------------------------------------------------------------------------

std::string AttentionMaskBuilder::visualize(const AttentionMask& mask) {
    std::ostringstream oss;
    for (int i = 0; i < mask.seq_len; ++i) {
        for (int j = 0; j < mask.seq_len; ++j) {
            oss << (mask.mask[i][j] ? '1' : '0');
            if (j < mask.seq_len - 1) oss << ' ';
        }
        oss << '\n';
    }
    return oss.str();
}

} // namespace llmquant

#endif // LLMQUANT_ATTENTION_MASK_ENABLED
