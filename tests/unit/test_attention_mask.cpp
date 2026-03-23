#include <gtest/gtest.h>

#ifdef LLMQUANT_ATTENTION_MASK_ENABLED

#include "AttentionMaskBuilder.h"

using namespace llmquant;

// ============================================================
// Causal mask
// ============================================================

TEST(AttentionMaskBuilder, CausalLowerTriangular) {
    auto m = AttentionMaskBuilder::build_causal(4);
    ASSERT_EQ(m.seq_len, 4);
    // Lower triangle should be true.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (j <= i) {
                EXPECT_TRUE(m.mask[i][j]) << "Expected mask[" << i << "][" << j << "] = true";
            } else {
                EXPECT_FALSE(m.mask[i][j]) << "Expected mask[" << i << "][" << j << "] = false";
            }
        }
    }
}

TEST(AttentionMaskBuilder, CausalDiagonalAlwaysTrue) {
    auto m = AttentionMaskBuilder::build_causal(6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(m.mask[i][i]);
    }
}

// ============================================================
// Full mask
// ============================================================

TEST(AttentionMaskBuilder, FullMaskAllTrue) {
    auto m = AttentionMaskBuilder::build_full(5);
    ASSERT_EQ(m.seq_len, 5);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_TRUE(m.mask[i][j]);
        }
    }
}

// ============================================================
// Sliding window mask
// ============================================================

TEST(AttentionMaskBuilder, SlidingWindowCorrect) {
    // window_size = 2: each position attends to itself and previous 1.
    auto m = AttentionMaskBuilder::build_sliding_window(5, 2);
    ASSERT_EQ(m.seq_len, 5);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            bool expected = (j >= std::max(0, i - 1)) && (j <= i);
            EXPECT_EQ(m.mask[i][j], expected)
                << "mask[" << i << "][" << j << "] mismatch";
        }
    }
}

TEST(AttentionMaskBuilder, SlidingWindowWindowOfOne) {
    // window_size = 1: each position attends only to itself.
    auto m = AttentionMaskBuilder::build_sliding_window(4, 1);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(m.mask[i][j], (i == j));
        }
    }
}

// ============================================================
// Prefix LM mask
// ============================================================

TEST(AttentionMaskBuilder, PrefixLMBoundaryCorrect) {
    // seq_len=6, prefix_len=3
    auto m = AttentionMaskBuilder::build_prefix_lm(6, 3);
    ASSERT_EQ(m.seq_len, 6);

    // Prefix rows (0, 1, 2): should attend bidirectionally within [0, 2].
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_TRUE(m.mask[i][j]) << "prefix[" << i << "][" << j << "]";
        }
        // Should NOT attend to non-prefix positions.
        for (int j = 3; j < 6; ++j) {
            EXPECT_FALSE(m.mask[i][j]) << "prefix row " << i << " -> non-prefix " << j;
        }
    }

    // Non-prefix rows (3, 4, 5): causal — attend to [0..i].
    for (int i = 3; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_EQ(m.mask[i][j], (j <= i))
                << "non-prefix[" << i << "][" << j << "]";
        }
    }
}

TEST(AttentionMaskBuilder, PrefixLMFullPrefixEqualsFullMask) {
    // When prefix_len == seq_len, every row should be fully attended.
    int n = 4;
    auto m = AttentionMaskBuilder::build_prefix_lm(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_TRUE(m.mask[i][j]);
        }
    }
}

TEST(AttentionMaskBuilder, PrefixLMZeroPrefixEqualsCausal) {
    int n = 5;
    auto causal = AttentionMaskBuilder::build_causal(n);
    auto prefix = AttentionMaskBuilder::build_prefix_lm(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_EQ(causal.mask[i][j], prefix.mask[i][j]);
        }
    }
}

// ============================================================
// apply_mask
// ============================================================

TEST(AttentionMaskBuilder, ApplyMaskReplacesCorrectly) {
    // 2×2 causal: masked position is [0][1].
    auto m = AttentionMaskBuilder::build_causal(2);
    std::vector<float> scores = {1.0f, 2.0f, 3.0f, 4.0f};
    auto result = AttentionMaskBuilder::apply_mask(scores, m, -1e9f);
    ASSERT_EQ(result.size(), 4u);
    // [0][0] = 1.0 (attended)
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    // [0][1] = -1e9 (masked)
    EXPECT_FLOAT_EQ(result[1], -1e9f);
    // [1][0] = 3.0 (attended)
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    // [1][1] = 4.0 (attended)
    EXPECT_FLOAT_EQ(result[3], 4.0f);
}

TEST(AttentionMaskBuilder, ApplyMaskFullPreservesAll) {
    auto m = AttentionMaskBuilder::build_full(3);
    std::vector<float> scores(9, 1.0f);
    auto result = AttentionMaskBuilder::apply_mask(scores, m, -1e9f);
    for (float v : result) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

// ============================================================
// visualize
// ============================================================

TEST(AttentionMaskBuilder, VisualizeContainsOnesAndZeros) {
    auto m = AttentionMaskBuilder::build_causal(3);
    std::string viz = AttentionMaskBuilder::visualize(m);
    EXPECT_FALSE(viz.empty());
    // Should contain '1' and '0'.
    EXPECT_NE(viz.find('1'), std::string::npos);
    EXPECT_NE(viz.find('0'), std::string::npos);
}

#endif // LLMQUANT_ATTENTION_MASK_ENABLED
