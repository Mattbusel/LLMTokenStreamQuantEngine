#include <gtest/gtest.h>

#ifdef LLMQUANT_POSITIONAL_ENCODING_ENABLED

#include "PositionalEncoding.h"
#include <cmath>
#include <numeric>

using namespace llmquant;

static constexpr float EPS = 1e-5f;

// ============================================================
// SinusoidalEncoding
// ============================================================

TEST(SinusoidalEncoding, PositionZeroDeterministic) {
    SinusoidalEncoding enc;
    auto pe1 = enc.encode(0, 16);
    auto pe2 = enc.encode(0, 16);
    ASSERT_EQ(pe1.size(), pe2.size());
    for (size_t i = 0; i < pe1.size(); ++i) {
        EXPECT_FLOAT_EQ(pe1[i], pe2[i]);
    }
}

TEST(SinusoidalEncoding, PositionZeroSinIsZero) {
    SinusoidalEncoding enc;
    auto pe = enc.encode(0, 8);
    // PE[0, 2k] = sin(0 / ...) = 0
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(pe[2 * k], 0.0f, EPS)
            << "PE[0, " << 2 * k << "] should be sin(0) = 0";
    }
}

TEST(SinusoidalEncoding, PositionZeroCosIsOne) {
    SinusoidalEncoding enc;
    auto pe = enc.encode(0, 8);
    // PE[0, 2k+1] = cos(0) = 1
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(pe[2 * k + 1], 1.0f, EPS)
            << "PE[0, " << 2 * k + 1 << "] should be cos(0) = 1";
    }
}

TEST(SinusoidalEncoding, DifferentPositionsDifferentEncodings) {
    SinusoidalEncoding enc;
    auto pe0 = enc.encode(0, 16);
    auto pe1 = enc.encode(1, 16);
    // They should not be identical.
    bool any_diff = false;
    for (size_t i = 0; i < pe0.size(); ++i) {
        if (std::abs(pe0[i] - pe1[i]) > EPS) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

// ============================================================
// LearnedEncoding
// ============================================================

TEST(LearnedEncoding, InitialisedWithSinusoidal) {
    LearnedEncoding learned(10, 8);
    SinusoidalEncoding sinusoidal;

    auto lpe = learned.encode(5, 8);
    auto spe = sinusoidal.encode(5, 8);
    ASSERT_EQ(lpe.size(), spe.size());
    for (size_t i = 0; i < lpe.size(); ++i) {
        EXPECT_NEAR(lpe[i], spe[i], EPS);
    }
}

TEST(LearnedEncoding, UpdateChangesEmbedding) {
    LearnedEncoding learned(10, 4);
    std::vector<float> new_emb = {1.0f, 2.0f, 3.0f, 4.0f};
    learned.update(3, new_emb);
    auto result = learned.encode(3, 4);
    for (size_t i = 0; i < new_emb.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], new_emb[i]);
    }
}

TEST(LearnedEncoding, UpdateDoesNotAffectOtherPositions) {
    LearnedEncoding learned(10, 4);
    SinusoidalEncoding sinusoidal;
    auto before = learned.encode(2, 4);

    std::vector<float> new_emb(4, 99.0f);
    learned.update(5, new_emb); // update a different position

    auto after = learned.encode(2, 4);
    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_FLOAT_EQ(before[i], after[i]);
    }
}

// ============================================================
// RotaryPositionalEncoding (RoPE)
// ============================================================

TEST(RoPE, PositionZeroIsIdentity) {
    RotaryPositionalEncoding rope(8);
    std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> k = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
    auto [rq, rk] = rope.apply_rope(q, k, 0);
    // At position 0: cos(0) = 1, sin(0) = 0, so rotation is identity.
    for (size_t i = 0; i < q.size(); ++i) {
        EXPECT_NEAR(rq[i], q[i], EPS);
        EXPECT_NEAR(rk[i], k[i], EPS);
    }
}

TEST(RoPE, PreservesNorm) {
    RotaryPositionalEncoding rope(8);
    std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> k = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};

    auto norm_sq = [](const std::vector<float>& v) {
        float s = 0.0f;
        for (float x : v) s += x * x;
        return s;
    };

    float q_norm_before = norm_sq(q);
    float k_norm_before = norm_sq(k);

    auto [rq, rk] = rope.apply_rope(q, k, 42);

    EXPECT_NEAR(norm_sq(rq), q_norm_before, 1e-3f);
    EXPECT_NEAR(norm_sq(rk), k_norm_before, 1e-3f);
}

TEST(RoPE, DifferentPositionsDifferentResults) {
    RotaryPositionalEncoding rope(8);
    std::vector<float> q(8, 1.0f);
    std::vector<float> k(8, 1.0f);

    auto [rq1, rk1] = rope.apply_rope(q, k, 1);
    auto [rq2, rk2] = rope.apply_rope(q, k, 10);

    bool any_diff = false;
    for (size_t i = 0; i < rq1.size(); ++i) {
        if (std::abs(rq1[i] - rq2[i]) > EPS) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

// ============================================================
// ALiBi
// ============================================================

TEST(ALiBi, SlopeDecreasesWithDistance) {
    AlibiPositionalBias alibi(4);
    auto biases = alibi.compute_bias(5, 4);
    ASSERT_EQ(biases.size(), 4u);

    for (int h = 0; h < 4; ++h) {
        const auto& b = biases[h];
        // For a fixed row i, |bias[i][j]| increases as |i-j| increases.
        int i = 2;
        float b_near = b[i * 5 + (i - 1)];  // distance 1
        float b_far  = b[i * 5 + 0];         // distance 2
        // All biases are ≤ 0.
        EXPECT_LE(b_near, 0.0f);
        EXPECT_LE(b_far, b_near) << "farther position should have more negative bias";
    }
}

TEST(ALiBi, DiagonalIsZero) {
    AlibiPositionalBias alibi(2);
    auto biases = alibi.compute_bias(4, 2);
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(biases[h][i * 4 + i], 0.0f)
                << "diagonal bias should be 0 for head " << h;
        }
    }
}

TEST(ALiBi, HeadSlopesDecrease) {
    // Higher-indexed heads have smaller slopes → less aggressive distance penalty.
    // slopes_[h] = 2^(-8*(h+1)/num_heads), so larger h → smaller slope (less negative bias per unit).
    AlibiPositionalBias alibi(4);
    auto biases = alibi.compute_bias(3, 4);
    // At distance 1, head 0 should have more negative bias than head 3.
    float b0 = biases[0][0 * 3 + 1]; // row 0, col 1, distance 1
    float b3 = biases[3][0 * 3 + 1];
    EXPECT_LT(b0, b3) << "head 0 should have more negative bias than head 3";
}

TEST(ALiBi, SymmetricAroundDiagonal) {
    AlibiPositionalBias alibi(1);
    auto biases = alibi.compute_bias(4, 1);
    const auto& b = biases[0];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(b[i * 4 + j], b[j * 4 + i])
                << "ALiBi bias should be symmetric";
        }
    }
}

#endif // LLMQUANT_POSITIONAL_ENCODING_ENABLED
