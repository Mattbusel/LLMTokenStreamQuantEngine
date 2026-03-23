#include <gtest/gtest.h>

#ifdef LLMQUANT_CONTEXT_ENCODER_ENABLED

#include "ContextEncoder.h"
#include <cmath>
#include <vector>

using namespace llmquant;

static constexpr float kEps = 1e-4f;

// ============================================================
// Construction
// ============================================================

TEST(ContextEncoder, ConstructsWithValidParams) {
    EXPECT_NO_THROW(ContextEncoder enc(512, 64));
}

TEST(ContextEncoder, ThrowsOnInvalidMaxSeqLen) {
    EXPECT_THROW(ContextEncoder(0, 64), std::invalid_argument);
}

TEST(ContextEncoder, ThrowsOnInvalidEmbedDim) {
    EXPECT_THROW(ContextEncoder(512, 0), std::invalid_argument);
}

// ============================================================
// sinusoidal_pe
// ============================================================

TEST(ContextEncoder, SinusoidalPEPositionZeroEvenIsZero) {
    ContextEncoder enc(128, 16);
    // PE(0, 2i) = sin(0) = 0
    for (int d = 0; d < 16; d += 2) {
        EXPECT_NEAR(enc.sinusoidal_pe(0, d), 0.0f, kEps)
            << "PE(0, " << d << ") should be 0";
    }
}

TEST(ContextEncoder, SinusoidalPEPositionZeroOddIsOne) {
    ContextEncoder enc(128, 16);
    // PE(0, 2i+1) = cos(0) = 1
    for (int d = 1; d < 16; d += 2) {
        EXPECT_NEAR(enc.sinusoidal_pe(0, d), 1.0f, kEps)
            << "PE(0, " << d << ") should be 1";
    }
}

TEST(ContextEncoder, SinusoidalPEBoundedInMinusOneToOne) {
    ContextEncoder enc(128, 32);
    for (int pos = 0; pos < 50; ++pos) {
        for (int d = 0; d < 32; ++d) {
            float v = enc.sinusoidal_pe(pos, d);
            EXPECT_GE(v, -1.01f) << "pos=" << pos << " d=" << d;
            EXPECT_LE(v, 1.01f)  << "pos=" << pos << " d=" << d;
        }
    }
}

// ============================================================
// encode_position
// ============================================================

TEST(ContextEncoder, EncodePositionReturnsCorrectSize) {
    ContextEncoder enc(128, 32);
    auto pe = enc.encode_position(5);
    EXPECT_EQ(static_cast<int>(pe.size()), 32);
}

TEST(ContextEncoder, EncodePositionDeterministic) {
    ContextEncoder enc(128, 16);
    auto pe1 = enc.encode_position(10);
    auto pe2 = enc.encode_position(10);
    ASSERT_EQ(pe1.size(), pe2.size());
    for (std::size_t i = 0; i < pe1.size(); ++i) {
        EXPECT_FLOAT_EQ(pe1[i], pe2[i]);
    }
}

TEST(ContextEncoder, DifferentPositionsDifferentEncodings) {
    ContextEncoder enc(128, 32);
    auto pe0 = enc.encode_position(0);
    auto pe1 = enc.encode_position(1);
    bool any_diff = false;
    for (std::size_t i = 0; i < pe0.size(); ++i) {
        if (std::abs(pe0[i] - pe1[i]) > kEps) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

// ============================================================
// encode_token
// ============================================================

TEST(ContextEncoder, EncodeTokenReturnsCorrectSize) {
    ContextEncoder enc(128, 32);
    auto tok = enc.encode_token(42, 5);
    EXPECT_EQ(static_cast<int>(tok.size()), 32);
}

TEST(ContextEncoder, EncodeTokenDeterministic) {
    ContextEncoder enc(128, 16);
    auto t1 = enc.encode_token(100, 3);
    auto t2 = enc.encode_token(100, 3);
    ASSERT_EQ(t1.size(), t2.size());
    for (std::size_t i = 0; i < t1.size(); ++i) {
        EXPECT_FLOAT_EQ(t1[i], t2[i]);
    }
}

// ============================================================
// encode_sequence
// ============================================================

TEST(ContextEncoder, EncodeSequenceCorrectLength) {
    ContextEncoder enc(128, 16);
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    auto result = enc.encode_sequence(tokens);
    EXPECT_EQ(result.size(), tokens.size());
    for (const auto& v : result) {
        EXPECT_EQ(static_cast<int>(v.size()), 16);
    }
}

// ============================================================
// RoPE
// ============================================================

TEST(ContextEncoder, RopeAnglePositionZeroIsZero) {
    ContextEncoder enc(128, 16);
    EXPECT_NEAR(enc.rope_angle(0, 0), 0.0f, kEps);
}

TEST(ContextEncoder, ApplyRopePositionZeroIdentity) {
    ContextEncoder enc(128, 16);
    // At position 0, angle=0, cos=1, sin=0 => q_rot=q, k_rot=k
    auto [q_rot, k_rot] = enc.apply_rope(1.0f, 0.5f, 0, 0);
    EXPECT_NEAR(q_rot, 1.0f, kEps);
    EXPECT_NEAR(k_rot, 0.5f, kEps);
}

// ============================================================
// ALiBi
// ============================================================

TEST(ContextEncoder, AlibiBiasCorrectSize) {
    ContextEncoder enc(128, 16);
    auto bias = enc.alibi_bias(10, 0);
    EXPECT_EQ(static_cast<int>(bias.size()), 10);
}

TEST(ContextEncoder, AlibiBiasLastElementIsZero) {
    ContextEncoder enc(128, 16);
    int seq_len = 8;
    auto bias = enc.alibi_bias(seq_len, 0);
    // Last position query at seq_len-1: bias[seq_len-1] = -|0| * slope = 0
    EXPECT_NEAR(bias[static_cast<std::size_t>(seq_len - 1)], 0.0f, kEps);
}

TEST(ContextEncoder, AlibiBiasMonotonicallyDecreasesFromEnd) {
    ContextEncoder enc(128, 16);
    auto bias = enc.alibi_bias(6, 1);
    // bias[i] = -|5-i| * slope, should increase toward last element
    for (int i = 0; i < 5; ++i) {
        EXPECT_LE(bias[static_cast<std::size_t>(i)],
                  bias[static_cast<std::size_t>(i + 1)] + kEps);
    }
}

TEST(ContextEncoder, AlibiBiasNonPositiveValues) {
    ContextEncoder enc(128, 16);
    auto bias = enc.alibi_bias(5, 2);
    for (const auto& v : bias) {
        EXPECT_LE(v, 0.0f + kEps);
    }
}

#else
// Stub tests when feature is disabled
TEST(ContextEncoder, FeatureDisabled) {
    SUCCEED();
}
#endif // LLMQUANT_CONTEXT_ENCODER_ENABLED
