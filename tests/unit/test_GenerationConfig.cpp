// test_GenerationConfig.cpp — GTest suite for GenerationConfig (12 tests).

#include "GenerationConfig.h"

#include <gtest/gtest.h>
#include <string>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class GenerationConfigTest : public ::testing::Test {
protected:
    ConfigValidator validator;

    GenerationConfig valid_config() {
        return ConfigValidator::with_defaults();
    }
};

// ---------------------------------------------------------------------------
// with_defaults
// ---------------------------------------------------------------------------

TEST_F(GenerationConfigTest, WithDefaultsIsValid) {
    auto cfg = ConfigValidator::with_defaults();
    EXPECT_TRUE(validator.is_valid(cfg));
}

TEST_F(GenerationConfigTest, WithDefaultsHasPositiveTemperature) {
    auto cfg = ConfigValidator::with_defaults();
    EXPECT_GT(cfg.temperature, 0.0f);
}

// ---------------------------------------------------------------------------
// validate — invalid configs
// ---------------------------------------------------------------------------

TEST_F(GenerationConfigTest, ZeroMaxNewTokensIsInvalid) {
    auto cfg = valid_config();
    cfg.max_new_tokens = 0;
    EXPECT_FALSE(validator.is_valid(cfg));
}

TEST_F(GenerationConfigTest, ZeroTemperatureIsInvalid) {
    auto cfg = valid_config();
    cfg.temperature = 0.0f;
    EXPECT_FALSE(validator.is_valid(cfg));
}

TEST_F(GenerationConfigTest, TopPAboveOneIsInvalid) {
    auto cfg = valid_config();
    cfg.top_p = 1.5f;
    EXPECT_FALSE(validator.is_valid(cfg));
}

TEST_F(GenerationConfigTest, ZeroRepetitionPenaltyIsInvalid) {
    auto cfg = valid_config();
    cfg.repetition_penalty = 0.0f;
    EXPECT_FALSE(validator.is_valid(cfg));
}

TEST_F(GenerationConfigTest, BeamsGreaterThanOneWithSamplingIsInvalid) {
    auto cfg = valid_config();
    cfg.num_beams = 4;
    cfg.do_sample = true;
    EXPECT_FALSE(validator.is_valid(cfg));
}

TEST_F(GenerationConfigTest, SamePadAndEosTokenIsInvalid) {
    auto cfg = valid_config();
    cfg.pad_token_id = 5;
    cfg.eos_token_id = 5;
    EXPECT_FALSE(validator.is_valid(cfg));
}

// ---------------------------------------------------------------------------
// from_json / to_json
// ---------------------------------------------------------------------------

TEST_F(GenerationConfigTest, ToJsonContainsAllKeys) {
    auto cfg = valid_config();
    std::string json = ConfigValidator::to_json(cfg);
    EXPECT_NE(json.find("max_new_tokens"),    std::string::npos);
    EXPECT_NE(json.find("temperature"),       std::string::npos);
    EXPECT_NE(json.find("top_p"),             std::string::npos);
    EXPECT_NE(json.find("repetition_penalty"), std::string::npos);
}

TEST_F(GenerationConfigTest, FromJsonRoundTrip) {
    auto original = valid_config();
    std::string json = ConfigValidator::to_json(original);
    auto parsed = ConfigValidator::from_json(json);
    EXPECT_EQ(parsed.max_new_tokens,  original.max_new_tokens);
    EXPECT_FLOAT_EQ(parsed.temperature, original.temperature);
    EXPECT_EQ(parsed.num_beams,       original.num_beams);
    EXPECT_EQ(parsed.do_sample,       original.do_sample);
}

TEST_F(GenerationConfigTest, FromJsonPartialValues) {
    std::string json = R"({"max_new_tokens": 512, "temperature": 0.5})";
    auto cfg = ConfigValidator::from_json(json);
    EXPECT_EQ(cfg.max_new_tokens, 512u);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.5f);
}

TEST_F(GenerationConfigTest, FromJsonUnknownKeysIgnored) {
    std::string json = R"({"max_new_tokens": 64, "unknown_key": 999})";
    auto cfg = ConfigValidator::from_json(json);
    EXPECT_EQ(cfg.max_new_tokens, 64u);
    // Should not throw and should have default for other fields.
    EXPECT_GT(cfg.top_k, 0u);
}
