// test_StreamCompressor.cpp — 15 GTest tests for StreamCompressor.
#include "gtest/gtest.h"
#include "StreamCompressor.h"

#include <stdexcept>

using namespace llmquant;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> vec(std::initializer_list<const char*> il) {
    std::vector<std::string> v;
    for (const char* s : il) v.emplace_back(s);
    return v;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(StreamCompressorTest, RleCompressEmptyInput) {
    StreamCompressor sc;
    auto result = sc.rle_compress({});
    EXPECT_TRUE(result.empty());
}

TEST(StreamCompressorTest, RleCompressAllDistinct) {
    StreamCompressor sc;
    auto tokens = vec({"a", "b", "c"});
    auto compressed = sc.rle_compress(tokens);
    EXPECT_EQ(compressed.size(), 3u);
    for (const auto& ct : compressed) {
        EXPECT_EQ(ct.count, 1u);
        EXPECT_FALSE(ct.is_run);
    }
}

TEST(StreamCompressorTest, RleCompressRunCollapsed) {
    StreamCompressor sc;
    auto tokens = vec({"x", "x", "x", "y"});
    auto compressed = sc.rle_compress(tokens);
    EXPECT_EQ(compressed.size(), 2u);
    EXPECT_TRUE(compressed[0].is_run);
    EXPECT_EQ(compressed[0].count, 3u);
    EXPECT_EQ(compressed[0].token, "x");
    EXPECT_EQ(compressed[1].token, "y");
}

TEST(StreamCompressorTest, RleDecompressRoundTrip) {
    StreamCompressor sc;
    auto tokens = vec({"foo", "foo", "bar", "baz", "baz", "baz"});
    auto compressed = sc.rle_compress(tokens);
    auto decompressed = sc.rle_decompress(compressed);
    EXPECT_EQ(decompressed, tokens);
}

TEST(StreamCompressorTest, RleDecompressEmptyInput) {
    StreamCompressor sc;
    auto result = sc.rle_decompress({});
    EXPECT_TRUE(result.empty());
}

TEST(StreamCompressorTest, CompressionRatioAllSame) {
    StreamCompressor sc;
    auto tokens = vec({"tok", "tok", "tok", "tok", "tok"});
    auto compressed = sc.rle_compress(tokens);
    double ratio = sc.compression_ratio(tokens, compressed);
    EXPECT_DOUBLE_EQ(ratio, 5.0);  // 5 tokens / 1 compressed entry
}

TEST(StreamCompressorTest, CompressionRatioNoReduction) {
    StreamCompressor sc;
    auto tokens = vec({"a", "b", "c"});
    auto compressed = sc.rle_compress(tokens);
    double ratio = sc.compression_ratio(tokens, compressed);
    EXPECT_DOUBLE_EQ(ratio, 1.0);
}

TEST(StreamCompressorTest, CompressionRatioEmptyCompressed) {
    StreamCompressor sc;
    double ratio = sc.compression_ratio({}, {});
    EXPECT_DOUBLE_EQ(ratio, 1.0);
}

TEST(StreamCompressorTest, DictCompressBasic) {
    StreamCompressor sc;
    auto tokens = vec({"the", "cat", "sat", "the", "cat"});
    auto [indices, dict] = sc.dict_compress(tokens);
    EXPECT_EQ(indices.size(), 5u);
    EXPECT_EQ(dict.size(), 3u);  // "the", "cat", "sat"
    EXPECT_EQ(indices[0], indices[3]);  // both "the"
    EXPECT_EQ(indices[1], indices[4]);  // both "cat"
}

TEST(StreamCompressorTest, DictCompressEmptyInput) {
    StreamCompressor sc;
    auto [indices, dict] = sc.dict_compress({});
    EXPECT_TRUE(indices.empty());
    EXPECT_TRUE(dict.empty());
}

TEST(StreamCompressorTest, DictDecompressRoundTrip) {
    StreamCompressor sc;
    auto tokens = vec({"alpha", "beta", "alpha", "gamma"});
    auto [indices, dict] = sc.dict_compress(tokens);
    auto reconstructed = sc.dict_decompress(indices, dict);
    EXPECT_EQ(reconstructed, tokens);
}

TEST(StreamCompressorTest, DictDecompressEmptyInput) {
    StreamCompressor sc;
    auto result = sc.dict_decompress({}, {});
    EXPECT_TRUE(result.empty());
}

TEST(StreamCompressorTest, DictDecompressOutOfRangeThrows) {
    StreamCompressor sc;
    std::vector<std::size_t> bad_indices = {0, 99};
    std::vector<std::string> dict        = {"only_one"};
    EXPECT_THROW(sc.dict_decompress(bad_indices, dict), std::out_of_range);
}

TEST(StreamCompressorTest, RleCompressSingleToken) {
    StreamCompressor sc;
    auto tokens = vec({"solo"});
    auto compressed = sc.rle_compress(tokens);
    ASSERT_EQ(compressed.size(), 1u);
    EXPECT_EQ(compressed[0].token, "solo");
    EXPECT_EQ(compressed[0].count, 1u);
    EXPECT_FALSE(compressed[0].is_run);
}

TEST(StreamCompressorTest, DictCompressPreservesOrder) {
    StreamCompressor sc;
    auto tokens = vec({"z", "a", "m", "z", "a"});
    auto [indices, dict] = sc.dict_compress(tokens);
    // First encountered order: z=0, a=1, m=2
    EXPECT_EQ(dict[0], "z");
    EXPECT_EQ(dict[1], "a");
    EXPECT_EQ(dict[2], "m");
    EXPECT_EQ(indices[0], 0u);
    EXPECT_EQ(indices[1], 1u);
    EXPECT_EQ(indices[2], 2u);
}

}  // namespace
