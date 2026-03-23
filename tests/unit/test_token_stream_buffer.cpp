#include <gtest/gtest.h>

#ifdef LLMQUANT_TOKEN_STREAM_BUFFER_ENABLED

#include "TokenStreamBuffer.h"
#include <thread>
#include <vector>
#include <atomic>

using namespace llmquant;

namespace {

// ============================================================
// Helpers
// ============================================================

static BufferedToken make_token(std::string s, double w = 0.0) {
    BufferedToken t;
    t.token    = std::move(s);
    t.weight   = w;
    return t;
}

// ============================================================
// Construction and defaults
// ============================================================

TEST(TokenStreamBuffer, DefaultConstructsOk) {
    TokenStreamBuffer buf;
    EXPECT_EQ(buf.size(), 0);
    EXPECT_GT(buf.capacity(), 0);
    EXPECT_NEAR(buf.fill_fraction(), 0.0, 1e-9);
    EXPECT_EQ(buf.enqueued(), 0u);
    EXPECT_EQ(buf.dequeued(), 0u);
    EXPECT_EQ(buf.dropped(), 0u);
    EXPECT_FALSE(buf.is_shutdown());
}

TEST(TokenStreamBuffer, CustomCapacity) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 8;
    TokenStreamBuffer buf(cfg);
    EXPECT_EQ(buf.capacity(), 8);
}

// ============================================================
// Basic enqueue / dequeue
// ============================================================

TEST(TokenStreamBuffer, EnqueueDequeueRoundtrip) {
    TokenStreamBuffer buf;
    EXPECT_TRUE(buf.enqueue("hello", 0.5));
    EXPECT_EQ(buf.size(), 1);
    EXPECT_EQ(buf.enqueued(), 1u);

    auto tok = buf.dequeue();
    ASSERT_TRUE(tok.has_value());
    EXPECT_EQ(tok->token, "hello");
    EXPECT_NEAR(tok->weight, 0.5, 1e-9);
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.dequeued(), 1u);
}

TEST(TokenStreamBuffer, DequeueEmptyReturnsNullopt) {
    TokenStreamBuffer buf;
    auto tok = buf.dequeue();
    EXPECT_FALSE(tok.has_value());
}

TEST(TokenStreamBuffer, PreservesOrder) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 16;
    TokenStreamBuffer buf(cfg);

    for (int i = 0; i < 5; ++i) {
        buf.enqueue(std::to_string(i));
    }

    for (int i = 0; i < 5; ++i) {
        auto tok = buf.dequeue();
        ASSERT_TRUE(tok.has_value());
        EXPECT_EQ(tok->token, std::to_string(i));
    }
}

// ============================================================
// Drop-Oldest mode
// ============================================================

TEST(TokenStreamBuffer, DropOldestMode) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 3;
    cfg.mode     = TokenStreamBuffer::BackpressureMode::DROP_OLDEST;
    TokenStreamBuffer buf(cfg);

    buf.enqueue("A");
    buf.enqueue("B");
    buf.enqueue("C");
    EXPECT_EQ(buf.size(), 3);

    // Buffer full — enqueue "D" should evict "A".
    int drop_count = 0;
    cfg.on_drop = [&](const BufferedToken&) { ++drop_count; };
    buf.update_config(cfg);  // Resets the buffer, re-fill.

    buf.enqueue("A");
    buf.enqueue("B");
    buf.enqueue("C");
    bool result = buf.enqueue("D");
    EXPECT_TRUE(result);
    EXPECT_EQ(buf.dropped(), 1u);

    auto tok = buf.dequeue();
    ASSERT_TRUE(tok.has_value());
    // "A" was evicted; "B" is now oldest.
    EXPECT_EQ(tok->token, "B");
}

// ============================================================
// Drop-Newest mode
// ============================================================

TEST(TokenStreamBuffer, DropNewestMode) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 2;
    cfg.mode     = TokenStreamBuffer::BackpressureMode::DROP_NEWEST;
    TokenStreamBuffer buf(cfg);

    buf.enqueue("A");
    buf.enqueue("B");
    // Buffer full.
    bool result = buf.enqueue("C");
    EXPECT_FALSE(result);
    EXPECT_EQ(buf.dropped(), 1u);
    EXPECT_EQ(buf.size(), 2);

    auto t1 = buf.dequeue();
    auto t2 = buf.dequeue();
    EXPECT_EQ(t1->token, "A");
    EXPECT_EQ(t2->token, "B");
}

// ============================================================
// Reject mode
// ============================================================

TEST(TokenStreamBuffer, RejectMode) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 2;
    cfg.mode     = TokenStreamBuffer::BackpressureMode::REJECT;
    TokenStreamBuffer buf(cfg);

    EXPECT_TRUE(buf.enqueue("A"));
    EXPECT_TRUE(buf.enqueue("B"));
    EXPECT_FALSE(buf.enqueue("C"));  // rejected
    EXPECT_EQ(buf.dropped(), 1u);
    EXPECT_EQ(buf.size(), 2);
}

// ============================================================
// Fill fraction and peak fill
// ============================================================

TEST(TokenStreamBuffer, FillFractionTracked) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 4;
    cfg.mode     = TokenStreamBuffer::BackpressureMode::REJECT;
    TokenStreamBuffer buf(cfg);

    EXPECT_NEAR(buf.fill_fraction(), 0.0, 1e-9);
    buf.enqueue("A");
    EXPECT_NEAR(buf.fill_fraction(), 0.25, 1e-9);
    buf.enqueue("B");
    buf.enqueue("C");
    buf.enqueue("D");
    EXPECT_NEAR(buf.fill_fraction(), 1.0, 1e-9);
    EXPECT_EQ(buf.peak_fill(), 4);
}

// ============================================================
// High-watermark callback
// ============================================================

TEST(TokenStreamBuffer, HighWatermarkCallbackFires) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity       = 4;
    cfg.mode           = TokenStreamBuffer::BackpressureMode::REJECT;
    cfg.high_watermark = 0.5;  // fires at 2/4 = 50%

    int hwm_fires    = 0;
    int last_fill    = 0;
    cfg.on_high_watermark = [&](int fill, int /*cap*/) {
        ++hwm_fires;
        last_fill = fill;
    };

    TokenStreamBuffer buf(cfg);
    buf.enqueue("A");
    buf.enqueue("B");  // ← crosses 50% watermark

    EXPECT_GE(hwm_fires, 1);
    EXPECT_GE(last_fill, 2);
}

// ============================================================
// Drain
// ============================================================

TEST(TokenStreamBuffer, DrainReturnsCorrectCount) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 16;
    TokenStreamBuffer buf(cfg);

    for (int i = 0; i < 10; ++i) buf.enqueue(std::to_string(i));

    std::vector<BufferedToken> out;
    int drained = buf.drain(out, 5);
    EXPECT_EQ(drained, 5);
    EXPECT_EQ(out.size(), 5u);
    EXPECT_EQ(buf.size(), 5);
}

TEST(TokenStreamBuffer, DrainAll) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 8;
    TokenStreamBuffer buf(cfg);

    for (int i = 0; i < 6; ++i) buf.enqueue(std::to_string(i));

    std::vector<BufferedToken> out;
    int drained = buf.drain(out, 100);
    EXPECT_EQ(drained, 6);
    EXPECT_EQ(buf.size(), 0);
}

// ============================================================
// Clear
// ============================================================

TEST(TokenStreamBuffer, ClearResetsAll) {
    TokenStreamBuffer buf;
    buf.enqueue("X");
    buf.enqueue("Y");
    buf.clear();
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.enqueued(), 0u);
    EXPECT_EQ(buf.dequeued(), 0u);
    EXPECT_EQ(buf.dropped(), 0u);
    EXPECT_FALSE(buf.is_shutdown());
}

// ============================================================
// Shutdown
// ============================================================

TEST(TokenStreamBuffer, ShutdownBlocksNoFurtherEnqueue) {
    TokenStreamBuffer buf;
    buf.shutdown();
    EXPECT_TRUE(buf.is_shutdown());
    EXPECT_FALSE(buf.enqueue("after_shutdown"));
}

TEST(TokenStreamBuffer, DequeueWaitReturnsNulloptAfterShutdown) {
    TokenStreamBuffer buf;
    buf.shutdown();
    auto tok = buf.dequeue_wait(50);
    EXPECT_FALSE(tok.has_value());
}

// ============================================================
// Concurrent producer / consumer (smoke test)
// ============================================================

TEST(TokenStreamBuffer, ConcurrentProducerConsumer) {
    TokenStreamBuffer::Config cfg;
    cfg.capacity = 64;
    cfg.mode     = TokenStreamBuffer::BackpressureMode::DROP_OLDEST;
    TokenStreamBuffer buf(cfg);

    constexpr int N = 500;
    std::atomic<int> consumed{0};

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            buf.enqueue(std::to_string(i));
        }
    });

    std::thread consumer([&] {
        int got = 0;
        while (got < N) {
            auto tok = buf.dequeue_wait(200);
            if (tok) ++got;
        }
        consumed.store(got);
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(consumed.load(), N);
}

// ============================================================
// Stats JSON contains expected keys
// ============================================================

TEST(TokenStreamBuffer, StatsJsonContainsKeys) {
    TokenStreamBuffer buf;
    buf.enqueue("tok");
    std::string json = buf.to_stats_json();
    EXPECT_NE(json.find("capacity"),      std::string::npos);
    EXPECT_NE(json.find("fill"),          std::string::npos);
    EXPECT_NE(json.find("enqueued"),      std::string::npos);
    EXPECT_NE(json.find("dropped"),       std::string::npos);
    EXPECT_NE(json.find("fill_fraction"), std::string::npos);
}

} // namespace

#else

TEST(TokenStreamBuffer, FeatureDisabled) {
    GTEST_SKIP() << "LLMQUANT_TOKEN_STREAM_BUFFER_ENABLED not defined";
}

#endif // LLMQUANT_TOKEN_STREAM_BUFFER_ENABLED
