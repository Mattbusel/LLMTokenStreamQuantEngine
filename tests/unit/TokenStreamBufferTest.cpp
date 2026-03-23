#include <gtest/gtest.h>
#include "TokenStreamBuffer.h"
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

using namespace llmquant;

// ============================================================
// SimpleTokenStreamBuffer Tests
// ============================================================

TEST(SimpleTokenStreamBuffer, ConstructsEmpty) {
    SimpleTokenStreamBuffer buf(10);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.capacity(), 10u);
    EXPECT_TRUE(buf.empty());
    EXPECT_FALSE(buf.full());
}

TEST(SimpleTokenStreamBuffer, PushAndPop) {
    SimpleTokenStreamBuffer buf(4);
    EXPECT_TRUE(buf.push(42));
    EXPECT_EQ(buf.size(), 1u);
    int out = 0;
    EXPECT_TRUE(buf.pop(out));
    EXPECT_EQ(out, 42);
    EXPECT_TRUE(buf.empty());
}

TEST(SimpleTokenStreamBuffer, FIFOOrder) {
    SimpleTokenStreamBuffer buf(4);
    buf.push(1);
    buf.push(2);
    buf.push(3);
    int out;
    buf.pop(out); EXPECT_EQ(out, 1);
    buf.pop(out); EXPECT_EQ(out, 2);
    buf.pop(out); EXPECT_EQ(out, 3);
}

TEST(SimpleTokenStreamBuffer, PushReturnsFalseWhenFull) {
    SimpleTokenStreamBuffer buf(2);
    EXPECT_TRUE(buf.push(1));
    EXPECT_TRUE(buf.push(2));
    EXPECT_FALSE(buf.push(3)); // full
    EXPECT_TRUE(buf.full());
}

TEST(SimpleTokenStreamBuffer, PopReturnsFalseWhenEmpty) {
    SimpleTokenStreamBuffer buf(4);
    int out;
    EXPECT_FALSE(buf.pop(out));
}

TEST(SimpleTokenStreamBuffer, Peek) {
    SimpleTokenStreamBuffer buf(4);
    buf.push(99);
    int out = 0;
    EXPECT_TRUE(buf.peek(out));
    EXPECT_EQ(out, 99);
    EXPECT_EQ(buf.size(), 1u); // not removed
}

TEST(SimpleTokenStreamBuffer, PeekEmptyReturnsFalse) {
    SimpleTokenStreamBuffer buf(4);
    int out;
    EXPECT_FALSE(buf.peek(out));
}

TEST(SimpleTokenStreamBuffer, Clear) {
    SimpleTokenStreamBuffer buf(4);
    buf.push(1);
    buf.push(2);
    buf.clear();
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
}

TEST(SimpleTokenStreamBuffer, Drain) {
    SimpleTokenStreamBuffer buf(4);
    buf.push(10);
    buf.push(20);
    buf.push(30);
    auto drained = buf.drain();
    EXPECT_EQ(drained.size(), 3u);
    EXPECT_EQ(drained[0], 10);
    EXPECT_EQ(drained[1], 20);
    EXPECT_EQ(drained[2], 30);
    EXPECT_TRUE(buf.empty());
}

TEST(SimpleTokenStreamBuffer, PushBatch) {
    SimpleTokenStreamBuffer buf(8);
    std::vector<int> tokens = {1, 2, 3, 4};
    EXPECT_TRUE(buf.push_batch(tokens));
    EXPECT_EQ(buf.size(), 4u);
}

TEST(SimpleTokenStreamBuffer, PushBatchReturnsFalseOnFull) {
    SimpleTokenStreamBuffer buf(2);
    std::vector<int> tokens = {1, 2, 3};
    EXPECT_FALSE(buf.push_batch(tokens));
}

TEST(SimpleTokenStreamBuffer, CircularWrap) {
    SimpleTokenStreamBuffer buf(3);
    buf.push(1);
    buf.push(2);
    int out;
    buf.pop(out); // head moves
    buf.push(3);
    buf.push(4); // wraps around
    buf.pop(out); EXPECT_EQ(out, 2);
    buf.pop(out); EXPECT_EQ(out, 3);
    buf.pop(out); EXPECT_EQ(out, 4);
}

// ============================================================
// ThreadSafeTokenBuffer Tests
// ============================================================

TEST(ThreadSafeTokenBuffer, ConstructsEmpty) {
    ThreadSafeTokenBuffer buf(8);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.capacity(), 8u);
    EXPECT_TRUE(buf.empty());
    EXPECT_FALSE(buf.full());
}

TEST(ThreadSafeTokenBuffer, BasicPushPop) {
    ThreadSafeTokenBuffer buf(4);
    EXPECT_TRUE(buf.push(7));
    int out = 0;
    EXPECT_TRUE(buf.pop(out));
    EXPECT_EQ(out, 7);
}

TEST(ThreadSafeTokenBuffer, FullReturnsTrue) {
    ThreadSafeTokenBuffer buf(2);
    buf.push(1);
    buf.push(2);
    EXPECT_TRUE(buf.full());
    EXPECT_FALSE(buf.push(3));
}

TEST(ThreadSafeTokenBuffer, Drain) {
    ThreadSafeTokenBuffer buf(4);
    buf.push(1); buf.push(2); buf.push(3);
    auto v = buf.drain();
    EXPECT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 1);
    EXPECT_TRUE(buf.empty());
}

TEST(ThreadSafeTokenBuffer, WaitAndPop_Timeout) {
    ThreadSafeTokenBuffer buf(4);
    int out = -1;
    bool got = buf.wait_and_pop(out, std::chrono::milliseconds(50));
    EXPECT_FALSE(got);
    EXPECT_EQ(out, -1);
}

TEST(ThreadSafeTokenBuffer, WaitAndPopGetsToken) {
    ThreadSafeTokenBuffer buf(4);
    buf.push(55);
    int out = 0;
    bool got = buf.wait_and_pop(out, std::chrono::milliseconds(100));
    EXPECT_TRUE(got);
    EXPECT_EQ(out, 55);
}

TEST(ThreadSafeTokenBuffer, WaitAndPushFromThread) {
    ThreadSafeTokenBuffer buf(2);
    buf.push(1);
    buf.push(2); // full

    std::atomic<bool> pushed{false};
    std::thread producer([&]() {
        buf.wait_and_push(3);
        pushed.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    // Consumer pops to make room
    int out;
    buf.pop(out);

    producer.join();
    EXPECT_TRUE(pushed.load());
}

TEST(ThreadSafeTokenBuffer, ConcurrentPushPop) {
    ThreadSafeTokenBuffer buf(100);
    constexpr int N = 50;
    std::atomic<int> popped{0};

    std::thread producer([&]() {
        for (int i = 0; i < N; ++i) {
            buf.push(i);
        }
    });

    std::thread consumer([&]() {
        int out;
        int count = 0;
        while (count < N) {
            if (buf.pop(out)) {
                ++count;
                ++popped;
            }
        }
    });

    producer.join();
    consumer.join();
    EXPECT_EQ(popped.load(), N);
}

TEST(ThreadSafeTokenBuffer, PushBatch) {
    ThreadSafeTokenBuffer buf(10);
    EXPECT_TRUE(buf.push_batch({1, 2, 3}));
    EXPECT_EQ(buf.size(), 3u);
}
