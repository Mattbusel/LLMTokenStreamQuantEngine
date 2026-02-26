#include "gtest/gtest.h"
#include "Deduplicator.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;
using ms = std::chrono::milliseconds;

// ---------------------------------------------------------------------------
// DedupKey tests
// ---------------------------------------------------------------------------

TEST(DeduplicatorTest, test_dedup_key_from_token_is_deterministic) {
    auto k1 = DedupKey::from_token("bullish", "ctx");
    auto k2 = DedupKey::from_token("bullish", "ctx");
    EXPECT_EQ(k1.value, k2.value);
}

TEST(DeduplicatorTest, test_dedup_key_different_tokens_produce_different_keys) {
    auto k1 = DedupKey::from_token("bullish");
    auto k2 = DedupKey::from_token("bearish");
    EXPECT_NE(k1.value, k2.value);
}

TEST(DeduplicatorTest, test_dedup_key_context_differentiates_same_token) {
    auto k1 = DedupKey::from_token("neutral", "market");
    auto k2 = DedupKey::from_token("neutral", "sentiment");
    EXPECT_NE(k1.value, k2.value);
}

TEST(DeduplicatorTest, test_dedup_key_empty_context_is_stable) {
    auto k1 = DedupKey::from_token("rally");
    auto k2 = DedupKey::from_token("rally", "");
    EXPECT_EQ(k1.value, k2.value);
}

// ---------------------------------------------------------------------------
// InProcessDeduplicator tests
// ---------------------------------------------------------------------------

TEST(DeduplicatorTest, test_in_process_dedup_novel_on_first_call) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("crash");
    EXPECT_EQ(dedup.check_and_register(key, ms{500}), DedupResult::Novel);
}

TEST(DeduplicatorTest, test_in_process_dedup_duplicate_on_second_call_within_ttl) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("surge");
    dedup.check_and_register(key, ms{5000});
    EXPECT_EQ(dedup.check_and_register(key, ms{5000}), DedupResult::Duplicate);
}

TEST(DeduplicatorTest, test_in_process_dedup_novel_after_ttl_expiry) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("volatile");
    dedup.check_and_register(key, ms{1});
    // Sleep longer than the TTL to ensure expiry.
    std::this_thread::sleep_for(ms{10});
    EXPECT_EQ(dedup.check_and_register(key, ms{500}), DedupResult::Novel);
}

TEST(DeduplicatorTest, test_in_process_dedup_evict_allows_reregistration) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("panic");
    dedup.check_and_register(key, ms{5000});
    dedup.evict(key);
    EXPECT_EQ(dedup.check_and_register(key, ms{5000}), DedupResult::Novel);
}

TEST(DeduplicatorTest, test_in_process_dedup_size_tracks_entries) {
    InProcessDeduplicator dedup;
    EXPECT_EQ(dedup.size(), 0u);
    dedup.check_and_register(DedupKey::from_token("a"), ms{5000});
    EXPECT_EQ(dedup.size(), 1u);
    dedup.check_and_register(DedupKey::from_token("b"), ms{5000});
    EXPECT_EQ(dedup.size(), 2u);
}

TEST(DeduplicatorTest, test_in_process_dedup_purge_expired_removes_stale_entries) {
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("plunge");
    dedup.check_and_register(key, ms{1});
    EXPECT_EQ(dedup.size(), 1u);
    std::this_thread::sleep_for(ms{10});
    dedup.purge_expired();
    EXPECT_EQ(dedup.size(), 0u);
}

TEST(DeduplicatorTest, test_in_process_dedup_concurrent_calls_are_safe) {
    // Four threads each check the same key 100 times.
    // Only the very first check across all threads should be Novel; all
    // subsequent ones should be Duplicate (TTL is long enough).
    InProcessDeduplicator dedup;
    auto key = DedupKey::from_token("concurrent_key");
    std::atomic<uint64_t> novel_count{0};

    constexpr int kThreads = 4;
    constexpr int kIters   = 100;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIters; ++i) {
                auto result = dedup.check_and_register(key, ms{60000});
                if (result == DedupResult::Novel) novel_count++;
            }
        });
    }
    for (auto& th : threads) th.join();

    // Exactly one thread wins the first registration; all others see Duplicate.
    EXPECT_EQ(novel_count.load(), 1u);
    EXPECT_EQ(dedup.total_duplicates(), static_cast<uint64_t>(kThreads * kIters - 1));
}

// ---------------------------------------------------------------------------
// RedisDeduplicator stub tests
// ---------------------------------------------------------------------------

TEST(DeduplicatorTest, test_redis_stub_delegates_to_inner) {
    RedisDeduplicator redis("redis://127.0.0.1:6379/0");
    auto key = DedupKey::from_token("rally");
    EXPECT_EQ(redis.check_and_register(key, ms{5000}), DedupResult::Novel);
    EXPECT_EQ(redis.check_and_register(key, ms{5000}), DedupResult::Duplicate);
}

TEST(DeduplicatorTest, test_redis_stub_stores_redis_url) {
    const std::string url = "redis://192.168.1.1:6379/1";
    RedisDeduplicator redis(url);
    EXPECT_EQ(redis.redis_url(), url);
}

TEST(DeduplicatorTest, test_redis_stub_evict_allows_reregistration) {
    RedisDeduplicator redis("redis://127.0.0.1:6379/0");
    auto key = DedupKey::from_token("breakout");
    redis.check_and_register(key, ms{5000});
    redis.evict(key);
    EXPECT_EQ(redis.check_and_register(key, ms{5000}), DedupResult::Novel);
}

// ---------------------------------------------------------------------------
// Deduplicator facade tests
// ---------------------------------------------------------------------------

TEST(DeduplicatorTest, test_deduplicator_facade_check_convenience) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, ms{5000});
    EXPECT_EQ(dedup.check("momentum"),              DedupResult::Novel);
    EXPECT_EQ(dedup.check("momentum"),              DedupResult::Duplicate);
    EXPECT_EQ(dedup.check("momentum", "equity"),    DedupResult::Novel);
}

TEST(DeduplicatorTest, test_deduplicator_facade_evict) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, ms{5000});
    dedup.check("support");
    dedup.evict("support");
    EXPECT_EQ(dedup.check("support"), DedupResult::Novel);
}

TEST(DeduplicatorTest, test_deduplicator_total_duplicate_count_accumulates) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, ms{5000});

    dedup.check("resistance");  // novel
    dedup.check("resistance");  // duplicate
    dedup.check("resistance");  // duplicate
    dedup.check("momentum");    // novel
    dedup.check("momentum");    // duplicate

    EXPECT_EQ(backend->total_duplicates(), 3u);
    EXPECT_EQ(backend->total_novel(),      2u);
}

TEST(DeduplicatorTest, test_deduplicator_check_with_custom_ttl) {
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, ms{5000});
    auto key = DedupKey::from_token("short_lived");
    EXPECT_EQ(dedup.check_with_ttl(key, ms{1}), DedupResult::Novel);
    std::this_thread::sleep_for(ms{10});
    EXPECT_EQ(dedup.check_with_ttl(key, ms{5000}), DedupResult::Novel);
}
