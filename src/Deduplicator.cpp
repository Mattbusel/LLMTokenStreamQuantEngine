#include "Deduplicator.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>

#ifdef LLMQUANT_REDIS_ENABLED
  #include <hiredis/hiredis.h>
#endif

namespace llmquant {

// ---------------------------------------------------------------------------
// DedupKey
// ---------------------------------------------------------------------------

DedupKey DedupKey::from_token(const std::string& token, const std::string& context) noexcept {
    // 128-bit dedup key: two independent FNV-1a 64-bit passes with different
    // seeds over "token|context".  Combining both halves gives collision
    // resistance of ~2^128 vs the original 2^64 single-pass approach.
    // Seed1 is the standard FNV-1a offset basis; seed2 is an alternate offset.
    constexpr uint64_t kSeed1 = 14695981039346656037ULL;
    constexpr uint64_t kSeed2 = 0xcbf29ce484222325ULL ^ 0xdeadbeefcafeULL;
    constexpr uint64_t kPrime = 1099511628211ULL;

    uint64_t lo = kSeed1;
    uint64_t hi = kSeed2;

    auto fnv_byte = [&](unsigned char b) {
        lo ^= static_cast<uint64_t>(b);
        lo *= kPrime;
        // Second pass: mix the byte differently for independence.
        hi ^= static_cast<uint64_t>(b) ^ (lo >> 32);
        hi *= kPrime;
    };

    for (unsigned char c : token)   fnv_byte(c);
    fnv_byte('|');
    for (unsigned char c : context) fnv_byte(c);

    return DedupKey{lo, hi};
}

// ---------------------------------------------------------------------------
// InProcessDeduplicator
// ---------------------------------------------------------------------------

DedupResult InProcessDeduplicator::check_and_register(const DedupKey& key,
                                                       std::chrono::milliseconds ttl) {
    // Clamp TTL to 24 hours to prevent unbounded expiry times.
    static constexpr auto kMaxTTL = std::chrono::hours(24);
    if (ttl > kMaxTTL) ttl = std::chrono::duration_cast<std::chrono::milliseconds>(kMaxTTL);

    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();

    auto it = table_.find(key);
    if (it != table_.end()) {
        if (it->second.expires_at > now) {
            total_duplicates_++;
            return DedupResult::Duplicate;
        }
        // Entry exists but has expired — refresh and treat as novel.
        it->second.expires_at = now + ttl;
        total_novel_++;
        return DedupResult::Novel;
    }

    table_.emplace(key, Entry{now + ttl});
    total_novel_++;
    return DedupResult::Novel;
}

void InProcessDeduplicator::evict(const DedupKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.erase(key);
}

size_t InProcessDeduplicator::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.size();
}

void InProcessDeduplicator::purge_expired() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->second.expires_at <= now) {
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}

void InProcessDeduplicator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.clear();
    total_novel_.store(0, std::memory_order_relaxed);
    total_duplicates_.store(0, std::memory_order_relaxed);
}

InProcessDeduplicator::~InProcessDeduplicator() {
    stop_background_purge();
}

void InProcessDeduplicator::start_background_purge(int interval_s) {
    bool expected = false;
    if (!purge_running_.compare_exchange_strong(expected, true)) return;
    purge_thread_ = std::thread([this, interval_s] {
        int elapsed = 0;
        while (purge_running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            elapsed += 200;
            if (elapsed >= interval_s * 1000) {
                elapsed = 0;
                if (purge_running_.load()) purge_expired();
            }
        }
    });
}

void InProcessDeduplicator::stop_background_purge() {
    purge_running_ = false;
    if (purge_thread_.joinable()) purge_thread_.join();
}

// ---------------------------------------------------------------------------
// RedisDeduplicator
// ---------------------------------------------------------------------------

RedisDeduplicator::RedisDeduplicator(std::string redis_url)
    : redis_url_(std::move(redis_url)) {
#ifdef LLMQUANT_REDIS_ENABLED
    try_connect();
#endif
}

RedisDeduplicator::~RedisDeduplicator() {
#ifdef LLMQUANT_REDIS_ENABLED
    redis_disconnect();
#endif
}

#ifdef LLMQUANT_REDIS_ENABLED
bool RedisDeduplicator::try_connect() {
    // Parse redis://host:port — minimal parser, no path/auth.
    std::string url = redis_url_;
    if (url.rfind("redis://", 0) == 0) url = url.substr(8);
    // Strip any trailing /db segment.
    size_t slash = url.find('/');
    if (slash != std::string::npos) url = url.substr(0, slash);

    std::string host = "127.0.0.1";
    int port = 6379;
    size_t colon = url.find(':');
    if (colon != std::string::npos) {
        host = url.substr(0, colon);
        port = std::stoi(url.substr(colon + 1));
    } else if (!url.empty()) {
        host = url;
    }

    auto* ctx = redisConnect(host.c_str(), port);
    if (!ctx || ctx->err) {
        if (ctx) redisFree(ctx);
        redis_connected_ = false;
        return false;
    }
    redis_ctx_       = ctx;
    redis_connected_ = true;
    return true;
}

void RedisDeduplicator::redis_disconnect() {
    if (redis_ctx_) {
        redisFree(static_cast<redisContext*>(redis_ctx_));
        redis_ctx_ = nullptr;
    }
    redis_connected_ = false;
}

bool RedisDeduplicator::try_reconnect() {
    std::lock_guard<std::mutex> lk(reconnect_mutex_);
    if (redis_connected_) return true;
    redis_reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);
    return try_connect();
}
#endif

void RedisDeduplicator::set_disconnect_callback(DisconnectCallback cb) {
    disconnect_cb_ = std::move(cb);
}

#ifndef LLMQUANT_REDIS_ENABLED
bool RedisDeduplicator::try_reconnect() { return false; }
#endif

bool RedisDeduplicator::is_connected() const {
#ifdef LLMQUANT_REDIS_ENABLED
    return redis_connected_;
#else
    return false;
#endif
}

DedupResult RedisDeduplicator::check_and_register(const DedupKey& key,
                                                   std::chrono::milliseconds ttl) {
#ifdef LLMQUANT_REDIS_ENABLED
    if (redis_connected_) {
        auto* ctx    = static_cast<redisContext*>(redis_ctx_);
        long long ttl_sec = std::max(1LL,
            static_cast<long long>(ttl.count()) / 1000);
        // SET key "" NX EX ttl_sec — atomic check-and-set with TTL.
        // Returns status "OK" if the key was newly set (novel), or nil if it
        // already existed (duplicate).
        // Use both 64-bit halves as the Redis key for full 128-bit uniqueness.
        // Hex encoding with fixed width avoids the separator collision that
        // plain decimal + "_" can cause (e.g. "12_3" vs "1_23").
        char redis_key_buf[35]; // "llmq:" + 16 hex + 16 hex + null
        std::snprintf(redis_key_buf, sizeof(redis_key_buf), "llmq:%016llx%016llx",
                      static_cast<unsigned long long>(key.value),
                      static_cast<unsigned long long>(key.value_hi));
        std::string redis_key = redis_key_buf;
        auto* reply = static_cast<redisReply*>(
            redisCommand(ctx, "SET %s \"\" NX EX %lld",
                         redis_key.c_str(), ttl_sec));
        if (!reply) {
            // Connection lost — fire disconnect callback then attempt reconnect.
            std::string err_msg = ctx->errstr ? ctx->errstr : "nil reply";
            bool was_connected = false;
            {
                std::lock_guard<std::mutex> lk(reconnect_mutex_);
                if (redis_connected_) {
                    redis_connected_ = false;
                    was_connected = true;
                }
            }
            if (was_connected && disconnect_cb_) {
                disconnect_cb_(err_msg);
            }
            // Attempt immediate single reconnect.
            try_reconnect();
            // Fall through to in-process backend.
        } else {
            bool is_novel = (reply->type == REDIS_REPLY_STATUS);
            freeReplyObject(reply);
            return is_novel ? DedupResult::Novel : DedupResult::Duplicate;
        }
    }
#endif
    return inner_.check_and_register(key, ttl);
}

void RedisDeduplicator::evict(const DedupKey& key) {
#ifdef LLMQUANT_REDIS_ENABLED
    if (redis_connected_) {
        auto* ctx   = static_cast<redisContext*>(redis_ctx_);
        // Hex encoding with fixed width avoids the separator collision that
        // plain decimal + "_" can cause (e.g. "12_3" vs "1_23").
        char redis_key_buf[35]; // "llmq:" + 16 hex + 16 hex + null
        std::snprintf(redis_key_buf, sizeof(redis_key_buf), "llmq:%016llx%016llx",
                      static_cast<unsigned long long>(key.value),
                      static_cast<unsigned long long>(key.value_hi));
        std::string redis_key = redis_key_buf;
        auto* reply = static_cast<redisReply*>(
            redisCommand(ctx, "DEL %s", redis_key.c_str()));
        if (reply) freeReplyObject(reply);
    }
#endif
    inner_.evict(key);
}

size_t RedisDeduplicator::size() const { return inner_.size(); }
void RedisDeduplicator::purge_expired() { inner_.purge_expired(); }

// ---------------------------------------------------------------------------
// Deduplicator facade
// ---------------------------------------------------------------------------

Deduplicator::Deduplicator(std::shared_ptr<DeduplicatorBackend> backend,
                            std::chrono::milliseconds default_ttl)
    : backend_(std::move(backend)), default_ttl_(default_ttl) {}

DedupResult Deduplicator::check(const std::string& token, const std::string& context) {
    return backend_->check_and_register(DedupKey::from_token(token, context), default_ttl_);
}

DedupResult Deduplicator::check_with_ttl(const DedupKey& key, std::chrono::milliseconds ttl) {
    return backend_->check_and_register(key, ttl);
}

void Deduplicator::evict(const std::string& token, const std::string& context) {
    backend_->evict(DedupKey::from_token(token, context));
}

void Deduplicator::purge_expired() { backend_->purge_expired(); }

void Deduplicator::start_background_purge(int interval_s) {
    if (auto* ip = dynamic_cast<InProcessDeduplicator*>(backend_.get())) {
        ip->start_background_purge(interval_s);
    } else {
        spdlog::info("[dedup] start_background_purge: backend is not InProcessDeduplicator; "
                     "purge not started. Redis TTL handles expiry server-side.");
    }
}

} // namespace llmquant
