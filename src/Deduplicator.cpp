#include "Deduplicator.h"

#include <cstdint>
#include <sstream>

#ifdef LLMQUANT_REDIS_ENABLED
  #include <hiredis/hiredis.h>
#endif

namespace llmquant {

// ---------------------------------------------------------------------------
// DedupKey
// ---------------------------------------------------------------------------

DedupKey DedupKey::from_token(const std::string& token, const std::string& context) {
    // Deterministic key: FNV-1a hash of "token|context".
    // For production, replace with SHA-256 of token+context if collision
    // resistance stronger than 64-bit is required.
    std::string raw = token + "|" + context;
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : raw) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex << hash;
    DedupKey k;
    k.value = oss.str();
    return k;
}

// ---------------------------------------------------------------------------
// InProcessDeduplicator
// ---------------------------------------------------------------------------

DedupResult InProcessDeduplicator::check_and_register(const DedupKey& key,
                                                       std::chrono::milliseconds ttl) {
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
        auto* reply = static_cast<redisReply*>(
            redisCommand(ctx, "SET %s \"\" NX EX %lld",
                         key.value.c_str(), ttl_sec));
        if (!reply) {
            // Connection lost — fall through to in-process backend.
            redis_connected_ = false;
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
        auto* reply = static_cast<redisReply*>(
            redisCommand(ctx, "DEL %s", key.value.c_str()));
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

} // namespace llmquant
