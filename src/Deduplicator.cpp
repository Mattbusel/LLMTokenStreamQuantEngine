#include "Deduplicator.h"

#include <cstdint>
#include <sstream>

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
// RedisDeduplicator (stub)
// ---------------------------------------------------------------------------

RedisDeduplicator::RedisDeduplicator(std::string redis_url)
    : redis_url_(std::move(redis_url)) {}

DedupResult RedisDeduplicator::check_and_register(const DedupKey& key,
                                                   std::chrono::milliseconds ttl) {
    return inner_.check_and_register(key, ttl);
}

void RedisDeduplicator::evict(const DedupKey& key) { inner_.evict(key); }
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
