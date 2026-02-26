#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llmquant {

/// Result of a deduplication check.
enum class DedupResult {
    Novel,      ///< This key has not been seen within the TTL window — process it.
    Duplicate,  ///< This key was seen recently — skip processing.
};

/// Key type for deduplication: a hash of token text + optional context string.
struct DedupKey {
    /// Hex-encoded FNV-1a hash of the token and context concatenation.
    std::string value;

    /// Construct a dedup key from a raw token string and optional context.
    ///
    /// The same (token, context) pair always produces the same key value
    /// (deterministic, no randomisation).
    ///
    /// # Arguments
    /// * `token`   — Raw token string.
    /// * `context` — Optional context string that scopes the key (default: "").
    static DedupKey from_token(const std::string& token,
                               const std::string& context = "");

    bool operator==(const DedupKey& other) const { return value == other.value; }
};

} // namespace llmquant

// Allow DedupKey to be used as an unordered_map key.
namespace std {
template<> struct hash<llmquant::DedupKey> {
    size_t operator()(const llmquant::DedupKey& k) const noexcept {
        return std::hash<std::string>{}(k.value);
    }
};
} // namespace std

namespace llmquant {

/// Abstract deduplication backend interface.
///
/// Concrete implementations: InProcessDeduplicator (in-memory, TTL eviction)
/// and RedisDeduplicator (stub — interface only, not wired to a live Redis).
class DeduplicatorBackend {
public:
    virtual ~DeduplicatorBackend() = default;

    /// Check whether `key` is a duplicate and register it if novel.
    ///
    /// Implementations must be thread-safe.
    ///
    /// # Arguments
    /// * `key` — The deduplication key derived from token + context.
    /// * `ttl` — How long the key should be considered live after registration.
    ///
    /// # Returns
    /// DedupResult::Novel if this is the first occurrence within the TTL.
    /// DedupResult::Duplicate otherwise.
    virtual DedupResult check_and_register(const DedupKey& key,
                                           std::chrono::milliseconds ttl) = 0;

    /// Explicitly remove a key (e.g. after processing completes).
    ///
    /// # Arguments
    /// * `key` — The key to remove from the live set.
    virtual void evict(const DedupKey& key) = 0;

    /// Return the number of entries currently tracked (including expired ones
    /// that have not yet been purged).
    virtual size_t size() const = 0;

    /// Remove all expired entries.  May be a no-op if the backend evicts lazily.
    virtual void purge_expired() = 0;
};

/// In-process deduplicator backed by an unordered_map with TTL entries.
///
/// Memory usage is bounded by the number of unique keys seen within the
/// configured TTL window.  purge_expired() should be called periodically
/// (e.g. once per second) to reclaim memory.
///
/// Thread safety: all public methods are safe to call concurrently.
class InProcessDeduplicator : public DeduplicatorBackend {
public:
    explicit InProcessDeduplicator() = default;

    /// Check and register a key; see DeduplicatorBackend::check_and_register.
    DedupResult check_and_register(const DedupKey& key,
                                   std::chrono::milliseconds ttl) override;

    /// Remove a key from the live set; see DeduplicatorBackend::evict.
    void evict(const DedupKey& key) override;

    /// Return the number of tracked entries (including not-yet-purged expired ones).
    size_t size() const override;

    /// Remove all entries whose TTL has elapsed.
    void purge_expired() override;

    /// Return the total number of duplicate hits since construction.
    uint64_t total_duplicates() const { return total_duplicates_.load(); }

    /// Return the total number of novel keys registered since construction.
    uint64_t total_novel() const { return total_novel_.load(); }

private:
    struct Entry {
        std::chrono::steady_clock::time_point expires_at;
    };

    mutable std::mutex mutex_;
    std::unordered_map<DedupKey, Entry> table_;
    std::atomic<uint64_t> total_duplicates_{0};
    std::atomic<uint64_t> total_novel_{0};
};

/// Redis deduplicator with optional live hiredis connection.
///
/// When built with LLMQUANT_REDIS_ENABLED (hiredis found at CMake time),
/// check_and_register and evict issue real Redis commands (SET NX EX / DEL).
/// If the connection is unavailable at construction time or drops mid-run,
/// the implementation falls back transparently to the in-process backend.
///
/// When built without hiredis, the class is a pure in-process stub: same
/// public interface, no network I/O.
class RedisDeduplicator : public DeduplicatorBackend {
public:
    /// Construct and optionally connect to Redis.
    ///
    /// When LLMQUANT_REDIS_ENABLED is defined, attempts to connect to the
    /// parsed host:port from `redis_url`. Falls back silently to in-process
    /// mode if the connection fails.
    ///
    /// # Arguments
    /// * `redis_url` — e.g. "redis://127.0.0.1:6379" or "127.0.0.1:6379".
    explicit RedisDeduplicator(std::string redis_url);

    /// Disconnect from Redis and free all resources.
    ~RedisDeduplicator();

    /// Check and register `key` with `ttl`.
    ///
    /// Uses Redis SET NX EX when connected; falls back to in-process backend
    /// on disconnection or when hiredis is absent.
    DedupResult check_and_register(const DedupKey& key,
                                   std::chrono::milliseconds ttl) override;

    /// Evict `key` from Redis (DEL) and from the in-process backend.
    void evict(const DedupKey& key) override;

    /// Return the number of entries in the in-process backend.
    size_t size() const override;

    /// Purge expired entries from the in-process backend.
    void purge_expired() override;

    /// Return the Redis URL this instance was constructed with.
    const std::string& redis_url() const { return redis_url_; }

    /// Returns true if a live Redis connection is active.
    ///
    /// Always returns false when built without hiredis (stub mode).
    bool is_connected() const;

private:
    std::string redis_url_;
    // Fallback: in-process deduplicator used when Redis is unavailable.
    InProcessDeduplicator inner_;

#ifdef LLMQUANT_REDIS_ENABLED
    void* redis_ctx_{nullptr};   ///< redisContext* — opaque to avoid hiredis header leaking.
    bool  redis_connected_{false};

    /// Parse redis_url_ and attempt redisConnect. Returns true on success.
    bool try_connect();

    /// Free the redisContext and mark disconnected.
    void redis_disconnect();
#endif
};

/// Facade that wraps a DeduplicatorBackend and adds convenience methods.
///
/// The default TTL can be set at construction and overridden per-call.
///
/// Example:
/// ```cpp
/// auto backend = std::make_shared<InProcessDeduplicator>();
/// Deduplicator dedup(backend, std::chrono::milliseconds{5000});
/// if (dedup.check("bullish") == DedupResult::Novel) { /* process */ }
/// ```
class Deduplicator {
public:
    /// Construct with the given backend and default TTL.
    ///
    /// # Arguments
    /// * `backend`     — Shared ownership of a DeduplicatorBackend.
    /// * `default_ttl` — TTL applied when check() is used (default: 5000 ms).
    explicit Deduplicator(std::shared_ptr<DeduplicatorBackend> backend,
                          std::chrono::milliseconds default_ttl =
                              std::chrono::milliseconds{5000});

    /// Check and register a token string using the default TTL.
    ///
    /// # Arguments
    /// * `token`   — Raw token string to deduplicate.
    /// * `context` — Optional context string (default: "").
    ///
    /// # Returns
    /// DedupResult::Novel or DedupResult::Duplicate.
    DedupResult check(const std::string& token,
                      const std::string& context = "");

    /// Check and register a pre-built key with a custom TTL.
    ///
    /// # Arguments
    /// * `key` — Pre-built DedupKey.
    /// * `ttl` — Per-call TTL override.
    DedupResult check_with_ttl(const DedupKey& key, std::chrono::milliseconds ttl);

    /// Evict a key by token string and optional context.
    ///
    /// # Arguments
    /// * `token`   — Raw token string.
    /// * `context` — Optional context string (default: "").
    void evict(const std::string& token, const std::string& context = "");

    /// Trigger expired-entry purge on the backend.
    void purge_expired();

    /// Return the underlying backend (for stats access).
    ///
    /// # Returns
    /// Reference to the concrete DeduplicatorBackend.
    DeduplicatorBackend& backend() { return *backend_; }

private:
    std::shared_ptr<DeduplicatorBackend> backend_;
    std::chrono::milliseconds default_ttl_;
};

} // namespace llmquant
