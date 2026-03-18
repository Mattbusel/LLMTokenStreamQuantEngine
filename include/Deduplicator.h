#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace llmquant {

/**
 * @brief Result of a deduplication check.
 */
enum class DedupResult {
    Novel,      ///< This key has not been seen within the TTL window -- process it.
    Duplicate,  ///< This key was seen recently -- skip processing.
};

/**
 * @brief Key type for deduplication: a raw FNV-1a 64-bit hash of token text + optional context.
 */
struct DedupKey {
    /** @brief Raw FNV-1a 64-bit hash of the token and context concatenation. */
    uint64_t value{0};

    /**
     * @brief Construct a dedup key from a raw token string and optional context.
     *
     * The same (token, context) pair always produces the same key value
     * (deterministic, no randomisation).
     *
     * @param token   Raw token string.
     * @param context Optional context string that scopes the key (default: "").
     * @return A DedupKey with the computed hash.
     */
    static DedupKey from_token(const std::string& token,
                               const std::string& context = "") noexcept;

    /**
     * @brief Equality comparison operator.
     *
     * @param other DedupKey to compare against.
     * @return true if both keys have the same hash value.
     */
    bool operator==(const DedupKey& other) const noexcept { return value == other.value; }
};

} // namespace llmquant

// Allow DedupKey to be used as an unordered_map key.
namespace std {
template<> struct hash<llmquant::DedupKey> {
    size_t operator()(const llmquant::DedupKey& k) const noexcept {
        return static_cast<size_t>(k.value);
    }
};
} // namespace std

namespace llmquant {

/**
 * @brief Abstract deduplication backend interface.
 *
 * Concrete implementations: InProcessDeduplicator (in-memory, TTL eviction)
 * and RedisDeduplicator (optional live Redis via hiredis; falls back to in-process).
 */
class DeduplicatorBackend {
public:
    virtual ~DeduplicatorBackend() = default;

    /**
     * @brief Check whether key is a duplicate and register it if novel.
     *
     * Implementations must be thread-safe.
     *
     * @param key The deduplication key derived from token + context.
     * @param ttl How long the key should be considered live after registration.
     * @return DedupResult::Novel if this is the first occurrence within the TTL;
     *         DedupResult::Duplicate otherwise.
     */
    virtual DedupResult check_and_register(const DedupKey& key,
                                           std::chrono::milliseconds ttl) = 0;

    /**
     * @brief Explicitly remove a key (e.g. after processing completes).
     *
     * @param key The key to remove from the live set.
     */
    virtual void evict(const DedupKey& key) = 0;

    /**
     * @brief Return the number of entries currently tracked (including expired ones
     *        that have not yet been purged).
     *
     * @return Current entry count.
     */
    virtual size_t size() const = 0;

    /**
     * @brief Remove all expired entries.
     *
     * May be a no-op if the backend evicts lazily.
     */
    virtual void purge_expired() = 0;
};

/**
 * @brief In-process deduplicator backed by an unordered_map with TTL entries.
 *
 * Memory usage is bounded by the number of unique keys seen within the
 * configured TTL window. Call start_background_purge() to automatically
 * reclaim memory at a regular interval; the destructor stops the thread.
 *
 * Thread safety: all public methods are safe to call concurrently.
 */
class InProcessDeduplicator : public DeduplicatorBackend {
public:
    explicit InProcessDeduplicator() = default;

    /**
     * @brief Destructor stops the background purge thread (if running) before destruction.
     */
    ~InProcessDeduplicator();

    /**
     * @brief Check and register a key; see DeduplicatorBackend::check_and_register.
     *
     * @param key The deduplication key.
     * @param ttl Time-to-live for this registration.
     * @return DedupResult::Novel or DedupResult::Duplicate.
     */
    DedupResult check_and_register(const DedupKey& key,
                                   std::chrono::milliseconds ttl) override;

    /**
     * @brief Remove a key from the live set; see DeduplicatorBackend::evict.
     *
     * @param key The key to remove.
     */
    void evict(const DedupKey& key) override;

    /**
     * @brief Return the number of tracked entries (including not-yet-purged expired ones).
     *
     * @return Current map size.
     */
    size_t size() const override;

    /**
     * @brief Remove all entries whose TTL has elapsed.
     */
    void purge_expired() override;

    /**
     * @brief Return the total number of duplicate hits since construction.
     *
     * @return Duplicate hit count.
     */
    uint64_t total_duplicates() const noexcept { return total_duplicates_.load(); }

    /**
     * @brief Return the total number of novel keys registered since construction.
     *
     * @return Novel registration count.
     */
    uint64_t total_novel() const noexcept { return total_novel_.load(); }

    /**
     * @brief Start a background thread that calls purge_expired() every interval_s seconds.
     *
     * No-op if the thread is already running.
     *
     * @param interval_s Purge interval in seconds (default: 60).
     */
    void start_background_purge(int interval_s = 60);

    /**
     * @brief Stop the background purge thread and join it.
     *
     * Safe to call even if start_background_purge() was never called.
     */
    void stop_background_purge();

private:
    struct Entry {
        std::chrono::steady_clock::time_point expires_at;
    };

    mutable std::mutex mutex_;
    std::unordered_map<DedupKey, Entry> table_;
    std::atomic<uint64_t> total_duplicates_{0};
    std::atomic<uint64_t> total_novel_{0};

    std::thread purge_thread_;
    std::atomic<bool> purge_running_{false};
};

/**
 * @brief Redis deduplicator with optional live hiredis connection.
 *
 * When built with LLMQUANT_REDIS_ENABLED (hiredis found at CMake time),
 * check_and_register and evict issue real Redis commands (SET NX EX / DEL).
 * If the connection is unavailable at construction time or drops mid-run,
 * the implementation falls back transparently to the in-process backend.
 *
 * When built without hiredis, the class is a pure in-process stub: same
 * public interface, no network I/O.
 */
class RedisDeduplicator : public DeduplicatorBackend {
public:
    /**
     * @brief Construct and optionally connect to Redis.
     *
     * When LLMQUANT_REDIS_ENABLED is defined, attempts to connect to the
     * parsed host:port from redis_url. Falls back silently to in-process
     * mode if the connection fails.
     *
     * @param redis_url Redis URL, e.g. "redis://127.0.0.1:6379" or "127.0.0.1:6379".
     */
    explicit RedisDeduplicator(std::string redis_url);

    /**
     * @brief Disconnect from Redis and free all resources.
     */
    ~RedisDeduplicator();

    /**
     * @brief Check and register key with ttl.
     *
     * Uses Redis SET NX EX when connected; falls back to in-process backend
     * on disconnection or when hiredis is absent.
     *
     * @param key The deduplication key.
     * @param ttl Time-to-live for this registration.
     * @return DedupResult::Novel or DedupResult::Duplicate.
     */
    DedupResult check_and_register(const DedupKey& key,
                                   std::chrono::milliseconds ttl) override;

    /**
     * @brief Evict key from Redis (DEL) and from the in-process backend.
     *
     * @param key The key to remove.
     */
    void evict(const DedupKey& key) override;

    /**
     * @brief Return the number of entries in the in-process backend.
     *
     * @return Entry count.
     */
    size_t size() const override;

    /**
     * @brief Purge expired entries from the in-process backend.
     */
    void purge_expired() override;

    /**
     * @brief Return the Redis URL this instance was constructed with.
     *
     * @return The redis_url string passed to the constructor.
     */
    const std::string& redis_url() const noexcept { return redis_url_; }

    /**
     * @brief Returns true if a live Redis connection is active.
     *
     * Always returns false when built without hiredis (stub mode).
     *
     * @return Connection status.
     */
    bool is_connected() const;

    /** @brief Callback type invoked when the Redis connection is lost. */
    using DisconnectCallback = std::function<void(const std::string& error)>;

    /**
     * @brief Register a callback invoked once per disconnection event.
     *
     * The callback fires from the thread that detected the failure.
     *
     * @param cb Callable matching DisconnectCallback.
     */
    void set_disconnect_callback(DisconnectCallback cb);

    /**
     * @brief Attempt to re-establish the Redis connection.
     *
     * @return true if reconnect succeeded; false otherwise. No-op stub when built without hiredis.
     */
    bool try_reconnect();

private:
    std::string redis_url_;
    InProcessDeduplicator inner_;

    DisconnectCallback disconnect_cb_;
    std::mutex reconnect_mutex_;   ///< Serialises reconnect attempts.

#ifdef LLMQUANT_REDIS_ENABLED
    void* redis_ctx_{nullptr};   ///< redisContext* -- opaque to avoid hiredis header leaking.
    bool  redis_connected_{false};

    /** @brief Parse redis_url_ and attempt redisConnect. Returns true on success. */
    bool try_connect();

    /** @brief Free the redisContext and mark disconnected. */
    void redis_disconnect();
#endif
};

/**
 * @brief Facade that wraps a DeduplicatorBackend and adds convenience methods.
 *
 * The default TTL can be set at construction and overridden per-call.
 *
 * Example:
 * @code
 * auto backend = std::make_shared<InProcessDeduplicator>();
 * Deduplicator dedup(backend, std::chrono::milliseconds{5000});
 * if (dedup.check("bullish") == DedupResult::Novel) { // process }
 * @endcode
 */
class Deduplicator {
public:
    /**
     * @brief Construct with the given backend and default TTL.
     *
     * @param backend     Shared ownership of a DeduplicatorBackend.
     * @param default_ttl TTL applied when check() is used (default: 5000 ms).
     */
    explicit Deduplicator(std::shared_ptr<DeduplicatorBackend> backend,
                          std::chrono::milliseconds default_ttl =
                              std::chrono::milliseconds{5000});

    /**
     * @brief Check and register a token string using the default TTL.
     *
     * @param token   Raw token string to deduplicate.
     * @param context Optional context string (default: "").
     * @return DedupResult::Novel or DedupResult::Duplicate.
     */
    DedupResult check(const std::string& token,
                      const std::string& context = "");

    /**
     * @brief Check and register a pre-built key with a custom TTL.
     *
     * @param key Pre-built DedupKey.
     * @param ttl Per-call TTL override.
     * @return DedupResult::Novel or DedupResult::Duplicate.
     */
    DedupResult check_with_ttl(const DedupKey& key, std::chrono::milliseconds ttl);

    /**
     * @brief Evict a key by token string and optional context.
     *
     * @param token   Raw token string.
     * @param context Optional context string (default: "").
     */
    void evict(const std::string& token, const std::string& context = "");

    /**
     * @brief Trigger expired-entry purge on the backend.
     */
    void purge_expired();

    /**
     * @brief Start a background purge thread on the backend if it is an
     *        InProcessDeduplicator. No-op for other backend types.
     *
     * @param interval_s Purge interval in seconds (default: 60).
     */
    void start_background_purge(int interval_s = 60);

    /**
     * @brief Return the underlying backend (for stats access).
     *
     * @return Reference to the concrete DeduplicatorBackend.
     */
    DeduplicatorBackend& backend() { return *backend_; }

private:
    std::shared_ptr<DeduplicatorBackend> backend_;
    std::chrono::milliseconds default_ttl_;
};

} // namespace llmquant
