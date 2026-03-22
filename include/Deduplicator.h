#pragma once

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
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
 * @brief Key type for deduplication: a 128-bit hash of token text + optional context.
 *
 * Uses two independent FNV-1a passes with different seeds to produce a 128-bit
 * (hi, lo) pair, giving collision resistance equivalent to 2^128 rather than 2^64.
 * The `value` field retains the low 64 bits for backwards compatibility where needed.
 */
struct DedupKey {
    /** @brief Low 64 bits of the 128-bit hash (same as the original FNV-1a value). */
    uint64_t value{0};
    /** @brief High 64 bits of the 128-bit hash (second FNV-1a pass with alternate seed). */
    uint64_t value_hi{0};

    /**
     * @brief Construct a dedup key from a raw token string and optional context.
     *
     * The same (token, context) pair always produces the same key value
     * (deterministic, no randomisation).
     *
     * @param token   Raw token string.
     * @param context Optional context string that scopes the key (default: "").
     * @return A DedupKey with the computed 128-bit hash.
     */
    static DedupKey from_token(const std::string& token,
                               const std::string& context = "") noexcept;

    /**
     * @brief Equality comparison operator (uses both 64-bit halves).
     *
     * @param other DedupKey to compare against.
     * @return true if both keys have the same 128-bit hash value.
     */
    bool operator==(const DedupKey& other) const noexcept {
        return value == other.value && value_hi == other.value_hi;
    }
};

} // namespace llmquant

// Allow DedupKey to be used as an unordered_map key.
// Combine both 64-bit halves into a single size_t hash via XOR-shift mixing.
namespace std {
template<> struct hash<llmquant::DedupKey> {
    size_t operator()(const llmquant::DedupKey& k) const noexcept {
        // Mix both halves: XOR the high bits into the low hash with a rotation.
        uint64_t h = k.value ^ (k.value_hi * 0x9e3779b97f4a7c15ULL);
        return static_cast<size_t>(h);
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
    [[nodiscard]] virtual DedupResult check_and_register(const DedupKey& key,
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

    // -----------------------------------------------------------------------
    // Stats interface (non-pure so backends may opt out).
    // -----------------------------------------------------------------------

    /** @brief Total novel (non-duplicate) registrations since construction. */
    virtual uint64_t total_novel()      const noexcept { return 0; }
    /** @brief Total duplicate hits since construction. */
    virtual uint64_t total_duplicates() const noexcept { return 0; }
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
    [[nodiscard]] DedupResult check_and_register(const DedupKey& key,
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
    uint64_t total_duplicates() const noexcept override { return total_duplicates_.load(); }

    /**
     * @brief Return the total number of novel keys registered since construction.
     *
     * @return Novel registration count.
     */
    uint64_t total_novel() const noexcept override { return total_novel_.load(); }

    /**
     * @brief Return the fraction of all seen keys that were duplicates, in [0.0, 1.0].
     *
     * Computed as total_duplicates / (total_duplicates + total_novel).
     * Returns 0.0 if no keys have been seen yet.
     * Thread-safe: reads atomic counters with relaxed ordering.
     *
     * @return Duplicate rate in [0.0, 1.0].
     */
    double get_duplicate_rate() const noexcept {
        uint64_t dups  = total_duplicates_.load(std::memory_order_relaxed);
        uint64_t novel = total_novel_.load(std::memory_order_relaxed);
        uint64_t total = dups + novel;
        return (total == 0) ? 0.0 : static_cast<double>(dups) / static_cast<double>(total);
    }

    /**
     * @brief Return the fraction of all seen keys that were novel, in [0.0, 1.0].
     *
     * Complement of get_duplicate_rate(): novel_rate + duplicate_rate == 1.0.
     * Returns 0.0 if no keys have been seen yet.
     * Thread-safe: reads atomic counters with relaxed ordering.
     *
     * @return Novel rate in [0.0, 1.0].
     */
    double get_novel_rate() const noexcept {
        uint64_t dups  = total_duplicates_.load(std::memory_order_relaxed);
        uint64_t novel = total_novel_.load(std::memory_order_relaxed);
        uint64_t total = dups + novel;
        return (total == 0) ? 0.0 : static_cast<double>(novel) / static_cast<double>(total);
    }

    /**
     * @brief Return the total number of check_and_register() calls since construction.
     *
     * Equivalent to total_duplicates() + total_novel().
     * Thread-safe: reads atomic counters with relaxed ordering.
     *
     * @return Total keys checked.
     */
    uint64_t total_checked() const noexcept {
        return total_duplicates_.load(std::memory_order_relaxed)
             + total_novel_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Aggregated statistics snapshot.
     */
    struct Stats {
        uint64_t total_novel{0};       ///< Novel registrations since construction/last reset.
        uint64_t total_duplicates{0};  ///< Duplicate hits since construction/last reset.
        size_t   current_size{0};      ///< Entries currently in the live table.
    };

    /**
     * @brief Return a snapshot of all deduplication statistics.
     *
     * @return Stats struct populated with current counter and table-size values.
     */
    Stats get_stats() const {
        Stats s;
        s.total_novel      = total_novel_.load(std::memory_order_relaxed);
        s.total_duplicates = total_duplicates_.load(std::memory_order_relaxed);
        s.current_size     = size();
        return s;
    }

    /**
     * @brief Reset all state: clear the key table and zero the novel/duplicate counters.
     *
     * Useful at session boundaries to reuse the same backend across sessions.
     * Thread-safe (acquires mutex_).  Does NOT stop any running background
     * purge thread — call stop_background_purge() separately if desired.
     */
    void reset();

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

    /**
     * @brief Serialise the current deduplicator statistics to a JSON string.
     *
     * Returns a JSON object with total_novel, total_duplicates, current_size,
     * and dup_rate_pct (duplicate percentage [0,100]).
     * Thread-safe (reads atomic counters with relaxed ordering; current_size
     * acquires the internal mutex briefly via size()).
     *
     * @return JSON object as std::string.
     */
    std::string to_stats_json() const {
        uint64_t nov  = total_novel_.load(std::memory_order_relaxed);
        uint64_t dup  = total_duplicates_.load(std::memory_order_relaxed);
        uint64_t tot  = nov + dup;
        double   pct  = (tot > 0) ? (static_cast<double>(dup) * 100.0 / static_cast<double>(tot)) : 0.0;
        size_t   cur  = size();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "{\"total_novel\":%" PRIu64
            ",\"total_duplicates\":%" PRIu64
            ",\"current_size\":%zu"
            ",\"dup_rate_pct\":%.2f}",
            nov, dup, cur, pct);
        return buf;
    }

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
    [[nodiscard]] DedupResult check_and_register(const DedupKey& key,
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
     * @brief Novel count — reflects in-process fallback entries and entries
     *        processed when Redis was unavailable.
     */
    uint64_t total_novel()      const noexcept override;

    /**
     * @brief Duplicate count — same scope as total_novel().
     */
    uint64_t total_duplicates() const noexcept override;

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

    /**
     * @brief Return the number of reconnect attempts since construction.
     *
     * @return Reconnect attempt count.
     */
    uint64_t reconnect_attempts() const noexcept {
        return redis_reconnect_attempts_.load(std::memory_order_relaxed);
    }

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
    std::atomic<uint64_t> redis_reconnect_attempts_{0};  ///< Total reconnect attempts since construction.

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

    /**
     * @brief Aggregated deduplication statistics.
     */
    struct Stats {
        uint64_t total_novel{0};       ///< Novel registrations since construction.
        uint64_t total_duplicates{0};  ///< Duplicate hits since construction.
        size_t   current_size{0};      ///< Entries currently in the live table.
    };

    /**
     * @brief Return aggregated deduplication statistics.
     *
     * total_novel and total_duplicates are populated only when the backend is
     * an InProcessDeduplicator; for other backend types they will be zero.
     * current_size is always available via the base interface.
     *
     * @return Stats snapshot.
     */
    Stats get_stats() const;

    /**
     * @brief Serialise statistics as a single-line JSON object.
     *
     * @return JSON string, e.g. {"novel":42,"duplicates":7,"size":35}
     */
    std::string to_stats_json() const {
        auto s = get_stats();
        return "{\"novel\":" + std::to_string(s.total_novel)
             + ",\"duplicates\":" + std::to_string(s.total_duplicates)
             + ",\"size\":" + std::to_string(s.current_size) + "}";
    }

private:
    std::shared_ptr<DeduplicatorBackend> backend_;
    std::chrono::milliseconds default_ttl_;
};

} // namespace llmquant
