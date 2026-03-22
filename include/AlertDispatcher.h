#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace llmquant {

/**
 * @brief Non-blocking async alert delivery for operational events.
 *
 * The AlertDispatcher decouples alert generation (hot path / callbacks) from
 * alert delivery (file, webhook, syslog) by maintaining an internal work queue
 * and a dedicated background thread.
 *
 * ## Alert structure
 * Each alert has:
 *   - `severity`  — DEBUG / INFO / WARN / CRITICAL
 *   - `category`  — Domain tag (e.g. "circuit_breaker", "regime", "stale")
 *   - `message`   — Human-readable description
 *   - `timestamp` — Wall-clock time of the event
 *
 * ## Sinks
 * Register any number of `AlertSink` callables.  Each is called in registration
 * order from the background thread.  The sink must be thread-safe with respect
 * to itself (it is never called concurrently with another invocation of itself,
 * but may be called concurrently with sink registration).
 *
 * ## Deduplication window
 * Identical (category, message) pairs within `dedup_window_ms` are suppressed
 * to prevent alert storms (e.g. regime flapping at a decision boundary).
 *
 * ## Thread safety
 * `post()` is lock-free for the common case (queue not full).
 * `add_sink()` / `remove_all_sinks()` / `start()` / `stop()` are safe to call
 * from any thread.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ALERT_DISPATCHER` (default ON).
 */
class AlertDispatcher {
public:
    /**
     * @brief Alert severity level.
     */
    enum class Severity { Debug, Info, Warn, Critical };

    /**
     * @brief A single alert event.
     */
    struct Alert {
        Severity                                severity{Severity::Info};
        std::string                             category;
        std::string                             message;
        std::chrono::system_clock::time_point   timestamp{std::chrono::system_clock::now()};
    };

    /**
     * @brief Alert sink callable type.
     *
     * Invoked from the background thread.  Must not throw.
     */
    using AlertSink = std::function<void(const Alert&)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Maximum number of alerts waiting in the queue.
         *
         * When the queue is full, new posts are silently dropped (non-blocking).
         * Default 1024.
         */
        int max_queue_depth{1024};

        /**
         * @brief Milliseconds within which identical (category, message) alerts
         *        are deduplicated (suppressed) to prevent storms.
         * 0 = no deduplication.  Default 5000 (5 s).
         */
        int dedup_window_ms{5000};

        /**
         * @brief Minimum severity level to accept.  Alerts below this level
         *        are silently discarded before entering the queue.
         * Default Debug (accept all).
         */
        Severity min_severity{Severity::Debug};
    };

    /**
     * @brief Construct with the given configuration.
     *
     * @param config Queue depth, dedup window, and severity filter.
     */
    explicit AlertDispatcher(Config config = Config{});

    /** @brief Destructor calls stop(). */
    ~AlertDispatcher();

    // Non-copyable.
    AlertDispatcher(const AlertDispatcher&)            = delete;
    AlertDispatcher& operator=(const AlertDispatcher&) = delete;

    /**
     * @brief Start the background dispatch thread.
     *
     * @return true on success; false if already running.
     */
    [[nodiscard]] bool start();

    /**
     * @brief Drain the queue and stop the background thread.
     *
     * Blocks until all queued alerts have been delivered.
     */
    void stop();

    /**
     * @brief Post an alert to the dispatch queue.
     *
     * Non-blocking.  If the queue is full or the dispatcher is stopped,
     * the alert is silently dropped (dropped_count increments).
     *
     * Thread-safe.
     *
     * @param severity Alert severity.
     * @param category Domain tag for this alert.
     * @param message  Human-readable description.
     */
    void post(Severity           severity,
              const std::string& category,
              const std::string& message);

    /**
     * @brief Convenience: post an Info alert.
     */
    void info(const std::string& category, const std::string& message) {
        post(Severity::Info, category, message);
    }

    /**
     * @brief Convenience: post a Warn alert.
     */
    void warn(const std::string& category, const std::string& message) {
        post(Severity::Warn, category, message);
    }

    /**
     * @brief Convenience: post a Critical alert.
     */
    void critical(const std::string& category, const std::string& message) {
        post(Severity::Critical, category, message);
    }

    /**
     * @brief Register an alert sink.
     *
     * Sinks are invoked in registration order from the background thread.
     * Thread-safe.
     *
     * @param sink AlertSink callable.
     */
    void add_sink(AlertSink sink);

    /**
     * @brief Remove all registered sinks.
     *
     * Thread-safe.
     */
    void remove_all_sinks();

    /**
     * @brief Build and add a file-based sink that writes NDJSON lines.
     *
     * Each line: {"ts":"...","sev":"...","cat":"...","msg":"..."}
     *
     * @param filepath Path to the output file (opened in append mode).
     * @return true if the file could be opened; false otherwise.
     */
    [[nodiscard]] bool add_file_sink(const std::string& filepath);

    /**
     * @brief Build and add a spdlog-based sink that logs via spdlog.
     *
     * Maps AlertDispatcher::Severity to the corresponding spdlog level.
     */
    void add_spdlog_sink();

    /**
     * @brief True while the background thread is running.
     */
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total alerts posted (including dropped and deduplicated).
     */
    [[nodiscard]] uint64_t total_posted() const noexcept {
        return total_posted_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Alerts dropped because the queue was full.
     */
    [[nodiscard]] uint64_t dropped_count() const noexcept {
        return dropped_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Alerts deduplicated (not dispatched, not counted as dropped).
     */
    [[nodiscard]] uint64_t dedup_count() const noexcept {
        return dedup_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total alerts successfully dispatched to sinks.
     */
    [[nodiscard]] uint64_t dispatched_count() const noexcept {
        return dispatched_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return a human-readable severity label.
     */
    [[nodiscard]] static const char* severity_name(Severity s) noexcept;

    /**
     * @brief Serialise current stats to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Update configuration at runtime.
     *
     * Thread-safe (takes internal mutex).
     */
    void update_config(const Config& config);

private:
    void dispatch_loop();
    [[nodiscard]] bool is_duplicate_locked(const Alert& a);

    Config config_;
    mutable std::mutex sinks_mutex_;
    mutable std::mutex queue_mutex_;
    std::vector<AlertSink> sinks_;

    std::queue<Alert>         queue_;
    std::condition_variable   cv_;
    std::thread               thread_;

    std::atomic<bool>     running_{false};
    std::atomic<bool>     stop_requested_{false};
    std::atomic<uint64_t> total_posted_{0};
    std::atomic<uint64_t> dropped_count_{0};
    std::atomic<uint64_t> dedup_count_{0};
    std::atomic<uint64_t> dispatched_count_{0};

    // Dedup state: (category+message) → last dispatch time.
    struct DedupEntry {
        std::string key;
        std::chrono::steady_clock::time_point last_seen;
    };
    std::vector<DedupEntry> dedup_cache_;  // protected by queue_mutex_
};

} // namespace llmquant
