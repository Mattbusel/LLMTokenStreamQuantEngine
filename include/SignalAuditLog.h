#pragma once

#include "TradeSignalEngine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace llmquant {

/**
 * @brief Feature-flagged async NDJSON audit log for every evaluated signal.
 *
 * Records each TradeSignal evaluation (pass/fail + reason) to a rotating
 * NDJSON file.  File I/O runs on a dedicated background thread via an
 * unbounded work queue, so callers on the hot path are never blocked.
 *
 * Each line is a JSON object:
 * @code{.json}
 * {"ts_ns":1704067200000000000,"passed":true,"reason":"",
 *  "bias":0.15,"vol":0.022,"conf":0.87,"spread":0.001,
 *  "cooldown_ms":100,"seq":42}
 * @endcode
 *
 * Compile-time feature flag: LLMQUANT_AUDIT_LOG_ENABLED
 * (controlled by CMake option LLMQUANT_ENABLE_AUDIT_LOG).
 *
 * Thread safety: log_signal() may be called from any thread concurrently.
 */
class SignalAuditLog {
public:
    /**
     * @brief Audit log configuration.
     */
    struct Config {
        /** @brief Path to the NDJSON output file. */
        std::string filepath{"signals_audit.ndjson"};

        /**
         * @brief Rotate the log file after this many bytes.
         *
         * The active file is renamed to "<filepath>.<n>" before a new one is
         * opened.  Set to 0 to disable rotation.
         */
        uint64_t rotate_bytes{50ULL * 1024ULL * 1024ULL};  // 50 MB

        /**
         * @brief Number of rotated files to retain (oldest deleted first).
         *
         * Set to 0 to keep all rotated files.
         */
        int keep_files{3};

        /** @brief Write a record for signals that passed all risk gates. */
        bool log_passed{true};

        /** @brief Write a record for signals blocked by a risk gate. */
        bool log_blocked{true};

        /**
         * @brief Maximum number of unwritten records that may be queued.
         *
         * When the queue reaches this depth the calling thread logs a warning
         * and the record is dropped.  0 = unlimited (bounded only by RAM).
         */
        size_t max_queue_depth{8192};
    };

    /**
     * @brief Construct the audit log and start the background writer thread.
     *
     * @param config File path, rotation policy, and filter settings.
     */
    explicit SignalAuditLog(Config config);

    /**
     * @brief Stop the writer thread and flush all queued records to disk.
     */
    ~SignalAuditLog();

    /**
     * @brief Enqueue an audit record for the given signal evaluation.
     *
     * Called from the hot path; never blocks.  The record is serialised to
     * JSON on the calling thread (cheap) then handed off to the writer queue.
     *
     * @param signal  The evaluated TradeSignal.
     * @param passed  true if the signal passed all risk gates; false if blocked.
     * @param reason  Human-readable rejection reason (empty string on pass).
     */
    void log_signal(const TradeSignal& signal,
                    bool               passed,
                    const std::string& reason = "");

    /**
     * @brief Total records enqueued for writing since construction.
     *
     * @return Enqueue count (includes both written and queued records).
     */
    [[nodiscard]] uint64_t records_enqueued() const noexcept {
        return records_enqueued_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total records successfully written to disk since construction.
     *
     * @return Write count.
     */
    [[nodiscard]] uint64_t records_written() const noexcept {
        return records_written_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total records dropped because the queue was full.
     *
     * @return Drop count.
     */
    [[nodiscard]] uint64_t records_dropped() const noexcept {
        return records_dropped_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of log rotations that have occurred since construction.
     *
     * @return Rotation count.
     */
    [[nodiscard]] int rotations() const noexcept {
        return rotations_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns true while the background writer thread is running.
     *
     * @return Running state.
     */
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_relaxed);
    }

private:
    void writer_thread();
    void write_record(const std::string& json_line);
    void rotate_if_needed();
    void prune_old_files();

    static std::string build_json(const TradeSignal& signal,
                                  bool               passed,
                                  const std::string& reason);

    Config config_;

    std::queue<std::string>   queue_;
    std::mutex                queue_mutex_;
    std::condition_variable   queue_cv_;
    std::thread               thread_;
    std::atomic<bool>         running_{false};
    std::atomic<bool>         stop_requested_{false};

    // Output file state (only accessed from the writer thread).
    std::FILE*    file_{nullptr};
    uint64_t      file_bytes_{0};    ///< Bytes written to the current file.

    // Counters (atomics, safe to read from any thread).
    std::atomic<uint64_t> records_enqueued_{0};
    std::atomic<uint64_t> records_written_{0};
    std::atomic<uint64_t> records_dropped_{0};
    std::atomic<int>      rotations_{0};
};

} // namespace llmquant
