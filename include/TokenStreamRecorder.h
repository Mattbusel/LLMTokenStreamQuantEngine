#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Append-only binary audit log for raw token streams.
 *
 * Writes a compact binary record for every token to a log file.  The record
 * format is fixed-size and timestamp-ordered, enabling exact replay and
 * regulatory audit trails.
 *
 * ## Binary record format (32 bytes per entry)
 * ```
 * Offset  Size  Field
 *      0     8  sequence_number (uint64_t, little-endian)
 *      8     8  timestamp_ns    (uint64_t, nanoseconds since epoch)
 *     16     8  weight          (double, IEEE 754 little-endian)
 *     24     4  token_len       (uint32_t, bytes in token string)
 *     28     4  crc32           (CRC-32 of the preceding 28 bytes)
 * ```
 * The variable-length token string follows the fixed header at offset 32.
 * Total record size = 32 + token_len bytes.
 *
 * ## File lifecycle
 * - `open(path)` creates or appends to the log file.
 * - `record(token, weight)` writes one entry.
 * - `close()` flushes and closes the file.
 * - `rotate(new_path)` closes the current file and opens a new one
 *   atomically (without losing records).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TOKEN_RECORDER` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class TokenStreamRecorder {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Initial log file path.  Empty = recorder starts closed.
         */
        std::string path;

        /**
         * @brief Maximum file size in bytes before auto-rotation.
         *
         * 0 = no auto-rotation.  Default 256 MiB.
         */
        uint64_t max_file_bytes{256ULL * 1024 * 1024};

        /**
         * @brief If true, flush after every record (safe but slower).
         *
         * Default false (flush on close / rotate).
         */
        bool sync_write{false};

        /**
         * @brief Called when the log file is rotated.
         *
         * Parameters: (old_path, new_path, records_written_to_old).
         */
        std::function<void(const std::string& old_path,
                           const std::string& new_path,
                           uint64_t records)> on_rotate;

        /**
         * @brief Called when a write fails (e.g. disk full).
         *
         * Parameters: (error_message).
         */
        std::function<void(const std::string& error)> on_error;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenStreamRecorder(Config cfg = Config{});
    ~TokenStreamRecorder();

    // Non-copyable.
    TokenStreamRecorder(const TokenStreamRecorder&)            = delete;
    TokenStreamRecorder& operator=(const TokenStreamRecorder&) = delete;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /**
     * @brief Open a log file for writing.
     *
     * @return true on success.
     */
    bool open(const std::string& path);

    /**
     * @brief Flush and close the current file.
     */
    void close();

    /**
     * @brief Rotate to a new log file.
     *
     * Closes the current file and opens `new_path`.  No records are lost.
     * @return true if the new file was opened successfully.
     */
    bool rotate(const std::string& new_path);

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /**
     * @brief Write one token record to the log.
     *
     * If no file is open, the call is silently ignored.
     *
     * @param token  Raw token string.
     * @param weight Signed weight (LLM adapter output).
     * @return true on success.
     */
    bool record(const std::string& token, double weight) noexcept;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief True if the log file is currently open.
     */
    [[nodiscard]] bool is_open() const noexcept {
        return is_open_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total records written to the current file.
     */
    [[nodiscard]] uint64_t records_written() const noexcept {
        return records_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Bytes written to the current file.
     */
    [[nodiscard]] uint64_t bytes_written() const noexcept {
        return bytes_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total write errors since construction.
     */
    [[nodiscard]] uint64_t error_count() const noexcept {
        return errors_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration without closing the file.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;
    std::string current_path_;

    // FILE* wrapped in void* to avoid including <cstdio> in the header.
    void* fp_{nullptr};

    std::atomic<bool>     is_open_{false};
    std::atomic<uint64_t> records_{0};
    std::atomic<uint64_t> bytes_{0};
    std::atomic<uint64_t> errors_{0};

    mutable std::mutex mutex_;

    static uint32_t crc32_compute(const void* data, std::size_t len) noexcept;
};

} // namespace llmquant
