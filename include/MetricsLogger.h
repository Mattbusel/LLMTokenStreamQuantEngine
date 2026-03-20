#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace llmquant {

/**
 * @brief Structured logging sink for the token-processing pipeline.
 *
 * Two output formats are supported: CSV (one row per event) and
 * newline-delimited JSON. An optional coloured console sink can be
 * attached alongside the file sink.
 *
 * Thread safety: all log_* methods are thread-safe (spdlog guarantees
 * internal locking per logger). flush() is also thread-safe.
 */
class MetricsLogger {
public:
    /**
     * @brief Serialisation format for the log file.
     */
    enum class OutputFormat {
        CSV,   ///< Comma-separated values with a header row.
        JSON   ///< One JSON object per line (newline-delimited JSON / NDJSON).
    };

    /**
     * @brief Construction-time parameters for the logger.
     */
    struct Config {
        /** @brief Path to the output log file (created or truncated at startup). */
        std::string log_file_path{"metrics.log"};
        /** @brief Serialisation format written to the file sink. */
        OutputFormat format{OutputFormat::CSV};
        /** @brief When true a coloured spdlog console logger is also attached. */
        bool enable_console_output{true};
        /** @brief How frequently the file sink should be flushed to disk. */
        std::chrono::milliseconds flush_interval{std::chrono::milliseconds{100}};
    };

    /**
     * @brief Construct and initialise both file and (optionally) console loggers.
     *
     * If the file sink cannot be created (e.g. permission denied), the logger
     * falls back to a null sink so callers are never interrupted by a throw.
     *
     * @param config Construction-time parameters.
     */
    explicit MetricsLogger(const Config& config);

    /**
     * @brief Flush all pending log entries before destruction.
     */
    ~MetricsLogger();

    /**
     * @brief Record a token-received event.
     *
     * @param token       The raw token string that arrived from the simulator.
     * @param sequence_id Monotonically increasing sequence number of the token.
     */
    void log_token_received(const std::string& token, uint64_t sequence_id);

    /**
     * @brief Record a trade-signal emission event.
     *
     * @param bias       delta_bias_shift from the emitted TradeSignal.
     * @param volatility volatility_adjustment from the emitted TradeSignal.
     * @param latency_us End-to-end token-to-signal latency in microseconds.
     */
    void log_signal_generated(double bias, double volatility, uint64_t latency_us);

    /**
     * @brief Record a raw latency measurement (e.g. from LatencyController).
     *
     * @param latency_us Measured latency in microseconds.
     */
    void log_latency_measurement(uint64_t latency_us);

    /**
     * @brief Record a system-resource snapshot.
     *
     * @param memory_usage Current RSS in bytes.
     * @param cpu_usage    Current CPU utilisation as a percentage (0-100).
     */
    void log_system_stats(uint64_t memory_usage, double cpu_usage);

    /**
     * @brief Record a risk gate rejection event.
     *
     * @param reason     Rejection reason string (e.g. "magnitude_exceeded").
     * @param bias       delta_bias_shift of the rejected signal.
     * @param confidence Confidence score of the rejected signal.
     */
    void log_risk_rejection(const std::string& reason, double bias, double confidence);

    /**
     * @brief Write a human-readable performance summary to the console (if enabled).
     */
    void log_performance_summary();

    /**
     * @brief Flush all pending log entries to their respective sinks immediately.
     */
    void flush();

    /**
     * @brief Return the total number of log entries written since construction.
     *
     * Thread-safe (atomic load). Counts every call to any log_* method that
     * successfully writes an entry.
     *
     * @return Total log entry count.
     */
    uint64_t get_log_entry_count() const noexcept {
        return log_entries_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset the log entry counter to zero.
     *
     * Useful at session boundaries to reuse the same logger without restarting
     * the process.  Does NOT truncate or rotate the underlying log file.
     * Thread-safe (atomic store with release ordering).
     */
    void reset_counters() noexcept {
        log_entries_.store(0, std::memory_order_release);
    }

private:
    void initialize_loggers();
    void write_csv_header();

    Config config_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::shared_ptr<spdlog::logger> console_logger_;
    std::atomic<uint64_t> log_entries_{0};
};

} // namespace llmquant
