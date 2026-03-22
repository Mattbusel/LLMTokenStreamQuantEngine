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
        /**
         * @brief When true, log_token_received() writes entries to the log file.
         *        Default is true. Set false to suppress per-token noise in high-throughput
         *        environments where only signal/risk events are relevant.
         */
        bool enable_token_logging{true};
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
     * @brief Record a structured trade signal emission event.
     *
     * Writes a single row (CSV) or JSON object containing the key fields of
     * the emitted signal.  Does not require a TradeSignal include — callers
     * extract the relevant fields before calling.
     *
     * @param bias         delta_bias_shift of the emitted signal.
     * @param volatility   volatility_adjustment of the emitted signal.
     * @param confidence   Confidence score in [0, 1].
     * @param latency_us   Token-to-signal latency in microseconds.
     * @param quality      Composite signal_quality score in [0, 1].
     */
    void log_trade_signal(double bias, double volatility,
                          double confidence, double latency_us,
                          double quality);

    /**
     * @brief Record a hot-reload event for the configuration file.
     *
     * Writes a structured log entry indicating that the configuration was
     * reloaded from @p source_path and whether it succeeded.  Increments
     * the log entry counter.
     *
     * @param source_path Path of the YAML configuration file that was reloaded.
     * @param success     true if the reload parsed and validated successfully.
     */
    void log_config_reload(const std::string& source_path, bool success);

    /**
     * @brief Log a pipeline health snapshot.
     *
     * Records latency SLO health, the current SLO breach rate, and the
     * backoff multiplier.  Useful for periodic health-check logging.
     *
     * @param healthy          true if the latency p99 is within the SLO target.
     * @param slo_breach_rate  Fraction of samples exceeding the target latency [0, 1].
     * @param backoff_multiplier Current backoff multiplier from LatencyController [1, 5].
     */
    void log_pipeline_health(bool healthy, double slo_breach_rate, double backoff_multiplier);

    /**
     * @brief Log a deduplication decision.
     *
     * Records whether a key was novel or a duplicate.  Useful for auditing
     * dedup behaviour in replay or integration scenarios.
     *
     * @param key          String representation of the dedup key.
     * @param is_duplicate true if the key was already registered (a duplicate).
     */
    void log_dedup_event(const std::string& key, bool is_duplicate);

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

    /**
     * @brief Return elapsed milliseconds since the logger was constructed.
     *
     * Useful for computing log throughput (entries / uptime_ms).
     * Thread-safe (construction_time_ is const after construction).
     *
     * @return Uptime in milliseconds.
     */
    double get_uptime_ms() const noexcept {
        auto now = std::chrono::high_resolution_clock::now();
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - construction_time_).count());
    }

    /**
     * @brief Return the average log entry rate since construction.
     *
     * Computed as log_entries / uptime_seconds.  Returns 0.0 if less than
     * 1 ms has elapsed.
     *
     * Thread-safe (atomic read + const construction_time_).
     *
     * @return Log entries per second since construction.
     */
    double get_log_rate() const noexcept {
        double uptime_ms = get_uptime_ms();
        if (uptime_ms < 1.0) return 0.0;
        return static_cast<double>(
            log_entries_.load(std::memory_order_relaxed)) / (uptime_ms / 1000.0);
    }

private:
    void initialize_loggers();
    void write_csv_header();

    Config config_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::shared_ptr<spdlog::logger> console_logger_;
    std::atomic<uint64_t> log_entries_{0};
    std::chrono::high_resolution_clock::time_point construction_time_{
        std::chrono::high_resolution_clock::now()};
};

} // namespace llmquant
