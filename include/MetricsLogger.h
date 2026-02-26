#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace llmquant {

/// Structured logging sink for the token-processing pipeline.
///
/// Two output formats are supported: CSV (one row per event) and
/// newline-delimited JSON.  An optional coloured console sink can be
/// attached alongside the file sink.
///
/// Thread safety: all log_* methods are thread-safe (spdlog guarantees
/// internal locking per logger).  flush() is also thread-safe.
class MetricsLogger {
public:
    /// Serialisation format for the log file.
    enum class OutputFormat {
        CSV,   ///< Comma-separated values with a header row.
        JSON   ///< One JSON object per line (newline-delimited JSON / NDJSON).
    };

    /// Construction-time parameters for the logger.
    struct Config {
        /// Path to the output log file (created or truncated at startup).
        std::string log_file_path{"metrics.log"};
        /// Serialisation format written to the file sink.
        OutputFormat format{OutputFormat::CSV};
        /// When true a coloured spdlog console logger is also attached.
        bool enable_console_output{true};
        /// How frequently the file sink should be flushed to disk.
        std::chrono::milliseconds flush_interval{std::chrono::milliseconds{100}};
    };

    /// Construct and initialise both file and (optionally) console loggers.
    ///
    /// # Throws
    /// `std::runtime_error` if the file sink cannot be created (e.g. permission denied).
    explicit MetricsLogger(const Config& config);

    /// Flush all pending log entries before destruction.
    ~MetricsLogger();

    /// Record a token-received event.
    ///
    /// # Arguments
    /// * `token`       — The raw token string that arrived from the simulator.
    /// * `sequence_id` — Monotonically increasing sequence number of the token.
    void log_token_received(const std::string& token, uint64_t sequence_id);

    /// Record a trade-signal emission event.
    ///
    /// # Arguments
    /// * `bias`       — delta_bias_shift from the emitted TradeSignal.
    /// * `volatility` — volatility_adjustment from the emitted TradeSignal.
    /// * `latency_us` — End-to-end token-to-signal latency in microseconds.
    void log_signal_generated(double bias, double volatility, uint64_t latency_us);

    /// Record a raw latency measurement (e.g. from LatencyController).
    ///
    /// # Arguments
    /// * `latency_us` — Measured latency in microseconds.
    void log_latency_measurement(uint64_t latency_us);

    /// Record a system-resource snapshot.
    ///
    /// # Arguments
    /// * `memory_usage` — Current RSS in bytes.
    /// * `cpu_usage`    — Current CPU utilisation as a percentage (0–100).
    void log_system_stats(uint64_t memory_usage, double cpu_usage);

    /// Write a human-readable performance summary to the console (if enabled).
    void log_performance_summary();

    /// Flush all pending log entries to their respective sinks immediately.
    void flush();

private:
    void initialize_loggers();
    void write_csv_header();

    Config config_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::shared_ptr<spdlog::logger> console_logger_;
    std::atomic<uint64_t> log_entries_{0};
};

} // namespace llmquant
