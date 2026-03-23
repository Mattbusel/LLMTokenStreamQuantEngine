#pragma once

/**
 * @file include/SignalReplayRunner.h
 * @brief JSONL signal replay and testing mode for LLMTokenStreamQuantEngine.
 *
 * Reads a JSONL file of `{"token": "...", "timestamp_us": 12345}` records,
 * replays them through the full pipeline, and outputs a signal trace.
 *
 * ## JSONL Input Format
 * ```jsonl
 * {"token": "bullish", "timestamp_us": 1704067200000000}
 * {"token": "growth",  "timestamp_us": 1704067200001200}
 * {"token": "rally",   "timestamp_us": 1704067200002500}
 * ```
 *
 * ## Usage
 * ```cpp
 * SignalReplayConfig cfg;
 * cfg.input_path  = "tokens.jsonl";
 * cfg.output_path = "signal_trace.jsonl";  // or empty for stdout
 * cfg.verbose     = true;
 *
 * SignalReplayRunner runner(cfg);
 * auto result = runner.run();
 *
 * if (result.success) {
 *     fmt::print("Processed {} tokens, emitted {} signals (efficiency={:.1f}%)\n",
 *         result.tokens_processed, result.signals_emitted,
 *         result.efficiency * 100.0);
 * }
 * ```
 *
 * ## CLI (via main --test-replay)
 * ```bash
 * ./llmquant --test-replay tokens.jsonl --output signal_trace.jsonl
 * ```
 */

#include <cstdint>
#include <string>

namespace llmquant {

/**
 * @brief Configuration for a replay run.
 */
struct SignalReplayConfig {
    /// Path to the JSONL input file (required).
    std::string input_path;

    /// Path to the JSONL output file. Empty = write to stdout.
    std::string output_path;

    /// If true, replay at wall-clock speed matching the recorded timestamps.
    /// If false (default), replay as fast as possible.
    bool realtime_replay{false};

    /// If true, log each skipped/malformed line to stderr.
    bool verbose{false};
};

/**
 * @brief Result of a replay run.
 */
struct SignalReplayResult {
    /// True if the replay completed without file I/O errors.
    bool success{false};

    /// Error message if success == false.
    std::string error_message;

    /// Total number of token records processed.
    uint64_t tokens_processed{0};

    /// Number of TradeSignal emissions.
    uint64_t signals_emitted{0};

    /// Number of lines skipped due to missing 'token' field.
    uint64_t malformed_lines{0};

    /// signals_emitted / tokens_processed.
    double efficiency{0.0};
};

/**
 * @brief Replays a JSONL token stream through the signal pipeline.
 *
 * Thread safety: Not thread-safe. Create one instance per replay session.
 */
class SignalReplayRunner {
public:
    explicit SignalReplayRunner(SignalReplayConfig config);

    /**
     * @brief Execute the replay and write the signal trace.
     *
     * @return SignalReplayResult with statistics and success flag.
     */
    [[nodiscard]] SignalReplayResult run() const;

private:
    SignalReplayConfig config_;
};

} // namespace llmquant
