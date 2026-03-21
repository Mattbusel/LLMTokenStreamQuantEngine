#pragma once

// Full definitions of the concrete OutputSink implementations.
// Include this header (not OutputSink.h) wherever you need CsvOutputSink,
// JsonOutputSink, or MemoryOutputSink.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "OutputSink.h"
#include "TradeSignalEngine.h"  // provides full TradeSignal definition

namespace llmquant {

// ---------------------------------------------------------------------------
// CsvOutputSink
// ---------------------------------------------------------------------------

/**
 * @brief CSV file sink — writes one signal per line as comma-separated values.
 *
 * A header row is written at construction time so the file is self-describing.
 * Not thread-safe; see OutputSink class documentation.
 */
class CsvOutputSink : public OutputSink {
public:
    /**
     * @brief Open (or create) the named file and write a CSV header row.
     *
     * @param filename Path to the output CSV file; existing file is truncated.
     * @throws std::runtime_error if the file cannot be opened.
     */
    explicit CsvOutputSink(const std::string& filename)
        : out_(filename, std::ios::out | std::ios::trunc)
    {
        if (!out_.is_open()) {
            throw std::runtime_error("CsvOutputSink: cannot open file: " + filename);
        }
        out_ << "timestamp_ns,delta_bias_shift,volatility_adjustment,"
                "spread_modifier,confidence,latency_us,"
                "strategy_toggle,strategy_weight,signal_quality\n";
        out_.flush();
    }

    /**
     * @brief Write one signal as a CSV row.
     *
     * NaN and Inf values are replaced with 0.0 so the output is always valid CSV.
     *
     * @param sig Fully populated TradeSignal to serialise.
     */
    void emit(const TradeSignal& sig) override {
        // Guard against NaN/Inf which are not valid CSV values.
        auto safe_d = [](double v) -> double {
            return std::isfinite(v) ? v : 0.0;
        };
        out_ << sig.timestamp_ns << ","
             << safe_d(sig.delta_bias_shift) << ","
             << safe_d(sig.volatility_adjustment) << ","
             << safe_d(sig.spread_modifier) << ","
             << safe_d(sig.confidence) << ","
             << safe_d(sig.latency_us) << ","
             << sig.strategy_toggle << ","
             << safe_d(sig.strategy_weight) << ","
             << safe_d(sig.signal_quality) << "\n";
        ++emit_count_;
    }

    /// @brief Flush the underlying file stream to disk.
    void flush() override { out_.flush(); }

    /// @brief Return the number of signals written so far (excluding the header row).
    size_t emit_count() const noexcept { return emit_count_; }

private:
    std::ofstream out_;
    size_t emit_count_{0};
};

// ---------------------------------------------------------------------------
// JsonOutputSink
// ---------------------------------------------------------------------------

/**
 * @brief JSON file sink — writes one JSON object per line (NDJSON format).
 *
 * Not thread-safe; see OutputSink class documentation.
 */
class JsonOutputSink : public OutputSink {
public:
    /**
     * @brief Open (or create) the named file.
     *
     * @param filename Path to the output NDJSON file; existing file is truncated.
     * @throws std::runtime_error if the file cannot be opened.
     */
    explicit JsonOutputSink(const std::string& filename)
        : out_(filename, std::ios::out | std::ios::trunc)
    {
        if (!out_.is_open()) {
            throw std::runtime_error("JsonOutputSink: cannot open file: " + filename);
        }
    }

    /**
     * @brief Write one signal as a JSON object followed by a newline.
     *
     * NaN and Inf values are replaced with 0.0 so the output is always valid JSON.
     *
     * @param sig Fully populated TradeSignal to serialise.
     */
    void emit(const TradeSignal& sig) override {
        // Guard against NaN/Inf which are not valid JSON values.
        auto safe_d = [](double v) -> double {
            return std::isfinite(v) ? v : 0.0;
        };
        out_ << "{"
             << "\"timestamp_ns\":"          << sig.timestamp_ns                      << ","
             << "\"delta_bias_shift\":"       << safe_d(sig.delta_bias_shift)          << ","
             << "\"volatility_adjustment\":"  << safe_d(sig.volatility_adjustment)     << ","
             << "\"spread_modifier\":"        << safe_d(sig.spread_modifier)           << ","
             << "\"confidence\":"             << safe_d(sig.confidence)                << ","
             << "\"latency_us\":"             << safe_d(sig.latency_us)                << ","
             << "\"strategy_toggle\":"        << sig.strategy_toggle                   << ","
             << "\"strategy_weight\":"        << safe_d(sig.strategy_weight)           << ","
             << "\"signal_quality\":"         << safe_d(sig.signal_quality)
             << "}\n";
        ++emit_count_;
    }

    /// @brief Flush the underlying file stream to disk.
    void flush() override { out_.flush(); }

    /// @brief Return the number of signals written so far.
    size_t emit_count() const noexcept { return emit_count_; }

private:
    std::ofstream out_;
    size_t emit_count_{0};
};

// ---------------------------------------------------------------------------
// MemoryOutputSink
// ---------------------------------------------------------------------------

/**
 * @brief In-memory buffer sink — accumulates all emitted signals in a vector.
 *
 * Intended for unit and integration tests where the full signal sequence
 * needs to be inspected after the fact without touching the filesystem.
 * Not thread-safe; see OutputSink class documentation.
 *
 * When constructed with a non-zero @p max_capacity the buffer acts as a
 * fixed-size ring: once full, the oldest signal is evicted and dropped_count_
 * is incremented before the new signal is stored.  A capacity of 0 (default)
 * means unlimited — the buffer grows without bound.
 */
class MemoryOutputSink : public OutputSink {
public:
    /**
     * @brief Construct the sink with an optional capacity cap.
     *
     * @param max_capacity Maximum number of signals to retain.  Pass 0 for
     *                     unlimited (the default, backward-compatible behaviour).
     */
    explicit MemoryOutputSink(size_t max_capacity = 0)
        : max_capacity_(max_capacity) {}

    /**
     * @brief Append the signal to the internal buffer.
     *
     * If a capacity cap is set and the buffer is full the oldest signal is
     * evicted first and the drop counter is incremented.
     *
     * @param sig Fully populated TradeSignal to store.
     */
    void emit(const TradeSignal& sig) override {
        if (max_capacity_ > 0 && signals_.size() >= max_capacity_) {
            signals_.erase(signals_.begin());
            ++dropped_count_;
        }
        signals_.push_back(sig);
        ++emit_count_;
    }

    /**
     * @brief Return a const reference to all stored signals.
     *
     * @return Const reference to the internal signal vector.
     */
    const std::vector<TradeSignal>& get_signals() const { return signals_; }

    /**
     * @brief Alias for get_signals() — provided for API symmetry.
     *
     * @return Const reference to the internal signal vector.
     */
    const std::vector<TradeSignal>& signals() const { return signals_; }

    /**
     * @brief Return the number of signals currently stored in the buffer.
     *
     * @return Signal count (never exceeds max_capacity if a cap is set).
     */
    size_t size() const noexcept { return signals_.size(); }

    /**
     * @brief Return the cumulative number of signals dropped due to capacity.
     *
     * @return Drop count (always 0 when max_capacity is 0).
     */
    size_t dropped_count() const noexcept { return dropped_count_; }

    /**
     * @brief Return the total number of emit() calls since construction or last clear().
     *
     * Unlike size(), this includes signals that were dropped due to capacity.
     *
     * @return Total signals received.
     */
    size_t emit_count() const noexcept { return emit_count_; }

    /**
     * @brief Return the fraction of emitted signals that were dropped, in [0.0, 1.0].
     *
     * Returns 0.0 if no signals have been emitted.
     *
     * @return Drop rate in [0.0, 1.0].
     */
    double drop_rate() const noexcept {
        return (emit_count_ == 0)
            ? 0.0
            : static_cast<double>(dropped_count_) / static_cast<double>(emit_count_);
    }

    /// @brief Clear the internal signal buffer and reset all counters.
    void clear() { signals_.clear(); dropped_count_ = 0; emit_count_ = 0; }

private:
    std::vector<TradeSignal> signals_;
    size_t max_capacity_{0};
    size_t dropped_count_{0};
    size_t emit_count_{0};
};

} // namespace llmquant
