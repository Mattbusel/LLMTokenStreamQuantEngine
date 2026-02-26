#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "TradeSignalEngine.h"

namespace llmquant {

/// Abstract output sink for routing trade signals to a destination.
///
/// Concrete sinks are provided for CSV files, newline-delimited JSON files,
/// and an in-memory buffer (primarily for unit testing).
///
/// Thread safety: individual sink implementations are NOT thread-safe.
/// External synchronisation is required if multiple threads call emit()
/// on the same sink instance concurrently.
class OutputSink {
public:
    virtual ~OutputSink() = default;

    /// Emit a single trade signal to this sink.
    ///
    /// # Arguments
    /// * `signal` — Fully populated TradeSignal to write.
    virtual void emit(const TradeSignal& signal) = 0;

    /// Flush any internally buffered output to the underlying destination.
    ///
    /// The default implementation is a no-op; file sinks override this.
    virtual void flush() {}
};

// ---------------------------------------------------------------------------
// CsvOutputSink
// ---------------------------------------------------------------------------

/// CSV file sink — writes one signal per line as comma-separated values.
///
/// A header row is written at construction time so the file is self-describing.
class CsvOutputSink : public OutputSink {
public:
    /// Open (or create) the named file and write a CSV header row.
    ///
    /// # Arguments
    /// * `filename` — Path to the output CSV file; existing file is truncated.
    ///
    /// # Throws
    /// `std::runtime_error` if the file cannot be opened.
    explicit CsvOutputSink(const std::string& filename)
        : out_(filename, std::ios::out | std::ios::trunc)
    {
        if (!out_.is_open()) {
            throw std::runtime_error("CsvOutputSink: cannot open file: " + filename);
        }
        out_ << "timestamp_ns,delta_bias_shift,volatility_adjustment,"
                "spread_modifier,confidence,latency_us,"
                "strategy_toggle,strategy_weight\n";
        out_.flush();
    }

    /// Write one CSV row for the given signal.
    void emit(const TradeSignal& signal) override {
        out_ << signal.timestamp_ns << ","
             << signal.delta_bias_shift << ","
             << signal.volatility_adjustment << ","
             << signal.spread_modifier << ","
             << signal.confidence << ","
             << signal.latency_us << ","
             << signal.strategy_toggle << ","
             << signal.strategy_weight << "\n";
    }

    /// Flush the underlying ofstream to disk.
    void flush() override { out_.flush(); }

private:
    std::ofstream out_;
};

// ---------------------------------------------------------------------------
// JsonOutputSink
// ---------------------------------------------------------------------------

/// JSON file sink — writes one JSON object per line (NDJSON format).
class JsonOutputSink : public OutputSink {
public:
    /// Open (or create) the named file.
    ///
    /// # Arguments
    /// * `filename` — Path to the output NDJSON file; existing file is truncated.
    ///
    /// # Throws
    /// `std::runtime_error` if the file cannot be opened.
    explicit JsonOutputSink(const std::string& filename)
        : out_(filename, std::ios::out | std::ios::trunc)
    {
        if (!out_.is_open()) {
            throw std::runtime_error("JsonOutputSink: cannot open file: " + filename);
        }
    }

    /// Write one JSON object (followed by a newline) for the given signal.
    void emit(const TradeSignal& signal) override {
        out_ << "{"
             << "\"timestamp_ns\":" << signal.timestamp_ns << ","
             << "\"delta_bias_shift\":" << signal.delta_bias_shift << ","
             << "\"volatility_adjustment\":" << signal.volatility_adjustment << ","
             << "\"spread_modifier\":" << signal.spread_modifier << ","
             << "\"confidence\":" << signal.confidence << ","
             << "\"latency_us\":" << signal.latency_us << ","
             << "\"strategy_toggle\":" << signal.strategy_toggle << ","
             << "\"strategy_weight\":" << signal.strategy_weight
             << "}\n";
    }

    /// Flush the underlying ofstream to disk.
    void flush() override { out_.flush(); }

private:
    std::ofstream out_;
};

// ---------------------------------------------------------------------------
// MemoryOutputSink
// ---------------------------------------------------------------------------

/// In-memory buffer sink — accumulates all emitted signals in a vector.
///
/// Intended for unit and integration tests where the full signal sequence
/// needs to be inspected after the fact without touching the filesystem.
class MemoryOutputSink : public OutputSink {
public:
    MemoryOutputSink() = default;

    /// Append the signal to the internal buffer.
    void emit(const TradeSignal& signal) override {
        signals_.push_back(signal);
    }

    /// Return a read-only view of all signals accumulated so far.
    const std::vector<TradeSignal>& get_signals() const { return signals_; }

    /// Clear the internal buffer.
    void clear() { signals_.clear(); }

private:
    std::vector<TradeSignal> signals_;
};

} // namespace llmquant
