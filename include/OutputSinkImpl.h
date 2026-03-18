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
             << sig.latency_us << ","
             << sig.strategy_toggle << ","
             << safe_d(sig.strategy_weight) << "\n";
    }

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
    explicit JsonOutputSink(const std::string& filename)
        : out_(filename, std::ios::out | std::ios::trunc)
    {
        if (!out_.is_open()) {
            throw std::runtime_error("JsonOutputSink: cannot open file: " + filename);
        }
    }

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
             << "\"latency_us\":"             << sig.latency_us                        << ","
             << "\"strategy_toggle\":"        << sig.strategy_toggle                   << ","
             << "\"strategy_weight\":"        << safe_d(sig.strategy_weight)
             << "}\n";
    }

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

    void emit(const TradeSignal& sig) override {
        signals_.push_back(sig);
    }

    const std::vector<TradeSignal>& get_signals() const { return signals_; }

    void clear() { signals_.clear(); }

private:
    std::vector<TradeSignal> signals_;
};

} // namespace llmquant
