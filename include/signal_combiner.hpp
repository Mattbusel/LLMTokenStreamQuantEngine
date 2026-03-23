#pragma once

/// @file include/signal_combiner.hpp
/// @brief Signal Combiner — Round 6 public API.
///
/// # Module: Signal Combiner
///
/// Combines multiple named trading signals (from sentiment analysis, alpha
/// decay, and execution timing) into a single actionable composite signal.
///
/// ## Signal inputs
/// Each signal carries:
///   - `name`:         source identifier (e.g. "sentiment", "alpha_decay")
///   - `value`:        directional strength in [-1, 1]
///   - `confidence`:   source reliability in [0, 1]
///   - `timestamp_ms`: wall-clock timestamp at generation
///
/// ## Combination methods
///
/// ### WeightedAverage
/// ```
///   action = Σ(confidence_i * value_i) / Σconfidence_i
/// ```
///
/// ### Majority
/// Count positive/negative votes weighted by confidence.
/// Result is positive if sum of confidences for positive signals exceeds
/// sum for negative signals.
///
/// ### Ensemble (Kalman-filter-style recursive update)
/// For each new signal `v` with confidence `c`, uncertainty `u = 1 - c`:
/// ```
///   K = c / (c + u)          Kalman gain
///   x = x + K * (v - x)      recursive update
/// ```
/// The initial state x=0 evolves toward the consensus of all signals.
///
/// ## Regime classification
/// After combining, the output is classified into:
///   - `TRENDING`:      |action| > 0.4
///   - `MEAN_REVERTING`: |action| < 0.15
///   - `UNCERTAIN`:     otherwise
///
/// ## SignalHistory
/// A fixed-capacity ring buffer of past `CombinedSignal`s that exposes:
///   - `trend_direction()`: +1 (up), -1 (down), 0 (flat)
///   - `volatility()`:      standard deviation of recent action values

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmquant {

// ─── SignalInput ──────────────────────────────────────────────────────────────

/// A named, confidence-annotated trading signal from one source.
struct SignalInput {
    std::string name;          ///< Source identifier
    double      value;         ///< Directional value in [-1, 1]
    double      confidence;    ///< Source reliability in [0, 1]
    int64_t     timestamp_ms;  ///< Wall-clock timestamp (ms since epoch)
};

// ─── SignalRegime ─────────────────────────────────────────────────────────────

/// Market regime as inferred from the combined signal.
enum class SignalRegime : uint8_t {
    TRENDING,        ///< |action| > 0.40 — strong directional conviction
    MEAN_REVERTING,  ///< |action| < 0.15 — weak / noisy signal
    UNCERTAIN,       ///< Between trending and mean-reverting thresholds
};

// ─── CombinedSignal ───────────────────────────────────────────────────────────

/// The output of a combination pass: a single actionable composite signal.
struct CombinedSignal {
    double                   action;               ///< Composite value in [-1, 1]
    double                   confidence;           ///< Weighted average confidence
    std::vector<std::string> contributing_signals; ///< Names of input signals used
    SignalRegime             regime;               ///< Inferred market regime
};

// ─── CombineMethod ────────────────────────────────────────────────────────────

/// Combination algorithm selector.
enum class CombineMethod : uint8_t {
    WEIGHTED_AVERAGE,  ///< confidence-weighted arithmetic mean
    MAJORITY,          ///< directional majority vote weighted by confidence
    ENSEMBLE,          ///< Kalman-style recursive update
};

// ─── SignalCombiner ───────────────────────────────────────────────────────────

/// Accumulates named signal inputs and combines them on demand.
///
/// Usage:
/// ```cpp
/// SignalCombiner sc;
/// sc.add_signal({"sentiment",  0.6, 0.8, ts});
/// sc.add_signal({"alpha_decay", 0.4, 0.6, ts});
/// auto result = sc.combine(CombineMethod::WEIGHTED_AVERAGE);
/// sc.reset();
/// ```
class SignalCombiner {
public:
    /// Thresholds for regime classification.
    static constexpr double kTrendingThreshold     = 0.40;
    static constexpr double kMeanRevertingThreshold = 0.15;

    /// Add a signal input to the current accumulation window.
    ///
    /// @param input  Signal to add (value clamped to [-1,1];
    ///               confidence clamped to [0,1]).
    void add_signal(const SignalInput& input);

    /// Combine accumulated signals using *method*.
    ///
    /// @param method  Combination algorithm.
    /// @return        Combined signal, or nullopt if no signals have been added.
    [[nodiscard]] std::optional<CombinedSignal>
    combine(CombineMethod method = CombineMethod::WEIGHTED_AVERAGE) const;

    /// Clear all accumulated signals and reset Kalman state.
    void reset() noexcept;

    /// Return the number of signals currently accumulated.
    [[nodiscard]] std::size_t signal_count() const noexcept {
        return signals_.size();
    }

private:
    std::vector<SignalInput> signals_;

    /// Classify a combined action value into a regime.
    static SignalRegime classify(double action) noexcept;

    /// Weighted-average combination.
    [[nodiscard]] CombinedSignal _weighted_average() const;

    /// Confidence-weighted majority vote.
    [[nodiscard]] CombinedSignal _majority() const;

    /// Kalman-filter-style ensemble update.
    [[nodiscard]] CombinedSignal _ensemble() const;
};

// ─── SignalHistory ────────────────────────────────────────────────────────────

/// Fixed-capacity ring buffer of recent `CombinedSignal`s.
///
/// Provides rolling summary statistics:
/// - `trend_direction()`: majority sign of recent action values
/// - `volatility()`:      standard deviation of recent action values
class SignalHistory {
public:
    /// Construct with the given ring-buffer capacity (default 64).
    explicit SignalHistory(std::size_t capacity = 64)
        : capacity_(capacity > 0 ? capacity : 1) {}

    /// Push a new combined signal into the history.
    void push(const CombinedSignal& signal);

    /// Return the most recent combined signal, or nullopt if empty.
    [[nodiscard]] std::optional<CombinedSignal> latest() const;

    /// Infer trend direction from recent action values.
    ///
    /// @return +1 if mean > 0.05, -1 if mean < -0.05, 0 otherwise.
    [[nodiscard]] int trend_direction() const noexcept;

    /// Compute the standard deviation of recent action values.
    ///
    /// @return Volatility >= 0.  Returns 0.0 if fewer than 2 entries.
    [[nodiscard]] double volatility() const noexcept;

    /// Number of entries currently held.
    [[nodiscard]] std::size_t size() const noexcept { return buffer_.size(); }

    /// Maximum capacity of the ring buffer.
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /// Remove all entries.
    void clear() noexcept { buffer_.clear(); }

private:
    std::size_t              capacity_;
    std::deque<CombinedSignal> buffer_;
};

}  // namespace llmquant
