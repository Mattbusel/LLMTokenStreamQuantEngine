#pragma once

/// @file include/sentiment_trend.hpp
/// @brief Sentiment Trend Tracker and Regime Switch Detector.
///
/// # Module: Sentiment Trend Tracker
///
/// ## Responsibility
/// Track how LLM-derived bias signals evolve over a trading session:
///
/// 1. **SentimentTrendTracker** — maintains a rolling window of
///    ``delta_bias_shift`` values from ``TradeSignal`` objects, fits an
///    online linear regression (OLS), and classifies the trend direction.
///
/// 2. **RegimeSwitchDetector** — monitors the rolling sign of the bias
///    signal and emits a ``RegimeSwitch`` event when sentiment crosses
///    from positive to negative territory (or vice versa) with sufficient
///    confidence.
///
/// ## Integration with RiskManager
/// The ``RegimeSwitchDetector`` exposes an ``is_switching()`` predicate
/// that ``RiskManager`` queries before allowing signal emission.  During
/// a regime switch, uncertainty is elevated and trading is disabled until
/// the new regime is confirmed.
///
/// ## Thread Safety
/// All public methods acquire a ``std::mutex`` and are safe for concurrent
/// access from multiple threads.

#include "TradeSignalEngine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llmquant {

// ─── TrendDirection ───────────────────────────────────────────────────────────

/// The directional label produced by linear regression on bias signals.
enum class TrendDirection {
    Bullish,   ///< slope > +threshold: sentiment rising
    Bearish,   ///< slope < −threshold: sentiment falling
    Flat,      ///< |slope| ≤ threshold: no meaningful trend
    Reversing, ///< R² is low AND slope has recently changed sign
};

/// Human-readable string for a TrendDirection.
[[nodiscard]] const char* to_string(TrendDirection d) noexcept;

// ─── TrendResult ──────────────────────────────────────────────────────────────

/// Result of a linear regression fit on the rolling bias window.
struct TrendResult {
    double slope{0.0};                ///< Regression slope (bias units / sample).
    double intercept{0.0};            ///< Regression intercept.
    double r_squared{0.0};            ///< Coefficient of determination [0, 1].
    TrendDirection trend_direction{TrendDirection::Flat}; ///< Classified trend.
    std::size_t n_samples{0};         ///< Number of samples in the window.
    double mean_bias{0.0};            ///< Mean bias over the window.
    std::chrono::high_resolution_clock::time_point timestamp; ///< Fit timestamp.

    /// Format a compact one-liner for logging.
    [[nodiscard]] std::string to_string() const;
};

// ─── SentimentTrendTracker ────────────────────────────────────────────────────

/// Tracks how LLM bias evolves over a rolling window, fitting a linear trend.
///
/// Usage:
/// ```cpp
/// SentimentTrendTracker tracker(window_size=60);
/// tracker.push(signal);
/// if (auto res = tracker.fit()) {
///     fmt::print("{}\n", res->to_string());
/// }
/// ```
class SentimentTrendTracker {
public:
    /// Configuration for the tracker.
    struct Config {
        /// Rolling window size (number of signals to keep).
        std::size_t window_size{60};

        /// Slope magnitude below which the trend is classified as Flat.
        double flat_slope_threshold{0.002};

        /// R² below this value alongside a recent sign change → Reversing.
        double reversing_r2_threshold{0.15};

        /// Minimum samples required before a fit is returned.
        std::size_t min_samples{10};
    };

    explicit SentimentTrendTracker(Config config = Config{}) noexcept;

    /// Push a new signal into the rolling window.
    ///
    /// Thread-safe.
    ///
    /// @param signal  The latest TradeSignal (uses delta_bias_shift).
    void push(const TradeSignal& signal);

    /// Push a raw bias value directly (for testing without full signals).
    ///
    /// Thread-safe.
    ///
    /// @param bias_value  Raw bias shift value.
    /// @param timestamp_ms  Millisecond timestamp of the observation.
    void push_raw(double bias_value,
                  uint64_t timestamp_ms = 0);

    /// Fit a linear regression on the current window and return the result.
    ///
    /// Returns nullopt if fewer than min_samples entries are in the window.
    /// Thread-safe.
    ///
    /// @returns TrendResult or std::nullopt during warm-up.
    [[nodiscard]] std::optional<TrendResult> fit() const;

    /// Return the current window size (number of samples stored).
    /// Thread-safe.
    [[nodiscard]] std::size_t window_fill() const;

    /// Return the most recently cached TrendResult (updated on each push).
    /// Thread-safe.
    [[nodiscard]] std::optional<TrendResult> latest_result() const;

    /// Reset the internal window and cached result.
    void reset();

private:
    Config config_;
    mutable std::mutex mutex_;
    std::deque<double> window_;     ///< Rolling bias values (oldest first).
    std::optional<TrendResult> latest_; ///< Cached result from last push.

    /// Internal OLS fit on the current window contents (called under lock).
    [[nodiscard]] std::optional<TrendResult>
    fit_locked() const;

    /// Classify a slope value as a TrendDirection.
    [[nodiscard]] TrendDirection
    classify(double slope, double r_squared, std::size_t n) const noexcept;
};

// ─── RegimeSwitchEvent ────────────────────────────────────────────────────────

/// A detected sentiment regime switch event.
struct RegimeSwitch {
    TrendDirection from;          ///< Previous regime (Bullish/Bearish).
    TrendDirection to;            ///< New regime.
    double         confidence;    ///< Estimated confidence in the switch [0, 1].
    uint64_t       timestamp_ms;  ///< Wall-clock milliseconds when detected.

    /// Format a one-liner for logging.
    [[nodiscard]] std::string to_string() const;
};

/// Callback type invoked when a regime switch is detected.
using RegimeSwitchCallback = std::function<void(const RegimeSwitch&)>;

// ─── RegimeSwitchDetector ─────────────────────────────────────────────────────

/// Detects when sentiment crosses from positive to negative territory.
///
/// Internally wraps a ``SentimentTrendTracker`` and monitors its output.
/// Emits a ``RegimeSwitch`` event when:
///   1. The bias mean changes sign (positive → negative or vice versa).
///   2. The new sign has been confirmed for at least ``confirm_bars`` samples.
///   3. The transition confidence (based on R² and slope magnitude) exceeds
///      ``min_confidence``.
///
/// During a confirmed switch, ``is_switching()`` returns true.  The window
/// of uncertainty lasts ``uncertainty_bars`` samples after detection.
///
/// Usage:
/// ```cpp
/// RegimeSwitchDetector detector;
/// detector.set_switch_callback([](const RegimeSwitch& sw) {
///     fmt::print("Regime switch: {}\n", sw.to_string());
///     risk_manager.disable_all_gates();  // halt trading
/// });
///
/// for (auto& signal : stream) {
///     detector.push(signal);
///     if (detector.is_switching()) continue;  // skip during uncertainty
/// }
/// ```
class RegimeSwitchDetector {
public:
    /// Configuration for the regime switch detector.
    struct Config {
        /// Underlying SentimentTrendTracker config.
        SentimentTrendTracker::Config tracker_cfg{};

        /// Number of consecutive same-sign samples required to confirm a switch.
        std::size_t confirm_bars{5};

        /// Number of bars after detection during which is_switching() returns true.
        std::size_t uncertainty_bars{20};

        /// Minimum confidence required to fire the switch callback.
        double min_confidence{0.30};
    };

    explicit RegimeSwitchDetector(Config config = Config{}) noexcept;

    /// Push a new signal into the detector.
    ///
    /// May fire the switch callback if a regime change is detected.
    /// Thread-safe.
    ///
    /// @param signal  Latest TradeSignal.
    void push(const TradeSignal& signal);

    /// Push a raw bias value directly.
    ///
    /// Thread-safe.
    ///
    /// @param bias_value    Raw bias shift value.
    /// @param timestamp_ms  Millisecond timestamp.
    void push_raw(double bias_value, uint64_t timestamp_ms = 0);

    /// Return true if the detector is currently in a regime switch uncertainty window.
    ///
    /// During this window callers should disable/reduce trading.
    /// Thread-safe.
    [[nodiscard]] bool is_switching() const noexcept;

    /// Return the most recent RegimeSwitch, if any has been detected.
    /// Thread-safe.
    [[nodiscard]] std::optional<RegimeSwitch> last_switch() const;

    /// Register a callback invoked when a regime switch is detected.
    ///
    /// @warning Callback is invoked synchronously inside push().  It must not
    ///          call push() or any other RegimeSwitchDetector method.
    void set_switch_callback(RegimeSwitchCallback cb);

    /// Return the current TrendResult from the underlying tracker.
    /// Thread-safe.
    [[nodiscard]] std::optional<TrendResult> current_trend() const;

    /// Reset all internal state.
    void reset();

private:
    Config config_;
    SentimentTrendTracker tracker_;
    mutable std::mutex mutex_;

    // Switch detection state.
    TrendDirection current_regime_{TrendDirection::Flat};
    int            same_sign_count_{0};   ///< Consecutive samples on the new side.
    bool           switching_{false};     ///< Currently in uncertainty window.
    std::size_t    uncertainty_remaining_{0}; ///< Bars left in uncertainty window.

    std::optional<RegimeSwitch> last_switch_;
    RegimeSwitchCallback switch_cb_;

    /// Internal push logic (called under lock).
    void push_locked(double bias_value, uint64_t timestamp_ms);

    /// Compute confidence from trend result.
    [[nodiscard]] static double compute_confidence(const TrendResult& tr) noexcept;
};

}  // namespace llmquant
