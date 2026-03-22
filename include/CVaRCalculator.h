#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Rolling Conditional Value-at-Risk (CVaR / Expected Shortfall) calculator.
 *
 * VaR at confidence level α is the threshold loss exceeded only in (1−α) of
 * scenarios.  CVaR (also called Expected Shortfall, ES) is the EXPECTED loss
 * conditional on loss exceeding VaR — it captures tail risk:
 *
 *   CVaR_α = E[L | L ≥ VaR_α]
 *
 * This is the standard coherent risk measure required by Basel III / FRTB.
 *
 * ## Algorithm
 * A rolling window of P&L observations is maintained.  To compute CVaR_α:
 *  1. Sort the observations (ascending).
 *  2. VaR_α is the (1−α)-quantile (e.g. α=0.95 → 5th percentile).
 *  3. CVaR_α is the mean of all observations ≤ VaR_α.
 *
 * ## Usage
 * Feed realised per-trade P&L after each position close.  Use `cvar()` to
 * gate new positions: if CVaR exceeds a drawdown threshold, halt trading.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CVAR` (default ON).
 */
class CVaRCalculator {
public:
    /**
     * @brief Callback fired when CVaR crosses a threshold.
     *
     * Parameters: (cvar_value, var_value, confidence_level).
     */
    using AlertCallback = std::function<void(double cvar, double var, double alpha)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Confidence level α ∈ (0, 1).
         *
         * CVaR is the expected loss in the worst (1−α) fraction of outcomes.
         * Default 0.95 (95% CVaR = expected loss in worst 5% of trades).
         */
        double alpha{0.95};

        /**
         * @brief Rolling window size (number of P&L observations to retain).
         *
         * Larger windows are more stable; smaller windows adapt faster.
         * Default 200.
         */
        int window_size{200};

        /**
         * @brief Minimum samples before CVaR/VaR are meaningful.
         *
         * Default 20.
         */
        int min_samples{20};

        /**
         * @brief CVaR threshold (negative for losses).
         *
         * When CVaR drops below this value, on_breach fires.
         * 0.0 = disabled.  Default -0.1 (-10% expected tail loss).
         */
        double breach_threshold{-0.1};

        /**
         * @brief Callback fired when CVaR drops below breach_threshold.
         *
         * Called from the thread invoking record().  Must not block.
         */
        AlertCallback on_breach;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit CVaRCalculator(Config config = Config{});

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /**
     * @brief Record a realised P&L observation.
     *
     * Positive = profit, negative = loss.  Updates the rolling window and
     * recomputes VaR and CVaR.  Thread-safe.
     *
     * @param pnl  Realised P&L (normalised, e.g. as fraction of position value).
     */
    void record(double pnl);

    /**
     * @brief Clear the window and reset all statistics.
     *
     * Thread-safe.
     */
    void reset();

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /**
     * @brief Current CVaR (Expected Shortfall).
     *
     * Returns 0 if fewer than min_samples observed.  Thread-safe.
     * Negative values indicate expected losses.
     */
    [[nodiscard]] double cvar() const noexcept;

    /**
     * @brief Current VaR at the configured α level.
     *
     * Returns 0 if fewer than min_samples observed.  Thread-safe.
     */
    [[nodiscard]] double var() const noexcept;

    /**
     * @brief Number of observations currently in the window.
     *
     * Thread-safe.
     */
    [[nodiscard]] int sample_count() const noexcept;

    /**
     * @brief True if CVaR is below breach_threshold.
     *
     * Thread-safe.
     */
    [[nodiscard]] bool is_in_breach() const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total P&L observations recorded (across all resets). */
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_records_.load(std::memory_order_relaxed);
    }

    /** @brief Number of times CVaR dropped below breach_threshold. */
    [[nodiscard]] uint64_t breach_events() const noexcept {
        return breach_events_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration; clears the window.
     *
     * Thread-safe.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config             config_;
    std::deque<double> window_;
    double             cvar_{0.0};
    double             var_{0.0};

    std::atomic<uint64_t> total_records_{0};
    std::atomic<uint64_t> breach_events_{0};

    // Recompute VaR + CVaR from the sorted window.  Called under lock.
    void recompute();
};

} // namespace llmquant
