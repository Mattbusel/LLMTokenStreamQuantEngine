#pragma once

#include "BacktestRunner.h"
#include "ParameterSweep.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Rolling walk-forward out-of-sample validation framework.
 *
 * Partitions a token sequence into successive train/test folds, optionally
 * optimising parameters on each training window and evaluating on the
 * subsequent hold-out test window.  Measures how much in-sample performance
 * degrades out-of-sample — the primary diagnostic for parameter overfitting.
 *
 * ## Methodology
 * 1. Split the full token sequence into (train_size + test_size) windows
 *    that advance by step_size each fold.
 * 2. For each fold, optionally run a ParameterSweep on the training slice.
 * 3. Evaluate the best (or fixed) config on the test slice.
 * 4. Aggregate OOS Sharpe, OOS P&L, and degradation vs. IS metrics.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_WALK_FORWARD` (default ON).
 * Requires `LLMQUANT_ENABLE_BACKTEST=ON` and `LLMQUANT_ENABLE_PARAMETER_SWEEP=ON`.
 *
 * ## Typical usage
 * @code
 * WalkForwardValidator wfv;
 * wfv.load_tokens({"bullish","rally","fear","crash",...});
 * wfv.add_axis("bias_sensitivity", {0.2, 0.5, 1.0});
 * auto result = wfv.run();
 * std::cout << result.to_summary_string();
 * @endcode
 *
 * Thread safety: not thread-safe; use one instance per thread.
 */
class WalkForwardValidator {
public:
    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    /**
     * @brief Result for a single train/test fold.
     */
    struct FoldResult {
        /** @brief Zero-based fold index. */
        int fold_id{0};

        /** @brief Token index range [begin, end) for the training window. */
        int train_begin{0};
        int train_end{0};

        /** @brief Token index range [begin, end) for the test window. */
        int test_begin{0};
        int test_end{0};

        /**
         * @brief Best parameter values found on the training window.
         *
         * Corresponds to WalkForwardValidator::axis_names() in order.
         * Empty when optimize=false.
         */
        std::vector<double> best_params;

        /** @brief Backtest result evaluated on the training window. */
        BacktestRunner::BacktestResult train_result;

        /** @brief Backtest result evaluated on the test (hold-out) window. */
        BacktestRunner::BacktestResult test_result;
    };

    /**
     * @brief Aggregate walk-forward validation statistics.
     */
    struct WalkForwardResult {
        /** @brief Per-fold outcomes in chronological order. */
        std::vector<FoldResult> folds;

        /**
         * @brief Mean out-of-sample Sharpe ratio across all folds.
         *
         * A reliable strategy should have OOS Sharpe > 0.
         */
        double mean_oos_sharpe{0.0};

        /**
         * @brief Mean in-sample Sharpe ratio across all folds.
         */
        double mean_is_sharpe{0.0};

        /**
         * @brief Sharpe degradation (IS − OOS).
         *
         * Large positive values indicate parameter overfitting.
         * Values near zero suggest genuinely predictive parameters.
         */
        double sharpe_degradation{0.0};

        /**
         * @brief Combined out-of-sample PnL (sum across all test windows).
         *
         * Each test window's PnL is added to produce a realistic equity curve
         * that is free of look-ahead bias.
         */
        double combined_oos_pnl{0.0};

        /** @brief Mean OOS win rate across folds (fraction of profitable signals). */
        double mean_oos_win_rate{0.0};

        /** @brief Number of folds where OOS Sharpe > 0. */
        int positive_oos_folds{0};

        /**
         * @brief Format a human-readable multi-line summary.
         */
        [[nodiscard]] std::string to_summary_string() const;
    };

    /**
     * @brief Walk-forward configuration.
     */
    struct Config {
        /**
         * @brief Number of tokens in each training window.
         *
         * Should be large enough for meaningful ParameterSweep optimisation.
         * Default 200.
         */
        int train_size{200};

        /**
         * @brief Number of tokens in each test (hold-out) window.
         *
         * Default 50.
         */
        int test_size{50};

        /**
         * @brief Number of tokens to advance between successive folds.
         *
         * Equal to test_size gives non-overlapping test windows (anchored
         * walk-forward).  Smaller values give more overlapping evaluations.
         * Default equals test_size.
         */
        int step_size{50};

        /**
         * @brief When true, run ParameterSweep on each training window
         *        and use the best params for the test window.
         *
         * When false, use base_config for all folds without optimisation.
         */
        bool optimize{true};

        /**
         * @brief Base BacktestRunner config (starting point for optimisation).
         */
        BacktestRunner::Config base_config;

        /**
         * @brief Optimisation objective for within-fold ParameterSweep.
         */
        ParameterSweep::Objective objective{ParameterSweep::Objective::SharpeRatio};

        /**
         * @brief Progress callback: called after each fold completes.
         *
         * Parameters: (completed_folds, total_folds).
         */
        std::function<void(int, int)> on_fold_complete;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit WalkForwardValidator(Config config = Config{});

    // -----------------------------------------------------------------------
    // Token input
    // -----------------------------------------------------------------------

    /**
     * @brief Set the full token sequence to partition into folds.
     */
    void load_tokens(std::vector<std::string> tokens);

    /**
     * @brief Load token sequence from a file (one token per line).
     *
     * @return true on success; false if file cannot be opened.
     */
    [[nodiscard]] bool load_tokens_from_file(const std::string& filepath);

    // -----------------------------------------------------------------------
    // Axes (used only when optimize=true)
    // -----------------------------------------------------------------------

    /**
     * @brief Add a parameter axis for within-fold optimisation.
     *
     * Supports the same axis names as ParameterSweep::add_axis().
     */
    void add_axis(const std::string& name, std::vector<double> values);

    /**
     * @brief Clear all registered axes.
     */
    void clear_axes();

    /**
     * @brief Return the ordered list of registered axis names.
     */
    [[nodiscard]] const std::vector<std::string>& axis_names() const noexcept {
        return axis_names_;
    }

    /**
     * @brief Compute the number of folds given the current token count and config.
     *
     * @return Number of folds that will be evaluated (0 if insufficient tokens).
     */
    [[nodiscard]] int num_folds() const noexcept;

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /**
     * @brief Execute the full walk-forward validation.
     *
     * For each fold:
     *  - If optimize=true, runs ParameterSweep on the training slice.
     *  - Applies the best (or base) config to a BacktestRunner on test slice.
     *  - Records per-fold IS and OOS statistics.
     *
     * @return Aggregate walk-forward result.
     */
    [[nodiscard]] WalkForwardResult run();

private:
    Config                          config_;
    std::vector<std::string>        tokens_;
    std::vector<ParameterSweep::Axis> axes_;
    std::vector<std::string>        axis_names_;

    [[nodiscard]] BacktestRunner::Config optimise_on_slice(
        const std::vector<std::string>& train_tokens) const;

    static WalkForwardResult aggregate(std::vector<FoldResult> folds);
};

} // namespace llmquant
