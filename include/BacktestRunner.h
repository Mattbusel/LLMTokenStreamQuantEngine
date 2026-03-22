#pragma once

#include "LLMAdapter.h"
#include "RiskManager.h"
#include "TradeSignalEngine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Full-pipeline backtesting harness.
 *
 * Replays a recorded token sequence through the entire signal-generation
 * pipeline (LLMAdapter → TradeSignalEngine → RiskManager) and produces a
 * detailed per-signal record plus aggregate statistics: cumulative PnL,
 * Sharpe ratio, maximum drawdown, and win rate.
 *
 * The simulated PnL model is deliberately simple:
 *   simulated_return = delta_bias_shift * price_impact_per_bias
 *                    - spread_cost * |delta_bias_shift|
 *
 * This model is suitable for relative comparisons between strategy
 * configurations; it is NOT a substitute for a full market simulation.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_BACKTEST` (default ON).
 *
 * ## Typical usage
 * @code
 * BacktestRunner runner;
 * runner.load_tokens({"bullish", "crash", "rally", "the", "fear"});
 * auto result = runner.run();
 * std::cout << result.to_summary_string();
 * @endcode
 *
 * Thread safety: not thread-safe; use one instance per thread or protect
 * externally.
 */
class BacktestRunner {
public:
    // -----------------------------------------------------------------------
    // Backtest simulation parameters
    // -----------------------------------------------------------------------

    /**
     * @brief Simulation cost and return parameters.
     */
    struct SimParams {
        /**
         * @brief Proportional transaction cost per unit of |delta_bias_shift|.
         *
         * Deducted from the gross return on every passed signal.
         * Default 0.0005 (half a basis point per unit of bias).
         */
        double spread_cost{0.0005};

        /**
         * @brief Price return per unit of delta_bias_shift.
         *
         * Scales the raw bias into a simulated price return.
         * Default 0.001 (10 basis points per unit of bias).
         */
        double price_impact_per_bias{0.001};
    };

    /**
     * @brief Composite configuration for the full backtest pipeline.
     */
    struct Config {
        /** @brief TradeSignalEngine tuning parameters. */
        TradeSignalEngine::Config engine;
        /** @brief Risk gate thresholds. */
        RiskManager::Config       risk;
        /** @brief Simulation return and cost parameters. */
        SimParams                 sim;
    };

    // -----------------------------------------------------------------------
    // Per-signal record
    // -----------------------------------------------------------------------

    /**
     * @brief Outcome record for a single signal evaluation.
     */
    struct SignalRecord {
        /** @brief Signal timestamp in nanoseconds (from TradeSignal::timestamp_ns). */
        uint64_t timestamp_ns{0};

        /** @brief Raw signal fields. */
        double bias{0.0};
        double vol{0.0};
        double confidence{0.0};
        double spread{0.0};
        double signal_quality{0.0};

        /** @brief True if all risk gates passed; false if blocked. */
        bool passed{false};

        /** @brief Human-readable block reason (empty when passed). */
        std::string block_reason;

        /**
         * @brief Simulated net return for this signal.
         *
         * Non-zero only for passed signals.
         * simulated_return = bias * price_impact - spread_cost * |bias|
         */
        double simulated_return{0.0};

        /** @brief Running cumulative PnL including this signal. */
        double cumulative_pnl{0.0};
    };

    // -----------------------------------------------------------------------
    // Aggregate statistics
    // -----------------------------------------------------------------------

    /**
     * @brief Aggregate statistics produced by a completed backtest run.
     */
    struct BacktestResult {
        /** @brief All per-signal records in emission order. */
        std::vector<SignalRecord> records;

        /** @brief Total number of signals emitted by TradeSignalEngine. */
        int total_signals{0};

        /** @brief Signals that passed all risk gates. */
        int passed_signals{0};

        /** @brief Signals blocked by at least one risk gate. */
        int blocked_signals{0};

        /** @brief Fraction of signals that passed: passed / total (NaN if 0 total). */
        double pass_rate{0.0};

        /** @brief Sum of all simulated_return values across passed signals. */
        double total_pnl{0.0};

        /**
         * @brief Per-signal Sharpe ratio: mean(returns) / std(returns).
         *
         * Computed over passed signals only.
         * Returns 0.0 if fewer than 2 passed signals exist.
         * Not annualised; represents the risk-adjusted return per signal.
         */
        double sharpe_ratio{0.0};

        /**
         * @brief Maximum peak-to-trough decline in the cumulative PnL curve.
         *
         * Measured in the same units as total_pnl.  Always >= 0.
         */
        double max_drawdown{0.0};

        /**
         * @brief Fraction of passed signals with simulated_return > 0.
         *
         * Returns 0.0 if no signals passed.
         */
        double win_rate{0.0};

        /** @brief Mean |delta_bias_shift| across all emitted signals. */
        double mean_abs_bias{0.0};

        /** @brief Mean confidence across all emitted signals. */
        double mean_confidence{0.0};

        /**
         * @brief Produce a human-readable multi-line summary.
         *
         * @return Formatted summary string suitable for console output or logging.
         */
        [[nodiscard]] std::string to_summary_string() const;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct a BacktestRunner with the given pipeline configuration.
     *
     * All pipeline components (LLMAdapter, TradeSignalEngine, RiskManager)
     * are constructed and owned internally using the provided configuration.
     *
     * @param config Combined configuration for all pipeline stages.
     */
    explicit BacktestRunner(Config config = Config{});

    // -----------------------------------------------------------------------
    // Input loading
    // -----------------------------------------------------------------------

    /**
     * @brief Set the token sequence to replay.
     *
     * Replaces any previously loaded sequence.
     *
     * @param tokens Sequence of raw token strings to process.
     */
    void load_tokens(std::vector<std::string> tokens);

    /**
     * @brief Load token sequence from a text file (one token per line).
     *
     * Lines starting with '#' and empty lines are skipped.
     *
     * @param filepath Path to the token file.
     * @return true on success; false if the file cannot be opened.
     */
    [[nodiscard]] bool load_tokens_from_file(const std::string& filepath);

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /**
     * @brief Replay the loaded token sequence through the full pipeline.
     *
     * Processes each token in order:
     *   1. LLMAdapter::map_token_to_weight
     *   2. TradeSignalEngine::process_semantic_weight (in backtest mode)
     *   3. RiskManager::evaluate_with_reason on each emitted signal
     *   4. Record outcome and compute simulated PnL
     *
     * The engine and risk manager are reset before each run so that
     * successive calls to run() produce independent results.
     *
     * @return Backtest result with per-signal records and aggregate statistics.
     */
    [[nodiscard]] BacktestResult run();

    /**
     * @brief Convenience: load tokens from a file and immediately run.
     *
     * @param filepath Path to the token file.
     * @return BacktestResult; records is empty and total_signals is -1 on file error.
     */
    [[nodiscard]] BacktestResult run_from_file(const std::string& filepath);

private:
    Config config_;
    std::vector<std::string> tokens_;

    static BacktestResult compute_stats(std::vector<SignalRecord> records,
                                        const SimParams& sim);
};

} // namespace llmquant
