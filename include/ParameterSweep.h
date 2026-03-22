#pragma once

#include "BacktestRunner.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Grid-search optimiser over BacktestRunner configurations.
 *
 * Runs a full Cartesian product sweep over user-specified parameter ranges and
 * ranks each configuration by Sharpe ratio, total PnL, win rate, and pass rate.
 *
 * ## Typical usage
 * @code
 * ParameterSweep sweep;
 * sweep.set_tokens({"bullish", "crash", "rally", "bearish", ...});
 * sweep.add_axis("spread_cost",   {0.0001, 0.0005, 0.001});
 * sweep.add_axis("price_impact",  {0.001, 0.005, 0.01});
 * sweep.set_objective(ParameterSweep::Objective::SharpeRatio);
 * auto results = sweep.run();
 * std::cout << results.to_table_string(10);
 * @endcode
 *
 * ## Parallelism
 * Each grid point is evaluated independently on the calling thread.
 * The sweep is single-threaded by design to avoid non-determinism; wrap in
 * a thread pool externally if parallel evaluation is needed.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_PARAMETER_SWEEP` (default ON).
 * Requires LLMQUANT_ENABLE_BACKTEST=ON.
 */
class ParameterSweep {
public:
    /**
     * @brief Ranking objective for grid search results.
     */
    enum class Objective {
        SharpeRatio,  ///< Maximise per-signal Sharpe ratio (risk-adjusted return).
        TotalPnL,     ///< Maximise raw cumulative PnL.
        WinRate,      ///< Maximise fraction of profitable signals.
        PassRate,     ///< Maximise fraction of signals passing risk gates.
        MinDrawdown,  ///< Minimise maximum peak-to-trough drawdown.
    };

    /**
     * @brief Named parameter axis for the grid.
     *
     * The axis name must match one of the supported sweep parameters (see below).
     */
    struct Axis {
        std::string          name;    ///< Parameter name (see supported list).
        std::vector<double>  values;  ///< Values to sweep over.
    };

    /**
     * @brief Outcome for one grid point.
     */
    struct GridPoint {
        std::vector<double>           param_values;  ///< Axis values for this point.
        BacktestRunner::BacktestResult result;        ///< Full backtest result.
        double                        objective_value{0.0}; ///< Value used for ranking.
    };

    /**
     * @brief Sweep results: all grid points, sorted best-first by objective.
     */
    struct SweepResult {
        std::vector<std::string> axis_names;    ///< Axis names in order.
        std::vector<GridPoint>   grid_points;   ///< Sorted best-first.
        Objective                objective;

        /**
         * @brief Format the top-N results as an ASCII table.
         *
         * @param top_n Number of rows to include (0 = all).
         * @return Multi-line table string.
         */
        [[nodiscard]] std::string to_table_string(int top_n = 0) const;

        /**
         * @brief Return the best grid point (lowest index after sorting).
         *
         * Returns nullptr if grid_points is empty.
         */
        [[nodiscard]] const GridPoint* best() const noexcept {
            return grid_points.empty() ? nullptr : &grid_points.front();
        }
    };

    /**
     * @brief Configuration for the sweep execution.
     */
    struct Config {
        /**
         * @brief Base BacktestRunner config applied before each sweep override.
         *
         * Axis parameter values overwrite the relevant fields in this config
         * for each grid point.
         */
        BacktestRunner::Config base_config;

        /**
         * @brief Ranking objective.
         *
         * Default: SharpeRatio.
         */
        Objective objective{Objective::SharpeRatio};

        /**
         * @brief Maximum number of grid points to evaluate.
         *
         * Limits computation time for large sweeps.
         * 0 = no limit.  Default 0.
         */
        int max_points{0};

        /**
         * @brief Progress callback: invoked after each grid point is evaluated.
         *
         * Parameters: (completed_count, total_count).
         */
        std::function<void(int, int)> progress_cb;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit ParameterSweep(Config config = Config{});

    // -----------------------------------------------------------------------
    // Input
    // -----------------------------------------------------------------------

    /**
     * @brief Set the token sequence used for all backtest runs.
     *
     * @param tokens Token strings (replayed in order for each grid point).
     */
    void set_tokens(std::vector<std::string> tokens);

    /**
     * @brief Load the token sequence from a file (one token per line).
     *
     * @param filepath Path to the token file.
     * @return true on success; false if the file cannot be opened.
     */
    [[nodiscard]] bool load_tokens_from_file(const std::string& filepath);

    /**
     * @brief Add a parameter axis to the sweep grid.
     *
     * Supported axis names:
     *   "spread_cost"        — SimParams::spread_cost
     *   "price_impact"       — SimParams::price_impact_per_bias
     *   "signal_threshold"   — TradeSignalEngine::Config::bias_sensitivity
     *   "signal_decay_rate"  — TradeSignalEngine::Config::signal_decay_rate
     *   "min_bias_threshold" — TradeSignalEngine::Config::min_bias_threshold
     *   "min_confidence"     — RiskManager::Config::min_confidence
     *   "max_bias_magnitude" — RiskManager::Config::max_bias_magnitude
     *
     * @param name   Axis name (must match one of the supported names above).
     * @param values Values to sweep.
     */
    void add_axis(const std::string& name, std::vector<double> values);

    /**
     * @brief Clear all axes (keeps tokens and base config).
     */
    void clear_axes();

    /**
     * @brief Return total number of grid points (Cartesian product of all axes).
     *
     * @return Total grid point count.
     */
    [[nodiscard]] int total_grid_points() const;

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /**
     * @brief Run the full parameter sweep.
     *
     * Evaluates every grid point (up to max_points if set), sorts by objective,
     * and returns the ranked results.
     *
     * @return Sweep results, sorted best-first.
     */
    [[nodiscard]] SweepResult run();

private:
    Config config_;
    std::vector<Axis> axes_;
    std::vector<std::string> tokens_;

    void apply_axis_values(BacktestRunner::Config& cfg,
                           const std::vector<std::string>& axis_names,
                           const std::vector<double>& values) const;

    [[nodiscard]] static double extract_objective(const BacktestRunner::BacktestResult& r,
                                                  Objective obj) noexcept;
};

} // namespace llmquant
