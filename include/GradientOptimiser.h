#pragma once

#include "BacktestRunner.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Finite-difference gradient-descent optimiser over BacktestRunner configs.
 *
 * Complements ParameterSweep (exhaustive grid search) with a local search
 * that converges orders of magnitude faster for high-dimensional parameter
 * spaces.  Starting from a user-supplied base configuration, the optimiser
 * iteratively estimates the gradient of the objective function via central
 * finite differences and steps in the ascent direction.
 *
 * ## Algorithm
 * Each iteration:
 * 1. For each parameter axis, run two backtests at ±h (the finite-difference
 *    step size, which is halved if results do not improve).
 * 2. Compute the numerical gradient: g_i = (f(+h) − f(−h)) / (2h).
 * 3. Update: p_i += learning_rate * g_i.
 * 4. Clamp each parameter to its [lo, hi] bounds.
 * 5. Accept the new point if it strictly improves the objective.
 *    Otherwise halve the step size and retry (Armijo-style backtracking).
 * 6. Terminate when |gradient| < tolerance or max_iterations is reached.
 *
 * ## Supported axes
 * Same as ParameterSweep::add_axis (spread_cost, price_impact,
 * signal_threshold, signal_decay_rate, min_bias_threshold,
 * min_confidence, max_bias_magnitude).
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_GRADIENT_OPTIMISER` (default ON).
 * Requires LLMQUANT_ENABLE_BACKTEST=ON.
 */
class GradientOptimiser {
public:
    /**
     * @brief Ranking objective — same as ParameterSweep::Objective.
     */
    enum class Objective {
        SharpeRatio,  ///< Maximise Sharpe ratio.
        TotalPnL,     ///< Maximise cumulative PnL.
        WinRate,      ///< Maximise fraction of profitable signals.
        PassRate,     ///< Maximise pass-through rate.
        MinDrawdown,  ///< Minimise maximum drawdown.
    };

    /**
     * @brief A bounded axis over which the optimiser can move.
     */
    struct Axis {
        std::string name;    ///< Parameter name (see BacktestRunner axis list).
        double      initial; ///< Starting value.
        double      lo;      ///< Lower bound (clamped during descent).
        double      hi;      ///< Upper bound (clamped during descent).
    };

    /**
     * @brief One step recorded during the optimisation run.
     */
    struct Step {
        int    iteration;                   ///< Step index (0-based).
        std::vector<double> params;         ///< Current parameter values.
        double objective_value;             ///< Objective at this point.
        double grad_norm;                   ///< Euclidean norm of gradient vector.
        int    backtests_run;               ///< Backtests evaluated this iteration.
        BacktestRunner::BacktestResult result; ///< Full backtest result.
    };

    /**
     * @brief Full result of one optimisation run.
     */
    struct OptimiseResult {
        std::vector<std::string> axis_names;  ///< Axis names in order.
        std::vector<Step>        steps;        ///< History of all accepted steps.
        int    total_backtests{0};             ///< Total backtests evaluated.
        bool   converged{false};               ///< True if |grad| < tolerance.

        /**
         * @brief Return the step with the highest objective value.
         *
         * Returns nullptr if steps is empty.
         */
        [[nodiscard]] const Step* best() const noexcept;

        /**
         * @brief Format optimisation trace as an ASCII table.
         */
        [[nodiscard]] std::string to_table_string() const;
    };

    /**
     * @brief Optimiser configuration.
     */
    struct Config {
        /**
         * @brief Base BacktestRunner config (engine + risk + sim settings).
         *
         * Axis parameter values overwrite the relevant fields for each
         * backtest evaluation.
         */
        BacktestRunner::Config base_config;

        /**
         * @brief Objective function to maximise.
         *
         * Default: SharpeRatio.
         */
        Objective objective{Objective::SharpeRatio};

        /**
         * @brief Maximum number of descent iterations.
         *
         * Default 50.
         */
        int max_iterations{50};

        /**
         * @brief Initial finite-difference step size h.
         *
         * Applied multiplicatively: h = initial_step * |param_value| + eps.
         * Default 0.05 (5% of the current parameter value).
         */
        double initial_step{0.05};

        /**
         * @brief Gradient descent learning rate.
         *
         * Scales the gradient update.  Default 0.5.
         */
        double learning_rate{0.5};

        /**
         * @brief Convergence tolerance on the gradient Euclidean norm.
         *
         * Iteration stops when ||∇f|| < tolerance.  Default 1e-4.
         */
        double tolerance{1e-4};

        /**
         * @brief Maximum number of step-size halvings per iteration (backtracking).
         *
         * Default 6 (step shrinks to initial_step / 64 at most).
         */
        int max_backtrack{6};

        /**
         * @brief Progress callback: (iteration, total_backtests, current_objective).
         */
        std::function<void(int, int, double)> progress_cb;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit GradientOptimiser(Config config = Config{});

    // -----------------------------------------------------------------------
    // Setup
    // -----------------------------------------------------------------------

    /**
     * @brief Set the token sequence for all internal backtest runs.
     */
    void set_tokens(std::vector<std::string> tokens);

    /**
     * @brief Add a bounded parameter axis.
     *
     * @param axis Axis descriptor (name, initial value, and bounds).
     */
    void add_axis(const Axis& axis);

    /**
     * @brief Clear all axes.
     */
    void clear_axes();

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /**
     * @brief Run the gradient-descent optimisation.
     *
     * @return Optimisation trace and best-found configuration.
     */
    [[nodiscard]] OptimiseResult run();

private:
    Config             config_;
    std::vector<Axis>  axes_;
    std::vector<std::string> tokens_;

    // Apply axis values to a BacktestRunner::Config copy.
    void apply_params(BacktestRunner::Config& cfg,
                      const std::vector<double>& params) const;

    // Run a single backtest and extract the objective value.
    [[nodiscard]] double evaluate(const std::vector<double>& params,
                                  BacktestRunner::BacktestResult* result_out);

    [[nodiscard]] static double extract_objective(
        const BacktestRunner::BacktestResult& r, Objective obj) noexcept;
};

} // namespace llmquant
