#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace llmquant {

// ---------------------------------------------------------------------------
// TokenBudget — configuration
// ---------------------------------------------------------------------------

/**
 * @brief Static budget limits and cost parameters for a single LLM session.
 */
struct TokenBudget {
    /// Maximum number of input (prompt) tokens allowed in the session.
    std::size_t max_input_tokens{4096};

    /// Maximum number of output (completion) tokens allowed in the session.
    std::size_t max_output_tokens{4096};

    /// Maximum combined token count (input + output).
    std::size_t max_total_tokens{8192};

    /// Cost per token in USD (applies uniformly to input and output tokens).
    double cost_per_token_usd{0.000002};
};

// ---------------------------------------------------------------------------
// TokenUsage — live accounting
// ---------------------------------------------------------------------------

/**
 * @brief Accumulated token usage and cost for a session.
 */
struct TokenUsage {
    /// Total input tokens consumed so far.
    std::size_t input_tokens{0};

    /// Total output tokens consumed so far.
    std::size_t output_tokens{0};

    /// Total cost accumulated (USD).
    double total_cost_usd{0.0};

    /// Convenience: total tokens (input + output).
    [[nodiscard]] std::size_t total_tokens() const noexcept {
        return input_tokens + output_tokens;
    }
};

// ---------------------------------------------------------------------------
// TokenBudgetManager
// ---------------------------------------------------------------------------

/**
 * @brief Tracks and enforces LLM API token budgets for a session.
 *
 * Thread safety: NOT thread-safe.  External synchronisation is required
 * if called from multiple threads.
 *
 * Usage example:
 * @code
 *   TokenBudget budget;
 *   budget.max_input_tokens  = 1024;
 *   budget.max_output_tokens = 512;
 *   budget.max_total_tokens  = 1536;
 *   budget.cost_per_token_usd = 0.000002;
 *
 *   TokenBudgetManager mgr(budget);
 *   if (mgr.check_input(100)) {
 *       mgr.consume_input(100);
 *   }
 * @endcode
 */
class TokenBudgetManager {
public:
    /**
     * @brief Construct with the given budget limits.
     * @param budget  Static limits and cost parameters.
     */
    explicit TokenBudgetManager(const TokenBudget& budget);

    // ------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------

    /**
     * @brief Return true if consuming ``tokens`` input tokens would remain
     *        within all budget limits (input cap and total cap).
     *
     * Does NOT modify any state.
     *
     * @param tokens  Number of input tokens to check.
     * @return true  if within budget.
     */
    [[nodiscard]] bool check_input(std::size_t tokens) const noexcept;

    /**
     * @brief Return true if consuming ``tokens`` output tokens would remain
     *        within all budget limits (output cap and total cap).
     *
     * Does NOT modify any state.
     *
     * @param tokens  Number of output tokens to check.
     * @return true  if within budget.
     */
    [[nodiscard]] bool check_output(std::size_t tokens) const noexcept;

    // ------------------------------------------------------------------
    // Consumption
    // ------------------------------------------------------------------

    /**
     * @brief Record consumption of ``tokens`` input tokens.
     *
     * Updates usage and cost accumulators.
     *
     * @param tokens  Number of input tokens consumed.
     * @throws std::runtime_error  If the consumption would exceed the input
     *                             or total token budgets.
     */
    void consume_input(std::size_t tokens);

    /**
     * @brief Record consumption of ``tokens`` output tokens.
     *
     * Updates usage and cost accumulators.
     *
     * @param tokens  Number of output tokens consumed.
     * @throws std::runtime_error  If the consumption would exceed the output
     *                             or total token budgets.
     */
    void consume_output(std::size_t tokens);

    // ------------------------------------------------------------------
    // Remaining budget
    // ------------------------------------------------------------------

    /**
     * @brief Number of output tokens still available within the output cap
     *        and total cap (whichever is tighter).
     */
    [[nodiscard]] std::size_t remaining_output_budget() const noexcept;

    /**
     * @brief Number of input tokens still available within the input cap
     *        and total cap (whichever is tighter).
     */
    [[nodiscard]] std::size_t remaining_input_budget() const noexcept;

    // ------------------------------------------------------------------
    // Utilisation
    // ------------------------------------------------------------------

    /**
     * @brief Fraction of the total token budget consumed, in [0.0, 1.0].
     *
     * Returns 0.0 if max_total_tokens is zero (avoid division-by-zero).
     */
    [[nodiscard]] double utilization() const noexcept;

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return a read-only snapshot of current usage.
    [[nodiscard]] const TokenUsage& usage() const noexcept { return usage_; }

    /// Return a read-only snapshot of the budget limits.
    [[nodiscard]] const TokenBudget& budget() const noexcept { return budget_; }

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    /**
     * @brief Reset all usage accumulators to zero.
     *
     * Budget limits (TokenBudget) are preserved.
     */
    void reset() noexcept;

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    /**
     * @brief Format a human-readable single-line summary of current state.
     *
     * Example: "input=100/1024 output=50/512 total=150/1536 cost=$0.000300 util=0.10"
     */
    [[nodiscard]] std::string format_summary() const;

private:
    TokenBudget budget_;
    TokenUsage  usage_;
};

} // namespace llmquant
