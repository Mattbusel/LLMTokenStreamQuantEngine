#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

/**
 * @brief Real-time API cost limiter for LLM token stream consumption.
 *
 * Tracks cumulative token count and estimated USD cost against configurable
 * soft and hard budget limits.  Emits tier callbacks and returns Action
 * recommendations to the caller (log, throttle, or halt processing).
 *
 * ## Cost model
 * Cost per token = `cost_per_1k_tokens / 1000.0`.  Set this to match the
 * pricing of the upstream LLM API (e.g. 0.002 USD/1k for GPT-3.5).
 *
 * ## Tiers
 *   - **Normal**    : cost < soft_budget_usd
 *   - **Warning**   : cost >= soft_budget_usd (approaching limit)
 *   - **Hard limit** : cost >= hard_budget_usd (halt recommended)
 *
 * ## Usage
 * ```cpp
 * TokenCostBudget budget;
 * budget.set_warning_callback([](double cost) {
 *     spdlog::warn("[budget] ${:.4f} — approaching limit", cost);
 * });
 * // Per token:
 * auto action = budget.consume(1);
 * if (action == TokenCostBudget::Action::Halt) stop_streaming();
 * ```
 *
 * ## Thread safety
 * `consume()` is lock-free.  Callbacks are fired at most once per tier.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_TOKEN_COST_BUDGET` (default ON).
 */
class TokenCostBudget {
public:
    /** @brief Recommended action from consume(). */
    enum class Action { Normal, Warn, Halt };

    using BudgetCallback = std::function<void(double cost_usd)>;

    /**
     * @brief Configuration.
     */
    struct Config {
        /**
         * @brief Cost per 1 000 tokens in USD.
         * Default: 0.002 (GPT-3.5-turbo pricing as a baseline).
         */
        double cost_per_1k_tokens{0.002};

        /**
         * @brief Soft budget: warn when cumulative cost reaches this.
         * Default: 1.00 USD.
         */
        double soft_budget_usd{1.00};

        /**
         * @brief Hard budget: halt when cumulative cost reaches this.
         * Default: 2.00 USD.
         */
        double hard_budget_usd{2.00};
    };

    explicit TokenCostBudget(Config cfg = Config{}) : cfg_(cfg) {}

    // Non-copyable.
    TokenCostBudget(const TokenCostBudget&)            = delete;
    TokenCostBudget& operator=(const TokenCostBudget&) = delete;

    /**
     * @brief Account for `count` tokens consumed and return the action tier.
     *
     * Thread-safe.
     *
     * @param count Number of tokens consumed (default 1).
     * @return Action recommendation.
     */
    Action consume(uint64_t count = 1);

    /**
     * @brief Total tokens consumed since construction or reset().
     */
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Cumulative estimated cost in USD.
     */
    [[nodiscard]] double total_cost_usd() const noexcept;

    /**
     * @brief Current action tier without consuming tokens.
     */
    [[nodiscard]] Action current_action() const noexcept;

    /**
     * @brief Fraction of hard budget consumed [0.0, 1.0+].
     */
    [[nodiscard]] double budget_fraction() const noexcept;

    /**
     * @brief Remaining tokens before the hard limit, or 0 if exceeded.
     */
    [[nodiscard]] uint64_t tokens_remaining() const noexcept;

    /**
     * @brief Reset token count and tier state.
     */
    void reset();

    /**
     * @brief Update configuration (resets state).
     */
    void update_config(const Config& cfg);

    /** @brief Human-readable action name. */
    [[nodiscard]] static const char* action_name(Action a) noexcept;

    /** @brief Callback fired once when the soft budget is first exceeded. */
    void set_warning_callback(BudgetCallback cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        warn_cb_ = std::move(cb);
    }

    /** @brief Callback fired once when the hard budget is first exceeded. */
    void set_halt_callback(BudgetCallback cb) {
        std::lock_guard<std::mutex> lk(mtx_);
        halt_cb_ = std::move(cb);
    }

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;
    mutable std::mutex mtx_;

    std::atomic<uint64_t> total_tokens_{0};
    std::atomic<bool>     warned_{false};
    std::atomic<bool>     halted_{false};

    BudgetCallback warn_cb_;
    BudgetCallback halt_cb_;
};

} // namespace llmquant
