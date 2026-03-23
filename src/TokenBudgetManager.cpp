#include "TokenBudgetManager.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace llmquant {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TokenBudgetManager::TokenBudgetManager(const TokenBudget& budget)
    : budget_(budget), usage_{}
{}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

bool TokenBudgetManager::check_input(std::size_t tokens) const noexcept {
    // Overflow-safe addition.
    if (tokens > budget_.max_input_tokens) return false;
    if (usage_.input_tokens > budget_.max_input_tokens - tokens) return false;

    std::size_t new_total = usage_.total_tokens() + tokens;
    if (new_total > budget_.max_total_tokens) return false;

    return true;
}

bool TokenBudgetManager::check_output(std::size_t tokens) const noexcept {
    if (tokens > budget_.max_output_tokens) return false;
    if (usage_.output_tokens > budget_.max_output_tokens - tokens) return false;

    std::size_t new_total = usage_.total_tokens() + tokens;
    if (new_total > budget_.max_total_tokens) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Consumption
// ---------------------------------------------------------------------------

void TokenBudgetManager::consume_input(std::size_t tokens) {
    if (!check_input(tokens)) {
        throw std::runtime_error(
            "TokenBudgetManager: input token budget exceeded "
            "(requested=" + std::to_string(tokens) +
            " used=" + std::to_string(usage_.input_tokens) +
            " max=" + std::to_string(budget_.max_input_tokens) + ")");
    }
    usage_.input_tokens += tokens;
    usage_.total_cost_usd += static_cast<double>(tokens) * budget_.cost_per_token_usd;
}

void TokenBudgetManager::consume_output(std::size_t tokens) {
    if (!check_output(tokens)) {
        throw std::runtime_error(
            "TokenBudgetManager: output token budget exceeded "
            "(requested=" + std::to_string(tokens) +
            " used=" + std::to_string(usage_.output_tokens) +
            " max=" + std::to_string(budget_.max_output_tokens) + ")");
    }
    usage_.output_tokens += tokens;
    usage_.total_cost_usd += static_cast<double>(tokens) * budget_.cost_per_token_usd;
}

// ---------------------------------------------------------------------------
// Remaining budget
// ---------------------------------------------------------------------------

std::size_t TokenBudgetManager::remaining_output_budget() const noexcept {
    std::size_t cap_output =
        (usage_.output_tokens < budget_.max_output_tokens)
            ? budget_.max_output_tokens - usage_.output_tokens
            : 0;

    std::size_t total_used = usage_.total_tokens();
    std::size_t cap_total  =
        (total_used < budget_.max_total_tokens)
            ? budget_.max_total_tokens - total_used
            : 0;

    return std::min(cap_output, cap_total);
}

std::size_t TokenBudgetManager::remaining_input_budget() const noexcept {
    std::size_t cap_input =
        (usage_.input_tokens < budget_.max_input_tokens)
            ? budget_.max_input_tokens - usage_.input_tokens
            : 0;

    std::size_t total_used = usage_.total_tokens();
    std::size_t cap_total  =
        (total_used < budget_.max_total_tokens)
            ? budget_.max_total_tokens - total_used
            : 0;

    return std::min(cap_input, cap_total);
}

// ---------------------------------------------------------------------------
// Utilisation
// ---------------------------------------------------------------------------

double TokenBudgetManager::utilization() const noexcept {
    if (budget_.max_total_tokens == 0) return 0.0;
    double util = static_cast<double>(usage_.total_tokens())
                / static_cast<double>(budget_.max_total_tokens);
    return util > 1.0 ? 1.0 : util;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void TokenBudgetManager::reset() noexcept {
    usage_ = TokenUsage{};
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string TokenBudgetManager::format_summary() const {
    char buf[256];
    std::snprintf(
        buf, sizeof(buf),
        "input=%zu/%zu output=%zu/%zu total=%zu/%zu cost=$%.6f util=%.2f",
        usage_.input_tokens,  budget_.max_input_tokens,
        usage_.output_tokens, budget_.max_output_tokens,
        usage_.total_tokens(), budget_.max_total_tokens,
        usage_.total_cost_usd,
        utilization());
    return buf;
}

} // namespace llmquant
