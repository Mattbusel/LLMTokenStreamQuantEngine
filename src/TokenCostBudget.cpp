#include "TokenCostBudget.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace llmquant {

const char* TokenCostBudget::action_name(Action a) noexcept {
    switch (a) {
        case Action::Normal: return "Normal";
        case Action::Warn:   return "Warn";
        case Action::Halt:   return "Halt";
    }
    return "Unknown";
}

double TokenCostBudget::total_cost_usd() const noexcept {
    return static_cast<double>(total_tokens_.load(std::memory_order_relaxed))
           * cfg_.cost_per_1k_tokens / 1000.0;
}

TokenCostBudget::Action TokenCostBudget::current_action() const noexcept {
    double cost = total_cost_usd();
    if (cost >= cfg_.hard_budget_usd) return Action::Halt;
    if (cost >= cfg_.soft_budget_usd) return Action::Warn;
    return Action::Normal;
}

double TokenCostBudget::budget_fraction() const noexcept {
    if (cfg_.hard_budget_usd <= 0.0) return 0.0;
    return total_cost_usd() / cfg_.hard_budget_usd;
}

uint64_t TokenCostBudget::tokens_remaining() const noexcept {
    double remaining_usd = cfg_.hard_budget_usd - total_cost_usd();
    if (remaining_usd <= 0.0 || cfg_.cost_per_1k_tokens <= 0.0) return 0;
    double tokens = remaining_usd * 1000.0 / cfg_.cost_per_1k_tokens;
    return static_cast<uint64_t>(tokens);
}

TokenCostBudget::Action TokenCostBudget::consume(uint64_t count) {
    total_tokens_.fetch_add(count, std::memory_order_relaxed);
    double cost = total_cost_usd();

    if (cost >= cfg_.hard_budget_usd) {
        bool was_halted = halted_.exchange(true, std::memory_order_relaxed);
        if (!was_halted) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (halt_cb_) halt_cb_(cost);
        }
        return Action::Halt;
    }

    if (cost >= cfg_.soft_budget_usd) {
        bool was_warned = warned_.exchange(true, std::memory_order_relaxed);
        if (!was_warned) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (warn_cb_) warn_cb_(cost);
        }
        return Action::Warn;
    }

    return Action::Normal;
}

void TokenCostBudget::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    total_tokens_.store(0, std::memory_order_relaxed);
    warned_.store(false, std::memory_order_relaxed);
    halted_.store(false, std::memory_order_relaxed);
}

void TokenCostBudget::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_ = cfg;
    total_tokens_.store(0, std::memory_order_relaxed);
    warned_.store(false, std::memory_order_relaxed);
    halted_.store(false, std::memory_order_relaxed);
}

std::string TokenCostBudget::to_stats_json() const {
    char buf[512];
    double cost   = total_cost_usd();
    double frac   = budget_fraction();
    uint64_t rem  = tokens_remaining();
    std::snprintf(buf, sizeof(buf),
        "{\"action\":\"%s\",\"total_tokens\":%llu,\"total_cost_usd\":%.6f,"
        "\"budget_fraction\":%.4f,\"tokens_remaining\":%llu,"
        "\"soft_budget_usd\":%.4f,\"hard_budget_usd\":%.4f,"
        "\"cost_per_1k\":%.6f}",
        action_name(current_action()),
        static_cast<unsigned long long>(total_tokens_.load(std::memory_order_relaxed)),
        cost, frac,
        static_cast<unsigned long long>(rem),
        cfg_.soft_budget_usd, cfg_.hard_budget_usd,
        cfg_.cost_per_1k_tokens);
    return buf;
}

} // namespace llmquant
