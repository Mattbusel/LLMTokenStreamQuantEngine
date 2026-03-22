#include "LatencyBudgetEnforcer.h"

#include <cstdio>
#include <mutex>

namespace llmquant {

// ---------------------------------------------------------------------------
// action_name
// ---------------------------------------------------------------------------

const char* LatencyBudgetEnforcer::action_name(Action a) noexcept {
    switch (a) {
        case Action::Normal:   return "Normal";
        case Action::Warn:     return "Warn";
        case Action::Throttle: return "Throttle";
        case Action::Drop:     return "Drop";
        case Action::Breaker:  return "Breaker";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

LatencyBudgetEnforcer::Action LatencyBudgetEnforcer::check(int64_t p99_us) {
    std::lock_guard<std::mutex> lk(mtx_);

    // Determine target tier from thresholds.
    Action target = Action::Normal;
    if      (p99_us >= cfg_.breaker_us)  target = Action::Breaker;
    else if (p99_us >= cfg_.drop_us)     target = Action::Drop;
    else if (p99_us >= cfg_.throttle_us) target = Action::Throttle;
    else if (p99_us >= cfg_.warn_us)     target = Action::Warn;

    Action current = static_cast<Action>(current_tier_.load(std::memory_order_relaxed));

    // Apply recovery hysteresis when de-escalating.
    if (target < current) {
        // Compute the recovery threshold for the current tier.
        int64_t tier_threshold = 0;
        switch (current) {
            case Action::Breaker:  tier_threshold = cfg_.breaker_us;  break;
            case Action::Drop:     tier_threshold = cfg_.drop_us;     break;
            case Action::Throttle: tier_threshold = cfg_.throttle_us; break;
            case Action::Warn:     tier_threshold = cfg_.warn_us;     break;
            default: break;
        }
        double recovery_level = static_cast<double>(tier_threshold) * cfg_.recovery_factor;
        if (static_cast<double>(p99_us) >= recovery_level) {
            // Not yet below hysteresis band — stay in current tier.
            target = current;
        }
    }

    if (target == current) {
        return current;
    }

    // Tier changed — update and fire callback.
    current_tier_.store(static_cast<int>(target), std::memory_order_relaxed);

    if (target > current) {
        escalation_count_.fetch_add(1, std::memory_order_relaxed);
        switch (target) {
            case Action::Warn:     if (warn_cb_)     warn_cb_(p99_us);     break;
            case Action::Throttle: if (throttle_cb_) throttle_cb_(p99_us); break;
            case Action::Drop:     if (drop_cb_)     drop_cb_(p99_us);     break;
            case Action::Breaker:  if (breaker_cb_)  breaker_cb_(p99_us);  break;
            default: break;
        }
    } else {
        recovery_count_.fetch_add(1, std::memory_order_relaxed);
        if (recovery_cb_) recovery_cb_(p99_us);
    }

    return target;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void LatencyBudgetEnforcer::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    current_tier_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string LatencyBudgetEnforcer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"action\":\"%s\",\"escalation_count\":%llu,\"recovery_count\":%llu,"
        "\"warn_us\":%lld,\"throttle_us\":%lld,\"drop_us\":%lld,\"breaker_us\":%lld}",
        action_name(static_cast<Action>(current_tier_.load(std::memory_order_relaxed))),
        static_cast<unsigned long long>(escalation_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(recovery_count_.load(std::memory_order_relaxed)),
        static_cast<long long>(cfg_.warn_us),
        static_cast<long long>(cfg_.throttle_us),
        static_cast<long long>(cfg_.drop_us),
        static_cast<long long>(cfg_.breaker_us));
    return buf;
}

} // namespace llmquant
