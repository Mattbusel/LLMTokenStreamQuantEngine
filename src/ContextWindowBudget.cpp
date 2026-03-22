#include "ContextWindowBudget.h"

#include <algorithm>
#include <sstream>

namespace llmquant {

ContextWindowBudget::ContextWindowBudget(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// consume
// ---------------------------------------------------------------------------
ContextWindowBudget::Tier ContextWindowBudget::consume(uint64_t tokens) {
    // Increment atomically; then check tiers under lock for callback state.
    uint64_t used = tokens_used_.fetch_add(tokens, std::memory_order_relaxed) + tokens;

    // Determine tier.
    auto cap  = config_.capacity;
    auto soft = static_cast<uint64_t>(static_cast<double>(cap) * config_.soft_fraction);
    auto hard = static_cast<uint64_t>(static_cast<double>(cap) * config_.hard_fraction);

    Tier tier = Tier::Normal;
    if (used >= cap)  tier = Tier::Overflow;
    else if (used >= hard) tier = Tier::Critical;
    else if (used >= soft) tier = Tier::Warn;

    // Fire callbacks (at most once per tier per session) under lock.
    std::function<void(uint64_t, uint64_t)> warn_cb, crit_cb, over_cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (tier >= Tier::Warn && !warned_) {
            warned_ = true;
            warn_events_.fetch_add(1, std::memory_order_relaxed);
            warn_cb = config_.on_warn;
        }
        if (tier >= Tier::Critical && !critiqued_) {
            critiqued_ = true;
            critical_events_.fetch_add(1, std::memory_order_relaxed);
            crit_cb = config_.on_critical;
        }
        if (tier >= Tier::Overflow && !overflowed_) {
            overflowed_ = true;
            overflow_events_.fetch_add(1, std::memory_order_relaxed);
            over_cb = config_.on_overflow;
        }
    }

    if (warn_cb) warn_cb(used, cap);
    if (crit_cb) crit_cb(used, cap);
    if (over_cb) over_cb(used, cap);

    return tier;
}

void ContextWindowBudget::reset() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        warned_     = false;
        critiqued_  = false;
        overflowed_ = false;
    }
    tokens_used_.store(0, std::memory_order_relaxed);
    total_resets_.fetch_add(1, std::memory_order_relaxed);
}

uint64_t ContextWindowBudget::tokens_remaining() const noexcept {
    uint64_t used = tokens_used_.load(std::memory_order_relaxed);
    auto     cap  = config_.capacity;
    return (used >= cap) ? 0 : cap - used;
}

double ContextWindowBudget::fill_fraction() const noexcept {
    if (config_.capacity == 0) return 1.0;
    double used = static_cast<double>(tokens_used_.load(std::memory_order_relaxed));
    return std::min(1.0, used / static_cast<double>(config_.capacity));
}

ContextWindowBudget::Tier ContextWindowBudget::current_tier() const noexcept {
    uint64_t used = tokens_used_.load(std::memory_order_relaxed);
    auto cap  = config_.capacity;
    auto soft = static_cast<uint64_t>(static_cast<double>(cap) * config_.soft_fraction);
    auto hard = static_cast<uint64_t>(static_cast<double>(cap) * config_.hard_fraction);

    if (used >= cap)  return Tier::Overflow;
    if (used >= hard) return Tier::Critical;
    if (used >= soft) return Tier::Warn;
    return Tier::Normal;
}

/* static */
const char* ContextWindowBudget::tier_name(Tier t) noexcept {
    switch (t) {
        case Tier::Normal:   return "Normal";
        case Tier::Warn:     return "Warn";
        case Tier::Critical: return "Critical";
        case Tier::Overflow: return "Overflow";
    }
    return "Unknown";
}

void ContextWindowBudget::update_config(const Config& config) {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        config_     = config;
        warned_     = false;
        critiqued_  = false;
        overflowed_ = false;
    }
    tokens_used_.store(0, std::memory_order_relaxed);
    total_resets_.fetch_add(1, std::memory_order_relaxed);
}

std::string ContextWindowBudget::to_stats_json() const {
    uint64_t used = tokens_used_.load(std::memory_order_relaxed);
    std::ostringstream os;
    os << "{"
       << "\"tokens_used\":"     << used << ","
       << "\"capacity\":"        << config_.capacity << ","
       << "\"fill_fraction\":"   << fill_fraction() << ","
       << "\"tier\":\""          << tier_name(current_tier()) << "\","
       << "\"total_resets\":"    << total_resets_.load(std::memory_order_relaxed) << ","
       << "\"warn_events\":"     << warn_events_.load(std::memory_order_relaxed) << ","
       << "\"critical_events\":" << critical_events_.load(std::memory_order_relaxed) << ","
       << "\"overflow_events\":" << overflow_events_.load(std::memory_order_relaxed)
       << "}";
    return os.str();
}

} // namespace llmquant
