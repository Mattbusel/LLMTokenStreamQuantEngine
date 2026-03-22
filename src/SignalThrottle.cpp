#include "SignalThrottle.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace llmquant {

SignalThrottle::SignalThrottle(Config cfg) : cfg_(cfg) {
    tokens_ = (cfg_.initial_tokens < 0.0) ? cfg_.capacity : cfg_.initial_tokens;
}

void SignalThrottle::refill_locked() {
    auto now = Clock::now();
    double elapsed_s = std::chrono::duration<double>(now - last_refill_).count();
    tokens_ += elapsed_s * cfg_.refill_rate;
    if (tokens_ > cfg_.capacity) tokens_ = cfg_.capacity;
    last_refill_ = now;
}

bool SignalThrottle::try_acquire() {
    total_attempts_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(mtx_);
    refill_locked();

    if (tokens_ >= 1.0) {
        tokens_ -= 1.0;
        passed_count_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    dropped_count_.fetch_add(1, std::memory_order_relaxed);
    if (drop_cb_) drop_cb_();
    return false;
}

double SignalThrottle::drop_rate() const noexcept {
    uint64_t total = total_attempts_.load(std::memory_order_relaxed);
    if (total == 0) return 0.0;
    return static_cast<double>(dropped_count_.load(std::memory_order_relaxed))
           / static_cast<double>(total);
}

double SignalThrottle::current_tokens() const {
    std::lock_guard<std::mutex> lk(mtx_);
    // Compute without advancing the last_refill_ clock.
    double elapsed_s = std::chrono::duration<double>(Clock::now() - last_refill_).count();
    double estimate = tokens_ + elapsed_s * cfg_.refill_rate;
    return std::min(estimate, cfg_.capacity);
}

void SignalThrottle::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    tokens_      = cfg_.capacity;
    last_refill_ = Clock::now();
    total_attempts_.store(0, std::memory_order_relaxed);
    passed_count_.store(0, std::memory_order_relaxed);
    dropped_count_.store(0, std::memory_order_relaxed);
}

void SignalThrottle::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_    = cfg;
    tokens_ = (cfg_.initial_tokens < 0.0) ? cfg_.capacity : cfg_.initial_tokens;
    last_refill_ = Clock::now();
    total_attempts_.store(0, std::memory_order_relaxed);
    passed_count_.store(0, std::memory_order_relaxed);
    dropped_count_.store(0, std::memory_order_relaxed);
}

std::string SignalThrottle::to_stats_json() const {
    double toks = current_tokens();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"tokens\":%.3f,\"capacity\":%.1f,\"refill_rate\":%.2f,"
        "\"total_attempts\":%llu,\"passed\":%llu,\"dropped\":%llu,"
        "\"drop_rate\":%.4f}",
        toks, cfg_.capacity, cfg_.refill_rate,
        static_cast<unsigned long long>(total_attempts_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(passed_count_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(dropped_count_.load(std::memory_order_relaxed)),
        drop_rate());
    return buf;
}

} // namespace llmquant
