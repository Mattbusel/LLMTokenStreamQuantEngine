#include "StaleTokenDetector.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <sstream>

namespace llmquant {

namespace {

int64_t epoch_ms_now() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
               .count();
}

} // anonymous namespace

StaleTokenDetector::StaleTokenDetector(Config config)
    : config_(config)
    , last_token_epoch_ms_(epoch_ms_now())
{}

void StaleTokenDetector::record_token() noexcept {
    last_token_epoch_ms_.store(epoch_ms_now(), std::memory_order_relaxed);
}

bool StaleTokenDetector::check() noexcept {
    int64_t gap_ms = ms_since_last_token();
    int threshold = config_.stale_threshold_ms;

    bool currently_stale = (gap_ms >= static_cast<int64_t>(threshold));
    bool was_stale       = stale_.load(std::memory_order_relaxed);

    if (currently_stale && !was_stale) {
        // Transition: Fresh → Stale
        stale_.store(true, std::memory_order_relaxed);
        stale_events_.fetch_add(1, std::memory_order_relaxed);
        if (stale_cb_) {
            try { stale_cb_(gap_ms); } catch (...) {}
        }
    } else if (!currently_stale && was_stale) {
        // Transition: Stale → Fresh (recovery)
        stale_.store(false, std::memory_order_relaxed);
        recovery_events_.fetch_add(1, std::memory_order_relaxed);
        if (recovery_cb_) {
            try { recovery_cb_(); } catch (...) {}
        }
    }

    return currently_stale;
}

int64_t StaleTokenDetector::ms_since_last_token() const noexcept {
    return epoch_ms_now() - last_token_epoch_ms_.load(std::memory_order_relaxed);
}

void StaleTokenDetector::set_stale_callback(StaleCallback cb) {
    stale_cb_ = std::move(cb);
}

void StaleTokenDetector::set_recovery_callback(RecoveryCallback cb) {
    recovery_cb_ = std::move(cb);
}

void StaleTokenDetector::update_config(const Config& config) {
    config_ = config;
}

void StaleTokenDetector::reset() noexcept {
    last_token_epoch_ms_.store(epoch_ms_now(), std::memory_order_relaxed);
    stale_.store(false, std::memory_order_relaxed);
}

std::string StaleTokenDetector::to_stats_json() const {
    int64_t gap   = ms_since_last_token();
    bool    stale = stale_.load(std::memory_order_relaxed);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"stale\":%s"
        ",\"ms_since_last_token\":%lld"
        ",\"stale_events\":%llu"
        ",\"recovery_events\":%llu"
        ",\"stale_threshold_ms\":%d}",
        stale ? "true" : "false",
        static_cast<long long>(gap),
        static_cast<unsigned long long>(stale_events_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(recovery_events_.load(std::memory_order_relaxed)),
        config_.stale_threshold_ms
    );
    return std::string(buf);
}

} // namespace llmquant
