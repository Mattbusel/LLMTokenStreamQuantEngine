#include "SignalBlindSpotDetector.h"

#include <algorithm>
#include <sstream>

namespace llmquant {

SignalBlindSpotDetector::SignalBlindSpotDetector(Config config)
    : config_(std::move(config)) {
    slots_.fill(SlotStats{});
}

void SignalBlindSpotDetector::record_outcome(int slot, bool won) {
    if (slot < 0 || slot >= kSlots) return;
    total_outcomes_.fetch_add(1, std::memory_order_relaxed);

    bool fire_cb  = false;
    double wr_cb  = 0.0;
    std::function<void(int, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto& s = slots_[static_cast<std::size_t>(slot)];

        // Update EMA count and win-rate.
        double f     = config_.forgetting;
        double win_v = won ? 1.0 : 0.0;

        // Effective sample count under exponential forgetting.
        s.ema_count    = f * s.ema_count + 1.0;

        // Standard EMA: new = forgetting * old + (1 - forgetting) * observation.
        s.ema_win_rate = f * s.ema_win_rate + (1.0 - f) * win_v;

        // Check if blind spot.
        double eff_count = s.ema_count;
        if (eff_count >= config_.min_samples &&
            s.ema_win_rate < config_.blind_spot_threshold) {
            if (!s.is_blind) {
                s.is_blind = true;
            }
            if (!s.alerted) {
                s.alerted = true;
                detection_events_.fetch_add(1, std::memory_order_relaxed);
                fire_cb = true;
                wr_cb   = s.ema_win_rate;
                cb      = config_.on_blind_spot_found;
            }
        } else if (s.ema_win_rate >= config_.blind_spot_threshold) {
            s.is_blind = false;
            s.alerted  = false;
        }
    }

    if (fire_cb && cb) cb(slot, wr_cb);
}

void SignalBlindSpotDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    slots_.fill(SlotStats{});
}

bool SignalBlindSpotDetector::is_blind_spot(int slot) const noexcept {
    if (slot < 0 || slot >= kSlots) return false;
    std::lock_guard<std::mutex> lk(mutex_);
    const auto& s = slots_[static_cast<std::size_t>(slot)];
    return s.is_blind && (s.ema_count >= config_.min_samples);
}

double SignalBlindSpotDetector::win_rate(int slot) const noexcept {
    if (slot < 0 || slot >= kSlots) return 0.5;
    std::lock_guard<std::mutex> lk(mutex_);
    const auto& s = slots_[static_cast<std::size_t>(slot)];
    return (s.ema_count >= 1.0) ? s.ema_win_rate : 0.5;
}

int SignalBlindSpotDetector::sample_count(int slot) const noexcept {
    if (slot < 0 || slot >= kSlots) return 0;
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(slots_[static_cast<std::size_t>(slot)].ema_count);
}

int SignalBlindSpotDetector::blind_spot_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int cnt = 0;
    for (const auto& s : slots_)
        if (s.is_blind && s.ema_count >= config_.min_samples) ++cnt;
    return cnt;
}

void SignalBlindSpotDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    slots_.fill(SlotStats{});
}

std::string SignalBlindSpotDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    int bsc = 0;
    for (const auto& s : slots_)
        if (s.is_blind && s.ema_count >= config_.min_samples) ++bsc;
    os << "{"
       << "\"blind_spot_count\":"  << bsc << ","
       << "\"total_outcomes\":"    << total_outcomes_.load(std::memory_order_relaxed) << ","
       << "\"detection_events\":"  << detection_events_.load(std::memory_order_relaxed) << ","
       << "\"slots\":[";
    for (int i = 0; i < kSlots; ++i) {
        const auto& s = slots_[static_cast<std::size_t>(i)];
        if (i > 0) os << ",";
        os << "{\"slot\":" << i
           << ",\"win_rate\":" << s.ema_win_rate
           << ",\"is_blind\":" << (s.is_blind ? "true" : "false")
           << "}";
    }
    os << "]}";
    return os.str();
}

} // namespace llmquant
