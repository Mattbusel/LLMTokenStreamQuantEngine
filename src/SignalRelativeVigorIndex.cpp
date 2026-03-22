#include "SignalRelativeVigorIndex.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalRelativeVigorIndex::SignalRelativeVigorIndex(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalRelativeVigorIndex::record(double bias) {
    double new_rvi = 0.0, new_sig = 0.0;
    bool bull_ev = false, bear_ev = false;
    bool new_bull = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            ema_num_ = 0.0;
            ema_den_ = std::fabs(bias);
            ema_sig_ = 0.0;
            seeded_  = true;
        } else {
            double change = bias - prev_bias_;
            double range  = std::fabs(bias);
            ema_num_ += cfg_.alpha_rvi * (change - ema_num_);
            ema_den_ += cfg_.alpha_rvi * (range  - ema_den_);
            double raw_rvi = (ema_den_ > 1e-12) ? ema_num_ / ema_den_ : 0.0;
            ema_sig_ += cfg_.alpha_signal * (raw_rvi - ema_sig_);
            new_rvi = raw_rvi;
            new_sig = ema_sig_;
        }
        prev_bias_ = bias;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<uint64_t>(cfg_.min_samples)) {
            new_bull = (new_rvi > new_sig + cfg_.cross_threshold);
            if (new_bull && !prev_bullish_) {
                bull_ev = true;
                bull_crosses_.fetch_add(1, std::memory_order_relaxed);
            } else if (!new_bull && prev_bullish_) {
                bear_ev = true;
                bear_crosses_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_bullish_ = new_bull;
        }
    }

    rvi_.store(new_rvi, std::memory_order_relaxed);
    sig_.store(new_sig, std::memory_order_relaxed);
    bullish_.store(new_bull, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (bull_ev && cfg_.on_bullish_cross) cfg_.on_bullish_cross(new_rvi);
    if (bear_ev && cfg_.on_bearish_cross) cfg_.on_bearish_cross(new_rvi);
}

double SignalRelativeVigorIndex::rvi()         const noexcept { return rvi_.load(std::memory_order_relaxed); }
double SignalRelativeVigorIndex::signal_line() const noexcept { return sig_.load(std::memory_order_relaxed); }
bool   SignalRelativeVigorIndex::is_bullish()  const noexcept { return bullish_.load(std::memory_order_relaxed); }

std::size_t SignalRelativeVigorIndex::bullish_crosses() const noexcept { return bull_crosses_.load(std::memory_order_relaxed); }
std::size_t SignalRelativeVigorIndex::bearish_crosses() const noexcept { return bear_crosses_.load(std::memory_order_relaxed); }
std::size_t SignalRelativeVigorIndex::total_records()   const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalRelativeVigorIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"rvi\":"           << rvi_.load(std::memory_order_relaxed)
       << ",\"signal_line\":"   << sig_.load(std::memory_order_relaxed)
       << ",\"is_bullish\":"    << (bullish_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"bullish_crosses\":" << bull_crosses_.load(std::memory_order_relaxed)
       << ",\"bearish_crosses\":" << bear_crosses_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalRelativeVigorIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false; prev_bias_ = 0.0;
    ema_num_ = 0.0; ema_den_ = 0.0; ema_sig_ = 0.0;
    prev_bullish_ = false;
    rvi_.store(0.0, std::memory_order_relaxed);
    sig_.store(0.0, std::memory_order_relaxed);
    bullish_.store(false, std::memory_order_relaxed);
    bull_crosses_.store(0, std::memory_order_relaxed);
    bear_crosses_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalRelativeVigorIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
