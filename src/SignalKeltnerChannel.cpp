#include "SignalKeltnerChannel.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalKeltnerChannel::SignalKeltnerChannel(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalKeltnerChannel::record(double bias) {
    double new_ema = 0.0, new_atr = 0.0, new_upper = 0.0, new_lower = 0.0;
    bool upper_ev = false, lower_ev = false;
    bool new_above = false, new_below = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            ema_raw_ = bias;
            atr_raw_ = 0.0;
            seeded_  = true;
        } else {
            double tr = std::fabs(bias - prev_bias_);
            atr_raw_ += cfg_.alpha_atr * (tr       - atr_raw_);
            ema_raw_ += cfg_.alpha_ema * (bias      - ema_raw_);
        }
        prev_bias_ = bias;

        new_ema   = ema_raw_;
        new_atr   = atr_raw_;
        new_upper = ema_raw_ + cfg_.multiplier * atr_raw_;
        new_lower = ema_raw_ - cfg_.multiplier * atr_raw_;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<uint64_t>(cfg_.min_samples)) {
            new_above = (bias > new_upper);
            new_below = (bias < new_lower);

            if (new_above && !prev_above_) {
                upper_ev = true;
                upper_ev_.fetch_add(1, std::memory_order_relaxed);
            }
            if (new_below && !prev_below_) {
                lower_ev = true;
                lower_ev_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_above_ = new_above;
            prev_below_ = new_below;
        }
    }

    ema_.store(new_ema, std::memory_order_relaxed);
    atr_.store(new_atr, std::memory_order_relaxed);
    upper_.store(new_upper, std::memory_order_relaxed);
    lower_.store(new_lower, std::memory_order_relaxed);
    above_.store(new_above, std::memory_order_relaxed);
    below_.store(new_below, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (upper_ev && cfg_.on_upper_break) cfg_.on_upper_break(bias, new_upper);
    if (lower_ev && cfg_.on_lower_break) cfg_.on_lower_break(bias, new_lower);
}

double SignalKeltnerChannel::ema()         const noexcept { return ema_.load(std::memory_order_relaxed); }
double SignalKeltnerChannel::atr()         const noexcept { return atr_.load(std::memory_order_relaxed); }
double SignalKeltnerChannel::upper_band()  const noexcept { return upper_.load(std::memory_order_relaxed); }
double SignalKeltnerChannel::lower_band()  const noexcept { return lower_.load(std::memory_order_relaxed); }
bool   SignalKeltnerChannel::above_upper() const noexcept { return above_.load(std::memory_order_relaxed); }
bool   SignalKeltnerChannel::below_lower() const noexcept { return below_.load(std::memory_order_relaxed); }

std::size_t SignalKeltnerChannel::upper_breaks()  const noexcept { return upper_ev_.load(std::memory_order_relaxed); }
std::size_t SignalKeltnerChannel::lower_breaks()  const noexcept { return lower_ev_.load(std::memory_order_relaxed); }
std::size_t SignalKeltnerChannel::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalKeltnerChannel::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"ema\":"         << ema_.load(std::memory_order_relaxed)
       << ",\"atr\":"         << atr_.load(std::memory_order_relaxed)
       << ",\"upper_band\":"  << upper_.load(std::memory_order_relaxed)
       << ",\"lower_band\":"  << lower_.load(std::memory_order_relaxed)
       << ",\"upper_breaks\":" << upper_ev_.load(std::memory_order_relaxed)
       << ",\"lower_breaks\":" << lower_ev_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalKeltnerChannel::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false; ema_raw_ = 0.0; atr_raw_ = 0.0; prev_bias_ = 0.0;
    prev_above_ = false; prev_below_ = false;
    ema_.store(0.0, std::memory_order_relaxed);
    atr_.store(0.0, std::memory_order_relaxed);
    upper_.store(0.0, std::memory_order_relaxed);
    lower_.store(0.0, std::memory_order_relaxed);
    above_.store(false, std::memory_order_relaxed);
    below_.store(false, std::memory_order_relaxed);
    upper_ev_.store(0, std::memory_order_relaxed);
    lower_ev_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalKeltnerChannel::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
