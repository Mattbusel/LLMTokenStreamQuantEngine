#include "SignalEchoSuppressor.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace llmquant {

SignalEchoSuppressor::SignalEchoSuppressor(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(2, cfg_.window);
    ring_.assign(static_cast<std::size_t>(w), 0);
}

bool SignalEchoSuppressor::record(double bias) {
    std::function<void(double)> det_cb, clr_cb;
    bool   fire_det = false, fire_clr = false;
    bool   is_echo  = false;
    double rate_val = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        det_cb = cfg_.on_echo_detected;
        clr_cb = cfg_.on_echo_cleared;

        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            // First signal is never an echo.
            is_echo = false;
        } else {
            is_echo = (std::abs(bias - prev_bias_) < cfg_.echo_threshold);
            prev_bias_ = bias;
        }

        if (is_echo) echo_count_.fetch_add(1, std::memory_order_relaxed);

        // Update rolling ring.
        int8_t new_val  = is_echo ? 1 : 0;
        int8_t old_val  = ring_[static_cast<std::size_t>(head_)];
        echo_sum_ -= old_val;
        echo_sum_ += new_val;
        ring_[static_cast<std::size_t>(head_)] = new_val;
        head_ = (head_ + 1) % static_cast<int>(ring_.size());
        if (fill_ < static_cast<int>(ring_.size())) ++fill_;

        if (fill_ > 0) {
            rate_val = static_cast<double>(echo_sum_) / fill_;
            rate_val = std::max(0.0, std::min(1.0, rate_val));
        }
        echo_rate_.store(rate_val, std::memory_order_relaxed);

        // Apply hysteresis to the echoing flag: enter when rate > threshold,
        // exit only when rate drops below threshold * clear_hysteresis.
        // This prevents a gap where the flag clears before the clear threshold
        // is reached, which would leave prev_echoing_=false and block the callback.
        bool now_echoing;
        if (prev_echoing_) {
            now_echoing = (rate_val >= cfg_.echo_rate_threshold * cfg_.clear_hysteresis);
        } else {
            now_echoing = (rate_val > cfg_.echo_rate_threshold);
        }
        echoing_.store(now_echoing, std::memory_order_relaxed);

        if (now_echoing && !prev_echoing_) {
            fire_det = true;
            echo_events_.fetch_add(1, std::memory_order_relaxed);
        } else if (!now_echoing && prev_echoing_) {
            fire_clr = true;
        }
        prev_echoing_ = now_echoing;
    }

    if (fire_det && det_cb) det_cb(rate_val);
    if (fire_clr && clr_cb) clr_cb(rate_val);
    return is_echo;
}

std::string SignalEchoSuppressor::to_stats_json() const {
    double rate;
    bool   echoing;
    uint64_t total, echos, events;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        rate   = echo_rate_.load(std::memory_order_relaxed);
        echoing = echoing_.load(std::memory_order_relaxed);
        total  = total_.load(std::memory_order_relaxed);
        echos  = echo_count_.load(std::memory_order_relaxed);
        events = echo_events_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"echo_rate\":"    << rate   << ","
       << "\"is_echoing\":"   << (echoing ? "true" : "false") << ","
       << "\"total_signals\":" << total  << ","
       << "\"echo_count\":"   << echos  << ","
       << "\"echo_events\":"  << events
       << "}";
    return ss.str();
}

void SignalEchoSuppressor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), 0);
    head_ = fill_ = echo_sum_ = 0;
    prev_bias_ = 0.0;
    seeded_    = false;
    prev_echoing_ = false;
    echo_rate_.store(0.0, std::memory_order_relaxed);
    echoing_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    echo_count_.store(0, std::memory_order_relaxed);
    echo_events_.store(0, std::memory_order_relaxed);
}

void SignalEchoSuppressor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_  = cfg;
    int w = std::max(2, cfg_.window);
    ring_.assign(static_cast<std::size_t>(w), 0);
    head_ = fill_ = echo_sum_ = 0;
    prev_bias_ = 0.0;
    seeded_    = false;
    prev_echoing_ = false;
    echo_rate_.store(0.0, std::memory_order_relaxed);
    echoing_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    echo_count_.store(0, std::memory_order_relaxed);
    echo_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
