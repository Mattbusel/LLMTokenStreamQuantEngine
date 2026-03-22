#include "SignalDrawdownMeter.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalDrawdownMeter::SignalDrawdownMeter(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalDrawdownMeter::record(double bias) {
    double new_dd   = 0.0;
    bool dd_event   = false;
    bool rec_event  = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        cum_bias_ += bias;

        if (cum_bias_ > peak_) {
            if (in_drawdown_) {
                // Recovered to new peak
                rec_event = true;
                recovery_events_.fetch_add(1, std::memory_order_relaxed);
                in_drawdown_ = false;
            }
            peak_ = cum_bias_;
        }

        double denom = std::fabs(peak_) + 1e-12;
        new_dd = (peak_ - cum_bias_) / denom;
        if (new_dd < 0.0) new_dd = 0.0;

        if (new_dd > max_drawdown_) {
            if (new_dd > cfg_.drawdown_threshold) {
                dd_event = true;
                drawdown_events_.fetch_add(1, std::memory_order_relaxed);
            }
            max_drawdown_ = new_dd;
            in_drawdown_  = true;
        }
    }

    cum_.store(cum_bias_, std::memory_order_relaxed);
    peak_atomic_.store(peak_, std::memory_order_relaxed);
    drawdown_.store(new_dd, std::memory_order_relaxed);
    max_dd_.store(max_drawdown_, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (dd_event  && cfg_.on_new_drawdown_high) cfg_.on_new_drawdown_high(new_dd);
    if (rec_event && cfg_.on_recovery)          cfg_.on_recovery(cum_bias_);
}

double SignalDrawdownMeter::cum_bias()     const noexcept { return cum_.load(std::memory_order_relaxed); }
double SignalDrawdownMeter::peak()         const noexcept { return peak_atomic_.load(std::memory_order_relaxed); }
double SignalDrawdownMeter::drawdown()     const noexcept { return drawdown_.load(std::memory_order_relaxed); }
double SignalDrawdownMeter::max_drawdown() const noexcept { return max_dd_.load(std::memory_order_relaxed); }

std::size_t SignalDrawdownMeter::drawdown_events()  const noexcept { return drawdown_events_.load(std::memory_order_relaxed); }
std::size_t SignalDrawdownMeter::recovery_events()  const noexcept { return recovery_events_.load(std::memory_order_relaxed); }
std::size_t SignalDrawdownMeter::total_records()    const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalDrawdownMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"cum_bias\":" << cum_.load(std::memory_order_relaxed)
       << ",\"peak\":" << peak_atomic_.load(std::memory_order_relaxed)
       << ",\"drawdown\":" << drawdown_.load(std::memory_order_relaxed)
       << ",\"max_drawdown\":" << max_dd_.load(std::memory_order_relaxed)
       << ",\"drawdown_events\":" << drawdown_events_.load(std::memory_order_relaxed)
       << ",\"recovery_events\":" << recovery_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalDrawdownMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cum_bias_ = 0.0; peak_ = 0.0; max_drawdown_ = 0.0; in_drawdown_ = false;
    cum_.store(0.0, std::memory_order_relaxed);
    peak_atomic_.store(0.0, std::memory_order_relaxed);
    drawdown_.store(0.0, std::memory_order_relaxed);
    max_dd_.store(0.0, std::memory_order_relaxed);
    drawdown_events_.store(0, std::memory_order_relaxed);
    recovery_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalDrawdownMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
