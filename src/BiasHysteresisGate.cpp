#include "BiasHysteresisGate.h"

#include <cmath>
#include <sstream>

namespace llmquant {

BiasHysteresisGate::BiasHysteresisGate(Config cfg)
    : cfg_(std::move(cfg))
{}

bool BiasHysteresisGate::evaluate(double bias) {
    std::function<void(double)> cb;
    bool fire_open  = false;
    bool fire_close = false;
    bool result     = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        double mag = std::abs(bias);

        if (!gate_open_) {
            if (mag >= cfg_.enter_threshold) {
                ++consecutive_;
                if (consecutive_ >= cfg_.required_ticks) {
                    gate_open_   = true;
                    consecutive_ = 0;
                    fire_open    = true;
                    open_events_.fetch_add(1, std::memory_order_relaxed);
                    cb = cfg_.on_gate_open;
                }
            } else {
                consecutive_ = 0;
            }
            result = gate_open_;
        } else {
            // Gate is open — check exit condition.
            if (mag < cfg_.exit_threshold) {
                gate_open_   = false;
                fire_close   = true;
                close_events_.fetch_add(1, std::memory_order_relaxed);
                cb = cfg_.on_gate_close;
            }
            result = gate_open_;
        }

        open_flag_.store(gate_open_, std::memory_order_relaxed);
        consec_flag_.store(consecutive_, std::memory_order_relaxed);
    }

    total_.fetch_add(1, std::memory_order_relaxed);

    if (fire_open  && cb) cb(bias);
    if (fire_close && cb) cb(bias);

    return result;
}

void BiasHysteresisGate::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    gate_open_   = false;
    consecutive_ = 0;
    open_flag_.store(false, std::memory_order_relaxed);
    consec_flag_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    open_events_.store(0, std::memory_order_relaxed);
    close_events_.store(0, std::memory_order_relaxed);
}

void BiasHysteresisGate::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_         = cfg;
    gate_open_   = false;
    consecutive_ = 0;
    open_flag_.store(false, std::memory_order_relaxed);
    consec_flag_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    open_events_.store(0, std::memory_order_relaxed);
    close_events_.store(0, std::memory_order_relaxed);
}

std::string BiasHysteresisGate::to_stats_json() const {
    bool     open   = open_flag_.load(std::memory_order_relaxed);
    int      consec = consec_flag_.load(std::memory_order_relaxed);
    uint64_t tot    = total_.load(std::memory_order_relaxed);
    uint64_t opens  = open_events_.load(std::memory_order_relaxed);
    uint64_t closes = close_events_.load(std::memory_order_relaxed);

    std::ostringstream ss;
    ss << "{"
       << "\"is_open\":"          << (open ? "true" : "false") << ","
       << "\"consecutive_ticks\":" << consec << ","
       << "\"total_evaluated\":"   << tot    << ","
       << "\"gate_open_events\":"  << opens  << ","
       << "\"gate_close_events\":" << closes
       << "}";
    return ss.str();
}

} // namespace llmquant
