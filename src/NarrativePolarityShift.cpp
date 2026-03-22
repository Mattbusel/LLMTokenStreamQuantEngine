#include "NarrativePolarityShift.h"
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativePolarityShift::NarrativePolarityShift(Config cfg)
    : cfg_(std::move(cfg))
    , win_buf_(cfg_.window, 0)
{}

void NarrativePolarityShift::record(double bias) {
    double new_frac = 0.5;
    bool bull_ev = false, bear_ev = false;
    bool new_bull = false, new_bear = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        win_buf_[win_head_] = (bias > 0.0) ? 1 : 0;
        win_head_ = (win_head_ + 1) % cfg_.window;
        if (win_fill_ < cfg_.window) ++win_fill_;

        if (win_fill_ >= cfg_.min_samples) {
            int pos_count = 0;
            for (int i = 0; i < win_fill_; ++i) pos_count += win_buf_[i];
            new_frac = static_cast<double>(pos_count) / win_fill_;

            new_bull = (new_frac >= cfg_.bull_threshold);
            new_bear = (new_frac <= cfg_.bear_threshold);

            if (new_bull && !prev_bullish_) {
                bull_ev = true;
                bull_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (new_bear && !prev_bearish_) {
                bear_ev = true;
                bear_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_bullish_ = new_bull;
            prev_bearish_ = new_bear;
        }
    }

    pos_frac_.store(new_frac, std::memory_order_relaxed);
    bullish_.store(new_bull, std::memory_order_relaxed);
    bearish_.store(new_bear, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (bull_ev && cfg_.on_bullish) cfg_.on_bullish(new_frac);
    if (bear_ev && cfg_.on_bearish) cfg_.on_bearish(new_frac);
}

double NarrativePolarityShift::positive_fraction() const noexcept { return pos_frac_.load(std::memory_order_relaxed); }
double NarrativePolarityShift::negative_fraction() const noexcept { return 1.0 - pos_frac_.load(std::memory_order_relaxed); }
bool   NarrativePolarityShift::is_bullish()        const noexcept { return bullish_.load(std::memory_order_relaxed); }
bool   NarrativePolarityShift::is_bearish()        const noexcept { return bearish_.load(std::memory_order_relaxed); }

std::size_t NarrativePolarityShift::bullish_events() const noexcept { return bull_events_.load(std::memory_order_relaxed); }
std::size_t NarrativePolarityShift::bearish_events() const noexcept { return bear_events_.load(std::memory_order_relaxed); }
std::size_t NarrativePolarityShift::total_records()  const noexcept { return total_.load(std::memory_order_relaxed); }

std::string NarrativePolarityShift::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"positive_fraction\":" << pos_frac_.load(std::memory_order_relaxed)
       << ",\"is_bullish\":"  << (bullish_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"is_bearish\":"  << (bearish_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"bullish_events\":" << bull_events_.load(std::memory_order_relaxed)
       << ",\"bearish_events\":" << bear_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativePolarityShift::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0);
    win_head_ = 0; win_fill_ = 0;
    prev_bullish_ = false; prev_bearish_ = false;
    pos_frac_.store(0.5, std::memory_order_relaxed);
    bullish_.store(false, std::memory_order_relaxed);
    bearish_.store(false, std::memory_order_relaxed);
    bull_events_.store(0, std::memory_order_relaxed);
    bear_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativePolarityShift::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_buf_.assign(cfg_.window, 0);
    win_head_ = 0; win_fill_ = 0;
    prev_bullish_ = false; prev_bearish_ = false;
    pos_frac_.store(0.5,   std::memory_order_relaxed);
    bullish_.store(false,  std::memory_order_relaxed);
    bearish_.store(false,  std::memory_order_relaxed);
    bull_events_.store(0,  std::memory_order_relaxed);
    bear_events_.store(0,  std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

} // namespace llmquant
