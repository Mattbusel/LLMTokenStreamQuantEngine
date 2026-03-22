#include "BiasMomentumIndex.h"

#include <iomanip>
#include <sstream>

namespace llmquant {

BiasMomentumIndex::BiasMomentumIndex(Config cfg)
    : cfg_(std::move(cfg))
{}

double BiasMomentumIndex::record(double bias) {
    std::function<void(double)> bull_cb, bear_cb;
    bool fire_bull = false, fire_bear = false;
    double hist_val = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        bull_cb = cfg_.on_bullish_crossover;
        bear_cb = cfg_.on_bearish_crossover;

        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            fast_ema_v_ = slow_ema_v_ = signal_v_ = bias;
            seeded_ = true;
        } else {
            fast_ema_v_ = cfg_.fast_alpha * bias + (1.0 - cfg_.fast_alpha) * fast_ema_v_;
            slow_ema_v_ = cfg_.slow_alpha * bias + (1.0 - cfg_.slow_alpha) * slow_ema_v_;
        }

        double macd = fast_ema_v_ - slow_ema_v_;
        signal_v_ = cfg_.signal_alpha * macd + (1.0 - cfg_.signal_alpha) * signal_v_;
        double hist = macd - signal_v_;
        hist_val = hist;

        fast_ema_.store(fast_ema_v_,  std::memory_order_relaxed);
        slow_ema_.store(slow_ema_v_,  std::memory_order_relaxed);
        macd_line_.store(macd,        std::memory_order_relaxed);
        signal_line_.store(signal_v_, std::memory_order_relaxed);
        histogram_.store(hist,        std::memory_order_relaxed);

        bool now_bullish = macd > signal_v_;
        bullish_.store(now_bullish, std::memory_order_relaxed);

        if (now_bullish && !prev_bullish_) {
            fire_bull = true;
            bullish_cross_.fetch_add(1, std::memory_order_relaxed);
        } else if (!now_bullish && prev_bullish_) {
            fire_bear = true;
            bearish_cross_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_bullish_ = now_bullish;
    }

    if (fire_bull && bull_cb) bull_cb(hist_val);
    if (fire_bear && bear_cb) bear_cb(hist_val);
    return hist_val;
}

std::string BiasMomentumIndex::to_stats_json() const {
    double fema, sema, macd, sig, hist;
    bool   bull;
    uint64_t bcross, becross, total;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        fema   = fast_ema_.load(std::memory_order_relaxed);
        sema   = slow_ema_.load(std::memory_order_relaxed);
        macd   = macd_line_.load(std::memory_order_relaxed);
        sig    = signal_line_.load(std::memory_order_relaxed);
        hist   = histogram_.load(std::memory_order_relaxed);
        bull   = bullish_.load(std::memory_order_relaxed);
        bcross = bullish_cross_.load(std::memory_order_relaxed);
        becross = bearish_cross_.load(std::memory_order_relaxed);
        total  = total_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"fast_ema\":"           << fema   << ","
       << "\"slow_ema\":"           << sema   << ","
       << "\"macd_line\":"          << macd   << ","
       << "\"signal_line\":"        << sig    << ","
       << "\"histogram\":"          << hist   << ","
       << "\"is_bullish\":"         << (bull ? "true" : "false") << ","
       << "\"bullish_crossovers\":" << bcross << ","
       << "\"bearish_crossovers\":" << becross << ","
       << "\"total_records\":"      << total
       << "}";
    return ss.str();
}

void BiasMomentumIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    fast_ema_v_ = slow_ema_v_ = signal_v_ = 0.0;
    seeded_       = false;
    prev_bullish_ = false;
    fast_ema_.store(0.0,   std::memory_order_relaxed);
    slow_ema_.store(0.0,   std::memory_order_relaxed);
    macd_line_.store(0.0,  std::memory_order_relaxed);
    signal_line_.store(0.0, std::memory_order_relaxed);
    histogram_.store(0.0,  std::memory_order_relaxed);
    bullish_.store(false,  std::memory_order_relaxed);
    bullish_cross_.store(0, std::memory_order_relaxed);
    bearish_cross_.store(0, std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

void BiasMomentumIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_          = cfg;
    fast_ema_v_ = slow_ema_v_ = signal_v_ = 0.0;
    seeded_       = false;
    prev_bullish_ = false;
    fast_ema_.store(0.0,   std::memory_order_relaxed);
    slow_ema_.store(0.0,   std::memory_order_relaxed);
    macd_line_.store(0.0,  std::memory_order_relaxed);
    signal_line_.store(0.0, std::memory_order_relaxed);
    histogram_.store(0.0,  std::memory_order_relaxed);
    bullish_.store(false,  std::memory_order_relaxed);
    bullish_cross_.store(0, std::memory_order_relaxed);
    bearish_cross_.store(0, std::memory_order_relaxed);
    total_.store(0,        std::memory_order_relaxed);
}

} // namespace llmquant
