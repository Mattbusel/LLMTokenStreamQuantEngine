#include "TokenSentimentMomentumIndex.h"

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenSentimentMomentumIndex::TokenSentimentMomentumIndex(Config cfg)
    : cfg_(std::move(cfg))
{}

void TokenSentimentMomentumIndex::record(double bias) {
    bool   fire    = false;
    double old_tsmi = 0.0, new_tsmi = 0.0;
    std::function<void(double,double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_momentum_shift;
        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_  = bias;
            seeded_     = true;
            return;
        }

        double delta = bias - prev_bias_;
        double vel   = delta;

        double acc = (rsi_warmup_ > 0) ? (delta - prev_delta_) : 0.0;
        prev_delta_ = delta;
        velocity_.store(vel, std::memory_order_relaxed);
        accel_.store(acc, std::memory_order_relaxed);

        // RSI-style signal strength (Wilder period 14)
        constexpr int period = 14;
        constexpr double alpha = 1.0 / period;
        double gain = (delta > 0.0) ? delta : 0.0;
        double loss = (delta < 0.0) ? -delta : 0.0;

        if (rsi_warmup_ < period) {
            avg_gain_ += gain;
            avg_loss_ += loss;
            ++rsi_warmup_;
            if (rsi_warmup_ == period) {
                avg_gain_ /= period;
                avg_loss_ /= period;
            }
        } else {
            avg_gain_ = avg_gain_ * (1.0 - alpha) + gain * alpha;
            avg_loss_ = avg_loss_ * (1.0 - alpha) + loss * alpha;
        }

        double rs  = (avg_loss_ > 0.0) ? avg_gain_ / avg_loss_ : (avg_gain_ > 0.0 ? 100.0 : 1.0);
        double ssi = 100.0 - 100.0 / (1.0 + rs);
        ssi_.store(ssi, std::memory_order_relaxed);

        double roc_norm = std::tanh(vel / (cfg_.roc_scale > 0.0 ? cfg_.roc_scale : 0.1));
        double acc_norm = std::tanh(acc / (cfg_.acc_scale > 0.0 ? cfg_.acc_scale : 0.05));
        double str_norm = (ssi - 50.0) / 50.0;

        new_tsmi = cfg_.w_roc * roc_norm + cfg_.w_acc * acc_norm + cfg_.w_str * str_norm;
        tsmi_.store(new_tsmi, std::memory_order_relaxed);

        if (tsmi_seeded_ && ((prev_tsmi_ < 0.0) != (new_tsmi < 0.0))) {
            old_tsmi = prev_tsmi_;
            fire = true;
        }
        prev_tsmi_   = new_tsmi;
        tsmi_seeded_ = true;

        prev_bias_ = bias;
    }

    if (fire && cb) cb(old_tsmi, new_tsmi);
}

std::string TokenSentimentMomentumIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"tsmi\":"           << tsmi_.load(std::memory_order_relaxed) << ","
       << "\"velocity\":"       << velocity_.load(std::memory_order_relaxed) << ","
       << "\"acceleration\":"   << accel_.load(std::memory_order_relaxed) << ","
       << "\"signal_strength\":" << ssi_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"  << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenSentimentMomentumIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    prev_bias_ = prev_delta_ = 0.0;
    avg_gain_ = avg_loss_ = 0.0;
    rsi_warmup_ = 0;
    seeded_ = tsmi_seeded_ = false;
    prev_tsmi_ = 0.0;
    tsmi_.store(0.0, std::memory_order_relaxed);
    velocity_.store(0.0, std::memory_order_relaxed);
    accel_.store(0.0, std::memory_order_relaxed);
    ssi_.store(50.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenSentimentMomentumIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    prev_bias_ = prev_delta_ = 0.0;
    avg_gain_ = avg_loss_ = 0.0;
    rsi_warmup_ = 0;
    seeded_ = tsmi_seeded_ = false;
}

} // namespace llmquant
