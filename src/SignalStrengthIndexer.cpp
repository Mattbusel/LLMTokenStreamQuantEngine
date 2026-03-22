#include "SignalStrengthIndexer.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalStrengthIndexer::SignalStrengthIndexer(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalStrengthIndexer::record(double bias) {
    std::function<void(double)> ob_cb, os_cb;
    bool   fire_ob = false, fire_os = false;
    double ssi_val = 50.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        ob_cb = cfg_.on_overbought;
        os_cb = cfg_.on_oversold;
        obs_count_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            return;
        }

        double delta = bias - prev_bias_;
        prev_bias_   = bias;

        double gain = (delta > 0.0) ? delta : 0.0;
        double loss = (delta < 0.0) ? -delta : 0.0;

        int period  = std::max(1, cfg_.period);
        double alpha = 1.0 / period;

        if (warmup_ < period) {
            // Simple average during warmup.
            avg_gain_ += gain;
            avg_loss_ += loss;
            ++warmup_;
            if (warmup_ == period) {
                avg_gain_ /= period;
                avg_loss_ /= period;
            } else {
                return;
            }
        } else {
            // Wilder's smoothing.
            avg_gain_ = avg_gain_ * (1.0 - alpha) + gain * alpha;
            avg_loss_ = avg_loss_ * (1.0 - alpha) + loss * alpha;
        }

        double rs  = (avg_loss_ > 0.0) ? (avg_gain_ / avg_loss_) : (avg_gain_ > 0.0 ? 100.0 : 1.0);
        ssi_val    = 100.0 - 100.0 / (1.0 + rs);

        ssi_.store(ssi_val, std::memory_order_relaxed);

        bool ob = ssi_val > cfg_.overbought_threshold;
        bool os = ssi_val < cfg_.oversold_threshold;

        overbought_.store(ob, std::memory_order_relaxed);
        oversold_.store(os,   std::memory_order_relaxed);

        if (ob && !prev_overbought_) fire_ob = true;
        if (os && !prev_oversold_)   fire_os = true;
        prev_overbought_ = ob;
        prev_oversold_   = os;
    }

    if (fire_ob && ob_cb) ob_cb(ssi_val);
    if (fire_os && os_cb) os_cb(ssi_val);
}

std::string SignalStrengthIndexer::to_stats_json() const {
    double s;
    bool ob, os;
    uint64_t oc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        s  = ssi_.load(std::memory_order_relaxed);
        ob = overbought_.load(std::memory_order_relaxed);
        os = oversold_.load(std::memory_order_relaxed);
        oc = obs_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"ssi\":"                << s << ","
       << "\"is_overbought\":"      << (ob ? "true" : "false") << ","
       << "\"is_oversold\":"        << (os ? "true" : "false") << ","
       << "\"observation_count\":"  << oc
       << "}";
    return ss.str();
}

void SignalStrengthIndexer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    prev_bias_       = 0.0;
    avg_gain_        = avg_loss_ = 0.0;
    seeded_          = false;
    warmup_          = 0;
    prev_overbought_ = prev_oversold_ = false;
    ssi_.store(50.0, std::memory_order_relaxed);
    overbought_.store(false, std::memory_order_relaxed);
    oversold_.store(false, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void SignalStrengthIndexer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_             = cfg;
    prev_bias_       = 0.0;
    avg_gain_        = avg_loss_ = 0.0;
    seeded_          = false;
    warmup_          = 0;
    prev_overbought_ = prev_oversold_ = false;
}

} // namespace llmquant
