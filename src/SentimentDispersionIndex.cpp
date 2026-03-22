#include "SentimentDispersionIndex.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SentimentDispersionIndex::SentimentDispersionIndex(Config cfg)
    : cfg_(std::move(cfg)) {}

void SentimentDispersionIndex::record(double abs_bias,
                                      double volatility_adj,
                                      double confidence) noexcept
{
    std::function<void(double)> cb_high, cb_low;
    bool fire_high = false, fire_low = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        // Compute instantaneous coefficient of variation of the 3 signals.
        double vals[3] = {abs_bias, volatility_adj, confidence};
        double sum = vals[0] + vals[1] + vals[2];
        double mean = sum / 3.0;
        if (mean < 1e-9) mean = 1e-9;

        double var = 0.0;
        for (double v : vals) { double d = v - mean; var += d * d; }
        var /= 3.0;
        double cv = std::sqrt(var) / mean; // coefficient of variation

        if (!seeded_) {
            sdi_ema_ = cv;
            seeded_  = true;
        } else {
            sdi_ema_ = cfg_.alpha * cv + (1.0 - cfg_.alpha) * sdi_ema_;
        }
        sdi_out_.store(sdi_ema_, std::memory_order_relaxed);
        total_.fetch_add(1, std::memory_order_relaxed);

        bool now_high = (sdi_ema_ > cfg_.high_threshold);
        bool now_low  = (sdi_ema_ < cfg_.low_threshold);

        if (now_high && !prev_high_) {
            prev_high_ = true;
            prev_low_  = false;
            fire_high  = true;
            high_events_.fetch_add(1, std::memory_order_relaxed);
        }
        if (now_low && !prev_low_) {
            prev_low_  = true;
            prev_high_ = false;
            fire_low   = true;
            low_events_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!now_high) prev_high_ = false;
        if (!now_low)  prev_low_  = false;

        cb_high = cfg_.on_high_dispersion;
        cb_low  = cfg_.on_low_dispersion;
    }
    double sdi_val = sdi_out_.load(std::memory_order_relaxed);
    if (fire_high && cb_high) cb_high(sdi_val);
    if (fire_low  && cb_low)  cb_low(sdi_val);
}

double SentimentDispersionIndex::sdi() const noexcept {
    return sdi_out_.load(std::memory_order_relaxed);
}

bool SentimentDispersionIndex::is_dispersed() const noexcept {
    return sdi_out_.load(std::memory_order_relaxed) > cfg_.high_threshold;
}

bool SentimentDispersionIndex::is_coherent() const noexcept {
    return sdi_out_.load(std::memory_order_relaxed) < cfg_.low_threshold;
}

void SentimentDispersionIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    sdi_ema_  = 0.0;
    seeded_   = false;
    prev_high_ = false;
    prev_low_  = true;
    sdi_out_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    high_events_.store(0, std::memory_order_relaxed);
    low_events_.store(0, std::memory_order_relaxed);
}

void SentimentDispersionIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    sdi_ema_ = 0.0;
    seeded_  = false;
}

std::string SentimentDispersionIndex::to_stats_json() const {
    double sdi_val = sdi_out_.load(std::memory_order_relaxed);
    uint64_t tot   = total_.load(std::memory_order_relaxed);
    uint64_t hi    = high_events_.load(std::memory_order_relaxed);
    uint64_t lo    = low_events_.load(std::memory_order_relaxed);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"sdi\":"            << sdi_val
       << ",\"is_dispersed\":"   << (sdi_val > cfg_.high_threshold ? "true" : "false")
       << ",\"is_coherent\":"    << (sdi_val < cfg_.low_threshold  ? "true" : "false")
       << ",\"total_records\":"  << tot
       << ",\"high_events\":"    << hi
       << ",\"low_events\":"     << lo
       << "}";
    return ss.str();
}

} // namespace llmquant
