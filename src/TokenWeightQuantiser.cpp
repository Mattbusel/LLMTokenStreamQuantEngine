#include "TokenWeightQuantiser.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenWeightQuantiser::TokenWeightQuantiser(Config cfg)
    : cfg_(std::move(cfg))
    , rng_(std::random_device{}())
{}

double TokenWeightQuantiser::quantise(double weight) {
    std::function<void(double, double)> cb;
    bool   fire        = false;
    double clamp_rate_snap = 0.0;

    double result;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_high_clamp_rate;

        total_count_.fetch_add(1, std::memory_order_relaxed);

        // Clamp.
        bool clamped = (weight < cfg_.range_min || weight > cfg_.range_max);
        if (clamped) {
            clamp_count_.fetch_add(1, std::memory_order_relaxed);
            weight = std::clamp(weight, cfg_.range_min, cfg_.range_max);
        }

        // Stochastic rounding.
        int N = std::max(2, cfg_.levels);
        double step = (cfg_.range_max - cfg_.range_min) / (N - 1);
        if (step <= 0.0) {
            result = cfg_.range_min;
        } else {
            double frac = (weight - cfg_.range_min) / step;
            int    lo   = static_cast<int>(std::floor(frac));
            lo = std::clamp(lo, 0, N - 2);
            double p_up = frac - lo;
            int level   = (uniform_(rng_) < p_up) ? (lo + 1) : lo;
            level = std::clamp(level, 0, N - 1);
            result = cfg_.range_min + level * step;
        }

        // Update quantisation error EMA.
        double err = std::abs(weight - result);
        quant_error_val_ += cfg_.error_alpha * (err - quant_error_val_);
        quant_error_.store(quant_error_val_, std::memory_order_relaxed);

        // Clamp rate alert.
        uint64_t tc = total_count_.load(std::memory_order_relaxed);
        uint64_t cc = clamp_count_.load(std::memory_order_relaxed);
        if (tc > 0) {
            double rate = static_cast<double>(cc) / static_cast<double>(tc);
            if (rate > cfg_.clamp_alert_fraction) {
                fire           = true;
                clamp_rate_snap = rate;
            }
        }
    }

    if (fire && cb) cb(clamp_rate_snap, weight);
    return result;
}

int TokenWeightQuantiser::quantise_level(double weight) {
    std::lock_guard<std::mutex> lk(mutex_);
    weight = std::clamp(weight, cfg_.range_min, cfg_.range_max);
    int N = std::max(2, cfg_.levels);
    double step = (cfg_.range_max - cfg_.range_min) / (N - 1);
    if (step <= 0.0) return 0;
    double frac  = (weight - cfg_.range_min) / step;
    int    lo    = static_cast<int>(std::floor(frac));
    lo = std::clamp(lo, 0, N - 2);
    double p_up  = frac - lo;
    int    level = (uniform_(rng_) < p_up) ? (lo + 1) : lo;
    return std::clamp(level, 0, N - 1);
}

double TokenWeightQuantiser::clamp_rate() const noexcept {
    uint64_t tc = total_count_.load(std::memory_order_relaxed);
    if (tc == 0) return 0.0;
    return static_cast<double>(clamp_count_.load(std::memory_order_relaxed)) /
           static_cast<double>(tc);
}

std::string TokenWeightQuantiser::to_stats_json() const {
    double qe, cr;
    uint64_t tc, cc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        qe = quant_error_.load(std::memory_order_relaxed);
        tc = total_count_.load(std::memory_order_relaxed);
        cc = clamp_count_.load(std::memory_order_relaxed);
        cr = (tc > 0) ? (static_cast<double>(cc) / static_cast<double>(tc)) : 0.0;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"quantisation_error\":" << qe << ","
       << "\"clamp_count\":"        << cc << ","
       << "\"total_count\":"        << tc << ","
       << "\"clamp_rate\":"         << cr
       << "}";
    return ss.str();
}

void TokenWeightQuantiser::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    quant_error_val_ = 0.0;
    quant_error_.store(0.0, std::memory_order_relaxed);
    clamp_count_.store(0, std::memory_order_relaxed);
    total_count_.store(0, std::memory_order_relaxed);
}

void TokenWeightQuantiser::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_             = cfg;
    quant_error_val_ = 0.0;
}

} // namespace llmquant
