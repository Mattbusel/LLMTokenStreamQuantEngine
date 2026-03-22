#include "AdaptivePositionSizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

AdaptivePositionSizer::AdaptivePositionSizer(Config cfg)
    : cfg_(std::move(cfg))
{}

void AdaptivePositionSizer::recompute_locked() {
    // coherence_factor: clamp raw coherence to [0,1]
    double cf = std::max(0.0, std::min(1.0, coherence_input_));
    // confidence_factor
    double kf = std::max(0.0, std::min(1.0, confidence_input_));
    // vol_factor: 1 - rv/rv_max
    double rv_max = (cfg_.rv_max > 0.0) ? cfg_.rv_max : 1.0;
    double vf = std::max(0.0, 1.0 - rv_input_ / rv_max);
    // regime_factor
    double rf = 1.0;
    switch (regime_input_) {
        case Regime::Trending: rf = cfg_.regime_factor_trending; break;
        case Regime::Ranging:  rf = cfg_.regime_factor_ranging;  break;
        case Regime::Crash:    rf = cfg_.regime_factor_crash;    break;
    }
    rf = std::max(0.0, std::min(1.0, rf));

    double wc = cfg_.coherence_weight;
    double wk = cfg_.confidence_weight;
    double wv = cfg_.vol_weight;
    double wr = cfg_.regime_weight;

    // Each factor is blended: weighted_i = w_i * factor_i + (1 - w_i)
    // Then multiplied together.
    double m = (wc * cf + (1.0 - wc))
             * (wk * kf + (1.0 - wk))
             * (wv * vf + (1.0 - wv))
             * (wr * rf + (1.0 - wr));

    m = std::max(cfg_.min_multiplier, std::min(1.0, m));
    mult_.store(m, std::memory_order_relaxed);

    std::function<void(double, double)> cb = cfg_.on_size_change;
    if (prev_mult_ >= 0.0 && std::abs(m - prev_mult_) > cfg_.change_threshold) {
        double old = prev_mult_;
        prev_mult_ = m;
        change_events_.fetch_add(1, std::memory_order_relaxed);
        // Fire outside lock — release via lock pattern not possible here since we're
        // already inside; defer by storing and returning.
        // We accept the callback fires inside lock for this module (infrequent).
        if (cb) cb(old, m);
    } else {
        prev_mult_ = m;
    }
}

void AdaptivePositionSizer::set_coherence(double coherence) {
    std::lock_guard<std::mutex> lk(mutex_);
    coherence_input_ = coherence;
    recompute_locked();
}

void AdaptivePositionSizer::set_confidence(double confidence) {
    std::lock_guard<std::mutex> lk(mutex_);
    confidence_input_ = confidence;
    recompute_locked();
}

void AdaptivePositionSizer::set_realized_vol(double rv) {
    std::lock_guard<std::mutex> lk(mutex_);
    rv_input_ = rv;
    recompute_locked();
}

void AdaptivePositionSizer::set_regime(Regime r) {
    std::lock_guard<std::mutex> lk(mutex_);
    regime_input_ = r;
    recompute_locked();
}

std::string AdaptivePositionSizer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"multiplier\":"     << mult_.load(std::memory_order_relaxed) << ","
       << "\"change_events\":"  << change_events_.load(std::memory_order_relaxed) << ","
       << "\"coherence_in\":"   << coherence_input_ << ","
       << "\"confidence_in\":"  << confidence_input_ << ","
       << "\"rv_in\":"          << rv_input_ << ","
       << "\"regime_in\":"      << static_cast<int>(regime_input_)
       << "}";
    return ss.str();
}

void AdaptivePositionSizer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    coherence_input_  = 1.0;
    confidence_input_ = 1.0;
    rv_input_         = 0.0;
    regime_input_     = Regime::Trending;
    prev_mult_        = -1.0;
    mult_.store(1.0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
}

void AdaptivePositionSizer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_       = cfg;
    prev_mult_ = -1.0;
    recompute_locked();
}

} // namespace llmquant
