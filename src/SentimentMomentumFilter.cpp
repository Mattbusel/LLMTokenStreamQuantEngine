#include "SentimentMomentumFilter.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>

namespace llmquant {

SentimentMomentumFilter::SentimentMomentumFilter(Config config)
    : config_{config}
    , analyzer_{config.analyzer} {}

void SentimentMomentumFilter::record_sample(double sentiment_score) {
    analyzer_.add_sample(sentiment_score);
}

TradeSignal SentimentMomentumFilter::filter_signal(const TradeSignal& signal) {
    signals_in_.fetch_add(1, std::memory_order_relaxed);

    Config cfg;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        cfg = config_;
    }

    if (cfg.mode == Mode::Disabled) {
        signals_passed_.fetch_add(1, std::memory_order_relaxed);
        return signal;
    }

    SentimentTrend trend = analyzer_.get_trend();

    // Determine whether the signal direction is consistent with the trend.
    bool bullish = (signal.delta_bias_shift > 0.0);
    bool pass    = false;

    switch (cfg.mode) {
        case Mode::Strict:
            pass = (bullish && trend == SentimentTrend::Improving)
                || (!bullish && trend == SentimentTrend::Declining);
            break;
        case Mode::Relaxed:
            pass = (bullish && (trend == SentimentTrend::Improving
                                || trend == SentimentTrend::Stable))
                || (!bullish && (trend == SentimentTrend::Declining
                                 || trend == SentimentTrend::Stable));
            break;
        case Mode::Disabled:
            pass = true;
            break;
    }

    if (!pass) {
        signals_blocked_.fetch_add(1, std::memory_order_relaxed);
        TradeSignal blocked = signal;
        blocked.delta_bias_shift = 0.0;
        return blocked;
    }

    // Optional slope-based scaling.
    if (cfg.scale_by_slope && cfg.slope_scale > 1e-15) {
        double slope = analyzer_.get_slope();
        double scale = std::min(1.0, std::abs(slope) / cfg.slope_scale);
        if (scale < 1.0 - 1e-9) {
            signals_scaled_.fetch_add(1, std::memory_order_relaxed);
            TradeSignal scaled = signal;
            scaled.delta_bias_shift = signal.delta_bias_shift * scale;
            signals_passed_.fetch_add(1, std::memory_order_relaxed);
            return scaled;
        }
    }

    signals_passed_.fetch_add(1, std::memory_order_relaxed);
    return signal;
}

SentimentTrend SentimentMomentumFilter::current_trend() const {
    return analyzer_.get_trend();
}

double SentimentMomentumFilter::current_slope() const {
    return analyzer_.get_slope();
}

SentimentMomentumFilter::Stats SentimentMomentumFilter::get_stats() const {
    Stats s;
    s.signals_in      = signals_in_.load(std::memory_order_relaxed);
    s.signals_passed  = signals_passed_.load(std::memory_order_relaxed);
    s.signals_blocked = signals_blocked_.load(std::memory_order_relaxed);
    s.signals_scaled  = signals_scaled_.load(std::memory_order_relaxed);
    s.pass_rate       = (s.signals_in > 0)
                        ? static_cast<double>(s.signals_passed)
                          / static_cast<double>(s.signals_in)
                        : 0.0;
    return s;
}

std::string SentimentMomentumFilter::to_stats_json() const {
    auto s = get_stats();
    const char* mode_str = "relaxed";
    {
        std::lock_guard<std::mutex> lk(mutex_);
        switch (config_.mode) {
            case Mode::Strict:   mode_str = "strict";   break;
            case Mode::Relaxed:  mode_str = "relaxed";  break;
            case Mode::Disabled: mode_str = "disabled"; break;
        }
    }
    const char* trend_str = SentimentTrajectoryAnalyzer::trend_name(analyzer_.get_trend());
    double slope = analyzer_.get_slope();

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"mode\":\"%s\",\"trend\":\"%s\",\"slope\":%.6f"
        ",\"signals_in\":%" PRIu64 ",\"signals_passed\":%" PRIu64
        ",\"signals_blocked\":%" PRIu64 ",\"pass_rate\":%.4f}",
        mode_str, trend_str, slope,
        s.signals_in, s.signals_passed, s.signals_blocked, s.pass_rate);
    return buf;
}

void SentimentMomentumFilter::reset() {
    analyzer_.reset();
    signals_in_.store(0, std::memory_order_relaxed);
    signals_passed_.store(0, std::memory_order_relaxed);
    signals_blocked_.store(0, std::memory_order_relaxed);
    signals_scaled_.store(0, std::memory_order_relaxed);
}

void SentimentMomentumFilter::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

SentimentMomentumFilter::Config SentimentMomentumFilter::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

} // namespace llmquant
