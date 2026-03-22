#include "MultiTimeframeAggregator.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MultiTimeframeAggregator::MultiTimeframeAggregator(Config config)
    : config_(std::move(config))
    , emas_(config_.timeframes.size(), 0.0) {}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

void MultiTimeframeAggregator::record(double bias_value) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    double fastest = 0.0, slowest = 0.0;
    bool any = false;
    std::function<void(double, double, double)> cb;
    bool fire_cb = false;
    double spread = 0.0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (size_t i = 0; i < config_.timeframes.size(); ++i) {
            double alpha = config_.timeframes[i].ema_alpha;
            emas_[i] = alpha * bias_value + (1.0 - alpha) * emas_[i];
            if (!any) {
                fastest = emas_[i];
                slowest = emas_[i];
                any = true;
            } else {
                if (emas_[i] > fastest) fastest = emas_[i];
                if (emas_[i] < slowest) slowest = emas_[i];
            }
        }
        if (any && config_.diverge_threshold > 0.0) {
            spread = std::abs(fastest - slowest);
            bool now_diverging = spread > config_.diverge_threshold;
            bool was_diverging = diverging_.exchange(now_diverging,
                                                      std::memory_order_relaxed);
            if (now_diverging && !was_diverging) {
                divergence_events_.fetch_add(1, std::memory_order_relaxed);
                cb = config_.on_divergence;
                fire_cb = true;
                spdlog::debug("[multi_tf] DIVERGE spread={:.3f} fast={:.3f} slow={:.3f}",
                              spread, fastest, slowest);
            }
        }
    }
    if (fire_cb && cb) cb(spread, fastest, slowest);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void MultiTimeframeAggregator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& e : emas_) e = 0.0;
    diverging_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// consensus
// ---------------------------------------------------------------------------

double MultiTimeframeAggregator::consensus() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    double weighted_sum = 0.0, weight_total = 0.0;
    for (size_t i = 0; i < config_.timeframes.size(); ++i) {
        double w = config_.timeframes[i].weight;
        weighted_sum  += w * emas_[i];
        weight_total  += w;
    }
    if (weight_total < 1e-15) return 0.0;
    return weighted_sum / weight_total;
}

// ---------------------------------------------------------------------------
// ema
// ---------------------------------------------------------------------------

double MultiTimeframeAggregator::ema(const std::string& name) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    for (size_t i = 0; i < config_.timeframes.size(); ++i) {
        if (config_.timeframes[i].name == name) return emas_[i];
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// timeframe_spread
// ---------------------------------------------------------------------------

double MultiTimeframeAggregator::timeframe_spread() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (emas_.empty()) return 0.0;
    double mn = emas_[0], mx = emas_[0];
    for (double e : emas_) {
        if (e < mn) mn = e;
        if (e > mx) mx = e;
    }
    return std::abs(mx - mn);
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void MultiTimeframeAggregator::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    emas_.assign(config_.timeframes.size(), 0.0);
    diverging_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string MultiTimeframeAggregator::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream s;
    s << "{\"total_records\":"    << total_records_.load(std::memory_order_relaxed)
      << ",\"divergence_events\":" << divergence_events_.load(std::memory_order_relaxed)
      << ",\"is_diverging\":"     << (diverging_.load(std::memory_order_relaxed) ? "true" : "false")
      << ",\"emas\":{";
    for (size_t i = 0; i < config_.timeframes.size(); ++i) {
        if (i > 0) s << ",";
        s << "\"" << config_.timeframes[i].name << "\":" << emas_[i];
    }
    s << "}}";
    return s.str();
}

} // namespace llmquant
