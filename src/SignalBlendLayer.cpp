#include "SignalBlendLayer.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SignalBlendLayer::SignalBlendLayer(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Source management
// ---------------------------------------------------------------------------

void SignalBlendLayer::register_source(const std::string& source_name, double weight) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (sources_.count(source_name)) return;  // already registered
    SourceState s;
    s.weight = (weight >= 0.0) ? weight : 1.0;
    sources_.emplace(source_name, std::move(s));
    spdlog::debug("[blend] registered source '{}' weight={:.2f}", source_name, s.weight);
}

void SignalBlendLayer::set_source_weight(const std::string& source_name, double weight) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = sources_.find(source_name);
    if (it == sources_.end()) return;
    it->second.weight = (weight >= 0.0) ? weight : 0.0;
}

void SignalBlendLayer::set_blend_callback(BlendCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    blend_cb_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// Hot path: push_signal
// ---------------------------------------------------------------------------

bool SignalBlendLayer::push_signal(const std::string& source_name,
                                   const TradeSignal& signal) {
    total_pushes_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(mutex_);

    auto it = sources_.find(source_name);
    if (it == sources_.end()) {
        spdlog::warn("[blend] push_signal from unregistered source '{}'", source_name);
        return false;
    }

    SourceState& s = it->second;
    s.last_signal    = signal;
    s.last_push_time = std::chrono::steady_clock::now();
    s.pending        = true;
    ++s.pushes;

    // Check quorum.
    if (count_pending_locked() < config_.min_sources) return false;

    // Emit blend.
    TradeSignal blended = compute_blend_locked();
    blends_emitted_.fetch_add(1, std::memory_order_relaxed);

    if (config_.reset_pending_after_blend) {
        for (auto& [name, src] : sources_) src.pending = false;
    }

    BlendCallback cb = blend_cb_;
    if (cb) {
        // cb must not call back into SignalBlendLayer (mutex is held).
        try { cb(blended); } catch (...) {}
    }
    return true;
}

// ---------------------------------------------------------------------------
// force_blend
// ---------------------------------------------------------------------------

TradeSignal SignalBlendLayer::force_blend() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return compute_blend_locked();
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

bool SignalBlendLayer::is_stale_locked(const SourceState& s) const {
    if (config_.staleness_ms <= 0) return false;
    if (s.pushes == 0) return true;
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - s.last_push_time).count();
    return age_ms > config_.staleness_ms;
}

int SignalBlendLayer::count_pending_locked() const {
    int count = 0;
    for (auto& [name, s] : sources_) {
        if (s.pending && !is_stale_locked(s)) ++count;
    }
    return count;
}

TradeSignal SignalBlendLayer::compute_blend_locked() const {
    TradeSignal out{};
    if (sources_.empty()) return out;

    double total_weight = 0.0;
    double sum_bias     = 0.0;
    double sum_vol      = 0.0;
    double sum_conf     = 0.0;
    double sum_spread   = 0.0;
    double sum_quality  = 0.0;
    int    count        = 0;

    for (auto& [name, s] : sources_) {
        if (s.pushes == 0 || is_stale_locked(s)) continue;

        double w = 1.0;
        switch (config_.mode) {
            case BlendMode::Equal:
                w = 1.0;
                break;
            case BlendMode::ConfidenceWeighted:
                w = s.last_signal.confidence;
                if (w < 1e-9) w = 1e-9;
                break;
            case BlendMode::CustomWeighted:
                w = s.weight;
                break;
        }

        sum_bias    += w * s.last_signal.delta_bias_shift;
        sum_vol     += w * s.last_signal.volatility_adjustment;
        sum_conf    += w * s.last_signal.confidence;
        sum_spread  += w * s.last_signal.spread_modifier;
        sum_quality += w * s.last_signal.signal_quality;
        total_weight += w;
        ++count;

        // Carry through the latest timestamp.
        if (s.last_signal.timestamp_ns > out.timestamp_ns)
            out.timestamp_ns = s.last_signal.timestamp_ns;
    }

    if (total_weight < 1e-12 || count == 0) return out;

    out.delta_bias_shift       = sum_bias    / total_weight;
    out.volatility_adjustment  = sum_vol     / total_weight;
    out.confidence             = sum_conf    / total_weight;
    out.spread_modifier        = sum_spread  / total_weight;
    out.signal_quality         = sum_quality / total_weight;
    return out;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::vector<SignalBlendLayer::SourceStats>
SignalBlendLayer::get_source_stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<SourceStats> out;
    out.reserve(sources_.size());
    for (auto& [name, s] : sources_) {
        SourceStats ss;
        ss.name            = name;
        ss.pushes          = s.pushes;
        ss.last_confidence = s.last_signal.confidence;
        ss.last_bias       = s.last_signal.delta_bias_shift;
        ss.weight          = s.weight;
        out.push_back(ss);
    }
    return out;
}

std::string SignalBlendLayer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream o;
    o << "{\"blends_emitted\":" << blends_emitted_.load(std::memory_order_relaxed)
      << ",\"total_pushes\":"   << total_pushes_.load(std::memory_order_relaxed)
      << ",\"source_count\":"   << sources_.size()
      << ",\"min_sources\":"    << config_.min_sources
      << "}";
    return o.str();
}

void SignalBlendLayer::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

void SignalBlendLayer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& [name, s] : sources_) {
        s.pending = false;
        s.pushes  = 0;
        s.last_signal = TradeSignal{};
    }
}

} // namespace llmquant
