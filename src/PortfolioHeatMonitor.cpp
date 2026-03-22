#include "PortfolioHeatMonitor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace llmquant {

// ---------------------------------------------------------------------------

PortfolioHeatMonitor::PortfolioHeatMonitor(Config cfg) : cfg_(std::move(cfg)) {}

// ---------------------------------------------------------------------------

const char* PortfolioHeatMonitor::level_name(Level l) noexcept {
    switch (l) {
        case Level::Cool:     return "Cool";
        case Level::Warm:     return "Warm";
        case Level::Hot:      return "Hot";
        case Level::Critical: return "Critical";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------

double PortfolioHeatMonitor::compute_heat_locked() const {
    // Base heat: sum of |bias| * |position| * heat_scale for all instruments.
    double heat = 0.0;
    for (const auto& [name, state] : instruments_) {
        heat += std::abs(state.bias) * std::abs(state.position) * cfg_.heat_scale;
    }

    if (!cfg_.double_correlated_heat) return heat;

    // Correlation penalty: for each pair of instruments both exceeding the
    // correlation gate threshold with the SAME sign, add extra heat equal to
    // the geometric mean of their individual contributions.
    std::vector<std::pair<std::string, InstrumentState>> vec(
        instruments_.begin(), instruments_.end());

    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = i + 1; j < vec.size(); ++j) {
            double bi = vec[i].second.bias;
            double bj = vec[j].second.bias;
            bool same_sign = (bi > 0.0 && bj > 0.0) || (bi < 0.0 && bj < 0.0);
            if (!same_sign) continue;
            if (std::abs(bi) < cfg_.correlation_gate_threshold) continue;
            if (std::abs(bj) < cfg_.correlation_gate_threshold) continue;

            double hi = std::abs(bi) * std::abs(vec[i].second.position) * cfg_.heat_scale;
            double hj = std::abs(bj) * std::abs(vec[j].second.position) * cfg_.heat_scale;
            heat += std::sqrt(hi * hj);  // geometric mean penalty
        }
    }

    return heat;
}

// ---------------------------------------------------------------------------

void PortfolioHeatMonitor::check_level_transition(double new_heat) {
    Level new_level = Level::Cool;
    if (new_heat >= cfg_.critical_heat)
        new_level = Level::Critical;
    else if (new_heat >= cfg_.warn_heat)
        new_level = Level::Hot;
    else if (new_heat >= cfg_.warn_heat * 0.8)
        new_level = Level::Warm;

    if (new_level == current_level_) return;

    Level old_level = current_level_;
    current_level_  = new_level;

    if (new_level > old_level) {
        // Escalation
        if (new_level == Level::Hot     && warn_cb_)     warn_cb_(new_heat);
        if (new_level == Level::Critical && critical_cb_) critical_cb_(new_heat);
    } else {
        // Recovery
        if (new_level == Level::Cool && recovery_cb_) recovery_cb_(new_heat);
    }
}

// ---------------------------------------------------------------------------

void PortfolioHeatMonitor::update_instrument(const std::string& name,
                                              double bias,
                                              double position) {
    std::lock_guard<std::mutex> lk(mtx_);
    instruments_[name] = {bias, position};
    double h = compute_heat_locked();
    check_level_transition(h);
}

void PortfolioHeatMonitor::remove_instrument(const std::string& name) {
    std::lock_guard<std::mutex> lk(mtx_);
    instruments_.erase(name);
    double h = compute_heat_locked();
    check_level_transition(h);
}

double PortfolioHeatMonitor::total_heat() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return compute_heat_locked();
}

PortfolioHeatMonitor::Level PortfolioHeatMonitor::heat_level() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return current_level_;
}

std::vector<PortfolioHeatMonitor::InstrumentSnapshot>
PortfolioHeatMonitor::snapshot() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<InstrumentSnapshot> out;
    out.reserve(instruments_.size());
    for (const auto& [name, state] : instruments_) {
        InstrumentSnapshot s;
        s.name                = name;
        s.bias                = state.bias;
        s.position            = state.position;
        s.heat_contribution   = std::abs(state.bias) * std::abs(state.position) * cfg_.heat_scale;
        s.correlated_burst    = std::abs(state.bias) >= cfg_.correlation_gate_threshold;
        out.push_back(s);
    }
    // Sort by heat contribution descending for readability.
    std::sort(out.begin(), out.end(),
              [](const InstrumentSnapshot& a, const InstrumentSnapshot& b) {
                  return a.heat_contribution > b.heat_contribution;
              });
    return out;
}

size_t PortfolioHeatMonitor::instrument_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return instruments_.size();
}

// ---------------------------------------------------------------------------
// Callback registration

void PortfolioHeatMonitor::set_warn_callback(HeatCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    warn_cb_ = std::move(cb);
}

void PortfolioHeatMonitor::set_critical_callback(HeatCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    critical_cb_ = std::move(cb);
}

void PortfolioHeatMonitor::set_recovery_callback(HeatCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    recovery_cb_ = std::move(cb);
}

// ---------------------------------------------------------------------------

void PortfolioHeatMonitor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_           = cfg;
    current_level_ = Level::Cool;
}

void PortfolioHeatMonitor::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    instruments_.clear();
    current_level_ = Level::Cool;
}

// ---------------------------------------------------------------------------

std::string PortfolioHeatMonitor::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mtx_);
    double h = compute_heat_locked();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"level\":\"%s\",\"total_heat\":%.4f,\"instruments\":%zu,"
        "\"warn_heat\":%.2f,\"critical_heat\":%.2f}",
        level_name(current_level_),
        h,
        instruments_.size(),
        cfg_.warn_heat,
        cfg_.critical_heat);
    return buf;
}

} // namespace llmquant
