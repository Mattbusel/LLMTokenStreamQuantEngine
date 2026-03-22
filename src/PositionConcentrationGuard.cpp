#include "PositionConcentrationGuard.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PositionConcentrationGuard::PositionConcentrationGuard() = default;

PositionConcentrationGuard::PositionConcentrationGuard(const Config& cfg)
    : cfg_(cfg) {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void PositionConcentrationGuard::recompute_locked() {
    double total = 0.0;
    for (auto& [name, ts] : themes_)
        total += ts.sum;

    if (total <= 0.0) {
        hhi_.store(0.0, std::memory_order_relaxed);
        return;
    }

    double hhi = 0.0;
    double best_share = -1.0;
    std::string best_name;
    for (auto& [name, ts] : themes_) {
        double share = ts.sum / total;
        hhi += share * share;
        if (share > best_share) {
            best_share = share;
            best_name = name;
        }
    }
    hhi_.store(hhi, std::memory_order_relaxed);
    dominant_ = best_name;
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void PositionConcentrationGuard::record_signal(const std::string& theme,
                                                double bias) {
    std::function<void(double, const std::string&)> on_conc;
    std::function<void(double)> on_div;
    bool fire_conc = false;
    bool fire_div  = false;
    double hhi_val = 0.0;
    std::string dom;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto& ts = themes_[theme];
        if (static_cast<int>(ts.window.size()) < cfg_.window_size) {
            ts.window.resize(cfg_.window_size, 0.0);
        }

        double abs_bias = std::abs(bias);
        // Subtract old sample from rolling sum
        ts.sum -= ts.window[ts.head];
        ts.window[ts.head] = abs_bias;
        ts.sum += abs_bias;
        if (ts.sum < 0.0) ts.sum = 0.0;  // guard floating-point underflow
        ts.head = (ts.head + 1) % cfg_.window_size;
        if (ts.count < cfg_.window_size) ++ts.count;

        recompute_locked();
        hhi_val = hhi_.load(std::memory_order_relaxed);
        dom = dominant_;

        if (!concentrated_ && hhi_val > cfg_.concentration_threshold) {
            concentrated_ = true;
            conc_events_.fetch_add(1, std::memory_order_relaxed);
            fire_conc = true;
            on_conc = cfg_.on_concentrated;
        } else if (concentrated_ && hhi_val < cfg_.concentration_threshold * cfg_.clear_hysteresis) {
            concentrated_ = false;
            fire_div = true;
            on_div = cfg_.on_diversified;
        }
        conc_flag_.store(concentrated_, std::memory_order_relaxed);
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (fire_conc && on_conc) on_conc(hhi_val, dom);
    if (fire_div  && on_div)  on_div(hhi_val);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double PositionConcentrationGuard::hhi() const noexcept {
    return hhi_.load(std::memory_order_relaxed);
}

bool PositionConcentrationGuard::is_concentrated() const noexcept {
    return conc_flag_.load(std::memory_order_relaxed);
}

uint64_t PositionConcentrationGuard::total_signals() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

uint64_t PositionConcentrationGuard::concentration_events() const noexcept {
    return conc_events_.load(std::memory_order_relaxed);
}

std::string PositionConcentrationGuard::dominant_theme() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return dominant_;
}

double PositionConcentrationGuard::dominant_share() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    double total = 0.0;
    for (const auto& [name, ts] : themes_) total += ts.sum;
    if (total <= 0.0 || dominant_.empty()) return 0.0;
    auto it = themes_.find(dominant_);
    if (it == themes_.end()) return 0.0;
    return it->second.sum / total;
}

int PositionConcentrationGuard::distinct_themes() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(themes_.size());
}

std::vector<std::pair<std::string, double>>
PositionConcentrationGuard::theme_shares() const {
    std::lock_guard<std::mutex> lk(mutex_);
    double total = 0.0;
    for (const auto& [name, ts] : themes_) total += ts.sum;
    std::vector<std::pair<std::string, double>> out;
    out.reserve(themes_.size());
    for (const auto& [name, ts] : themes_)
        out.emplace_back(name, total > 0.0 ? ts.sum / total : 0.0);
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    return out;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void PositionConcentrationGuard::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    themes_.clear();
    concentrated_ = false;
    dominant_.clear();

    hhi_.store(0.0, std::memory_order_relaxed);
    conc_flag_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    conc_events_.store(0, std::memory_order_relaxed);
}

void PositionConcentrationGuard::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    themes_.clear();
    concentrated_ = false;
    dominant_.clear();

    hhi_.store(0.0, std::memory_order_relaxed);
    conc_flag_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    conc_events_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string PositionConcentrationGuard::to_stats_json() const {
    double h    = hhi();
    bool   conc = is_concentrated();
    auto   tot  = total_signals();
    auto   evts = concentration_events();
    std::string dom = dominant_theme();
    double share = dominant_share();

    std::ostringstream os;
    os << "{"
       << "\"hhi\":"                    << h    << ","
       << "\"is_concentrated\":"        << (conc ? "true" : "false") << ","
       << "\"dominant_theme\":\""       << dom  << "\","
       << "\"dominant_share\":"         << share << ","
       << "\"concentration_events\":"   << evts  << ","
       << "\"total_signals\":"          << tot
       << "}";
    return os.str();
}

}  // namespace llmquant
