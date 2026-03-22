#include "BiasConcentrationRisk.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasConcentrationRisk::BiasConcentrationRisk(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void BiasConcentrationRisk::record(double bias) {
    double new_hhi = 0.0;
    bool conc_ev = false, disp_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = std::fabs(bias);
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            double total = 0.0;
            for (int i = 0; i < n; ++i) total += buf_[i];

            if (total > 1e-12) {
                for (int i = 0; i < n; ++i) {
                    double share = buf_[i] / total;
                    new_hhi += share * share;
                }
            } else {
                new_hhi = 1.0 / n; // uniform
            }

            bool conc = (new_hhi > cfg_.concentration_threshold);
            if (conc && !prev_concentrated_) {
                conc_ev = true;
                conc_events_.fetch_add(1, std::memory_order_relaxed);
            } else if (!conc && prev_concentrated_) {
                disp_ev = true;
                disp_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_concentrated_ = conc;
        }
    }

    hhi_.store(new_hhi, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (conc_ev && cfg_.on_concentrated) cfg_.on_concentrated(new_hhi);
    if (disp_ev && cfg_.on_dispersed)    cfg_.on_dispersed(new_hhi);
}

double BiasConcentrationRisk::hhi() const noexcept { return hhi_.load(std::memory_order_relaxed); }

std::size_t BiasConcentrationRisk::concentration_events() const noexcept { return conc_events_.load(std::memory_order_relaxed); }
std::size_t BiasConcentrationRisk::dispersed_events()     const noexcept { return disp_events_.load(std::memory_order_relaxed); }
std::size_t BiasConcentrationRisk::total_records()        const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasConcentrationRisk::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"hhi\":" << hhi_.load(std::memory_order_relaxed)
       << ",\"concentration_events\":" << conc_events_.load(std::memory_order_relaxed)
       << ",\"dispersed_events\":" << disp_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasConcentrationRisk::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0; prev_concentrated_ = false;
    hhi_.store(0.0, std::memory_order_relaxed);
    conc_events_.store(0, std::memory_order_relaxed);
    disp_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasConcentrationRisk::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
