#include "BiasLocalExtrema.h"
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasLocalExtrema::BiasLocalExtrema(Config cfg)
    : cfg_(std::move(cfg))
{}

void BiasLocalExtrema::record(double bias) {
    bool peak_ev = false, trough_ev = false;
    double new_lp = last_peak_.load(std::memory_order_relaxed);
    double new_lt = last_trough_.load(std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            extremum_ = bias;
            seeded_ = true;
        } else if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples)) {
            if (seek_peak_) {
                // Track highest point; reverse when bias drops by threshold
                if (bias > extremum_) {
                    extremum_ = bias;
                } else if ((extremum_ - bias) >= cfg_.reversal_threshold) {
                    // Peak confirmed
                    new_lp = extremum_;
                    peak_ev = true;
                    peak_ev_.fetch_add(1, std::memory_order_relaxed);
                    seek_peak_ = false;
                    extremum_  = bias;
                }
            } else {
                // Looking for trough; track lowest, reverse when bias rises by threshold
                if (bias < extremum_) {
                    extremum_ = bias;
                } else if ((bias - extremum_) >= cfg_.reversal_threshold) {
                    // Trough confirmed
                    new_lt = extremum_;
                    trough_ev = true;
                    trough_ev_.fetch_add(1, std::memory_order_relaxed);
                    seek_peak_ = true;
                    extremum_  = bias;
                }
            }
        }
    }

    if (peak_ev)   last_peak_.store(new_lp, std::memory_order_relaxed);
    if (trough_ev) last_trough_.store(new_lt, std::memory_order_relaxed);
    seeking_peak_.store(seek_peak_, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (peak_ev   && cfg_.on_peak)   cfg_.on_peak(new_lp);
    if (trough_ev && cfg_.on_trough) cfg_.on_trough(new_lt);
}

double BiasLocalExtrema::last_peak()    const noexcept { return last_peak_.load(std::memory_order_relaxed); }
double BiasLocalExtrema::last_trough()  const noexcept { return last_trough_.load(std::memory_order_relaxed); }
bool   BiasLocalExtrema::seeking_peak() const noexcept { return seeking_peak_.load(std::memory_order_relaxed); }

std::size_t BiasLocalExtrema::peak_events()   const noexcept { return peak_ev_.load(std::memory_order_relaxed); }
std::size_t BiasLocalExtrema::trough_events() const noexcept { return trough_ev_.load(std::memory_order_relaxed); }
std::size_t BiasLocalExtrema::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string BiasLocalExtrema::to_stats_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"last_peak\":"     << last_peak_.load(std::memory_order_relaxed)
       << ",\"last_trough\":"   << last_trough_.load(std::memory_order_relaxed)
       << ",\"seeking_peak\":"  << (seeking_peak_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"peak_events\":"   << peak_ev_.load(std::memory_order_relaxed)
       << ",\"trough_events\":" << trough_ev_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasLocalExtrema::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false; seek_peak_ = true; extremum_ = 0.0;
    last_peak_.store(0.0, std::memory_order_relaxed);
    last_trough_.store(0.0, std::memory_order_relaxed);
    seeking_peak_.store(true, std::memory_order_relaxed);
    peak_ev_.store(0, std::memory_order_relaxed);
    trough_ev_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasLocalExtrema::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
