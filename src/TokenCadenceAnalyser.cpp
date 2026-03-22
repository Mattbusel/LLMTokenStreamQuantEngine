#include "TokenCadenceAnalyser.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenCadenceAnalyser::TokenCadenceAnalyser(Config cfg)
    : cfg_(std::move(cfg))
    , iai_buf_(cfg_.window, 0.0)
{}

void TokenCadenceAnalyser::record(double /*bias*/) {
    double new_iai_ms = 0.0;
    double new_mean   = 0.0;
    double new_cv     = 0.0;
    bool   burst_ev   = false;
    bool   gap_ev     = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        auto now = Clock::now();
        if (has_prev_) {
            auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - prev_time_).count();
            new_iai_ms = static_cast<double>(dur_ns) / 1e6;

            iai_buf_[iai_head_] = new_iai_ms;
            iai_head_ = (iai_head_ + 1) % cfg_.window;
            if (iai_fill_ < cfg_.window) ++iai_fill_;

            if (iai_fill_ >= cfg_.min_samples) {
                int n = iai_fill_;
                double sum = 0.0;
                for (int i = 0; i < n; ++i) sum += iai_buf_[i];
                new_mean = sum / n;

                double var = 0.0;
                for (int i = 0; i < n; ++i) {
                    double d = iai_buf_[i] - new_mean;
                    var += d * d;
                }
                var /= n;
                double std_dev = std::sqrt(var);
                new_cv = (new_mean > 1e-12) ? std_dev / new_mean : 0.0;

                if (new_iai_ms < cfg_.burst_threshold_ms) {
                    burst_ev = true;
                    burst_events_.fetch_add(1, std::memory_order_relaxed);
                }
                if (new_iai_ms > cfg_.gap_threshold_ms) {
                    gap_ev = true;
                    gap_events_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        prev_time_ = now;
        has_prev_  = true;
    }

    mean_iai_.store(new_mean, std::memory_order_relaxed);
    cv_iai_.store(new_cv, std::memory_order_relaxed);
    last_iai_.store(new_iai_ms, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (burst_ev && cfg_.on_burst) cfg_.on_burst(new_iai_ms);
    if (gap_ev   && cfg_.on_gap)   cfg_.on_gap(new_iai_ms);
}

double TokenCadenceAnalyser::mean_iai_ms() const noexcept { return mean_iai_.load(std::memory_order_relaxed); }
double TokenCadenceAnalyser::cv_iai()      const noexcept { return cv_iai_.load(std::memory_order_relaxed); }
double TokenCadenceAnalyser::last_iai_ms() const noexcept { return last_iai_.load(std::memory_order_relaxed); }

std::size_t TokenCadenceAnalyser::burst_events()  const noexcept { return burst_events_.load(std::memory_order_relaxed); }
std::size_t TokenCadenceAnalyser::gap_events()    const noexcept { return gap_events_.load(std::memory_order_relaxed); }
std::size_t TokenCadenceAnalyser::total_records() const noexcept { return total_.load(std::memory_order_relaxed); }

std::string TokenCadenceAnalyser::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "{\"mean_iai_ms\":" << mean_iai_.load(std::memory_order_relaxed)
       << ",\"cv_iai\":" << cv_iai_.load(std::memory_order_relaxed)
       << ",\"last_iai_ms\":" << last_iai_.load(std::memory_order_relaxed)
       << ",\"burst_events\":" << burst_events_.load(std::memory_order_relaxed)
       << ",\"gap_events\":" << gap_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenCadenceAnalyser::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(iai_buf_.begin(), iai_buf_.end(), 0.0);
    iai_head_ = 0; iai_fill_ = 0; has_prev_ = false;
    mean_iai_.store(0.0, std::memory_order_relaxed);
    cv_iai_.store(0.0, std::memory_order_relaxed);
    last_iai_.store(0.0, std::memory_order_relaxed);
    burst_events_.store(0, std::memory_order_relaxed);
    gap_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenCadenceAnalyser::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    iai_buf_.assign(cfg_.window, 0.0);
    iai_head_ = 0; iai_fill_ = 0;
}

} // namespace llmquant
