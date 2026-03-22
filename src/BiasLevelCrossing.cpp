#include "BiasLevelCrossing.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasLevelCrossing::BiasLevelCrossing(Config cfg)
    : cfg_(std::move(cfg))
    , win_buf_(cfg_.window, 0)  // 0 = no crossing, 1 = crossing
{}

void BiasLevelCrossing::record(double bias) {
    double new_zcr{};
    double new_mcr{};
    bool   changed = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        int crossed_zero = 0;
        if (seeded_) {
            crossed_zero = ((prev_bias_ >= 0.0) != (bias >= 0.0)) ? 1 : 0;
        }
        seeded_   = true;
        prev_bias_ = bias;

        win_buf_[win_head_] = crossed_zero;
        win_head_ = (win_head_ + 1) % cfg_.window;
        if (win_fill_ < cfg_.window) ++win_fill_;

        if (win_fill_ >= cfg_.min_samples) {
            int n = win_fill_;
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += win_buf_[i];
            new_zcr = sum / n;
            new_mcr = new_zcr; // simplified: single level (zero)

            if (prev_zcr_ >= 0.0 && std::fabs(new_zcr - prev_zcr_) > cfg_.zcr_change_threshold) {
                changed = true;
                change_events_.fetch_add(1, std::memory_order_relaxed);
            }
            double old_zcr_cb = prev_zcr_;
            prev_zcr_ = new_zcr;
            (void)old_zcr_cb;
        }
    }

    zcr_.store(new_zcr, std::memory_order_relaxed);
    mcr_.store(new_mcr, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (changed && cfg_.on_zcr_change) {
        // Retrieve old_zcr under lock is tricky — pass prev and new approx
        double old_approx{};
        {
            std::lock_guard<std::mutex> lk(mutex_);
            old_approx = prev_zcr_;
        }
        cfg_.on_zcr_change(old_approx, new_zcr);
    }
}

std::string BiasLevelCrossing::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"zero_crossing_rate\":" << zcr_.load(std::memory_order_relaxed)
       << ",\"mean_crossing_rate\":" << mcr_.load(std::memory_order_relaxed)
       << ",\"zcr_change_events\":" << change_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void BiasLevelCrossing::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0);
    win_head_  = 0;
    win_fill_  = 0;
    prev_bias_ = 0.0;
    seeded_    = false;
    prev_zcr_  = -1.0;
    zcr_.store(0.0, std::memory_order_relaxed);
    mcr_.store(0.0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BiasLevelCrossing::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_buf_.assign(cfg_.window, 0);
    win_head_ = 0;
    win_fill_ = 0;
}

} // namespace llmquant
