#include "SignalTailRiskMeter.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalTailRiskMeter::SignalTailRiskMeter(Config cfg)
    : cfg_(std::move(cfg))
    , win_buf_(cfg_.window, 0.0)
{}

void SignalTailRiskMeter::record(double bias) {
    double new_var{};
    double new_es{};
    bool alarm = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        win_buf_[win_head_] = bias;
        win_head_ = (win_head_ + 1) % cfg_.window;
        if (win_fill_ < cfg_.window) ++win_fill_;

        if (win_fill_ >= cfg_.min_samples) {
            // Copy window, partial sort to find quantile
            std::vector<double> sorted(win_buf_.begin(),
                                       win_buf_.begin() + win_fill_);
            int q_idx = static_cast<int>(cfg_.alpha * sorted.size());
            if (q_idx >= static_cast<int>(sorted.size())) q_idx = static_cast<int>(sorted.size()) - 1;
            std::nth_element(sorted.begin(), sorted.begin() + q_idx, sorted.end());
            new_var = sorted[q_idx]; // VaR at alpha (lower quantile)

            // ES = mean of all values ≤ VaR
            double sum_tail = 0.0;
            int    cnt_tail = 0;
            for (int i = 0; i < win_fill_; ++i) {
                if (win_buf_[i] <= new_var) {
                    sum_tail += win_buf_[i];
                    ++cnt_tail;
                }
            }
            new_es = (cnt_tail > 0) ? sum_tail / cnt_tail : new_var;

            bool ok = (new_es >= cfg_.es_threshold);
            if (!ok && prev_ok_) {
                alarm = true;
                tail_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_ok_ = ok;
        }
    }

    var_.store(new_var, std::memory_order_relaxed);
    es_.store(new_es, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (alarm && cfg_.on_tail_event) cfg_.on_tail_event(new_es);
}

std::string SignalTailRiskMeter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"var_alpha\":" << var_.load(std::memory_order_relaxed)
       << ",\"expected_shortfall\":" << es_.load(std::memory_order_relaxed)
       << ",\"tail_events\":" << tail_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalTailRiskMeter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(win_buf_.begin(), win_buf_.end(), 0.0);
    win_head_ = 0;
    win_fill_ = 0;
    prev_ok_  = true;
    var_.store(0.0, std::memory_order_relaxed);
    es_.store(0.0, std::memory_order_relaxed);
    tail_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalTailRiskMeter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    win_buf_.assign(cfg_.window, 0.0);
    win_head_ = 0;
    win_fill_ = 0;
}

} // namespace llmquant
