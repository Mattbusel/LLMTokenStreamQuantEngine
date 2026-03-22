#include "NarrativeCoherenceTracker.h"
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llmquant {

NarrativeCoherenceTracker::NarrativeCoherenceTracker(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void NarrativeCoherenceTracker::record(double bias) {
    double new_coh = 0.0;
    bool high_ev = false, low_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            // Majority direction = sign of rolling mean
            double sum = 0.0;
            for (int i = 0; i < fill_; ++i) sum += buf_[i];
            double mean = sum / fill_;
            int dir = (mean >= 0.0) ? 1 : -1;

            // Coherence = fraction of window agreeing with majority direction
            int agree = 0;
            for (int i = 0; i < fill_; ++i) {
                if (dir > 0 && buf_[i] >= 0.0) ++agree;
                else if (dir < 0 && buf_[i] < 0.0) ++agree;
            }
            new_coh = static_cast<double>(agree) / fill_;

            bool cur_high = (new_coh >= cfg_.high_threshold);
            bool cur_low  = (new_coh <= cfg_.low_threshold);

            if (cur_high && !prev_high_) {
                high_ev = true;
                high_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (cur_low && !prev_low_) {
                low_ev = true;
                low_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_high_ = cur_high;
            prev_low_  = cur_low;
        }
    }

    coherence_.store(new_coh, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (high_ev && cfg_.on_high_coherence) cfg_.on_high_coherence(new_coh);
    if (low_ev  && cfg_.on_low_coherence)  cfg_.on_low_coherence(new_coh);
}

double NarrativeCoherenceTracker::coherence() const noexcept {
    return coherence_.load(std::memory_order_relaxed);
}

std::size_t NarrativeCoherenceTracker::high_coherence_events() const noexcept {
    return high_events_.load(std::memory_order_relaxed);
}
std::size_t NarrativeCoherenceTracker::low_coherence_events() const noexcept {
    return low_events_.load(std::memory_order_relaxed);
}
std::size_t NarrativeCoherenceTracker::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

std::string NarrativeCoherenceTracker::to_stats_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"coherence\":"       << coherence_.load(std::memory_order_relaxed)
       << ",\"high_events\":"     << high_events_.load(std::memory_order_relaxed)
       << ",\"low_events\":"      << low_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":"   << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeCoherenceTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0;
    prev_high_ = prev_low_ = false;
    coherence_.store(0.0, std::memory_order_relaxed);
    high_events_.store(0, std::memory_order_relaxed);
    low_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeCoherenceTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
