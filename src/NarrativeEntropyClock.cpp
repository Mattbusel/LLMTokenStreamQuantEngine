#include "NarrativeEntropyClock.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeEntropyClock::NarrativeEntropyClock(Config cfg)
    : cfg_(std::move(cfg))
{
    init_ref_dist();
}

void NarrativeEntropyClock::init_ref_dist() {
    int b = std::max(2, cfg_.n_bins);
    cfg_.n_bins = b;
    ref_dist_.assign(static_cast<std::size_t>(b),
                     1.0 / static_cast<double>(b));  // uniform prior
}

int NarrativeEntropyClock::bin_for(double bias) const {
    int b = cfg_.n_bins;
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (bias - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * b);
}

void NarrativeEntropyClock::record(double bias) {
    bool   fire_exhausted = false;
    double nats_val       = 0.0;
    double delta_kl       = 0.0;
    std::function<void(double)>         exhausted_cb;
    std::function<void(double, double)> update_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        exhausted_cb = cfg_.on_budget_exhausted;
        update_cb    = cfg_.on_kl_update;

        uint64_t n = total_.fetch_add(1, std::memory_order_relaxed) + 1;

        int b = bin_for(bias);

        // Compute surprisal of current observation under reference distribution
        double q_b = ref_dist_[static_cast<std::size_t>(b)] + cfg_.epsilon;
        // Normalize
        double sum_q = cfg_.epsilon * cfg_.n_bins;
        for (double v : ref_dist_) sum_q += v;
        q_b /= sum_q;

        delta_kl = -std::log(q_b);  // surprisal = -log q(x)

        // Update reference distribution with EMA
        double alpha = cfg_.ref_ema_alpha;
        for (auto& v : ref_dist_) v *= (1.0 - alpha);
        ref_dist_[static_cast<std::size_t>(b)] += alpha;

        // Accumulate KL
        accum_kl_v_ += delta_kl;
        nats_val = accum_kl_v_;
        accum_kl_.store(accum_kl_v_, std::memory_order_relaxed);
        last_dkl_.store(delta_kl,    std::memory_order_relaxed);

        // Check budget exhaustion
        if (n >= static_cast<uint64_t>(cfg_.min_samples) &&
            accum_kl_v_ >= cfg_.budget_nats) {
            fire_exhausted = true;
            exhaustion_events_.fetch_add(1, std::memory_order_relaxed);
            accum_kl_v_ = 0.0;
            accum_kl_.store(0.0, std::memory_order_relaxed);
        }
    }

    if (update_cb) try { update_cb(delta_kl, nats_val); } catch (...) {}
    if (fire_exhausted && exhausted_cb) try { exhausted_cb(nats_val); } catch (...) {}
}

std::string NarrativeEntropyClock::to_stats_json() const {
    double kl  = accum_kl_.load(std::memory_order_relaxed);
    double dkl = last_dkl_.load(std::memory_order_relaxed);
    uint64_t ev = exhaustion_events_.load(std::memory_order_relaxed);
    uint64_t t  = total_.load(std::memory_order_relaxed);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"accumulated_kl\":%.6f,\"last_delta_kl\":%.6f,"
        "\"exhaustion_events\":%llu,\"total_records\":%llu}",
        kl, dkl,
        static_cast<unsigned long long>(ev),
        static_cast<unsigned long long>(t));
    return buf;
}

void NarrativeEntropyClock::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    accum_kl_v_ = 0.0;
    init_ref_dist();
    accum_kl_.store(0.0, std::memory_order_relaxed);
    last_dkl_.store(0.0, std::memory_order_relaxed);
    exhaustion_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeEntropyClock::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    accum_kl_v_ = 0.0;
    init_ref_dist();
    accum_kl_.store(0.0, std::memory_order_relaxed);
    last_dkl_.store(0.0, std::memory_order_relaxed);
    exhaustion_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
