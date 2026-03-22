#include "BayesianSentimentPrior.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

// Student-t quantile approximation for t_{0.975, df} (95% two-sided CI)
// Using rational approximation valid for df >= 1.
static double t_quantile_975(double df) {
    if (df <= 0.0) return 1.960;
    if (df >= 1000.0) return 1.960;
    // Cornish-Fisher-like approximation
    // t_{0.975, df} ≈ z + g₁(z)/df + g₂(z)/df² + ...
    // where z = 1.959964
    // Simple table lookup via polynomial fit:
    if (df < 2.0)  return 12.706;
    if (df < 3.0)  return 4.303;
    if (df < 4.0)  return 3.182;
    if (df < 5.0)  return 2.776;
    if (df < 6.0)  return 2.571;
    if (df < 7.0)  return 2.447;
    if (df < 8.0)  return 2.365;
    if (df < 9.0)  return 2.306;
    if (df < 10.0) return 2.262;
    if (df < 15.0) return 2.145;
    if (df < 20.0) return 2.093;
    if (df < 30.0) return 2.042;
    if (df < 60.0) return 2.000;
    if (df < 120.0) return 1.980;
    return 1.960;
}

BayesianSentimentPrior::BayesianSentimentPrior(Config cfg)
    : cfg_(std::move(cfg))
{
    // Initialise posterior to prior
    post_mean_.store(cfg_.prior_mean, std::memory_order_relaxed);
    post_std_.store(std::sqrt(cfg_.prior_beta / (cfg_.prior_alpha * cfg_.prior_kappa)),
                   std::memory_order_relaxed);
    double t975 = t_quantile_975(2.0 * cfg_.prior_alpha);
    double ps   = std::sqrt(cfg_.prior_beta / (cfg_.prior_alpha * cfg_.prior_kappa));
    ci_lo_.store(cfg_.prior_mean - t975 * ps, std::memory_order_relaxed);
    ci_hi_.store(cfg_.prior_mean + t975 * ps, std::memory_order_relaxed);
}

void BayesianSentimentPrior::recompute_locked() {
    if (n_ == 0) {
        post_kappa_ = cfg_.prior_kappa;
        post_alpha_ = cfg_.prior_alpha;
        post_beta_  = cfg_.prior_beta;
        post_mean_.store(cfg_.prior_mean, std::memory_order_relaxed);
        double ps = std::sqrt(cfg_.prior_beta / (cfg_.prior_alpha * cfg_.prior_kappa));
        post_std_.store(ps, std::memory_order_relaxed);
        double t975 = t_quantile_975(2.0 * cfg_.prior_alpha);
        ci_lo_.store(cfg_.prior_mean - t975 * ps, std::memory_order_relaxed);
        ci_hi_.store(cfg_.prior_mean + t975 * ps, std::memory_order_relaxed);
        return;
    }

    double n  = static_cast<double>(n_);
    double x_bar = sum_x_ / n;
    // Sample variance (biased)
    double var_n = sum_x2_ / n - x_bar * x_bar;
    if (var_n < 0.0) var_n = 0.0;

    // Normal-Gamma posterior update
    double kn = cfg_.prior_kappa + n;
    double an = cfg_.prior_alpha + n / 2.0;
    double mn = (cfg_.prior_kappa * cfg_.prior_mean + sum_x_) / kn;
    double bn = cfg_.prior_beta
                + 0.5 * n * var_n
                + (cfg_.prior_kappa * n * (x_bar - cfg_.prior_mean) * (x_bar - cfg_.prior_mean))
                  / (2.0 * kn);

    post_kappa_ = kn;
    post_alpha_ = an;
    post_beta_  = bn;

    // Marginal posterior for μ: scaled t with 2α_n degrees of freedom
    // std dev = sqrt(β_n / (α_n × κ_n))
    double ps = (an > 0.0 && kn > 0.0) ? std::sqrt(bn / (an * kn)) : 0.0;
    double t975 = t_quantile_975(2.0 * an);

    post_mean_.store(mn, std::memory_order_relaxed);
    post_std_.store(ps, std::memory_order_relaxed);
    ci_lo_.store(mn - t975 * ps, std::memory_order_relaxed);
    ci_hi_.store(mn + t975 * ps, std::memory_order_relaxed);
}

void BayesianSentimentPrior::update(double x) {
    bool   fire_shift    = false;
    bool   fire_restored = false;
    double pm_val = 0.0, ps_val = 0.0;
    std::function<void(double, double)> shift_cb;
    std::function<void()>               restored_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        shift_cb    = cfg_.on_belief_shift;
        restored_cb = cfg_.on_belief_restored;

        total_.fetch_add(1, std::memory_order_relaxed);
        ++n_;
        sum_x_  += x;
        sum_x2_ += x * x;

        recompute_locked();

        pm_val = post_mean_.load(std::memory_order_relaxed);
        ps_val = post_std_.load(std::memory_order_relaxed);

        if (n_ >= static_cast<uint64_t>(cfg_.min_samples) && ps_val > 1e-12) {
            double dev = std::abs(pm_val - cfg_.prior_mean) / ps_val;
            bool now_shifted = (dev > cfg_.shift_threshold);

            if (now_shifted && !prev_shifted_) {
                fire_shift = true;
                shifts_.fetch_add(1, std::memory_order_relaxed);
            } else if (!now_shifted && prev_shifted_) {
                fire_restored = true;
            }
            shifted_.store(now_shifted, std::memory_order_relaxed);
            prev_shifted_ = now_shifted;
        }
    }

    if (fire_shift    && shift_cb)    try { shift_cb(pm_val, ps_val); } catch (...) {}
    if (fire_restored && restored_cb) try { restored_cb(); }           catch (...) {}
}

std::string BayesianSentimentPrior::to_stats_json() const {
    double pm  = post_mean_.load(std::memory_order_relaxed);
    double ps  = post_std_.load(std::memory_order_relaxed);
    double lo  = ci_lo_.load(std::memory_order_relaxed);
    double hi  = ci_hi_.load(std::memory_order_relaxed);
    bool   sh  = shifted_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);
    uint64_t s = shifts_.load(std::memory_order_relaxed);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"posterior_mean\":%.6f,\"posterior_std\":%.6f,"
        "\"ci_lower\":%.6f,\"ci_upper\":%.6f,"
        "\"is_shifted\":%s,\"belief_shifts\":%llu,\"total_updates\":%llu}",
        pm, ps, lo, hi,
        sh ? "true" : "false",
        static_cast<unsigned long long>(s),
        static_cast<unsigned long long>(t));
    return buf;
}

void BayesianSentimentPrior::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    n_ = 0; sum_x_ = sum_x2_ = 0.0;
    prev_shifted_ = false;
    recompute_locked();
    shifted_.store(false, std::memory_order_relaxed);
    shifts_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void BayesianSentimentPrior::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    n_ = 0; sum_x_ = sum_x2_ = 0.0;
    prev_shifted_ = false;
    recompute_locked();
    shifted_.store(false, std::memory_order_relaxed);
    shifts_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
