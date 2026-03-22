#include "MarketRegimeProbabilityEstimator.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

namespace {
constexpr double kPi = 3.14159265358979323846;
} // namespace

MarketRegimeProbabilityEstimator::MarketRegimeProbabilityEstimator(Config cfg)
    : cfg_(std::move(cfg))
{
    // Initialise uniform prior.
    pi_[0] = 0.5;  // risk-OFF
    pi_[1] = 0.5;  // risk-ON

    // Risk-OFF (state 0): bearish sentiment, higher vol.
    mu_sent_[0] = -0.3;
    var_sent_[0] = 0.1;
    mu_vol_[0]  =  0.3;
    var_vol_[0] =  0.05;

    // Risk-ON (state 1): bullish sentiment, lower vol.
    mu_sent_[1] =  0.3;
    var_sent_[1] = 0.1;
    mu_vol_[1]  =  0.1;
    var_vol_[1] =  0.02;

    prob_risk_on_.store(0.5, std::memory_order_relaxed);
}

/*static*/ double MarketRegimeProbabilityEstimator::gaussian_pdf(
    double x, double mu, double var, double min_var) noexcept
{
    double v = std::max(var, min_var);
    double diff = x - mu;
    return std::exp(-0.5 * diff * diff / v) / std::sqrt(2.0 * kPi * v);
}

void MarketRegimeProbabilityEstimator::update(double sentiment,
                                               double volatility) noexcept
{
    double transition_prob;
    double emission_alpha;
    double min_variance;
    double threshold;
    int min_obs;
    std::function<void(double, bool)> cb;

    double new_prob_on;
    bool fire_cb  = false;
    bool is_on_cb = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        transition_prob = cfg_.transition_prob;
        emission_alpha  = cfg_.emission_alpha;
        min_variance    = cfg_.min_variance;
        threshold       = cfg_.transition_threshold;
        min_obs         = cfg_.min_observations;
        cb              = cfg_.on_regime_change;

        uint64_t n = obs_count_.load(std::memory_order_relaxed) + 1;
        obs_count_.store(n, std::memory_order_relaxed);

        // --- Predict step: apply transition matrix ---
        // A = [[1-p, p], [p, 1-p]]  (symmetric)
        double p = transition_prob;
        double pred0 = (1.0 - p) * pi_[0] + p * pi_[1];
        double pred1 = p * pi_[0] + (1.0 - p) * pi_[1];

        // --- Update emission parameter estimates (online EMA) ---
        double alpha = emission_alpha;
        for (int s = 0; s < kStates; ++s) {
            // Soft responsibility: use current pi as weight.
            double r = (s == 0) ? pred0 : pred1;
            double sent_diff = sentiment  - mu_sent_[s];
            double vol_diff  = volatility - mu_vol_[s];
            mu_sent_[s]  += alpha * r * sent_diff;
            mu_vol_[s]   += alpha * r * vol_diff;
            var_sent_[s] += alpha * r * (sent_diff * sent_diff - var_sent_[s]);
            var_vol_[s]  += alpha * r * (vol_diff  * vol_diff  - var_vol_[s]);
            var_sent_[s] = std::max(var_sent_[s], min_variance);
            var_vol_[s]  = std::max(var_vol_[s],  min_variance);
        }

        // --- Update step: multiply by emission likelihoods ---
        double b0 = gaussian_pdf(sentiment, mu_sent_[0], var_sent_[0], min_variance)
                  * gaussian_pdf(volatility, mu_vol_[0],  var_vol_[0],  min_variance);
        double b1 = gaussian_pdf(sentiment, mu_sent_[1], var_sent_[1], min_variance)
                  * gaussian_pdf(volatility, mu_vol_[1],  var_vol_[1],  min_variance);

        pi_[0] = pred0 * b0;
        pi_[1] = pred1 * b1;

        // --- Normalise ---
        double total = pi_[0] + pi_[1];
        if (total > 0.0) {
            pi_[0] /= total;
            pi_[1] /= total;
        } else {
            pi_[0] = pi_[1] = 0.5;
        }

        new_prob_on = pi_[1];
        prob_risk_on_.store(new_prob_on, std::memory_order_relaxed);

        // --- Regime change detection ---
        if (static_cast<int>(n) >= min_obs && cb) {
            bool now_on = (new_prob_on > threshold);
            bool now_off = (new_prob_on < (1.0 - threshold));
            bool changed = (now_on && !prev_risk_on_) || (now_off && prev_risk_on_);
            if (changed) {
                prev_risk_on_ = now_on;
                transition_count_.fetch_add(1, std::memory_order_relaxed);
                fire_cb  = true;
                is_on_cb = now_on;
            }
        }
    }

    if (fire_cb) cb(new_prob_on, is_on_cb);
}

double MarketRegimeProbabilityEstimator::prob_risk_on() const noexcept {
    return prob_risk_on_.load(std::memory_order_relaxed);
}

double MarketRegimeProbabilityEstimator::prob_risk_off() const noexcept {
    return 1.0 - prob_risk_on_.load(std::memory_order_relaxed);
}

bool MarketRegimeProbabilityEstimator::is_risk_on() const noexcept {
    return prob_risk_on_.load(std::memory_order_relaxed)
           > cfg_.transition_threshold;
}

void MarketRegimeProbabilityEstimator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    pi_[0] = pi_[1] = 0.5;
    prob_risk_on_.store(0.5, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
    transition_count_.store(0, std::memory_order_relaxed);
    prev_risk_on_ = false;
}

void MarketRegimeProbabilityEstimator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

std::string MarketRegimeProbabilityEstimator::to_stats_json() const {
    double pon, poff;
    uint64_t obs, trans;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        pon   = pi_[1];
        poff  = pi_[0];
        obs   = obs_count_.load(std::memory_order_relaxed);
        trans = transition_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"prob_risk_on\":"  << pon   << ","
       << "\"prob_risk_off\":" << poff  << ","
       << "\"observations\":"  << obs   << ","
       << "\"transitions\":"   << trans << ","
       << "\"is_risk_on\":"    << (pon > cfg_.transition_threshold ? "true" : "false")
       << "}";
    return ss.str();
}

} // namespace llmquant
