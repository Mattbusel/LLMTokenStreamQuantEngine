#include "BayesianSignalFilter.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

BayesianSignalFilter::BayesianSignalFilter(Config config)
    : config_(std::move(config))
    , bull_alpha_(config_.prior_alpha)
    , bull_beta_(config_.prior_beta)
    , bear_alpha_(config_.prior_alpha)
    , bear_beta_(config_.prior_beta) {}

// ---------------------------------------------------------------------------
// update_posterior (internal, called under lock)
// ---------------------------------------------------------------------------

void BayesianSignalFilter::update_posterior(double& alpha, double& beta, bool win) {
    // Exponential forgetting: decay counts toward the prior mass.
    double f = config_.forgetting;
    if (f < 1.0) {
        alpha *= f;
        beta  *= f;
    }
    if (win)
        alpha += 1.0;
    else
        beta  += 1.0;
}

// ---------------------------------------------------------------------------
// record_signal
// ---------------------------------------------------------------------------

void BayesianSignalFilter::record_signal(bool /*bullish*/) {
    total_signals_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// record_outcome
// ---------------------------------------------------------------------------

void BayesianSignalFilter::record_outcome(bool bullish, bool win) {
    total_outcomes_.fetch_add(1, std::memory_order_relaxed);

    std::function<void(bool, double)> cb;
    bool fire_cb = false;
    double post = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (bullish) {
            update_posterior(bull_alpha_, bull_beta_, win);
            post = bull_alpha_ / (bull_alpha_ + bull_beta_);
        } else {
            update_posterior(bear_alpha_, bear_beta_, win);
            post = bear_alpha_ / (bear_alpha_ + bear_beta_);
        }
        if (post < config_.low_confidence_threshold) {
            cb = config_.on_low_confidence;
            fire_cb = true;
        }
    }
    if (fire_cb) {
        low_confidence_events_.fetch_add(1, std::memory_order_relaxed);
        spdlog::warn("[bayes_filter] low posterior={:.3f} dir={}", post,
                     bullish ? "BULL" : "BEAR");
        if (cb) cb(bullish, post);
    }
}

// ---------------------------------------------------------------------------
// posterior_confidence
// ---------------------------------------------------------------------------

double BayesianSignalFilter::posterior_confidence(bool bullish) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (bullish) return bull_alpha_ / (bull_alpha_ + bull_beta_);
    return bear_alpha_ / (bear_alpha_ + bear_beta_);
}

// ---------------------------------------------------------------------------
// credible_interval_half_width
// ---------------------------------------------------------------------------

double BayesianSignalFilter::credible_interval_half_width(bool bullish) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    double a = bullish ? bull_alpha_ : bear_alpha_;
    double b = bullish ? bull_beta_  : bear_beta_;
    double n = a + b;
    double var = (a * b) / (n * n * (n + 1.0));
    return 1.96 * std::sqrt(var);
}

// ---------------------------------------------------------------------------
// posterior_precision
// ---------------------------------------------------------------------------

double BayesianSignalFilter::posterior_precision(bool bullish) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (bullish) return bull_alpha_ + bull_beta_;
    return bear_alpha_ + bear_beta_;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void BayesianSignalFilter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    bull_alpha_ = config_.prior_alpha;
    bull_beta_  = config_.prior_beta;
    bear_alpha_ = config_.prior_alpha;
    bear_beta_  = config_.prior_beta;
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void BayesianSignalFilter::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_     = config;
    bull_alpha_ = config_.prior_alpha;
    bull_beta_  = config_.prior_beta;
    bear_alpha_ = config_.prior_alpha;
    bear_beta_  = config_.prior_beta;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string BayesianSignalFilter::to_stats_json() const {
    double bull_post = 0.0, bear_post = 0.0, bull_prec = 0.0, bear_prec = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        bull_post = bull_alpha_ / (bull_alpha_ + bull_beta_);
        bear_post = bear_alpha_ / (bear_alpha_ + bear_beta_);
        bull_prec = bull_alpha_ + bull_beta_;
        bear_prec = bear_alpha_ + bear_beta_;
    }
    std::ostringstream s;
    s << "{\"bull_posterior\":"      << bull_post
      << ",\"bear_posterior\":"      << bear_post
      << ",\"bull_precision\":"      << bull_prec
      << ",\"bear_precision\":"      << bear_prec
      << ",\"total_signals\":"       << total_signals()
      << ",\"total_outcomes\":"      << total_outcomes()
      << ",\"low_confidence_events\":" << low_confidence_events()
      << "}";
    return s.str();
}

} // namespace llmquant
