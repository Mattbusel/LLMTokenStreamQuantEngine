#include "KellyPositionSizer.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

KellyPositionSizer::KellyPositionSizer()
    : KellyPositionSizer(Config{}) {}

KellyPositionSizer::KellyPositionSizer(Config config)
    : config_(config)
    , win_rate_ema_(config.prior_win_rate)
    , avg_win_ema_(config.prior_avg_win)
    , avg_loss_ema_(config.prior_avg_loss)
{
    if (config.max_fraction <= 0.0)
        throw std::invalid_argument("KellyPositionSizer: max_fraction must be > 0");
}

// ---------------------------------------------------------------------------
// Outcome recording
// ---------------------------------------------------------------------------

void KellyPositionSizer::record_outcome(bool win, double return_pct) {
    if (return_pct < 0.0) return_pct = 0.0;
    if (config_.min_outcome_return > 0.0 && return_pct < config_.min_outcome_return) return;

    std::lock_guard<std::mutex> lk(mutex_);

    const double alpha = config_.ema_alpha;

    if (win) {
        win_rate_ema_ = alpha * 1.0 + (1.0 - alpha) * win_rate_ema_;
        avg_win_ema_  = alpha * return_pct + (1.0 - alpha) * avg_win_ema_;
        total_wins_.fetch_add(1, std::memory_order_relaxed);
    } else {
        win_rate_ema_ = alpha * 0.0 + (1.0 - alpha) * win_rate_ema_;
        avg_loss_ema_ = alpha * return_pct + (1.0 - alpha) * avg_loss_ema_;
        total_losses_.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Kelly computation (mutex must be held by caller)
// ---------------------------------------------------------------------------

double KellyPositionSizer::compute_kelly_locked() const {
    // Kelly formula: f* = p/l - (1-p)/w
    // where p = win_rate, w = avg_win, l = avg_loss
    double p = win_rate_ema_;
    double w = avg_win_ema_;
    double l = avg_loss_ema_;

    if (w < 1e-12 || l < 1e-12) return 0.0;

    double edge = p * w - (1.0 - p) * l;
    if (edge <= 0.0) return 0.0;

    double f = edge / (w * l);
    return std::clamp(f, 0.0, config_.max_fraction);
}

// ---------------------------------------------------------------------------
// Signal sizing
// ---------------------------------------------------------------------------

TradeSignal KellyPositionSizer::size_signal(const TradeSignal& signal) const {
    total_sized_.fetch_add(1, std::memory_order_relaxed);

    uint64_t outcomes = total_wins_.load(std::memory_order_relaxed)
                      + total_losses_.load(std::memory_order_relaxed);

    if (static_cast<int>(outcomes) < config_.min_history) {
        // Not enough history — return signal unchanged.
        sized_full_.fetch_add(1, std::memory_order_relaxed);
        return signal;
    }

    double f;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        f = compute_kelly_locked();
    }

    if (f <= 0.0) {
        sized_zero_.fetch_add(1, std::memory_order_relaxed);
        TradeSignal out = signal;
        out.delta_bias_shift = 0.0;
        return out;
    }

    TradeSignal out = signal;
    out.delta_bias_shift = signal.delta_bias_shift * f;
    return out;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double KellyPositionSizer::get_kelly_fraction() const {
    uint64_t outcomes = total_wins_.load(std::memory_order_relaxed)
                      + total_losses_.load(std::memory_order_relaxed);
    if (static_cast<int>(outcomes) < config_.min_history) return 0.0;
    std::lock_guard<std::mutex> lk(mutex_);
    return compute_kelly_locked();
}

double KellyPositionSizer::get_edge() const {
    std::lock_guard<std::mutex> lk(mutex_);
    double p = win_rate_ema_;
    double w = avg_win_ema_;
    double l = avg_loss_ema_;
    if (w < 1e-12 || l < 1e-12) return 0.0;
    return p * w - (1.0 - p) * l;
}

double KellyPositionSizer::get_win_rate() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return win_rate_ema_;
}

KellyPositionSizer::Stats KellyPositionSizer::get_stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    Stats s;
    s.total_wins      = total_wins_.load(std::memory_order_relaxed);
    s.total_losses    = total_losses_.load(std::memory_order_relaxed);
    s.win_rate_ema    = win_rate_ema_;
    s.avg_win_ema     = avg_win_ema_;
    s.avg_loss_ema    = avg_loss_ema_;
    s.kelly_fraction  = compute_kelly_locked();
    s.edge            = win_rate_ema_ * avg_win_ema_ - (1.0 - win_rate_ema_) * avg_loss_ema_;
    s.total_sized     = total_sized_.load(std::memory_order_relaxed);
    s.sized_full      = sized_full_.load(std::memory_order_relaxed);
    s.sized_zero      = sized_zero_.load(std::memory_order_relaxed);
    return s;
}

std::string KellyPositionSizer::to_stats_json() const {
    auto s = get_stats();
    std::ostringstream o;
    o << "{\"win_rate\":" << s.win_rate_ema
      << ",\"avg_win\":"  << s.avg_win_ema
      << ",\"avg_loss\":" << s.avg_loss_ema
      << ",\"kelly_fraction\":" << s.kelly_fraction
      << ",\"edge\":"     << s.edge
      << ",\"total_wins\":" << s.total_wins
      << ",\"total_losses\":" << s.total_losses
      << ",\"total_sized\":" << s.total_sized
      << "}";
    return o.str();
}

void KellyPositionSizer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    win_rate_ema_ = config_.prior_win_rate;
    avg_win_ema_  = config_.prior_avg_win;
    avg_loss_ema_ = config_.prior_avg_loss;
    total_wins_.store(0, std::memory_order_relaxed);
    total_losses_.store(0, std::memory_order_relaxed);
    total_sized_.store(0, std::memory_order_relaxed);
    sized_full_.store(0, std::memory_order_relaxed);
    sized_zero_.store(0, std::memory_order_relaxed);
}

void KellyPositionSizer::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

KellyPositionSizer::Config KellyPositionSizer::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

} // namespace llmquant
