#include "PositionTracker.h"

#include <cinttypes>
#include <cstdio>

namespace llmquant {

PositionTracker::PositionTracker(KellyPositionSizer& kelly_sizer, Config config)
    : kelly_sizer_(kelly_sizer)
    , config_(std::move(config)) {}

uint64_t PositionTracker::open_trade(const TradeSignal& signal, double entry_price) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (static_cast<int>(open_trades_.size()) >= config_.max_open_trades) {
        ++rejected_opens_;
        return 0;
    }

    uint64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
    Trade t;
    t.id            = id;
    t.entry_price   = entry_price;
    t.signal_bias   = signal.delta_bias_shift;
    t.size_fraction = kelly_sizer_.get_kelly_fraction();
    t.entry_time    = std::chrono::steady_clock::now();
    open_trades_[id] = std::move(t);
    return id;
}

bool PositionTracker::close_trade(uint64_t trade_id, double exit_price) {
    TradeCloseCallback cb;
    double return_pct = 0.0;
    bool   win        = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = open_trades_.find(trade_id);
        if (it == open_trades_.end()) return false;

        const Trade& t = it->second;

        // Compute raw return.  If signal was bearish (negative bias), flip sign.
        double raw = (t.entry_price > 1e-15)
                     ? (exit_price - t.entry_price) / t.entry_price
                     : 0.0;
        return_pct = (t.signal_bias < 0.0) ? -raw : raw;
        win        = (return_pct >= 0.0);

        // Update statistics.
        ++total_trades_;
        cumulative_pnl_ += return_pct;
        if (cumulative_pnl_ > peak_pnl_) peak_pnl_ = cumulative_pnl_;
        double drawdown = peak_pnl_ - cumulative_pnl_;
        if (drawdown > max_drawdown_) max_drawdown_ = drawdown;
        if (return_pct > best_trade_)  best_trade_  = return_pct;
        if (return_pct < worst_trade_) worst_trade_ = return_pct;

        if (win) {
            ++total_wins_;
            sum_win_returns_ += return_pct;
        } else {
            ++total_losses_;
            sum_loss_returns_ += (-return_pct);
        }

        open_trades_.erase(it);
        cb = close_cb_;
    }

    // Feed back into Kelly sizer outside the lock (sizer has its own mutex).
    if (config_.auto_kelly_feedback) {
        double abs_return = (return_pct >= 0.0) ? return_pct : -return_pct;
        if (abs_return >= config_.min_return_for_kelly) {
            kelly_sizer_.record_outcome(win, abs_return);
        }
    }

    if (cb) cb(trade_id, return_pct, win);
    return true;
}

PositionTracker::Stats PositionTracker::get_stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    Stats s;
    s.total_trades    = total_trades_;
    s.total_wins      = total_wins_;
    s.total_losses    = total_losses_;
    s.win_rate        = (total_trades_ > 0)
                        ? static_cast<double>(total_wins_) / static_cast<double>(total_trades_)
                        : 0.0;
    s.avg_win_return  = (total_wins_ > 0)
                        ? sum_win_returns_ / static_cast<double>(total_wins_)
                        : 0.0;
    s.avg_loss_return = (total_losses_ > 0)
                        ? sum_loss_returns_ / static_cast<double>(total_losses_)
                        : 0.0;
    s.cumulative_pnl  = cumulative_pnl_;
    s.max_drawdown    = max_drawdown_;
    s.best_trade      = best_trade_;
    s.worst_trade     = worst_trade_;
    s.open_trades     = static_cast<uint64_t>(open_trades_.size());
    s.rejected_opens  = rejected_opens_;
    return s;
}

uint64_t PositionTracker::open_trade_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<uint64_t>(open_trades_.size());
}

void PositionTracker::set_trade_close_callback(TradeCloseCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    close_cb_ = std::move(cb);
}

std::string PositionTracker::to_stats_json() const {
    auto s = get_stats();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"total_trades\":%" PRIu64 ",\"win_rate\":%.6f"
        ",\"avg_win_return\":%.6f,\"avg_loss_return\":%.6f"
        ",\"cumulative_pnl\":%.6f,\"max_drawdown\":%.6f"
        ",\"best_trade\":%.6f,\"worst_trade\":%.6f"
        ",\"open_trades\":%" PRIu64 ",\"rejected_opens\":%" PRIu64 "}",
        s.total_trades, s.win_rate,
        s.avg_win_return, s.avg_loss_return,
        s.cumulative_pnl, s.max_drawdown,
        s.best_trade, s.worst_trade,
        s.open_trades, s.rejected_opens);
    return buf;
}

void PositionTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    open_trades_.clear();
    total_trades_    = 0;
    total_wins_      = 0;
    total_losses_    = 0;
    sum_win_returns_ = 0.0;
    sum_loss_returns_= 0.0;
    cumulative_pnl_  = 0.0;
    peak_pnl_        = 0.0;
    max_drawdown_    = 0.0;
    best_trade_      = 0.0;
    worst_trade_     = 0.0;
    rejected_opens_  = 0;
}

void PositionTracker::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

PositionTracker::Config PositionTracker::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

} // namespace llmquant
