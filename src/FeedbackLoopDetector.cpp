#include "FeedbackLoopDetector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

FeedbackLoopDetector::FeedbackLoopDetector(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// cross_correlation (static)
// ---------------------------------------------------------------------------
/* static */
double FeedbackLoopDetector::cross_correlation(
        const std::deque<double>& x,
        const std::deque<double>& y,
        int lag) noexcept {
    int n = static_cast<int>(x.size());
    if (n - lag < 2 || lag < 0) return 0.0;

    int count = n - lag;

    // Means over the overlapping region.
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < count; ++i) {
        mx += x[static_cast<std::size_t>(i)];
        my += y[static_cast<std::size_t>(i + lag)];
    }
    mx /= count;
    my /= count;

    // Pearson numerator + denominators.
    double cov = 0.0, vx = 0.0, vy = 0.0;
    for (int i = 0; i < count; ++i) {
        double dx = x[static_cast<std::size_t>(i)] - mx;
        double dy = y[static_cast<std::size_t>(i + lag)] - my;
        cov += dx * dy;
        vx  += dx * dx;
        vy  += dy * dy;
    }

    double denom = std::sqrt(vx * vy);
    if (denom < 1e-14) return 0.0;
    return cov / denom;
}

// ---------------------------------------------------------------------------
// recompute (must be called under lock)
// ---------------------------------------------------------------------------
bool FeedbackLoopDetector::recompute() {
    int n = static_cast<int>(activity_win_.size());
    int m = static_cast<int>(sentiment_win_.size());
    int min_n = std::min(n, m);

    if (min_n < config_.min_samples) {
        feedback_score_ = 0.0;
        peak_lag_       = 0;
        return false;
    }

    // Use the most recent min_n samples from both windows.
    std::deque<double> act(activity_win_.end() - min_n, activity_win_.end());
    std::deque<double> sent(sentiment_win_.end() - min_n, sentiment_win_.end());

    double best_r   = 0.0;
    int    best_lag = 0;

    for (int lag = 1; lag <= std::min(config_.max_lag, min_n - 1); ++lag) {
        double r = cross_correlation(act, sent, lag);
        if (std::abs(r) > std::abs(best_r)) {
            best_r   = r;
            best_lag = lag;
        }
    }

    feedback_score_ = best_r;
    peak_lag_       = best_lag;

    bool over_thresh = (best_r >= config_.threshold);
    bool fire = over_thresh && !in_feedback_episode_;
    if (over_thresh) {
        in_feedback_episode_ = true;
    } else {
        in_feedback_episode_ = false;
    }
    return fire;
}

// ---------------------------------------------------------------------------
// record_own_activity
// ---------------------------------------------------------------------------
void FeedbackLoopDetector::record_own_activity(double activity) {
    total_activity_.fetch_add(1, std::memory_order_relaxed);

    bool fire_cb = false;
    double score = 0.0;
    int    lag   = 0;
    FeedbackCallback cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        activity_win_.push_back(activity);
        if (static_cast<int>(activity_win_.size()) > config_.window_size)
            activity_win_.pop_front();

        if (static_cast<int>(sentiment_win_.size()) >= config_.min_samples) {
            fire_cb = recompute();
            if (fire_cb) {
                feedback_events_.fetch_add(1, std::memory_order_relaxed);
                score = feedback_score_;
                lag   = peak_lag_;
                cb    = config_.on_feedback;
            }
        }
    }

    if (fire_cb && cb) cb(score, lag);
}

// ---------------------------------------------------------------------------
// record_sentiment
// ---------------------------------------------------------------------------
void FeedbackLoopDetector::record_sentiment(double sentiment) {
    total_sentiment_.fetch_add(1, std::memory_order_relaxed);

    bool fire_cb = false;
    double score = 0.0;
    int    lag   = 0;
    FeedbackCallback cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        sentiment_win_.push_back(sentiment);
        if (static_cast<int>(sentiment_win_.size()) > config_.window_size)
            sentiment_win_.pop_front();

        if (static_cast<int>(activity_win_.size()) >= config_.min_samples) {
            fire_cb = recompute();
            if (fire_cb) {
                feedback_events_.fetch_add(1, std::memory_order_relaxed);
                score = feedback_score_;
                lag   = peak_lag_;
                cb    = config_.on_feedback;
            }
        }
    }

    if (fire_cb && cb) cb(score, lag);
}

void FeedbackLoopDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    activity_win_.clear();
    sentiment_win_.clear();
    feedback_score_     = 0.0;
    peak_lag_           = 0;
    in_feedback_episode_ = false;
}

double FeedbackLoopDetector::feedback_score() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return feedback_score_;
}

int FeedbackLoopDetector::peak_lag() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return peak_lag_;
}

bool FeedbackLoopDetector::feedback_detected() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return in_feedback_episode_;
}

void FeedbackLoopDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    activity_win_.clear();
    sentiment_win_.clear();
    feedback_score_     = 0.0;
    peak_lag_           = 0;
    in_feedback_episode_ = false;
}

std::string FeedbackLoopDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream os;
    os << std::fixed;
    os << "{"
       << "\"feedback_score\":"         << feedback_score_ << ","
       << "\"peak_lag\":"               << peak_lag_ << ","
       << "\"feedback_detected\":"      << (in_feedback_episode_ ? "true" : "false") << ","
       << "\"total_activity\":"         << total_activity_.load(std::memory_order_relaxed) << ","
       << "\"total_sentiment\":"        << total_sentiment_.load(std::memory_order_relaxed) << ","
       << "\"feedback_events\":"        << feedback_events_.load(std::memory_order_relaxed) << ","
       << "\"activity_window_size\":"   << activity_win_.size() << ","
       << "\"sentiment_window_size\":"  << sentiment_win_.size()
       << "}";
    return os.str();
}

} // namespace llmquant
