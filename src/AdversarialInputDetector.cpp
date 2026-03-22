#include "AdversarialInputDetector.h"

#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

AdversarialInputDetector::AdversarialInputDetector(Config cfg)
    : cfg_(std::move(cfg))
{}

bool AdversarialInputDetector::inspect(const std::string& token, double weight) {
    bool anomaly   = false;
    AnomalyKind kind = AnomalyKind::WeightAnomaly;
    double score   = 0.0;
    std::function<void(AnomalyKind, const std::string&, double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_anomaly;

        total_tokens_.fetch_add(1, std::memory_order_relaxed);
        double abs_w = std::fabs(weight);

        // --- Welford update ---
        ++w_n_;
        double delta  = abs_w - w_mean_;
        w_mean_      += delta / w_n_;
        double delta2 = abs_w - w_mean_;
        w_M2_        += delta * delta2;
        double var    = (w_n_ > 1) ? w_M2_ / (w_n_ - 1) : 0.0;
        double std_   = std::sqrt(var);
        weight_mean_snap_ = w_mean_;
        weight_std_snap_  = std_;

        // --- Weight anomaly detection ---
        if (w_n_ >= cfg_.min_warmup_tokens && std_ > 1e-9) {
            double z = std::fabs(abs_w - w_mean_) / std_;
            if (z > cfg_.anomaly_threshold) {
                anomaly = true;
                kind    = AnomalyKind::WeightAnomaly;
                score   = z;
            }
        }

        // --- Repetition attack detection ---
        repeat_buf_.push_back(token);
        repeat_counts_[token]++;
        if (static_cast<int>(repeat_buf_.size()) > cfg_.repeat_window) {
            const std::string& old = repeat_buf_.front();
            if (--repeat_counts_[old] == 0) repeat_counts_.erase(old);
            repeat_buf_.pop_front();
        }
        if (!anomaly && static_cast<int>(repeat_buf_.size()) == cfg_.repeat_window) {
            int max_cnt = 0;
            for (auto& [tok, cnt] : repeat_counts_)
                if (cnt > max_cnt) max_cnt = cnt;
            double frac = static_cast<double>(max_cnt) / cfg_.repeat_window;
            if (frac > cfg_.max_repeat_fraction) {
                anomaly = true;
                kind    = AnomalyKind::RepetitionAttack;
                score   = frac;
            }
        }

        // --- Vocabulary inflation detection ---
        bool is_novel = (seen_vocab_.find(token) == seen_vocab_.end());
        seen_vocab_[token] = true;
        novel_buf_.push_back(is_novel);
        if (is_novel) ++novel_in_window_;
        if (static_cast<int>(novel_buf_.size()) > cfg_.vocab_window) {
            if (novel_buf_.front()) --novel_in_window_;
            novel_buf_.pop_front();
        }
        if (!anomaly && static_cast<int>(novel_buf_.size()) == cfg_.vocab_window) {
            double novel_frac = static_cast<double>(novel_in_window_) / cfg_.vocab_window;
            if (novel_frac > cfg_.max_novel_fraction) {
                anomaly = true;
                kind    = AnomalyKind::VocabularyInflation;
                score   = novel_frac;
            }
        }

        if (anomaly) {
            armed_.store(true, std::memory_order_relaxed);
            anomaly_count_.fetch_add(1, std::memory_order_relaxed);
            clean_streak_ = 0;
        } else if (armed_.load(std::memory_order_relaxed)) {
            ++clean_streak_;
            if (cfg_.reset_after_n_clean > 0 &&
                clean_streak_ >= cfg_.reset_after_n_clean) {
                armed_.store(false, std::memory_order_relaxed);
                clean_streak_ = 0;
            }
        }
    }

    if (anomaly && cb) cb(kind, token, score);
    return anomaly;
}

double AdversarialInputDetector::weight_mean() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return weight_mean_snap_;
}

double AdversarialInputDetector::weight_std() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return weight_std_snap_;
}

void AdversarialInputDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    w_mean_ = w_M2_ = 0.0;
    w_n_    = 0;
    repeat_buf_.clear();
    repeat_counts_.clear();
    novel_buf_.clear();
    novel_in_window_ = 0;
    // Keep seen_vocab_ so re-injecting known tokens doesn't appear novel.
    armed_.store(false, std::memory_order_relaxed);
    anomaly_count_.store(0, std::memory_order_relaxed);
    clean_streak_ = 0;
    weight_mean_snap_ = weight_std_snap_ = 0.0;
}

void AdversarialInputDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

std::string AdversarialInputDetector::to_stats_json() const {
    bool arm;
    uint64_t ac, tt;
    double wm, ws;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        arm = armed_.load(std::memory_order_relaxed);
        ac  = anomaly_count_.load(std::memory_order_relaxed);
        tt  = total_tokens_.load(std::memory_order_relaxed);
        wm  = weight_mean_snap_;
        ws  = weight_std_snap_;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"is_armed\":"      << (arm ? "true" : "false") << ","
       << "\"anomaly_count\":" << ac  << ","
       << "\"total_tokens\":"  << tt  << ","
       << "\"weight_mean\":"   << wm  << ","
       << "\"weight_std\":"    << ws
       << "}";
    return ss.str();
}

} // namespace llmquant
