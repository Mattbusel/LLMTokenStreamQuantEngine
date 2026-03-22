#include "NarrativeTopicClassifier.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

namespace llmquant {

NarrativeTopicClassifier::NarrativeTopicClassifier(Config cfg)
    : cfg_(std::move(cfg))
{}

bool NarrativeTopicClassifier::register_topic(const TopicDef& def) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (static_cast<int>(topics_.size()) >= kMaxTopics) return false;
    TopicState ts;
    ts.def      = def;
    ts.freq_ema = 0.0;
    topics_.push_back(std::move(ts));
    return true;
}

int NarrativeTopicClassifier::classify(const std::string& /*token*/, double weight) {
    std::function<void(const std::string&, const std::string&, double)> cb;
    bool   fire      = false;
    int    best_idx  = -1;
    int    new_dom   = -1;
    int    old_dom   = -1;
    double new_mult  = 1.0;
    std::string old_name, new_name;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_topic_change;

        if (topics_.empty()) return -1;

        // Find nearest topic by absolute distance in weight space.
        double best_dist = std::numeric_limits<double>::max();
        best_idx = 0;
        for (int i = 0; i < static_cast<int>(topics_.size()); ++i) {
            double d = std::abs(weight - topics_[static_cast<std::size_t>(i)].def.weight_centroid);
            if (d < best_dist) {
                best_dist = d;
                best_idx  = i;
            }
        }

        // Update EMA frequencies: decay all, boost winner.
        double alpha = cfg_.freq_alpha;
        for (int i = 0; i < static_cast<int>(topics_.size()); ++i) {
            double target = (i == best_idx) ? 1.0 : 0.0;
            topics_[static_cast<std::size_t>(i)].freq_ema +=
                alpha * (target - topics_[static_cast<std::size_t>(i)].freq_ema);
        }

        classified_count_.fetch_add(1, std::memory_order_relaxed);
        ++classified_;

        if (classified_ >= cfg_.min_warmup) {
            // Find new dominant.
            int max_idx = 0;
            double max_freq = topics_[0].freq_ema;
            for (int i = 1; i < static_cast<int>(topics_.size()); ++i) {
                if (topics_[static_cast<std::size_t>(i)].freq_ema > max_freq) {
                    max_freq = topics_[static_cast<std::size_t>(i)].freq_ema;
                    max_idx  = i;
                }
            }

            if (max_idx != dominant_idx_) {
                old_dom       = dominant_idx_;
                dominant_idx_ = max_idx;
                new_dom       = max_idx;
                new_mult      = topics_[static_cast<std::size_t>(max_idx)].def.signal_multiplier;
                dominant_mult_.store(new_mult, std::memory_order_relaxed);
                old_name = (old_dom >= 0 && old_dom < static_cast<int>(topics_.size()))
                    ? topics_[static_cast<std::size_t>(old_dom)].def.name : "";
                new_name = topics_[static_cast<std::size_t>(max_idx)].def.name;
                fire = true;
            }
        }
    }

    if (fire && cb) cb(old_name, new_name, new_mult);
    return best_idx;
}

std::string NarrativeTopicClassifier::dominant_topic() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (dominant_idx_ < 0 || dominant_idx_ >= static_cast<int>(topics_.size()))
        return {};
    return topics_[static_cast<std::size_t>(dominant_idx_)].def.name;
}

double NarrativeTopicClassifier::signal_multiplier(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& ts : topics_)
        if (ts.def.name == name) return ts.def.signal_multiplier;
    return 1.0;
}

double NarrativeTopicClassifier::topic_frequency(int idx) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (idx < 0 || idx >= static_cast<int>(topics_.size())) return 0.0;
    return topics_[static_cast<std::size_t>(idx)].freq_ema;
}

std::string NarrativeTopicClassifier::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"dominant_topic\":\"" << (dominant_idx_ >= 0
            ? topics_[static_cast<std::size_t>(dominant_idx_)].def.name : "") << "\","
       << "\"dominant_multiplier\":"
       << (dominant_idx_ >= 0
            ? topics_[static_cast<std::size_t>(dominant_idx_)].def.signal_multiplier : 1.0) << ","
       << "\"classified_count\":"  << classified_count_.load(std::memory_order_relaxed) << ","
       << "\"topics\":[";
    for (std::size_t i = 0; i < topics_.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"name\":\"" << topics_[i].def.name << "\","
           << "\"freq_ema\":" << topics_[i].freq_ema << ","
           << "\"multiplier\":" << topics_[i].def.signal_multiplier << "}";
    }
    ss << "]}";
    return ss.str();
}

void NarrativeTopicClassifier::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& ts : topics_) ts.freq_ema = 0.0;
    dominant_idx_ = -1;
    classified_   = 0;
    dominant_mult_.store(1.0, std::memory_order_relaxed);
    classified_count_.store(0, std::memory_order_relaxed);
}

void NarrativeTopicClassifier::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    for (auto& ts : topics_) ts.freq_ema = 0.0;
    dominant_idx_ = -1;
    classified_   = 0;
    dominant_mult_.store(1.0, std::memory_order_relaxed);
    classified_count_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
