#include "SentimentDivergenceDetector.h"

#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace llmquant {

SentimentDivergenceDetector::SentimentDivergenceDetector(Config cfg)
    : cfg_(std::move(cfg))
{
    sources_.reserve(kMaxSources);
}

void SentimentDivergenceDetector::register_source(const std::string& source_name) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (index_.count(source_name)) return;
    if (sources_.size() >= kMaxSources)
        throw std::runtime_error("SentimentDivergenceDetector: max sources exceeded");
    index_[source_name] = sources_.size();
    sources_.push_back({source_name, 0.0, 0});
}

void SentimentDivergenceDetector::record(const std::string& source_name,
                                          double sentiment)
{
    double new_div      = 0.0;
    bool   fire_div     = false;
    bool   fire_rec     = false;
    std::string src_a, src_b;
    std::function<void(double, const std::string&, const std::string&)> cb_div;
    std::function<void(double)> cb_rec;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        auto it = index_.find(source_name);
        if (it == index_.end()) return;

        SourceState& s = sources_[it->second];
        double alpha = cfg_.ema_alpha;
        s.ema = s.ema + alpha * (sentiment - s.ema);
        ++s.obs_count;

        // Find worst-case pair.
        double worst = 0.0;
        std::size_t ia = 0, ib = 0;
        for (std::size_t i = 0; i < sources_.size(); ++i) {
            if (sources_[i].obs_count < cfg_.min_observations) continue;
            for (std::size_t j = i + 1; j < sources_.size(); ++j) {
                if (sources_[j].obs_count < cfg_.min_observations) continue;
                double d = std::fabs(sources_[i].ema - sources_[j].ema);
                if (d > worst) { worst = d; ia = i; ib = j; }
            }
        }

        new_div = worst;
        divergence_.store(new_div, std::memory_order_relaxed);

        bool was_diverged = diverged_.load(std::memory_order_relaxed);
        double thresh     = cfg_.divergence_threshold;

        if (!was_diverged && new_div > thresh) {
            diverged_.store(true, std::memory_order_relaxed);
            divergence_events_.fetch_add(1, std::memory_order_relaxed);
            fire_div = true;
            src_a = sources_[ia].name;
            src_b = sources_[ib].name;
            cb_div = cfg_.on_divergence;
        } else if (was_diverged && new_div < thresh * cfg_.recovery_hysteresis) {
            diverged_.store(false, std::memory_order_relaxed);
            recovery_events_.fetch_add(1, std::memory_order_relaxed);
            fire_rec = true;
            cb_rec = cfg_.on_recovery;
        }
    }

    if (fire_div && cb_div) cb_div(new_div, src_a, src_b);
    if (fire_rec && cb_rec) cb_rec(new_div);
}

double SentimentDivergenceDetector::source_ema(const std::string& source_name) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = index_.find(source_name);
    return (it == index_.end()) ? -2.0 : sources_[it->second].ema;
}

std::size_t SentimentDivergenceDetector::source_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return sources_.size();
}

void SentimentDivergenceDetector::update_divergence_locked() {
    // (intentionally left minimal — divergence is computed inline in record())
}

void SentimentDivergenceDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& s : sources_) { s.ema = 0.0; s.obs_count = 0; }
    divergence_.store(0.0, std::memory_order_relaxed);
    diverged_.store(false, std::memory_order_relaxed);
}

void SentimentDivergenceDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

std::string SentimentDivergenceDetector::to_stats_json() const {
    double div;
    bool is_div;
    uint64_t de, re;
    std::vector<SourceState> snap;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        div    = divergence_.load(std::memory_order_relaxed);
        is_div = diverged_.load(std::memory_order_relaxed);
        de     = divergence_events_.load(std::memory_order_relaxed);
        re     = recovery_events_.load(std::memory_order_relaxed);
        snap   = sources_;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"divergence\":"        << div    << ","
       << "\"is_diverged\":"       << (is_div ? "true" : "false") << ","
       << "\"divergence_events\":" << de     << ","
       << "\"recovery_events\":"   << re     << ","
       << "\"sources\":[";
    for (std::size_t i = 0; i < snap.size(); ++i) {
        if (i) ss << ",";
        ss << "{\"name\":\"" << snap[i].name << "\","
           << "\"ema\":"     << snap[i].ema  << ","
           << "\"obs\":"     << snap[i].obs_count << "}";
    }
    ss << "]}";
    return ss.str();
}

} // namespace llmquant
