#include "TokenSentimentHeatmap.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TokenSentimentHeatmap::TokenSentimentHeatmap(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// sentiment_to_bucket (internal)
// ---------------------------------------------------------------------------

int TokenSentimentHeatmap::sentiment_to_bucket(double sentiment) const noexcept {
    int n = config_.num_buckets;
    if (n <= 0) return 0;
    double range = config_.max_sentiment - config_.min_sentiment;
    if (range <= 0.0) return 0;
    double frac = (sentiment - config_.min_sentiment) / range;
    int bucket = static_cast<int>(frac * n);
    return std::max(0, std::min(n - 1, bucket));
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

void TokenSentimentHeatmap::record(const std::string& token_id, double sentiment) {
    total_records_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(mutex_);

    auto it = rows_.find(token_id);
    if (it == rows_.end()) {
        // New token: allocate and zero-fill.
        it = rows_.emplace(token_id,
                           std::vector<double>(static_cast<std::size_t>(config_.num_buckets),
                                               0.0)).first;
    }

    auto& counts = it->second;

    // Apply forgetting.
    if (config_.forgetting < 1.0) {
        for (double& c : counts)
            c *= config_.forgetting;
    }

    int b = sentiment_to_bucket(sentiment);
    counts[static_cast<std::size_t>(b)] += 1.0;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void TokenSentimentHeatmap::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    rows_.clear();
}

// ---------------------------------------------------------------------------
// bucket_counts
// ---------------------------------------------------------------------------

std::vector<double> TokenSentimentHeatmap::bucket_counts(const std::string& token_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = rows_.find(token_id);
    if (it == rows_.end()) return {};
    return it->second;
}

// ---------------------------------------------------------------------------
// peak_bucket
// ---------------------------------------------------------------------------

int TokenSentimentHeatmap::peak_bucket(const std::string& token_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = rows_.find(token_id);
    if (it == rows_.end()) return -1;
    const auto& counts = it->second;
    auto max_it = std::max_element(counts.begin(), counts.end());
    return static_cast<int>(max_it - counts.begin());
}

// ---------------------------------------------------------------------------
// bucket_centre
// ---------------------------------------------------------------------------

double TokenSentimentHeatmap::bucket_centre(int bucket_idx) const noexcept {
    int n = config_.num_buckets;
    if (n <= 0) return 0.0;
    double range = config_.max_sentiment - config_.min_sentiment;
    double width  = range / n;
    return config_.min_sentiment + (bucket_idx + 0.5) * width;
}

// ---------------------------------------------------------------------------
// dominant_sentiment
// ---------------------------------------------------------------------------

double TokenSentimentHeatmap::dominant_sentiment(const std::string& token_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = rows_.find(token_id);
    if (it == rows_.end()) return 0.0;
    const auto& counts = it->second;
    double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    if (total <= 0.0) return 0.0;

    int n = static_cast<int>(counts.size());
    double range = config_.max_sentiment - config_.min_sentiment;
    double width  = (n > 0) ? range / n : 0.0;
    double weighted_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double centre = config_.min_sentiment + (i + 0.5) * width;
        weighted_sum += counts[static_cast<std::size_t>(i)] * centre;
    }
    double mean = weighted_sum / total;
    if (mean > 0.0) return 1.0;
    if (mean < 0.0) return -1.0;
    return 0.0;
}

// ---------------------------------------------------------------------------
// token_count
// ---------------------------------------------------------------------------

int TokenSentimentHeatmap::token_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(rows_.size());
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void TokenSentimentHeatmap::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    rows_.clear();
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string TokenSentimentHeatmap::to_stats_json(int top_n) const {
    // Build sorted list of (total_count, token_id) pairs, pick top_n.
    std::vector<std::pair<double, std::string>> ranked;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        ranked.reserve(rows_.size());
        for (const auto& [tok, counts] : rows_) {
            double total = std::accumulate(counts.begin(), counts.end(), 0.0);
            ranked.emplace_back(total, tok);
        }
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (top_n > 0 && static_cast<int>(ranked.size()) > top_n)
        ranked.resize(static_cast<std::size_t>(top_n));

    std::ostringstream s;
    s << "{\"token_count\":" << token_count()
      << ",\"total_records\":" << total_records()
      << ",\"top_tokens\":[";
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        if (i) s << ",";
        double dom = dominant_sentiment(ranked[i].second);
        s << "{\"token\":\"" << ranked[i].second << "\""
          << ",\"total_count\":" << ranked[i].first
          << ",\"dominant\":" << dom
          << ",\"peak_bucket\":" << peak_bucket(ranked[i].second)
          << "}";
    }
    s << "]}";
    return s.str();
}

} // namespace llmquant
