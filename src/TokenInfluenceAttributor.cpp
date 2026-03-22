#include "TokenInfluenceAttributor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenInfluenceAttributor::TokenInfluenceAttributor(Config cfg)
    : cfg_(std::move(cfg))
{
    window_.resize(cfg_.window_size);
}

void TokenInfluenceAttributor::record(const std::string& token, double weight) {
    std::function<void(const std::string&, double)> cb;
    double frac = 0.0;
    bool fire   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Resize buffer if config changed.
        if (window_.size() != cfg_.window_size) {
            window_.clear();
            window_.resize(cfg_.window_size);
            head_  = 0;
            count_ = 0;
        }

        window_[head_] = {token, weight, seq_++};
        head_ = (head_ + 1) % cfg_.window_size;
        if (count_ < cfg_.window_size) ++count_;

        total_recorded_.fetch_add(1, std::memory_order_relaxed);

        // Check for dominant token.
        if (cfg_.on_dominant_token && cfg_.dominant_token_fraction < 1.0 && count_ > 1) {
            auto results = compute_attribution_locked();
            if (!results.empty()) {
                double top_frac = results[0].abs_influence;
                double total_abs = 0.0;
                for (auto& r : results) total_abs += r.abs_influence;
                if (total_abs > 0.0) {
                    frac = top_frac / total_abs;
                    if (frac >= cfg_.dominant_token_fraction) {
                        cb   = cfg_.on_dominant_token;
                        fire = true;
                        dominant_events_.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    if (fire) cb(token, frac);
}

std::vector<TokenInfluenceAttributor::AttributionResult>
TokenInfluenceAttributor::compute_attribution_locked() const
{
    if (count_ == 0) return {};

    // Collect active records.
    std::vector<const TokenRecord*> active;
    active.reserve(count_);
    std::size_t start = (head_ >= count_) ? (head_ - count_) : (cfg_.window_size - (count_ - head_));
    for (std::size_t i = 0; i < count_; ++i)
        active.push_back(&window_[(start + i) % cfg_.window_size]);

    // Compute mean weight.
    double sum = 0.0;
    for (auto* r : active) sum += r->weight;
    double mean = sum / static_cast<double>(count_);

    // Build results.
    std::vector<AttributionResult> results;
    results.reserve(count_);
    for (auto* r : active) {
        AttributionResult ar;
        ar.token        = r->token;
        ar.weight       = r->weight;
        ar.influence    = r->weight - mean;
        ar.abs_influence = std::fabs(ar.influence);
        ar.sequence     = r->sequence;
        ar.cumulative_fraction = 0.0;
        results.push_back(std::move(ar));
    }

    // Sort by |influence| descending.
    std::sort(results.begin(), results.end(),
        [](const AttributionResult& a, const AttributionResult& b) {
            return a.abs_influence > b.abs_influence;
        });

    // Fill cumulative fraction.
    double total_abs = 0.0;
    for (auto& r : results) total_abs += r.abs_influence;
    double running = 0.0;
    for (auto& r : results) {
        running += r.abs_influence;
        r.cumulative_fraction = (total_abs > 0.0) ? running / total_abs : 0.0;
    }

    // Limit to top_k.
    if (results.size() > cfg_.top_k) results.resize(cfg_.top_k);

    return results;
}

std::vector<TokenInfluenceAttributor::AttributionResult>
TokenInfluenceAttributor::attribute() const
{
    std::lock_guard<std::mutex> lk(mutex_);
    return compute_attribution_locked();
}

void TokenInfluenceAttributor::attribute(
    std::function<void(const AttributionResult&)> visitor) const
{
    auto results = attribute();
    for (auto& r : results) visitor(r);
}

std::size_t TokenInfluenceAttributor::window_size() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return count_;
}

void TokenInfluenceAttributor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(window_.begin(), window_.end(), TokenRecord{});
    head_  = 0;
    count_ = 0;
    seq_   = 0;
}

void TokenInfluenceAttributor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    window_.resize(cfg_.window_size);
}

std::string TokenInfluenceAttributor::to_stats_json() const {
    auto results = attribute();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"total_recorded\":"  << total_recorded_.load(std::memory_order_relaxed) << ","
       << "\"dominant_events\":" << dominant_events_.load(std::memory_order_relaxed) << ","
       << "\"window_count\":"    << window_size() << ","
       << "\"top_tokens\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) ss << ",";
        const auto& r = results[i];
        ss << "{"
           << "\"token\":\"" << r.token << "\","
           << "\"weight\":"  << r.weight << ","
           << "\"influence\":" << r.influence << ","
           << "\"cumulative_fraction\":" << r.cumulative_fraction
           << "}";
    }
    ss << "]}";
    return ss.str();
}

} // namespace llmquant
