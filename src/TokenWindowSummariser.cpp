#include "TokenWindowSummariser.hpp"

#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TokenWindowSummariser::TokenWindowSummariser(size_t window_size, float decay)
    : window_size_(window_size), decay_(decay)
{
    if (window_size_ == 0)
        throw std::invalid_argument("TokenWindowSummariser: window_size must be >= 1");
    if (decay_ <= 0.0f || decay_ > 1.0f)
        throw std::invalid_argument("TokenWindowSummariser: decay must be in (0, 1]");
}

// ---------------------------------------------------------------------------
// Data intake
// ---------------------------------------------------------------------------

void TokenWindowSummariser::push(const SemanticWeight& w)
{
    std::lock_guard<std::mutex> lk(mutex_);

    // Age existing entries by the decay factor.
    for (auto& e : window_)
        e.bias *= decay_;

    // Evict oldest when at capacity.
    if (window_.size() >= window_size_)
        window_.pop_front();

    window_.push_back(Entry{
        static_cast<float>(w.directional_bias),
        static_cast<float>(w.confidence_score),
        static_cast<float>(w.volatility_score)
    });
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

TokenWindowSummariser::WindowSummary TokenWindowSummariser::summarise() const
{
    std::lock_guard<std::mutex> lk(mutex_);

    WindowSummary s;
    s.token_count = window_.size();

    if (window_.empty())
        return s;

    // --- net_bias: mean of the (possibly decayed) directional_bias values ---
    float bias_sum = 0.0f;
    float conf_sum = 0.0f;
    for (const auto& e : window_) {
        bias_sum += e.bias;
        conf_sum += e.confidence;
    }
    const float n = static_cast<float>(window_.size());
    s.net_bias   = bias_sum / n;
    s.confidence = conf_sum / n;

    // --- volatility_est: sample std-dev of the *original* (un-decayed) bias ---
    // We use the stored (decayed) biases as a recency-weighted proxy; the
    // variance of this series captures how spread-out recent bias values are.
    if (window_.size() >= 2) {
        float mean = s.net_bias;
        float ss   = 0.0f;
        for (const auto& e : window_) {
            float d = e.bias - mean;
            ss += d * d;
        }
        s.volatility_est = std::sqrt(ss / (n - 1.0f));
    }

    return s;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool TokenWindowSummariser::is_ready() const noexcept
{
    std::lock_guard<std::mutex> lk(mutex_);
    return window_.size() >= window_size_;
}

size_t TokenWindowSummariser::size() const noexcept
{
    std::lock_guard<std::mutex> lk(mutex_);
    return window_.size();
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

void TokenWindowSummariser::reset()
{
    std::lock_guard<std::mutex> lk(mutex_);
    window_.clear();
}

std::string TokenWindowSummariser::to_stats_json() const
{
    std::lock_guard<std::mutex> lk(mutex_);

    // Compute summary inline (mutex already held).
    float bias_sum = 0.0f;
    float conf_sum = 0.0f;
    for (const auto& e : window_) {
        bias_sum += e.bias;
        conf_sum += e.confidence;
    }
    const float n       = window_.empty() ? 1.0f : static_cast<float>(window_.size());
    float net_bias      = window_.empty() ? 0.0f : bias_sum / n;
    float conf          = window_.empty() ? 0.0f : conf_sum / n;
    float vol_est       = 0.0f;
    if (window_.size() >= 2) {
        float ss = 0.0f;
        for (const auto& e : window_) {
            float d = e.bias - net_bias;
            ss += d * d;
        }
        vol_est = std::sqrt(ss / (n - 1.0f));
    }
    bool ready = (window_.size() >= window_size_);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"net_bias\":"        << net_bias    << ","
       << "\"confidence\":"      << conf        << ","
       << "\"volatility_est\":"  << vol_est     << ","
       << "\"token_count\":"     << window_.size() << ","
       << "\"window_size\":"     << window_size_ << ","
       << "\"decay\":"           << decay_       << ","
       << "\"is_ready\":"        << (ready ? "true" : "false")
       << "}";
    return ss.str();
}

} // namespace llmquant
