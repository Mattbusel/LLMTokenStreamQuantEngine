#include "BootstrapSignalCI.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

BootstrapSignalCI::BootstrapSignalCI(size_t n_bootstrap,
                                     float  confidence_level,
                                     size_t window_size,
                                     size_t min_observations)
    : n_bootstrap_(n_bootstrap)
    , confidence_level_(confidence_level)
    , window_size_(window_size)
    , min_observations_(min_observations)
{
    if (n_bootstrap_ < 2)
        throw std::invalid_argument("BootstrapSignalCI: n_bootstrap must be >= 2");
    if (confidence_level_ <= 0.0f || confidence_level_ >= 1.0f)
        throw std::invalid_argument("BootstrapSignalCI: confidence_level must be in (0, 1)");
    if (window_size_ == 0)
        throw std::invalid_argument("BootstrapSignalCI: window_size must be >= 1");
    if (min_observations_ < 2)
        throw std::invalid_argument("BootstrapSignalCI: min_observations must be >= 2");
    if (min_observations_ > window_size_)
        throw std::invalid_argument("BootstrapSignalCI: min_observations must be <= window_size");

    reservoir_.resize(window_size_, 0.0f);
}

// ---------------------------------------------------------------------------
// Data intake
// ---------------------------------------------------------------------------

void BootstrapSignalCI::add_observation(float signal_value)
{
    std::lock_guard<std::mutex> lk(mutex_);
    reservoir_[head_] = signal_value;
    head_ = (head_ + 1) % window_size_;
    if (count_ < window_size_) ++count_;
    total_recorded_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Computation
// ---------------------------------------------------------------------------

std::vector<float> BootstrapSignalCI::bootstrap_means_locked() const
{
    // Collect active observations into a contiguous vector.
    std::vector<float> obs;
    obs.reserve(count_);
    size_t start = (count_ < window_size_) ? 0 : head_;
    for (size_t i = 0; i < count_; ++i)
        obs.push_back(reservoir_[(start + i) % window_size_]);

    const size_t n = obs.size();
    std::vector<float> means;
    means.reserve(n_bootstrap_);

    // Deterministic seed derived from current observation count so results are
    // reproducible per-window-fill but vary as new data arrives.
    std::mt19937 rng(static_cast<uint32_t>(
        total_recorded_.load(std::memory_order_relaxed) ^ (count_ * 2654435761ULL)));
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    for (size_t b = 0; b < n_bootstrap_; ++b) {
        float sum = 0.0f;
        for (size_t s = 0; s < n; ++s)
            sum += obs[dist(rng)];
        means.push_back(sum / static_cast<float>(n));
    }

    std::sort(means.begin(), means.end());
    return means;
}

BootstrapSignalCI::ConfidenceInterval BootstrapSignalCI::compute() const
{
    std::lock_guard<std::mutex> lk(mutex_);

    ConfidenceInterval ci;

    if (count_ == 0)
        return ci;

    // Collect active observations.
    std::vector<float> obs;
    obs.reserve(count_);
    size_t start = (count_ < window_size_) ? 0 : head_;
    for (size_t i = 0; i < count_; ++i)
        obs.push_back(reservoir_[(start + i) % window_size_]);

    const float simple_mean = std::accumulate(obs.begin(), obs.end(), 0.0f)
                              / static_cast<float>(obs.size());

    if (count_ < min_observations_) {
        // Not enough data: degenerate interval at simple mean.
        ci.lower = ci.upper = ci.median = simple_mean;
        return ci;
    }

    // Run bootstrap.
    std::vector<float> means = bootstrap_means_locked();

    // Percentile indices.
    float alpha       = 1.0f - confidence_level_;
    float lower_frac  = alpha / 2.0f;
    float upper_frac  = 1.0f - alpha / 2.0f;
    float median_frac = 0.5f;

    auto percentile = [&](float frac) -> float {
        // Linear interpolation.
        float idx = frac * static_cast<float>(means.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = lo + 1;
        if (hi >= means.size()) return means.back();
        float t = idx - static_cast<float>(lo);
        return means[lo] * (1.0f - t) + means[hi] * t;
    };

    ci.lower  = percentile(lower_frac);
    ci.upper  = percentile(upper_frac);
    ci.median = percentile(median_frac);
    return ci;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool BootstrapSignalCI::is_ready() const noexcept
{
    std::lock_guard<std::mutex> lk(mutex_);
    return count_ >= min_observations_;
}

size_t BootstrapSignalCI::observation_count() const noexcept
{
    std::lock_guard<std::mutex> lk(mutex_);
    return count_;
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

void BootstrapSignalCI::reset()
{
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(reservoir_.begin(), reservoir_.end(), 0.0f);
    head_  = 0;
    count_ = 0;
}

std::string BootstrapSignalCI::to_stats_json() const
{
    ConfidenceInterval ci = compute();   // acquires mutex internally
    uint64_t tot;
    size_t   cnt;
    bool     ready;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        tot   = total_recorded_.load(std::memory_order_relaxed);
        cnt   = count_;
        ready = cnt >= min_observations_;
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"lower\":"            << ci.lower            << ","
       << "\"median\":"           << ci.median           << ","
       << "\"upper\":"            << ci.upper            << ","
       << "\"confidence_level\":" << confidence_level_   << ","
       << "\"n_bootstrap\":"      << n_bootstrap_        << ","
       << "\"observation_count\":" << cnt                << ","
       << "\"total_recorded\":"   << tot                 << ","
       << "\"is_ready\":"         << (ready ? "true" : "false")
       << "}";
    return ss.str();
}

} // namespace llmquant
