#include "CrossTokenCorrelationMatrix.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

CrossTokenCorrelationMatrix::CrossTokenCorrelationMatrix(Config cfg)
    : cfg_(std::move(cfg))
{
    int n = std::min(std::max(1, cfg_.n_channels), kMaxChannels);
    int w = std::max(2, cfg_.window);
    bufs_.assign(static_cast<std::size_t>(n),
                 std::vector<double>(static_cast<std::size_t>(w), 0.0));
    heads_.assign(static_cast<std::size_t>(n), 0);
    fills_.assign(static_cast<std::size_t>(n), 0);
    corr_cache_.assign(static_cast<std::size_t>(n * n), 0.0);
    prev_diverged_.assign(static_cast<std::size_t>(n * n), false);
    prev_converged_.assign(static_cast<std::size_t>(n * n), false);
}

void CrossTokenCorrelationMatrix::record(int channel, double bias) {
    if (channel < 0 || channel >= static_cast<int>(bufs_.size())) return;

    std::vector<std::pair<int,int>> fire_div, fire_conv;
    std::vector<double> fire_div_r, fire_conv_r;
    std::function<void(int,int,double)> div_cb, con_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        div_cb = cfg_.on_divergence;
        con_cb = cfg_.on_convergence;
        total_.fetch_add(1, std::memory_order_relaxed);

        int n = static_cast<int>(bufs_.size());
        int w = static_cast<int>(bufs_[0].size());
        auto ch = static_cast<std::size_t>(channel);

        // Update ring buffer for this channel
        bufs_[ch][static_cast<std::size_t>(heads_[ch])] = bias;
        heads_[ch] = (heads_[ch] + 1) % w;
        if (fills_[ch] < w) ++fills_[ch];

        // Recompute all pairs involving this channel
        for (int j = 0; j < n; ++j) {
            if (j == channel) continue;
            recompute_pair_locked(channel, j);

            int key = channel * n + j;
            double r = corr_cache_[static_cast<std::size_t>(key)];
            double abs_r = std::abs(r);

            bool now_div = abs_r < cfg_.divergence_threshold;
            bool now_con = abs_r > cfg_.convergence_threshold;

            if (now_div && !prev_diverged_[static_cast<std::size_t>(key)]) {
                fire_div.push_back({channel, j});
                fire_div_r.push_back(r);
            }
            if (now_con && !prev_converged_[static_cast<std::size_t>(key)]) {
                fire_conv.push_back({channel, j});
                fire_conv_r.push_back(r);
            }
            prev_diverged_[static_cast<std::size_t>(key)] = now_div;
            prev_converged_[static_cast<std::size_t>(key)] = now_con;

            // Symmetric entry
            int key2 = j * n + channel;
            corr_cache_[static_cast<std::size_t>(key2)] = r;
            prev_diverged_[static_cast<std::size_t>(key2)]  = now_div;
            prev_converged_[static_cast<std::size_t>(key2)] = now_con;
        }
    }

    for (std::size_t k = 0; k < fire_div.size(); ++k)
        if (div_cb) div_cb(fire_div[k].first, fire_div[k].second, fire_div_r[k]);
    for (std::size_t k = 0; k < fire_conv.size(); ++k)
        if (con_cb) con_cb(fire_conv[k].first, fire_conv[k].second, fire_conv_r[k]);
}

void CrossTokenCorrelationMatrix::recompute_pair_locked(int i, int j) {
    int n_ch  = static_cast<int>(bufs_.size());
    int w     = static_cast<int>(bufs_[0].size());
    int ni    = fills_[static_cast<std::size_t>(i)];
    int nj    = fills_[static_cast<std::size_t>(j)];
    int count = std::min(ni, nj);

    int key = i * n_ch + j;

    if (count < 2) {
        corr_cache_[static_cast<std::size_t>(key)] = 0.0;
        return;
    }

    // Aligned suffix (most recent `count` samples from both channels)
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int t = 0; t < count; ++t) {
        int xi = (heads_[static_cast<std::size_t>(i)] - 1 - t + w * 2) % w;
        int yi = (heads_[static_cast<std::size_t>(j)] - 1 - t + w * 2) % w;
        double x = bufs_[static_cast<std::size_t>(i)][static_cast<std::size_t>(xi)];
        double y = bufs_[static_cast<std::size_t>(j)][static_cast<std::size_t>(yi)];
        sx  += x; sy  += y;
        sxx += x * x; syy += y * y;
        sxy += x * y;
    }

    double n  = static_cast<double>(count);
    double denom = std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    double r = (denom > 0.0) ? (n * sxy - sx * sy) / denom : 0.0;
    r = std::max(-1.0, std::min(1.0, r));  // clamp numerical noise
    corr_cache_[static_cast<std::size_t>(key)] = r;
}

double CrossTokenCorrelationMatrix::correlation(int ch_i, int ch_j) const {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(bufs_.size());
    if (ch_i < 0 || ch_i >= n || ch_j < 0 || ch_j >= n) return 0.0;
    return corr_cache_[static_cast<std::size_t>(ch_i * n + ch_j)];
}

std::string CrossTokenCorrelationMatrix::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(bufs_.size());
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"total_records\":" << total_.load(std::memory_order_relaxed)
       << ",\"correlations\":[";
    bool first = true;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (!first) ss << ",";
            first = false;
            ss << "{\"i\":" << i << ",\"j\":" << j
               << ",\"r\":" << corr_cache_[static_cast<std::size_t>(i * n + j)] << "}";
        }
    }
    ss << "]}";
    return ss.str();
}

void CrossTokenCorrelationMatrix::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(bufs_.size());
    int w = static_cast<int>(bufs_[0].size());
    for (int i = 0; i < n; ++i) {
        std::fill(bufs_[static_cast<std::size_t>(i)].begin(),
                  bufs_[static_cast<std::size_t>(i)].end(), 0.0);
        heads_[static_cast<std::size_t>(i)] = 0;
        fills_[static_cast<std::size_t>(i)] = 0;
    }
    std::fill(corr_cache_.begin(), corr_cache_.end(), 0.0);
    std::fill(prev_diverged_.begin(), prev_diverged_.end(), false);
    std::fill(prev_converged_.begin(), prev_converged_.end(), false);
    total_.store(0, std::memory_order_relaxed);
    (void)w;
}

void CrossTokenCorrelationMatrix::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int n = std::min(std::max(1, cfg_.n_channels), kMaxChannels);
    int w = std::max(2, cfg_.window);
    bufs_.assign(static_cast<std::size_t>(n),
                 std::vector<double>(static_cast<std::size_t>(w), 0.0));
    heads_.assign(static_cast<std::size_t>(n), 0);
    fills_.assign(static_cast<std::size_t>(n), 0);
    corr_cache_.assign(static_cast<std::size_t>(n * n), 0.0);
    prev_diverged_.assign(static_cast<std::size_t>(n * n), false);
    prev_converged_.assign(static_cast<std::size_t>(n * n), false);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
