/**
 * @file src/cross_asset_sentiment.cpp
 * @brief Cross-Asset Sentiment Correlation implementation.
 *
 * See include/cross_asset_sentiment.hpp for the public API.
 */

#include "cross_asset_sentiment.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ─── SentimentCorrelationMatrix ───────────────────────────────────────────────

SentimentCorrelationMatrix::SentimentCorrelationMatrix(Config config) noexcept
    : config_(std::move(config)) {}

bool SentimentCorrelationMatrix::register_asset(const std::string& asset_id) {
    if (windows_.count(asset_id)) return false;
    windows_[asset_id] = AssetWindow{};
    id_order_.push_back(asset_id);
    return true;
}

bool SentimentCorrelationMatrix::update(const AssetSentiment& sentiment) {
    auto it = windows_.find(sentiment.asset_id);
    if (it == windows_.end()) return false;

    auto& w = it->second.values;
    w.push_back(sentiment.clamped_score());
    while (static_cast<int>(w.size()) > config_.window_size) {
        w.pop_front();
    }
    return true;
}

void SentimentCorrelationMatrix::update(
    const std::vector<AssetSentiment>& sentiments
) {
    for (const auto& s : sentiments) {
        update(s);
    }
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

// static
std::vector<double> SentimentCorrelationMatrix::tail(
    const std::deque<double>& dq,
    int n
) noexcept {
    int sz = static_cast<int>(dq.size());
    int take = std::min(n, sz);
    std::vector<double> out(static_cast<std::size_t>(take));
    auto it = dq.end() - take;
    for (int i = 0; i < take; ++i, ++it) out[i] = *it;
    return out;
}

// static
double SentimentCorrelationMatrix::pearson(
    const std::vector<double>& xs,
    const std::vector<double>& ys
) noexcept {
    const int n = static_cast<int>(std::min(xs.size(), ys.size()));
    if (n < 2) return 0.0;

    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_x  += xs[i];
        sum_y  += ys[i];
        sum_xx += xs[i] * xs[i];
        sum_yy += ys[i] * ys[i];
        sum_xy += xs[i] * ys[i];
    }

    const double inv_n = 1.0 / n;
    const double mean_x = sum_x * inv_n;
    const double mean_y = sum_y * inv_n;
    const double var_x  = sum_xx * inv_n - mean_x * mean_x;
    const double var_y  = sum_yy * inv_n - mean_y * mean_y;
    const double cov_xy = sum_xy * inv_n - mean_x * mean_y;

    const double denom = std::sqrt(var_x * var_y);
    if (denom < 1e-12) return 0.0;

    return std::max(-1.0, std::min(1.0, cov_xy / denom));
}

// static
double SentimentCorrelationMatrix::lagged_pearson(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    int lag
) noexcept {
    // xs leads ys by `lag` steps:
    //   compare xs[0..N-lag-1] with ys[lag..N-1]
    if (lag <= 0) return pearson(xs, ys);

    const int n = static_cast<int>(std::min(xs.size(), ys.size()));
    if (n <= lag) return 0.0;

    const int aligned = n - lag;
    std::vector<double> xs_lead(xs.begin(), xs.begin() + aligned);
    std::vector<double> ys_lagged(ys.begin() + lag, ys.begin() + aligned + lag);

    return pearson(xs_lead, ys_lagged);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

std::optional<double> SentimentCorrelationMatrix::correlation(
    const std::string& asset_a,
    const std::string& asset_b
) const noexcept {
    auto ia = windows_.find(asset_a);
    auto ib = windows_.find(asset_b);
    if (ia == windows_.end() || ib == windows_.end()) return std::nullopt;

    const int n = std::min(
        static_cast<int>(ia->second.values.size()),
        static_cast<int>(ib->second.values.size())
    );
    if (n < config_.min_samples) return std::nullopt;

    const auto xs = tail(ia->second.values, n);
    const auto ys = tail(ib->second.values, n);

    return pearson(xs, ys);
}

std::optional<std::string> SentimentCorrelationMatrix::leading_asset(
    const std::string& target
) const noexcept {
    auto it_target = windows_.find(target);
    if (it_target == windows_.end()) return std::nullopt;

    const auto& target_vals = it_target->second.values;
    const int target_n = static_cast<int>(target_vals.size());

    std::string best_asset;
    double best_abs_corr = config_.min_lag_corr - 1e-12;  // strictly less

    for (const auto& [id, window] : windows_) {
        if (id == target) continue;

        const int n = std::min(
            static_cast<int>(window.values.size()),
            target_n
        );
        if (n < config_.min_samples + config_.max_lag) continue;

        const auto xs = tail(window.values, n);
        const auto ys = tail(target_vals, n);

        for (int lag = 1; lag <= config_.max_lag; ++lag) {
            const double r = lagged_pearson(xs, ys, lag);
            const double abs_r = std::fabs(r);
            if (abs_r > best_abs_corr) {
                best_abs_corr = abs_r;
                best_asset = id;
            }
        }
    }

    if (best_asset.empty()) return std::nullopt;
    return best_asset;
}

std::vector<std::string> SentimentCorrelationMatrix::asset_ids() const {
    return id_order_;
}

int SentimentCorrelationMatrix::window_size_for(
    const std::string& asset_id
) const noexcept {
    auto it = windows_.find(asset_id);
    if (it == windows_.end()) return 0;
    return static_cast<int>(it->second.values.size());
}

std::vector<SentimentCorrelationMatrix::CorrelationEntry>
SentimentCorrelationMatrix::full_matrix() const noexcept {
    std::vector<CorrelationEntry> result;
    for (std::size_t i = 0; i < id_order_.size(); ++i) {
        for (std::size_t j = i + 1; j < id_order_.size(); ++j) {
            const auto& a = id_order_[i];
            const auto& b = id_order_[j];
            auto rho_opt = correlation(a, b);
            result.push_back({a, b, rho_opt.value_or(0.0)});
        }
    }
    return result;
}

double SentimentCorrelationMatrix::average_correlation() const noexcept {
    const auto entries = full_matrix();
    if (entries.empty()) return 0.0;

    double sum = 0.0;
    for (const auto& e : entries) sum += e.rho;
    return sum / static_cast<double>(entries.size());
}

// ─── SentimentCluster ─────────────────────────────────────────────────────────

// static
std::vector<SentimentCluster::Cluster> SentimentCluster::build(
    const SentimentCorrelationMatrix& matrix,
    double threshold
) {
    const auto ids = matrix.asset_ids();
    const int n = static_cast<int>(ids.size());

    // Union-Find for single-linkage clustering
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int(int)> find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx != ry) parent[rx] = ry;
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto rho_opt = matrix.correlation(ids[i], ids[j]);
            if (rho_opt.has_value() &&
                std::fabs(rho_opt.value()) >= threshold) {
                unite(i, j);
            }
        }
    }

    // Collect clusters
    std::unordered_map<int, std::vector<std::string>> groups;
    for (int i = 0; i < n; ++i) {
        groups[find(i)].push_back(ids[i]);
    }

    std::vector<Cluster> clusters;
    clusters.reserve(groups.size());
    for (auto& [root, members] : groups) {
        Cluster c;
        c.members = std::move(members);
        clusters.push_back(std::move(c));
    }

    // Sort: larger clusters first
    std::sort(clusters.begin(), clusters.end(),
              [](const Cluster& a, const Cluster& b) {
                  return a.size() > b.size();
              });

    return clusters;
}

// ─── SentimentRegimeDetector ──────────────────────────────────────────────────

SentimentRegimeDetector::SentimentRegimeDetector(Config config) noexcept
    : config_(std::move(config)) {}

double SentimentRegimeDetector::update(
    const SentimentCorrelationMatrix& matrix
) {
    ++steps_since_event_;

    const double avg = matrix.average_correlation();
    const bool above_threshold = avg >= config_.spike_threshold;
    const bool cooldown_expired = steps_since_event_ > config_.cooldown_steps;

    if (above_threshold && !in_contagion_ && cooldown_expired) {
        in_contagion_ = true;
        ++event_count_;
        steps_since_event_ = 0;
        if (config_.on_contagion_event) {
            config_.on_contagion_event(avg);
        }
    } else if (!above_threshold && in_contagion_) {
        in_contagion_ = false;
    }

    last_avg_corr_ = avg;
    return avg;
}

void SentimentRegimeDetector::reset() noexcept {
    event_count_ = 0;
    steps_since_event_ = 0;
    last_avg_corr_ = 0.0;
    in_contagion_ = false;
}

}  // namespace llmquant
