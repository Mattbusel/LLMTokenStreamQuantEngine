#include "MultiFeedSignalAggregator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MultiFeedSignalAggregator::MultiFeedSignalAggregator(Config cfg)
    : cfg_(std::move(cfg))
{
    for (const auto& fs : cfg_.feeds) {
        FeedState st;
        st.weight    = fs.weight;
        st.ema_alpha = fs.ema_alpha;
        feeds_[fs.name] = st;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void MultiFeedSignalAggregator::recompute_locked() {
    // Collect active feeds.
    std::vector<std::pair<double, double>> active;  // (weight, ema_bias)
    active.reserve(feeds_.size());
    double total_w = 0.0;
    double wsum_b  = 0.0;
    double wsum_c  = 0.0;

    for (const auto& [name, st] : feeds_) {
        if (!st.active) continue;
        double w = std::max(st.weight, 1e-9);
        active.emplace_back(w, st.ema_bias);
        total_w += w;
        wsum_b  += w * st.ema_bias;
        wsum_c  += w * st.ema_conf;
    }

    int n_active = static_cast<int>(active.size());
    if (n_active < cfg_.min_feeds_active || total_w <= 0.0) {
        agg_bias_.store(0.0, std::memory_order_relaxed);
        agg_conf_.store(0.0, std::memory_order_relaxed);
        divergence_.store(0.0, std::memory_order_relaxed);
        return;
    }

    double agg_b = wsum_b / total_w;
    double agg_c = wsum_c / total_w;
    agg_bias_.store(agg_b, std::memory_order_relaxed);
    agg_conf_.store(agg_c, std::memory_order_relaxed);

    // Divergence: normalised std dev of feed EMAs.
    double var = 0.0;
    for (const auto& [w, b] : active) {
        double d = b - agg_b;
        var += (w / total_w) * d * d;
    }
    double sd = std::sqrt(var);
    double ref = std::max(std::abs(agg_b), 0.01);
    double div = std::min(sd / ref, 1.0);
    divergence_.store(div, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void MultiFeedSignalAggregator::record(const std::string& feed_name,
                                        double bias,
                                        double confidence)
{
    std::function<void(double, const std::string&)> on_div;
    std::function<void(double)> on_conv;
    bool fire_div  = false;
    bool fire_conv = false;
    double div_val = 0.0;
    std::string dominant;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        auto it = feeds_.find(feed_name);
        if (it == feeds_.end()) return;  // unknown feed

        FeedState& st = it->second;
        if (!st.active) {
            // Seed the EMA on first record.
            st.ema_bias = bias;
            st.ema_conf = confidence;
            st.active   = true;
        } else {
            st.ema_bias = st.ema_alpha * bias + (1.0 - st.ema_alpha) * st.ema_bias;
            st.ema_conf = st.ema_alpha * confidence + (1.0 - st.ema_alpha) * st.ema_conf;
        }

        recompute_locked();
        div_val = divergence_.load(std::memory_order_relaxed);

        // Find dominant feed (largest |ema_bias|).
        double best = -1.0;
        for (const auto& [name, fst] : feeds_) {
            if (!fst.active) continue;
            if (std::abs(fst.ema_bias) > best) {
                best     = std::abs(fst.ema_bias);
                dominant = name;
            }
        }

        bool now_div = (div_val > cfg_.divergence_threshold);
        if (now_div && !prev_diverged_) {
            div_events_.fetch_add(1, std::memory_order_relaxed);
            fire_div = true;
            on_div   = cfg_.on_divergence;
        } else if (!now_div &&
                   div_val < cfg_.divergence_threshold * cfg_.clear_hysteresis &&
                   prev_diverged_) {
            fire_conv = true;
            on_conv   = cfg_.on_convergence;
        }
        prev_diverged_ = now_div;
        diverged_.store(now_div, std::memory_order_relaxed);
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (fire_div  && on_div)  on_div(div_val, dominant);
    if (fire_conv && on_conv) on_conv(div_val);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::string MultiFeedSignalAggregator::dominant_feed() const {
    std::lock_guard<std::mutex> lk(mutex_);
    double best = -1.0;
    std::string name;
    for (const auto& [n, st] : feeds_) {
        if (!st.active) continue;
        if (std::abs(st.ema_bias) > best) {
            best = std::abs(st.ema_bias);
            name = n;
        }
    }
    return name;
}

int MultiFeedSignalAggregator::active_feed_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int cnt = 0;
    for (const auto& [n, st] : feeds_)
        if (st.active) ++cnt;
    return cnt;
}

std::string MultiFeedSignalAggregator::to_stats_json() const {
    double ab   = aggregated_bias();
    double ac   = aggregated_confidence();
    double div  = divergence();
    bool   divd = is_diverged();
    auto   tot  = total_records();
    auto   evt  = divergence_events();
    std::string dom = dominant_feed();

    std::ostringstream os;
    os << std::fixed << std::setprecision(4);
    os << "{"
       << "\"aggregated_bias\":"  << ab   << ","
       << "\"aggregated_conf\":"  << ac   << ","
       << "\"divergence\":"       << div  << ","
       << "\"is_diverged\":"      << (divd ? "true" : "false") << ","
       << "\"dominant_feed\":\"" << dom  << "\","
       << "\"divergence_events\":" << evt << ","
       << "\"total_records\":"     << tot
       << "}";
    return os.str();
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void MultiFeedSignalAggregator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& [n, st] : feeds_) {
        st.ema_bias = 0.0;
        st.ema_conf = 0.0;
        st.active   = false;
    }
    prev_diverged_ = false;
    agg_bias_.store(0.0, std::memory_order_relaxed);
    agg_conf_.store(0.0, std::memory_order_relaxed);
    divergence_.store(0.0, std::memory_order_relaxed);
    diverged_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    div_events_.store(0, std::memory_order_relaxed);
}

void MultiFeedSignalAggregator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    feeds_.clear();
    for (const auto& fs : cfg_.feeds) {
        FeedState st;
        st.weight    = fs.weight;
        st.ema_alpha = fs.ema_alpha;
        feeds_[fs.name] = st;
    }
    prev_diverged_ = false;
    agg_bias_.store(0.0, std::memory_order_relaxed);
    agg_conf_.store(0.0, std::memory_order_relaxed);
    divergence_.store(0.0, std::memory_order_relaxed);
    diverged_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    div_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
