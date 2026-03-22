#include "TokenBiasPhaseSpace.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

namespace llmquant {

TokenBiasPhaseSpace::TokenBiasPhaseSpace(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    int g = std::max(2, cfg_.grid_size);
    ring_.assign(static_cast<std::size_t>(w), 0);
    grid_.assign(static_cast<std::size_t>(g * g), 0);
    cfg_grid_size_.store(g, std::memory_order_relaxed);
}

int TokenBiasPhaseSpace::bin_for(double v) const {
    int g   = std::max(2, cfg_.grid_size);
    double rng = cfg_.range_max - cfg_.range_min;
    if (rng <= 0.0) return 0;
    double frac = (v - cfg_.range_min) / rng;
    frac = std::max(0.0, std::min(1.0 - 1e-12, frac));
    return static_cast<int>(frac * g);
}

void TokenBiasPhaseSpace::recompute_grid_locked() {
    int g = std::max(2, cfg_.grid_size);
    int total_cells = g * g;

    // Find dominant cell
    int dom = 0, dom_count = 0;
    for (int i = 0; i < total_cells; ++i) {
        if (grid_[static_cast<std::size_t>(i)] > dom_count) {
            dom_count = grid_[static_cast<std::size_t>(i)];
            dom = i;
        }
    }
    dominant_packed_.store(dom, std::memory_order_relaxed);

    // Occupancy entropy H = -Σ p_i log2(p_i)
    double total_occ = static_cast<double>(ring_fill_);
    double H = 0.0;
    if (total_occ > 0.0) {
        for (int i = 0; i < total_cells; ++i) {
            double c = static_cast<double>(grid_[static_cast<std::size_t>(i)]);
            if (c > 0.0) {
                double p = c / total_occ;
                H -= p * std::log2(p);
            }
        }
    }
    entropy_.store(H, std::memory_order_relaxed);
}

void TokenBiasPhaseSpace::record(double bias) {
    bool fire_shift      = false;
    bool fire_conc       = false;
    bool fire_diffuse    = false;
    double entropy_val   = 0.0;
    CellId old_cell{}, new_cell{};
    std::function<void(CellId, CellId)> shift_cb;
    std::function<void(double)>         conc_cb, diff_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        shift_cb = cfg_.on_attractor_shift;
        conc_cb  = cfg_.on_attractor_concentrated;
        diff_cb  = cfg_.on_attractor_diffuse;
        total_.fetch_add(1, std::memory_order_relaxed);

        if (!seeded_) {
            prev_bias_ = bias;
            seeded_    = true;
            return;
        }

        int g    = std::max(2, cfg_.grid_size);
        int row  = bin_for(prev_bias_);
        int col  = bin_for(bias);
        int cell = row * g + col;

        int w = static_cast<int>(ring_.size());
        if (ring_fill_ == w) {
            // Evict oldest
            int old_cell_id = ring_[static_cast<std::size_t>(ring_head_)];
            if (grid_[static_cast<std::size_t>(old_cell_id)] > 0)
                --grid_[static_cast<std::size_t>(old_cell_id)];
        } else {
            ++ring_fill_;
        }
        ring_[static_cast<std::size_t>(ring_head_)] = cell;
        ring_head_ = (ring_head_ + 1) % w;
        ++grid_[static_cast<std::size_t>(cell)];

        prev_bias_ = bias;

        recompute_grid_locked();

        int cur_dom = dominant_packed_.load(std::memory_order_relaxed);
        entropy_val = entropy_.load(std::memory_order_relaxed);

        if (prev_dominant_ >= 0 && cur_dom != prev_dominant_) {
            int p_row = prev_dominant_ / g, p_col = prev_dominant_ % g;
            int n_row = cur_dom / g,        n_col = cur_dom % g;
            old_cell = {p_row, p_col};
            new_cell = {n_row, n_col};
            fire_shift = true;
            shift_events_.fetch_add(1, std::memory_order_relaxed);
        }
        prev_dominant_ = cur_dom;

        if (entropy_val < cfg_.entropy_low_threshold)  fire_conc   = true;
        if (entropy_val > cfg_.entropy_high_threshold) fire_diffuse = true;
    }

    if (fire_shift   && shift_cb) shift_cb(old_cell, new_cell);
    if (fire_conc    && conc_cb)  conc_cb(entropy_val);
    if (fire_diffuse && diff_cb)  diff_cb(entropy_val);
}

std::string TokenBiasPhaseSpace::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    int g   = std::max(1, cfg_.grid_size);
    int dom = dominant_packed_.load(std::memory_order_relaxed);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"dominant_row\":"      << (dom / g) << ","
       << "\"dominant_col\":"      << (dom % g) << ","
       << "\"occupancy_entropy\":" << entropy_.load(std::memory_order_relaxed) << ","
       << "\"shift_events\":"      << shift_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"     << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenBiasPhaseSpace::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_.begin(), ring_.end(), 0);
    std::fill(grid_.begin(), grid_.end(), 0);
    ring_head_ = ring_fill_ = 0;
    seeded_       = false;
    prev_dominant_= -1;
    dominant_packed_.store(0, std::memory_order_relaxed);
    entropy_.store(0.0, std::memory_order_relaxed);
    shift_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenBiasPhaseSpace::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    int g = std::max(2, cfg_.grid_size);
    ring_.assign(static_cast<std::size_t>(w), 0);
    grid_.assign(static_cast<std::size_t>(g * g), 0);
    ring_head_ = ring_fill_ = 0;
    seeded_       = false;
    prev_dominant_= -1;
    cfg_grid_size_.store(g, std::memory_order_relaxed);
}

} // namespace llmquant
