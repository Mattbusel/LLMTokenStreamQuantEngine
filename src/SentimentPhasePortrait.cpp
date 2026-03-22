#include "SentimentPhasePortrait.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SentimentPhasePortrait::SentimentPhasePortrait() {
    int n = cfg_.grid_size;
    counts_.assign(n, std::vector<double>(n, 0.0));
    visit_history_.assign(cfg_.cycle_window, -1);
}

SentimentPhasePortrait::SentimentPhasePortrait(const Config& cfg)
    : cfg_(cfg) {
    int n = cfg_.grid_size;
    counts_.assign(n, std::vector<double>(n, 0.0));
    visit_history_.assign(cfg_.cycle_window, -1);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

int SentimentPhasePortrait::coord_to_bin(double value, double lo, double hi,
                                          int n) const noexcept {
    if (n <= 0) return 0;
    double frac = (value - lo) / (hi - lo);
    int b = static_cast<int>(frac * n);
    if (b < 0) b = 0;
    if (b >= n) b = n - 1;
    return b;
}

void SentimentPhasePortrait::recompute_attractor_locked() {
    int n = static_cast<int>(counts_.size());
    double best = -1.0;
    double total = 0.0;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            total += counts_[r][c];
    if (total < static_cast<double>(cfg_.min_visits)) return;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            double frac = counts_[r][c] / total;
            if (frac > best && frac >= cfg_.attractor_threshold) {
                best = frac;
                attractor_row_val_ = r;
                attractor_col_val_ = c;
            }
        }
    }
}

void SentimentPhasePortrait::push_visit_locked(int encoded) {
    int cap = static_cast<int>(visit_history_.size());
    if (cap <= 0) return;
    visit_history_[visit_head_] = encoded;
    visit_head_ = (visit_head_ + 1) % cap;
    if (visit_count_ < cap) ++visit_count_;
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

void SentimentPhasePortrait::record(double bias) {
    std::function<void(int, int)>         attr_cb;
    std::function<void(int,int,int,int)>  cycle_cb;
    int new_row = 0, new_col = 0;
    int old_row = -1, old_col = -1;
    bool attr_changed = false;
    bool cycle_found  = false;
    int  c1r = -1, c1c = -1, c2r = -1, c2c = -1;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        int n = static_cast<int>(counts_.size());

        // EMA velocity
        double alpha = cfg_.velocity_alpha;
        if (!seeded_) {
            bias_ema_      = bias;
            prev_bias_ema_ = bias;
            seeded_        = true;
        } else {
            prev_bias_ema_ = bias_ema_;
            bias_ema_      = alpha * bias + (1.0 - alpha) * bias_ema_;
        }
        double vel = bias_ema_ - prev_bias_ema_;

        // Map to grid cell
        new_row = coord_to_bin(bias_ema_, cfg_.bias_min, cfg_.bias_max, n);
        new_col = coord_to_bin(vel,       cfg_.vel_min,  cfg_.vel_max,  n);

        bool moved = (new_row != current_row_val_ || new_col != current_col_val_);
        if (total_.load(std::memory_order_relaxed) > 0) {
            old_row = current_row_val_;
            old_col = current_col_val_;
        }

        counts_[new_row][new_col] += 1.0;
        push_visit_locked(new_row * n + new_col);

        int prev_attr_r = attractor_row_val_;
        int prev_attr_c = attractor_col_val_;
        recompute_attractor_locked();
        if (attractor_row_val_ != prev_attr_r || attractor_col_val_ != prev_attr_c) {
            if (prev_attr_r >= 0) {
                attr_changed = true;
                attr_cb = cfg_.on_attractor_change;
            }
        }

        // Period-2 cycle detection: look for A-B-A-B pattern in recent window
        if (visit_count_ >= 4) {
            int cap = static_cast<int>(visit_history_.size());
            // Walk recent history backwards
            auto at = [&](int ago) -> int {
                int idx = ((visit_head_ - 1 - ago) % cap + cap) % cap;
                return visit_history_[idx];
            };
            int a = at(0), b = at(1), c2 = at(2), d = at(3);
            if (a != b && a == c2 && b == d && a >= 0 && b >= 0) {
                cycle_found = true;
                cycle_cb = cfg_.on_cycle_detected;
                c1r = a / n;  c1c = a % n;
                c2r = b / n;  c2c = b % n;
            }
        }

        if (moved && (old_row >= 0)) {
            transitions_.fetch_add(1, std::memory_order_relaxed);
        }
        current_row_val_ = new_row;
        current_col_val_ = new_col;
    }

    row_out_.store(new_row, std::memory_order_relaxed);
    col_out_.store(new_col, std::memory_order_relaxed);
    // velocity is stored as bias EMA difference — read inside lock would race,
    // so recompute: vel = bias_ema_ - prev_bias_ema_ already stored in col
    total_.fetch_add(1, std::memory_order_relaxed);

    if (attr_changed && attr_cb) attr_cb(attractor_row_val_, attractor_col_val_);
    if (cycle_found  && cycle_cb) cycle_cb(c1r, c1c, c2r, c2c);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int SentimentPhasePortrait::current_row() const noexcept {
    return row_out_.load(std::memory_order_relaxed);
}

int SentimentPhasePortrait::current_col() const noexcept {
    return col_out_.load(std::memory_order_relaxed);
}

double SentimentPhasePortrait::velocity() const noexcept {
    return vel_out_.load(std::memory_order_relaxed);
}

uint64_t SentimentPhasePortrait::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

uint64_t SentimentPhasePortrait::cell_transitions() const noexcept {
    return transitions_.load(std::memory_order_relaxed);
}

double SentimentPhasePortrait::dwell_fraction(int row, int col) const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(counts_.size());
    if (row < 0 || row >= n || col < 0 || col >= n) return 0.0;
    double total = 0.0;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            total += counts_[r][c];
    if (total == 0.0) return 0.0;
    return counts_[row][col] / total;
}

int SentimentPhasePortrait::attractor_row() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return attractor_row_val_;
}

int SentimentPhasePortrait::attractor_col() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return attractor_col_val_;
}

bool SentimentPhasePortrait::cycle_detected() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (visit_count_ < 4) return false;
    int n = static_cast<int>(counts_.size());
    int cap = static_cast<int>(visit_history_.size());
    auto at = [&](int ago) -> int {
        int idx = ((visit_head_ - 1 - ago) % cap + cap) % cap;
        return visit_history_[idx];
    };
    int a = at(0), b = at(1), c = at(2), d = at(3);
    return (a != b && a == c && b == d && a >= 0 && b >= 0 && n > 0);
}

double SentimentPhasePortrait::divergence_index() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    if (attractor_row_val_ < 0 || visit_count_ < 2) return 0.0;
    int n   = static_cast<int>(counts_.size());
    int cap = static_cast<int>(visit_history_.size());
    int aenc = attractor_row_val_ * n + attractor_col_val_;
    int away = 0;
    for (int i = 0; i < visit_count_; ++i) {
        int idx = ((visit_head_ - 1 - i) % cap + cap) % cap;
        if (visit_history_[idx] != aenc) ++away;
    }
    return static_cast<double>(away) / static_cast<double>(visit_count_);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void SentimentPhasePortrait::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = static_cast<int>(counts_.size());
    for (auto& row : counts_) std::fill(row.begin(), row.end(), 0.0);
    std::fill(visit_history_.begin(), visit_history_.end(), -1);
    visit_head_         = 0;
    visit_count_        = 0;
    bias_ema_           = 0.0;
    prev_bias_ema_      = 0.0;
    seeded_             = false;
    current_row_val_    = 0;
    current_col_val_    = 0;
    attractor_row_val_  = -1;
    attractor_col_val_  = -1;
    (void)n;

    row_out_.store(0,  std::memory_order_relaxed);
    col_out_.store(0,  std::memory_order_relaxed);
    vel_out_.store(0.0, std::memory_order_relaxed);
    total_.store(0,    std::memory_order_relaxed);
    transitions_.store(0, std::memory_order_relaxed);
}

void SentimentPhasePortrait::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int n = cfg_.grid_size;
    counts_.assign(n, std::vector<double>(n, 0.0));
    visit_history_.assign(cfg_.cycle_window, -1);
    visit_head_         = 0;
    visit_count_        = 0;
    bias_ema_           = 0.0;
    prev_bias_ema_      = 0.0;
    seeded_             = false;
    current_row_val_    = 0;
    current_col_val_    = 0;
    attractor_row_val_  = -1;
    attractor_col_val_  = -1;

    row_out_.store(0,  std::memory_order_relaxed);
    col_out_.store(0,  std::memory_order_relaxed);
    vel_out_.store(0.0, std::memory_order_relaxed);
    total_.store(0,    std::memory_order_relaxed);
    transitions_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string SentimentPhasePortrait::to_stats_json() const {
    int row  = current_row();
    int col  = current_col();
    int ar   = attractor_row();
    int ac   = attractor_col();
    bool cyc = cycle_detected();
    double di = divergence_index();
    auto tot = total_records();
    auto tr  = cell_transitions();

    std::ostringstream os;
    os << "{"
       << "\"row\":"             << row  << ","
       << "\"col\":"             << col  << ","
       << "\"attractor_row\":"   << ar   << ","
       << "\"attractor_col\":"   << ac   << ","
       << "\"cycle_detected\":"  << (cyc ? "true" : "false") << ","
       << "\"divergence_index\":" << di  << ","
       << "\"cell_transitions\":" << tr  << ","
       << "\"total_records\":"   << tot
       << "}";
    return os.str();
}

}  // namespace llmquant
