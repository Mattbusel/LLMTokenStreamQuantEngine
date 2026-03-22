#pragma once

/// @file TokenBiasPhaseSpace.h
/// @brief Plots the bias stream in a 2-D phase space (bias_t vs bias_{t-1})
///        and detects attractor migration — a sign of regime change.
///
/// ## Motivation
/// Plotting x[t] against x[t-1] (a delay embedding) reveals nonlinear
/// structure invisible to linear statistics.  A stable trending regime
/// produces a dense diagonal attractor; a mean-reverting regime produces
/// an anti-diagonal one; a noisy regime fills the space uniformly.
/// By tracking which grid cell has the highest density, we detect when
/// the dominant attractor migrates — a leading indicator of regime
/// transitions.
///
/// ## Algorithm
/// - Maintains a rolling ring buffer of the last `window` (prev, curr) pairs.
/// - Discretises each axis into `grid_size` bins over [range_min, range_max].
/// - Counts occupancy per cell with a ring-buffer eviction scheme.
/// - Identifies the dominant cell (highest count); fires
///   `on_attractor_shift(old_cell, new_cell)` when it changes.
/// - Also tracks occupancy entropy (spread of density) as a regime noise proxy.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_PHASE_SPACE` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenBiasPhaseSpace {
public:
    struct CellId {
        int row{0};  // bin index for bias_{t-1}
        int col{0};  // bin index for bias_t
    };

    struct Config {
        /// Rolling window of (prev, curr) pairs. Default 128.
        int window{128};

        /// Grid dimension (grid_size × grid_size cells). Default 8.
        int grid_size{8};

        /// Bias range for grid binning. Default -1..+1.
        double range_min{-1.0};
        double range_max{1.0};

        /// Fired when the dominant cell changes (outside lock).
        std::function<void(CellId old_cell, CellId new_cell)> on_attractor_shift;

        /// Fired when occupancy entropy drops below entropy_low_threshold
        /// (concentrated attractor, strong regime). Default 1.0.
        double entropy_low_threshold{1.0};
        std::function<void(double entropy)> on_attractor_concentrated;

        /// Fired when occupancy entropy rises above entropy_high_threshold
        /// (diffuse — noise regime). Default 2.5.
        double entropy_high_threshold{2.5};
        std::function<void(double entropy)> on_attractor_diffuse;
    };

    explicit TokenBiasPhaseSpace(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Dominant cell (row, col) — packed as row*grid_size+col in the atomic.
    [[nodiscard]] CellId dominant_cell() const noexcept {
        int packed = dominant_packed_.load(std::memory_order_relaxed);
        int g      = cfg_grid_size_.load(std::memory_order_relaxed);
        if (g < 1) g = 1;
        return {packed / g, packed % g};
    }

    /// Occupancy entropy of the phase-space grid (bits). High → diffuse/random.
    [[nodiscard]] double occupancy_entropy() const noexcept {
        return entropy_.load(std::memory_order_relaxed);
    }

    /// Total attractor-shift events fired.
    [[nodiscard]] uint64_t shift_events() const noexcept {
        return shift_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    int  bin_for(double v) const;
    void recompute_grid_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    // Ring buffer of packed cell ids for eviction
    std::vector<int> ring_;
    int ring_head_{0};
    int ring_fill_{0};

    // Flat grid[row*g+col] occupancy counts
    std::vector<int> grid_;

    double prev_bias_{0.0};
    bool   seeded_{false};

    int prev_dominant_{-1};

    std::atomic<int>      dominant_packed_{0};
    std::atomic<int>      cfg_grid_size_{8};
    std::atomic<double>   entropy_{0.0};
    std::atomic<uint64_t> shift_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
