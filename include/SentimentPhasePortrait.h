#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Phase-portrait analyser for the (bias, velocity) dynamical system.
 *
 * Treats the sentiment stream as a 2-D continuous dynamical system where
 * bias is the state and velocity (EMA of first derivative) is the rate of
 * change.  The phase plane is discretised into an N×N grid of *cells*; each
 * observation plots a point in this space.
 *
 * ## What it measures
 * - **Dwell time** per cell — how long the system spends in each region.
 * - **Attractor detection** — a cell qualifies as an attractor if its dwell
 *   fraction exceeds `attractor_threshold` AND the system returns to it most
 *   often from its neighbours.
 * - **Cycle detection** — a simple period-2 oscillation is flagged when the
 *   two most-visited cells alternate in the recent visit history.
 * - **Divergence index** — fraction of recent steps that moved *away* from
 *   the dominant attractor; high values indicate an unstable / trending regime.
 *
 * ## Design
 * - O(1) per `record()` call; all heavy computation is deferred to accessors.
 * - Thread-safe: atomic outputs for hot-path reads; mutex for grid updates.
 * - Fully resettable and reconfigurable at runtime.
 */
class SentimentPhasePortrait {
public:
    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    static constexpr int kDefaultGridSize = 8;   ///< Default NxN grid resolution.

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    struct Config {
        int    grid_size           = kDefaultGridSize; ///< N for the N×N phase grid.
        double bias_min            = -1.0;  ///< Bias axis lower bound.
        double bias_max            =  1.0;  ///< Bias axis upper bound.
        double vel_min             = -0.5;  ///< Velocity axis lower bound.
        double vel_max             =  0.5;  ///< Velocity axis upper bound.
        double velocity_alpha      = 0.20;  ///< EMA coefficient for velocity.
        double attractor_threshold = 0.15;  ///< Min dwell fraction to qualify as attractor.
        int    cycle_window        = 20;    ///< Recent-visit window for cycle detection.
        int    min_visits          = 10;    ///< Minimum total visits before reporting.

        /// Fired when the dominant attractor cell changes (row, col).
        std::function<void(int, int)> on_attractor_change;
        /// Fired when a period-2 cycle is first confirmed.
        std::function<void(int, int, int, int)> on_cycle_detected;  // (r1,c1,r2,c2)
    };

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    SentimentPhasePortrait();
    explicit SentimentPhasePortrait(const Config& cfg);

    // -------------------------------------------------------------------------
    // Core
    // -------------------------------------------------------------------------

    /**
     * @brief Record a new bias observation; velocity is derived from EMA diff.
     * @param bias  Normalised sentiment bias (expected within [bias_min, bias_max]).
     */
    void record(double bias);

    // -------------------------------------------------------------------------
    // Accessors (lock-free hot path)
    // -------------------------------------------------------------------------

    /// Current grid row of the trajectory (bias axis).
    int current_row() const noexcept;
    /// Current grid column of the trajectory (velocity axis).
    int current_col() const noexcept;
    /// Current EMA-smoothed velocity value.
    double velocity() const noexcept;
    /// Total observations recorded.
    uint64_t total_records() const noexcept;
    /// Number of cell transitions in the phase plane.
    uint64_t cell_transitions() const noexcept;

    // -------------------------------------------------------------------------
    // Accessors (mutex-guarded; slower)
    // -------------------------------------------------------------------------

    /// Dwell fraction for cell (row, col) — fraction of total visits.
    double dwell_fraction(int row, int col) const noexcept;
    /// Row of the dominant attractor (-1 if insufficient data).
    int attractor_row() const noexcept;
    /// Column of the dominant attractor (-1 if insufficient data).
    int attractor_col() const noexcept;
    /// True if a period-2 cycle has been detected in the recent visit window.
    bool cycle_detected() const noexcept;
    /// Divergence index [0,1]: fraction of recent steps moving away from attractor.
    double divergence_index() const noexcept;

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    /** Reset all accumulated state (counts, history, EMAs). */
    void reset();
    /** Replace config and reset state. */
    void update_config(const Config& cfg);

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /** Compact JSON with key phase-portrait metrics. */
    std::string to_stats_json() const;

private:
    Config cfg_;
    mutable std::mutex mutex_;

    // Phase grid — counts_[row][col] = visit count
    std::vector<std::vector<double>> counts_;

    // Recent-visit ring buffer for cycle detection
    std::vector<int> visit_history_;  // encoded as row*N+col
    int   visit_head_  = 0;
    int   visit_count_ = 0;

    // EMA state
    double bias_ema_   = 0.0;
    double prev_bias_ema_ = 0.0;
    bool   seeded_     = false;

    int  current_row_val_ = 0;
    int  current_col_val_ = 0;
    int  attractor_row_val_ = -1;
    int  attractor_col_val_ = -1;

    // Atomic outputs for lock-free reads
    std::atomic<int>      row_out_{0};
    std::atomic<int>      col_out_{0};
    std::atomic<double>   vel_out_{0.0};
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> transitions_{0};

    // Helpers
    int coord_to_bin(double value, double lo, double hi, int n) const noexcept;
    void recompute_attractor_locked();
    void push_visit_locked(int encoded);
};

}  // namespace llmquant
