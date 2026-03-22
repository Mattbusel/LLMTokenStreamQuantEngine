#pragma once

/// @file SentimentTopologyMapper.h
/// @brief Maps the bias stream's topological structure using persistent
///        homology proxies — specifically zero-dimensional persistence
///        (connected components / "islands") in the bias level-set.
///
/// ## Motivation
/// Persistent homology provides a multi-scale shape descriptor of a signal.
/// Zero-dimensional persistence counts how many distinct "peaks" (connected
/// components at a given threshold) survive across thresholds.  For a bias
/// stream this approximates: how many distinct sentiment clusters coexist?
///   - One persistent island: single, coherent sentiment state.
///   - Multiple islands: fragmented / conflicting sentiment structure.
///
/// ## Algorithm (simplified sublevel-set persistence on a 1-D signal)
/// Over a rolling window of length `window`:
///   1. Sort unique values as thresholds ascending.
///   2. Walk thresholds bottom-up; track when adjacent samples merge into
///      the same component (union-find).  Each merge kills one component
///      (death = current threshold, birth = component creation threshold).
///   3. Persistence of component i = death_i − birth_i.
///   4. Count components with persistence > min_persistence.
///   5. Compute total persistence (Betti-0 proxy) = Σ persistence_i.
///
/// Fires `on_topology_change(old_count, new_count)` when the component count
/// changes, signalling a structural shift in the sentiment landscape.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TOPOLOGY_MAPPER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SentimentTopologyMapper {
public:
    struct Config {
        /// Rolling window length. Default 64.
        int window{64};

        /// Minimum persistence (death - birth) for a component to be counted. Default 0.05.
        double min_persistence{0.05};

        /// Minimum fill before computing. Default 16.
        int min_samples{16};

        /// Fired when component count changes (outside lock).
        std::function<void(int old_count, int new_count)> on_topology_change;
    };

    explicit SentimentTopologyMapper(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    void record(double bias);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Number of persistent sentiment components (Betti-0 count).
    [[nodiscard]] int component_count() const noexcept {
        return components_.load(std::memory_order_relaxed);
    }

    /// Total persistence Σ(death_i - birth_i) for all persistent components.
    [[nodiscard]] double total_persistence() const noexcept {
        return total_pers_.load(std::memory_order_relaxed);
    }

    /// Total topology-change events fired.
    [[nodiscard]] uint64_t topology_events() const noexcept {
        return topo_events_.load(std::memory_order_relaxed);
    }

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void compute_persistence_locked();

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    int prev_components_{-1};

    std::atomic<int>      components_{0};
    std::atomic<double>   total_pers_{0.0};
    std::atomic<uint64_t> topo_events_{0};
    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
