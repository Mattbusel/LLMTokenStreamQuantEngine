#pragma once

/// @file ReinforcementSignalWeighter.h
/// @brief UCB1 multi-armed bandit for adaptive signal component weighting.
///
/// ## Motivation
/// In a multi-model ensemble (several LLMs or signal generators), not all arms
/// contribute equally to PnL.  Static weights are a guess; this module lets the
/// system learn which signal arms have been producing reliable rewards and
/// concentrates weight on them while maintaining exploration of undersampled arms.
///
/// ## Algorithm (UCB1)
/// For arm i after T total updates:
///   UCB_i = mean_reward_i + c * sqrt(ln(T) / n_i)
///
/// where c is the exploration constant (sqrt(2) by default).
/// Arm weights are the softmax of UCB scores, so all arms receive some weight.
/// When the dominant arm (highest UCB) changes, `on_dominant_arm_change` fires.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_RL_SIGNAL_WEIGHTER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class ReinforcementSignalWeighter {
public:
    struct ArmSpec {
        std::string name;
        double initial_weight{1.0};
    };

    struct Config {
        /// Arms to track.  Must be non-empty.
        std::vector<ArmSpec> arms;

        /// UCB exploration constant.  Higher → more exploration. Default sqrt(2).
        double exploration_c{1.4142135623730951};

        /// Softmax temperature.  Higher → more uniform weights. Default 1.0.
        double softmax_temp{1.0};

        /// Fired (outside lock) when the dominant arm (highest UCB score) changes.
        /// Parameter: new dominant arm name.
        std::function<void(const std::string& arm_name)> on_dominant_arm_change;
    };

    explicit ReinforcementSignalWeighter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a reward observation for the named arm.
    /// Unknown arm names are silently ignored.
    void update(const std::string& arm_name, double reward);

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Softmax-normalised weight for the named arm (0 if unknown or no updates).
    [[nodiscard]] double weight(const std::string& arm_name) const;

    /// Arm with the highest UCB score.
    [[nodiscard]] std::string dominant_arm() const;

    /// Total reward updates across all arms.
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of times the dominant arm changed.
    [[nodiscard]] uint64_t dominant_arm_changes() const noexcept {
        return dom_changes_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    struct ArmState {
        std::string name;
        double      mean_reward{0.0};
        double      sum_reward{0.0};
        uint64_t    pulls{0};
        double      weight{1.0};
        double      ucb_score{0.0};
    };

    mutable std::mutex mutex_;
    Config             cfg_;

    std::vector<ArmState> arms_;
    int                   dominant_idx_{-1};

    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> dom_changes_{0};

    void recompute_weights_locked();
    int  find_arm(const std::string& name) const;
};

} // namespace llmquant
