#pragma once
/// TokenInfluenceDecay — exponential influence decay tracker for positive vs negative bias tokens.
/// Maintains separate EMA accumulators for positive and negative bias observations.
/// The net influence = pos_ema - |neg_ema|. Fires on_positive_dominant / on_negative_dominant
/// when one side's influence persistently exceeds the other by dominance_threshold.
///
/// Feature flag: LLMQUANT_ENABLE_INFLUENCE_DECAY / LLMQUANT_INFLUENCE_DECAY_ENABLED

#ifdef LLMQUANT_INFLUENCE_DECAY_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace llmquant {

class TokenInfluenceDecay {
public:
    struct Config {
        double alpha_pos            = 0.15;  ///< EMA decay for positive bias tokens
        double alpha_neg            = 0.15;  ///< EMA decay for negative bias tokens
        double dominance_threshold  = 0.02;  ///< net influence above this → on_positive_dominant
        int    min_samples          = 5;     ///< minimum before firing events

        /// Fires on transition to positive-dominant state.
        std::function<void(double net_influence)> on_positive_dominant;
        /// Fires on transition to negative-dominant state.
        std::function<void(double net_influence)> on_negative_dominant;
    };

    explicit TokenInfluenceDecay(Config cfg = {});

    void record(double bias);

    double pos_influence()  const noexcept; ///< positive EMA accumulator
    double neg_influence()  const noexcept; ///< negative EMA accumulator (magnitude)
    double net_influence()  const noexcept; ///< pos_influence - neg_influence

    bool   is_positive_dominant() const noexcept;
    bool   is_negative_dominant() const noexcept;

    std::size_t positive_dominant_events() const noexcept;
    std::size_t negative_dominant_events() const noexcept;
    std::size_t total_records()            const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    double ema_pos_ = 0.0;
    double ema_neg_ = 0.0;
    bool   prev_pos_dom_ = false;
    bool   prev_neg_dom_ = false;

    std::atomic<double>      pos_inf_{0.0};
    std::atomic<double>      neg_inf_{0.0};
    std::atomic<double>      net_inf_{0.0};
    std::atomic<bool>        pos_dom_{false};
    std::atomic<bool>        neg_dom_{false};
    std::atomic<std::size_t> pos_dom_events_{0};
    std::atomic<std::size_t> neg_dom_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_INFLUENCE_DECAY_ENABLED
