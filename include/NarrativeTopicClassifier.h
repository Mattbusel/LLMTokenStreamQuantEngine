#pragma once

/// @file NarrativeTopicClassifier.h
/// @brief Online bag-of-centroids topic classifier that buckets incoming tokens
///        into macro topics and adjusts per-topic signal weights.
///
/// ## Motivation
/// Not all sentiment is equal: an "earnings beat" has very different signal
/// persistence than a "geopolitical risk" headline.  This classifier maintains
/// a small set of named topics, each with a centroid in token-weight space,
/// and assigns each incoming token to its nearest centroid using cosine-like
/// distance on the token's `(weight, position_in_sequence)` feature vector.
/// Per-topic signal multipliers allow the downstream TradeSignalEngine to
/// amplify or suppress signals based on which narrative is dominant.
///
/// ## Algorithm
/// 1. Topics are registered with `register_topic(name, weight_centroid)`.
/// 2. `classify(token, weight)` finds the nearest topic and increments its
///    running frequency count.
/// 3. `dominant_topic()` returns the topic with the highest smoothed frequency.
/// 4. `signal_multiplier(topic)` returns the user-configured per-topic weight.
/// 5. `on_topic_change` callback fires when the dominant topic shifts.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_NARRATIVE_CLASSIFIER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeTopicClassifier {
public:
    static constexpr int kMaxTopics = 16;

    struct TopicDef {
        std::string name;
        double      weight_centroid{0.0}; ///< Center of this topic in |weight| space.
        double      signal_multiplier{1.0}; ///< Downstream signal gain for this topic.
    };

    struct Config {
        /// EMA alpha for per-topic frequency smoothing. Default 0.1.
        double freq_alpha{0.1};

        /// EMA alpha for dominant-topic smoothing (prevents rapid flipping). Default 0.05.
        double dominant_alpha{0.05};

        /// Minimum classified tokens before dominant_topic is considered reliable. Default 20.
        int min_warmup{20};

        /// Fired when the dominant topic changes.
        /// Parameters: (old_topic_name, new_topic_name, new_topic_multiplier).
        std::function<void(const std::string& old_topic,
                           const std::string& new_topic,
                           double multiplier)> on_topic_change;
    };

    explicit NarrativeTopicClassifier(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Setup
    // -----------------------------------------------------------------------

    /// Register a topic.  Must be called before `classify`.
    /// @return false if topic limit reached (kMaxTopics).
    bool register_topic(const TopicDef& def);

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Classify one token and update topic frequencies.
    /// @return Index of the assigned topic, or -1 if no topics registered.
    int classify(const std::string& token, double weight);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Name of the currently dominant topic, or empty if undetermined.
    [[nodiscard]] std::string dominant_topic() const;

    /// Signal multiplier for the dominant topic (1.0 if undetermined).
    [[nodiscard]] double dominant_multiplier() const noexcept {
        return dominant_mult_.load(std::memory_order_relaxed);
    }

    /// Signal multiplier for a named topic (1.0 if not found).
    [[nodiscard]] double signal_multiplier(const std::string& topic_name) const;

    /// Smoothed frequency estimate for a topic index ∈ [0, n_topics).
    [[nodiscard]] double topic_frequency(int idx) const noexcept;

    /// Total tokens classified.
    [[nodiscard]] uint64_t classified_count() const noexcept {
        return classified_count_.load(std::memory_order_relaxed);
    }

    /// JSON diagnostics snapshot.
    [[nodiscard]] std::string to_stats_json() const;

    /// Reset frequencies and state (keeps topic definitions).
    void reset();

    /// Update configuration (resets state, keeps topic definitions).
    void update_config(const Config& cfg);

private:
    struct TopicState {
        TopicDef def;
        double   freq_ema{0.0};
    };

    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<TopicState> topics_;
    int    dominant_idx_{-1};
    int    classified_{0};

    std::atomic<double>   dominant_mult_{1.0};
    std::atomic<uint64_t> classified_count_{0};
};

} // namespace llmquant
