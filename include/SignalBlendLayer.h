#pragma once

#include "TradeSignalEngine.h"  // TradeSignal

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Multi-source trade signal blending layer.
 *
 * Accepts TradeSignal contributions from multiple named sources (e.g.
 * one per subreddit LLM stream), normalises their weights, and emits a
 * single blended signal whenever a quorum of sources has updated.
 *
 * ## Blending modes
 * | Mode              | delta_bias_shift computation                                |
 * |-------------------|-------------------------------------------------------------|
 * | Equal             | Simple average across all registered sources               |
 * | ConfidenceWeighted| Weighted by each source's current signal.confidence         |
 * | CustomWeighted    | Weighted by caller-supplied per-source weight               |
 *
 * Volatility, spread, and confidence of the blended output are also
 * averaged (confidence-weighted for ConfidenceWeighted mode, plain
 * average otherwise).
 *
 * ## Quorum
 * A blend is produced only when at least `min_sources` have contributed
 * a signal since the last blend was emitted.  This prevents a single
 * active source from dominating when others are silent.
 *
 * ## Thread safety
 * `push_signal()` and all accessors are safe to call from multiple threads
 * concurrently.  The blend callback is invoked with the internal mutex held
 * so it must be short and must NOT call back into SignalBlendLayer.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_SIGNAL_BLEND` (default ON).
 */
class SignalBlendLayer {
public:
    /**
     * @brief Blending strategy.
     */
    enum class BlendMode {
        Equal,             ///< All sources have equal weight.
        ConfidenceWeighted,///< Weight proportional to signal.confidence.
        CustomWeighted,    ///< Weight supplied per source via set_source_weight().
    };

    /**
     * @brief Configuration for the blend layer.
     */
    struct Config {
        /**
         * @brief Minimum number of distinct sources that must have pushed a
         *        signal before a blend is emitted.
         *
         * Default 1 = emit as soon as any source updates (pass-through mode).
         * Set to the number of registered sources to require a full quorum.
         */
        int min_sources{1};

        /**
         * @brief Blending strategy applied to produce the output signal.
         *
         * Default: ConfidenceWeighted.
         */
        BlendMode mode{BlendMode::ConfidenceWeighted};

        /**
         * @brief Maximum staleness of a source signal in milliseconds.
         *
         * Sources whose last push is older than this threshold are excluded
         * from the blend.  0 = never expire (default).
         */
        int staleness_ms{0};

        /**
         * @brief When true, reset the per-source pending flags after each
         *        blend so that every blend requires a fresh quorum update.
         *
         * When false, sources retain their "has updated" flag across blends,
         * so min_sources acts as a one-time threshold.
         * Default true.
         */
        bool reset_pending_after_blend{true};
    };

    /**
     * @brief Callback invoked with the blended signal.
     *
     * Signature: void(const TradeSignal& blended)
     */
    using BlendCallback = std::function<void(const TradeSignal&)>;

    /**
     * @brief Per-source statistics snapshot.
     */
    struct SourceStats {
        std::string name;
        uint64_t    pushes{0};          ///< Total signals pushed by this source.
        double      last_confidence{0}; ///< Most recent signal confidence.
        double      last_bias{0};       ///< Most recent delta_bias_shift.
        double      weight{1.0};        ///< Current effective weight.
    };

    /**
     * @brief Construct the blend layer with the given configuration.
     *
     * @param config Quorum, mode, and staleness parameters.
     */
    explicit SignalBlendLayer(Config config = Config{});

    /**
     * @brief Register a named source.
     *
     * Sources must be registered before push_signal() is called.
     * Registering the same name twice is a no-op.
     * Thread-safe.
     *
     * @param source_name Unique identifier for this signal source.
     * @param weight      Initial custom weight (used in CustomWeighted mode).
     */
    void register_source(const std::string& source_name, double weight = 1.0);

    /**
     * @brief Update the custom weight for a registered source.
     *
     * Only meaningful in CustomWeighted mode.  Thread-safe.
     *
     * @param source_name The source to update.
     * @param weight      New weight (must be >= 0).
     */
    void set_source_weight(const std::string& source_name, double weight);

    /**
     * @brief Register the callback invoked each time a blended signal is produced.
     *
     * @param cb Callable invoked with the blended TradeSignal.
     */
    void set_blend_callback(BlendCallback cb);

    /**
     * @brief Submit a new signal from a named source.
     *
     * Updates the source's cached signal, marks it as pending, checks
     * the quorum, and — if the quorum is met — invokes the blend callback
     * with a freshly computed blended signal.
     *
     * Thread-safe.
     *
     * @param source_name The registered source identifier.
     * @param signal      The signal to blend in.
     * @return true if a blended signal was emitted this call; false otherwise.
     */
    [[nodiscard]] bool push_signal(const std::string& source_name,
                                   const TradeSignal& signal);

    /**
     * @brief Force an immediate blend over all non-stale sources, regardless
     *        of quorum.  Does not reset pending flags.
     *
     * Useful for flushing the blend layer at shutdown or for testing.
     * Thread-safe.
     *
     * @return The blended signal; fields are zero if no sources are registered.
     */
    [[nodiscard]] TradeSignal force_blend() const;

    /**
     * @brief Return per-source statistics snapshots.
     *
     * Thread-safe.
     */
    [[nodiscard]] std::vector<SourceStats> get_source_stats() const;

    /**
     * @brief Total number of blended signals emitted since construction.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t blends_emitted() const noexcept {
        return blends_emitted_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of signals pushed (across all sources) since construction.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t total_pushes() const noexcept {
        return total_pushes_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Update the blend configuration at runtime.
     *
     * The registered sources and their cached signals are preserved.
     * Thread-safe.
     *
     * @param config New configuration.
     */
    void update_config(const Config& config);

    /**
     * @brief Serialise layer statistics to a JSON object string.
     *
     * Fields: blends_emitted, total_pushes, source_count, mode, min_sources.
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

    /**
     * @brief Reset all source signals and pending flags.
     *
     * Source registrations and weights are preserved.
     * Thread-safe.
     */
    void reset();

private:
    struct SourceState {
        double      weight{1.0};
        uint64_t    pushes{0};
        bool        pending{false};
        bool        registered{true};
        TradeSignal last_signal{};
        std::chrono::steady_clock::time_point last_push_time{};
    };

    [[nodiscard]] TradeSignal compute_blend_locked() const;
    [[nodiscard]] int count_pending_locked() const;
    [[nodiscard]] bool is_stale_locked(const SourceState& s) const;

    mutable std::mutex mutex_;
    Config             config_;
    BlendCallback      blend_cb_;

    std::unordered_map<std::string, SourceState> sources_;

    std::atomic<uint64_t> blends_emitted_{0};
    std::atomic<uint64_t> total_pushes_{0};
};

} // namespace llmquant
