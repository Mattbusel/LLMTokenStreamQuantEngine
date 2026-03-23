#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace llmquant {

/**
 * @brief Multi-model LLM ensemble aggregator with cross-model agreement scoring.
 *
 * Accepts bias signals from up to 8 named LLM provider channels in parallel.
 * For each update cycle the ensemble:
 *
 *  1. Computes the mean bias across all registered channels.
 *  2. Computes the cross-model agreement score:
 *       agreement = 1.0 - mean(|bias_i - mean_bias|) / normalisation_range
 *     A value of 1.0 means all models agree exactly; 0.0 means maximum spread.
 *  3. Applies the agreement score as a confidence multiplier on the aggregated
 *     signal, producing a risk-adjusted final bias.
 *
 * ## Design constraints
 *
 * - Maximum 8 channels (compile-time constant `kMaxChannels`).
 * - Lock-free aggregation: channel bias values are stored as `std::atomic<uint64_t>`
 *   using bit-cast from `double`, avoiding a mutex on the hot update path.
 * - No exceptions on the hot path — all hot-path methods return `bool` or
 *   `std::optional`.
 * - Channel registration (add_channel / remove_channel) uses a lightweight
 *   spinlock only during the cold setup phase.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_MODEL_ENSEMBLE_ENABLED` (defined when
 * `LLMQUANT_ENABLE_MODEL_ENSEMBLE=ON` in CMake).
 *
 * Thread safety: update_channel() is fully lock-free; all other public methods
 * are protected by a lightweight spinlock and are safe to call from any thread.
 */
class ModelEnsemble {
public:
    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /// Maximum number of LLM provider channels supported.
    static constexpr std::size_t kMaxChannels = 8;

    // -----------------------------------------------------------------------
    // Data types
    // -----------------------------------------------------------------------

    /**
     * @brief Result produced by aggregate().
     */
    struct AggregateResult {
        double   mean_bias;       ///< Simple mean of all active channel biases.
        double   agreement;       ///< Cross-model agreement score ∈ [0, 1].
        double   final_signal;    ///< mean_bias × agreement (confidence-adjusted).
        uint32_t channels_active; ///< Number of channels that have posted a bias.
        uint64_t update_count;    ///< Total channel updates since construction.
    };

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Normalisation range for the agreement score denominator.
         *
         * The raw mean absolute deviation is divided by this value before
         * computing 1 - MAD/range.  Should be set to the maximum expected
         * bias range (e.g. 2.0 for biases in [-1, 1]).  Must be > 0.
         * Default 2.0.
         */
        double normalisation_range{2.0};

        /**
         * @brief Minimum number of active channels required before aggregate()
         * returns a result.  Returns std::nullopt if fewer channels have posted.
         * Default 1.
         */
        uint32_t min_channels{1};

        /**
         * @brief Called outside the internal lock whenever aggregate() produces
         * a result.  May be nullptr.
         */
        std::function<void(const AggregateResult&)> on_aggregate;
    };

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    explicit ModelEnsemble(Config cfg = Config{});
    ~ModelEnsemble() = default;

    ModelEnsemble(const ModelEnsemble&)            = delete;
    ModelEnsemble& operator=(const ModelEnsemble&) = delete;
    ModelEnsemble(ModelEnsemble&&)                 = delete;
    ModelEnsemble& operator=(ModelEnsemble&&)      = delete;

    // -----------------------------------------------------------------------
    // Channel management (cold path — uses spinlock)
    // -----------------------------------------------------------------------

    /**
     * @brief Register a named provider channel.
     *
     * @param name Unique channel identifier (max 31 characters; truncated if longer).
     * @return true if the channel was added; false if name is duplicate or the
     *         kMaxChannels limit has been reached.
     */
    bool add_channel(const std::string& name);

    /**
     * @brief Remove a previously registered channel.
     *
     * @param name Channel identifier passed to add_channel().
     * @return true if found and removed; false if not found.
     */
    bool remove_channel(const std::string& name);

    /**
     * @brief Return the number of currently registered channels.
     */
    [[nodiscard]] uint32_t channel_count() const noexcept;

    /**
     * @brief Return the slot index for the named channel, or -1 if not found.
     */
    [[nodiscard]] int find_channel(const std::string& name) const noexcept;

    // -----------------------------------------------------------------------
    // Hot path — lock-free bias update
    // -----------------------------------------------------------------------

    /**
     * @brief Post a new bias value for the named channel.
     *
     * Lock-free.  Bias is stored via an atomic bit-cast of the double to
     * uint64_t using std::atomic<uint64_t>::store(relaxed).
     *
     * @param name  Channel identifier.
     * @param bias  Directional bias value (typically ∈ [-1, 1]).
     * @return true if the channel was found and updated; false otherwise.
     */
    bool update_channel(const std::string& name, double bias) noexcept;

    /**
     * @brief Post a new bias value by pre-resolved channel slot index.
     *
     * Faster variant for callers that cache the slot from find_channel().
     *
     * @param slot  Index returned by find_channel().
     * @param bias  Directional bias value.
     * @return true if slot is valid and was updated; false if out of range.
     */
    bool update_channel_by_slot(int slot, double bias) noexcept;

    // -----------------------------------------------------------------------
    // Aggregation
    // -----------------------------------------------------------------------

    /**
     * @brief Compute and return the current ensemble aggregate.
     *
     * Reads all channel biases with relaxed atomics, computes mean, MAD-based
     * agreement score, and applies the score as a multiplier.
     *
     * @return AggregateResult if min_channels active channels have posted;
     *         std::nullopt otherwise.
     */
    [[nodiscard]] std::optional<AggregateResult> aggregate() const noexcept;

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /**
     * @brief Return the total number of update_channel() calls since construction.
     */
    [[nodiscard]] uint64_t total_updates() const noexcept {
        return update_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset all channel biases to zero and clear the active-channel mask.
     *
     * Channel registrations are preserved.
     */
    void reset() noexcept;

    /**
     * @brief Update the ensemble configuration.
     *
     * normalisation_range must be > 0; invalid values are silently ignored.
     */
    void update_config(const Config& cfg);

private:
    // -----------------------------------------------------------------------
    // Internal channel slot
    // -----------------------------------------------------------------------

    struct alignas(64) ChannelSlot {
        // Bias stored as bit-cast uint64_t for lock-free atomic access.
        std::atomic<uint64_t> bias_bits{0};
        // True once update_channel() has been called for this slot at least once.
        std::atomic<bool>     active{false};
        // Name stored as fixed-length array to avoid heap allocation.
        char name[32]{};
        // Padding to avoid false sharing between slots.
        char _pad[64 - sizeof(std::atomic<uint64_t>)
                     - sizeof(std::atomic<bool>)
                     - 32]{};
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Bit-cast helpers: reinterpret double bits as uint64_t and back.
    static uint64_t double_to_bits(double v) noexcept;
    static double   bits_to_double(uint64_t b) noexcept;

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    Config cfg_;

    std::array<ChannelSlot, kMaxChannels> slots_{};

    // Number of registered (named) slots.  Written only under the spinlock.
    std::atomic<uint32_t> num_channels_{0};

    // Total hot-path updates.
    std::atomic<uint64_t> update_count_{0};

    // Lightweight spinlock protecting add_channel / remove_channel.
    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;

    void spinlock_acquire() const noexcept;
    void spinlock_release() const noexcept;
};

} // namespace llmquant
