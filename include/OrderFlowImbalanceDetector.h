#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llmquant {

/**
 * @brief Detects buy/sell order flow imbalance from LLM token streams.
 *
 * Token-stream signals like "buyers stepping up", "ask lifted", "bid held",
 * "sellers dominating", "heavy selling" carry implicit information about
 * near-term order flow direction.  This detector maintains an EMA of signed
 * flow pressure and fires a callback when a significant imbalance is detected.
 *
 * Each token contributes a signed pressure weight:
 *  - Positive (buy) tokens push the EMA above zero.
 *  - Negative (sell) tokens push it below zero.
 *  - Neutral tokens add zero pressure.
 *
 * Token pressures are set via the built-in keyword dictionary (which covers
 * common order-flow language) and can be extended with `add_token_weight()`.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ORDER_FLOW_IMBALANCE` (default ON).
 *
 * ## Typical usage
 * @code
 * OrderFlowImbalanceDetector det;
 * det.set_imbalance_callback([](double imb, bool buy_heavy) {
 *     std::cout << "Imbalance=" << imb << (buy_heavy ? " BUY" : " SELL") << "\n";
 * });
 * for (auto& tok : stream) det.record(tok);
 * @endcode
 *
 * Thread safety: all public methods are thread-safe.
 */
class OrderFlowImbalanceDetector {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief EMA smoothing factor for the imbalance signal.
         *
         * Larger = faster response, noisier.  Default 0.1.
         */
        double ema_alpha{0.1};

        /**
         * @brief Imbalance level (absolute value) above which callbacks fire.
         *
         * Range (0, 1].  Default 0.6.
         */
        double imbalance_threshold{0.6};

        /**
         * @brief Minimum tokens recorded before callbacks can fire.
         *
         * Prevents spurious alerts from the first few tokens.  Default 10.
         */
        int min_tokens{10};

        /**
         * @brief Callback invoked when |imbalance| crosses the threshold.
         *
         * Parameters: (imbalance ∈ [-1,1], is_buy_heavy).
         * Called outside the internal lock; do not call back into the detector.
         */
        std::function<void(double imbalance, bool is_buy_heavy)> on_imbalance;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit OrderFlowImbalanceDetector(Config config = Config{});

    // -----------------------------------------------------------------------
    // Token intake
    // -----------------------------------------------------------------------

    /**
     * @brief Record a token from the LLM stream.
     *
     * Looks up the token in the pressure dictionary (case-insensitive prefix
     * match on the first recognised keyword), updates the EMA, and fires the
     * imbalance callback if the threshold is crossed.
     *
     * @param token Raw token string from the LLM stream.
     */
    void record(const std::string& token) noexcept;

    /**
     * @brief Record a token with an explicit signed pressure weight.
     *
     * Bypasses the keyword dictionary.  Useful for externally pre-weighted
     * tokens (e.g., from a downstream sentiment model).
     *
     * @param pressure Signed pressure in [-1, 1]; positive = buy side.
     */
    void record_pressure(double pressure) noexcept;

    // -----------------------------------------------------------------------
    // Custom dictionary
    // -----------------------------------------------------------------------

    /**
     * @brief Add or override the pressure weight for a keyword.
     *
     * The key is matched case-insensitively via exact string equality against
     * lowercased tokens.
     *
     * @param keyword  Lowercase keyword string.
     * @param pressure Signed pressure in [-1, 1].
     */
    void add_token_weight(const std::string& keyword, double pressure);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Current EMA imbalance in [-1, 1].
     *
     * Positive = buy-side pressure; negative = sell-side pressure.
     */
    [[nodiscard]] double imbalance() const noexcept;

    /**
     * @brief True when imbalance > threshold (buy-side dominant).
     */
    [[nodiscard]] bool is_buy_imbalance() const noexcept;

    /**
     * @brief True when imbalance < -threshold (sell-side dominant).
     */
    [[nodiscard]] bool is_sell_imbalance() const noexcept;

    /**
     * @brief Total tokens recorded (including zero-pressure tokens).
     */
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total imbalance events fired since last reset.
     */
    [[nodiscard]] uint64_t imbalance_events() const noexcept {
        return imbalance_events_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Config / reset
    // -----------------------------------------------------------------------

    /**
     * @brief Update configuration without resetting the EMA state.
     */
    void update_config(const Config& config);

    /**
     * @brief Get current configuration.
     */
    [[nodiscard]] Config get_config() const;

    /**
     * @brief Reset EMA and counters.
     */
    void reset();

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

    // -----------------------------------------------------------------------
    // Callback registration (alternative to Config::on_imbalance)
    // -----------------------------------------------------------------------

    /**
     * @brief Register the imbalance callback after construction.
     */
    void set_imbalance_callback(
        std::function<void(double, bool)> cb);

private:
    Config  config_;

    // Pressure lookup dictionary: lowercased keyword → signed weight.
    std::unordered_map<std::string, double> dict_;

    double ema_imbalance_{0.0};

    std::atomic<uint64_t> total_tokens_{0};
    std::atomic<uint64_t> imbalance_events_{0};
    std::atomic<double>   imbalance_snapshot_{0.0};

    mutable std::mutex mutex_;

    void build_default_dict();
    [[nodiscard]] double lookup_pressure(const std::string& token) const;
    void update_ema_locked(double pressure);
};

} // namespace llmquant
