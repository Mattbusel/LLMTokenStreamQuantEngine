#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

/**
 * @brief Portfolio-level risk heat aggregator across multiple signal streams.
 *
 * In a multi-instrument deployment (e.g. AAPL + MSFT + QQQ all feeding
 * separate TradeSignalEngine instances), individual per-engine risk gates
 * operate in isolation.  PortfolioHeatMonitor provides a cross-instrument
 * view by aggregating "heat units" contributed by each engine.
 *
 * ### Heat model
 * Each registered instrument contributes heat according to its current
 * absolute bias and position exposure.  Total portfolio heat is the sum of
 * per-instrument heat.  When heat exceeds configurable warn/critical levels
 * callbacks fire — allowing the caller to tighten risk across all engines.
 *
 * ### Correlation limiter
 * When two instruments' biases are simultaneously directional and
 * above `correlation_gate_threshold`, they are flagged as "correlated burst"
 * and count as double heat.  This prevents concentrating in highly correlated
 * names during volatile markets.
 *
 * ## Usage
 * ```cpp
 * PortfolioHeatMonitor heat;
 * heat.set_warn_callback([](double h) {
 *     spdlog::warn("[portfolio] heat={:.2f} — widening spreads", h);
 * });
 * // In each engine's signal callback:
 * heat.update_instrument("AAPL", signal.delta_bias_shift, position_size);
 * auto level = heat.heat_level();
 * if (level >= PortfolioHeatMonitor::Level::Critical)
 *     stop_all_engines();
 * ```
 *
 * ## Thread safety
 * All public methods are thread-safe.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_PORTFOLIO_HEAT` (default ON).
 */
class PortfolioHeatMonitor {
public:
    /** @brief Portfolio heat severity tier. */
    enum class Level {
        Cool,     ///< Heat within normal bounds.
        Warm,     ///< Approaching warn threshold.
        Hot,      ///< At warn threshold — log and consider action.
        Critical  ///< At critical threshold — shed risk.
    };

    using HeatCallback = std::function<void(double total_heat)>;

    /**
     * @brief Per-instrument snapshot for diagnostics.
     */
    struct InstrumentSnapshot {
        std::string  name;
        double       bias{0.0};
        double       position{0.0};
        double       heat_contribution{0.0};
        bool         correlated_burst{false};
    };

    /**
     * @brief Configuration.
     */
    struct Config {
        /** @brief Heat units per (|bias| * |position|). */
        double heat_scale{1.0};

        /** @brief Absolute bias above which directional correlation is counted. */
        double correlation_gate_threshold{0.3};

        /** @brief Heat at which level transitions to Hot. */
        double warn_heat{5.0};

        /** @brief Heat at which level transitions to Critical. */
        double critical_heat{10.0};

        /** @brief If true, correlated pairs count 2× heat. */
        bool double_correlated_heat{true};
    };

    explicit PortfolioHeatMonitor(Config cfg = Config{});

    // Non-copyable.
    PortfolioHeatMonitor(const PortfolioHeatMonitor&)            = delete;
    PortfolioHeatMonitor& operator=(const PortfolioHeatMonitor&) = delete;

    /**
     * @brief Update an instrument's current bias and position size.
     *
     * If the instrument is unknown it is added automatically.
     * Thread-safe.
     *
     * @param name       Instrument identifier (e.g. "AAPL").
     * @param bias       Current signal bias (signed, e.g. delta_bias_shift).
     * @param position   Absolute position size (notional or shares).
     */
    void update_instrument(const std::string& name, double bias, double position);

    /**
     * @brief Remove an instrument from the monitor.
     *
     * Thread-safe.
     */
    void remove_instrument(const std::string& name);

    /**
     * @brief Compute and return total portfolio heat.
     *
     * Thread-safe.
     */
    [[nodiscard]] double total_heat() const;

    /**
     * @brief Current heat severity tier.
     *
     * Thread-safe.
     */
    [[nodiscard]] Level heat_level() const;

    /**
     * @brief Human-readable name for a Level.
     */
    [[nodiscard]] static const char* level_name(Level l) noexcept;

    /**
     * @brief Snapshot of all registered instruments.
     *
     * Thread-safe.
     */
    [[nodiscard]] std::vector<InstrumentSnapshot> snapshot() const;

    /**
     * @brief Number of registered instruments.
     */
    [[nodiscard]] size_t instrument_count() const;

    // ----------------------------------------------------------------
    // Callbacks
    // ----------------------------------------------------------------

    /** @brief Fired once when heat crosses the warn threshold upward. */
    void set_warn_callback(HeatCallback cb);
    /** @brief Fired once when heat crosses the critical threshold upward. */
    void set_critical_callback(HeatCallback cb);
    /** @brief Fired once when heat drops below warm threshold (recovery). */
    void set_recovery_callback(HeatCallback cb);

    /**
     * @brief Update configuration at runtime.
     *
     * Resets heat level state.
     */
    void update_config(const Config& cfg);

    /**
     * @brief Reset all instruments and heat state.
     */
    void reset();

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    struct InstrumentState {
        double bias{0.0};
        double position{0.0};
    };

    [[nodiscard]] double compute_heat_locked() const;
    void check_level_transition(double new_heat);

    Config cfg_;
    mutable std::mutex mtx_;

    std::unordered_map<std::string, InstrumentState> instruments_;

    Level current_level_{Level::Cool};

    HeatCallback warn_cb_;
    HeatCallback critical_cb_;
    HeatCallback recovery_cb_;
};

} // namespace llmquant
