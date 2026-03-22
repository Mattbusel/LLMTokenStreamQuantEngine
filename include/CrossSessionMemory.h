#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace llmquant {

/**
 * @brief Persists adaptive module state across process restarts.
 *
 * Without persistence, every restart loses the Kelly win/loss history and
 * drawdown state accumulated in the previous session.  The first session after
 * a restart then behaves as if it has never seen any trades — the Kelly sizer
 * uses its cold-start pass-through behaviour, and the drawdown protector has
 * no baseline to reference.  Under volatile market conditions this can produce
 * oversized positions exactly when caution is most needed.
 *
 * CrossSessionMemory stores a small JSON snapshot to disk on shutdown and
 * restores it on startup, giving the engine a warm start.
 *
 * ## What is persisted
 * - Kelly sizer: total outcomes, win count, win/loss EMA values.
 * - Drawdown protector: peak P&L, cumulative P&L.
 * - Metadata: save timestamp, session counter, engine version.
 *
 * ## File format
 * A single JSON file (default `llmquant_session.json` in the working
 * directory).  Human-readable so operators can inspect or reset it.
 *
 * ## Usage
 * ```cpp
 * CrossSessionMemory mem;
 * mem.load();                         // restore on startup
 * mem.restore_kelly(kelly_sizer);
 * // ... run engine ...
 * mem.capture_kelly(kelly_sizer);
 * mem.save();                         // persist on shutdown
 * ```
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_CROSS_SESSION_MEMORY` (default ON).
 */
class CrossSessionMemory {
public:
    struct Config {
        /** @brief Path to the state file.  Default "llmquant_session.json". */
        std::string path{"llmquant_session.json"};

        /**
         * @brief If true, a missing or corrupt file is silently ignored
         * (cold start).  If false, load() returns false on error.
         * Default true.
         */
        bool ignore_missing{true};

        /** @brief Called when the file is loaded successfully. */
        std::function<void(const std::string& path, uint64_t session)> on_load;
        /** @brief Called when state is saved successfully. */
        std::function<void(const std::string& path, uint64_t session)> on_save;
    };

    /**
     * @brief In-memory representation of persisted Kelly state.
     *
     * These values are written/read by capture_kelly() / restore_kelly()
     * and do not depend on the KellyPositionSizer header being available.
     */
    struct KellyState {
        uint64_t total_outcomes{0};
        uint64_t win_count{0};
        double   win_rate_ema{0.5};
        double   avg_win_ema{0.01};
        double   avg_loss_ema{0.01};
    };

    /**
     * @brief In-memory representation of persisted drawdown state.
     */
    struct DrawdownState {
        double cumulative_pnl{0.0};
        double peak_pnl{0.0};
    };

    explicit CrossSessionMemory(Config cfg = Config{}) : cfg_(std::move(cfg)) {}

    // Non-copyable.
    CrossSessionMemory(const CrossSessionMemory&)            = delete;
    CrossSessionMemory& operator=(const CrossSessionMemory&) = delete;

    /**
     * @brief Load state from disk.
     *
     * @return true on success or if the file is missing and ignore_missing=true.
     *         false if the file exists but is corrupt.
     */
    bool load();

    /**
     * @brief Write current state to disk atomically (write temp file, rename).
     *
     * @return true on success.
     */
    bool save();

    // ----------------------------------------------------------------
    // Kelly state capture / restore
    // ----------------------------------------------------------------

    /**
     * @brief Copy Kelly state from the loaded snapshot into @p state.
     * @return false if no Kelly state was loaded (cold start).
     */
    bool restore_kelly(KellyState& state) const;

    /**
     * @brief Capture Kelly state into the in-memory snapshot for save().
     */
    void capture_kelly(const KellyState& state);

    // ----------------------------------------------------------------
    // Drawdown state capture / restore
    // ----------------------------------------------------------------

    /**
     * @brief Copy drawdown state from the loaded snapshot.
     * @return false if no drawdown state was loaded.
     */
    bool restore_drawdown(DrawdownState& state) const;

    /**
     * @brief Capture drawdown state for save().
     */
    void capture_drawdown(const DrawdownState& state);

    // ----------------------------------------------------------------
    // Session metadata
    // ----------------------------------------------------------------

    /** @brief Session counter (incremented on each save). */
    [[nodiscard]] uint64_t session_number() const noexcept { return session_; }

    /** @brief ISO-8601 timestamp of the last save (empty if none). */
    [[nodiscard]] const std::string& last_save_time() const noexcept { return last_save_time_; }

    /** @brief Returns true if a snapshot was successfully loaded. */
    [[nodiscard]] bool has_loaded_state() const noexcept { return loaded_; }

    /**
     * @brief Update configuration.
     */
    void update_config(const Config& cfg) { cfg_ = cfg; }

    /**
     * @brief Serialise metadata to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    // Loaded state flags.
    bool loaded_{false};
    bool has_kelly_{false};
    bool has_drawdown_{false};

    KellyState    kelly_state_;
    DrawdownState drawdown_state_;

    uint64_t    session_{0};
    std::string last_save_time_;

    // Simple JSON helpers (no external dependency).
    [[nodiscard]] static std::string make_iso8601_now();
    [[nodiscard]] bool parse_json(const std::string& json);
    [[nodiscard]] std::string to_json() const;
};

} // namespace llmquant
