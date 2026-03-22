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
 * @brief Weighted vote aggregation across multiple named signal strategies.
 *
 * In a real-world sentiment trading system, several independent sub-models may
 * produce separate trade signal votes (e.g. Reddit-NLP, earnings transcript,
 * macro-news, Twitter firehose).  This class aggregates those votes into a
 * single consensus output using one of three modes:
 *
 *  - **Majority** — sign of Σ weight_i × sign(vote_i)
 *  - **ConfidenceWeighted** — Σ weight_i × vote_i  (weighted mean)
 *  - **RankWeighted** — votes ranked by |vote_i|; higher rank → more influence
 *
 * Each sub-model is registered with a name and a fixed weight.  When all
 * registered models have submitted their votes for the current epoch, the voter
 * produces a consensus and fires `on_consensus`.  Partial votes (not all models
 * present) produce an interim result based on the received votes only.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ENSEMBLE_VOTER` (default ON).
 *
 * Thread safety: all public methods are thread-safe.
 */
class EnsembleSignalVoter {
public:
    // -----------------------------------------------------------------------
    // Data types
    // -----------------------------------------------------------------------

    enum class VotingMode {
        Majority,          ///< sign(Σ weight × sign(vote))
        ConfidenceWeighted,///< Σ weight × vote  (weighted average)
        RankWeighted,      ///< rank-weighted: rank 1 (highest |vote|) gets n-1 points
    };

    struct ConsensusResult {
        double   consensus;       ///< Aggregated signal ∈ [-1, 1].
        double   agreement;       ///< Fraction of voters on same side as consensus.
        int      votes_received;  ///< Number of votes in this epoch.
        int      votes_total;     ///< Total registered voters.
        uint64_t epoch;           ///< Monotonic epoch counter.
    };

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Aggregation mode.  Default ConfidenceWeighted.
         */
        VotingMode mode{VotingMode::ConfidenceWeighted};

        /**
         * @brief If true, emit a consensus result whenever a vote arrives,
         * even if not all models have voted.  Default true.
         */
        bool allow_partial{true};

        /**
         * @brief Minimum agreement fraction required to consider the consensus
         * "strong" (reported via is_strong_consensus()).  Default 0.6.
         */
        double strong_threshold{0.6};

        /**
         * @brief Fired when a consensus result is produced.
         *
         * Called outside the internal lock.
         */
        std::function<void(const ConsensusResult&)> on_consensus;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit EnsembleSignalVoter(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Voter registration
    // -----------------------------------------------------------------------

    /**
     * @brief Register a named sub-model with a relative weight.
     *
     * @param name   Unique model identifier.
     * @param weight Non-negative weight (normalized internally).
     */
    void register_voter(const std::string& name, double weight = 1.0);

    // -----------------------------------------------------------------------
    // Voting
    // -----------------------------------------------------------------------

    /**
     * @brief Submit a vote from the named model.
     *
     * @param name  Model name passed to register_voter().
     * @param vote  Signed signal value (typically ∈ [-1, 1]).
     * @return Consensus result (may be partial if allow_partial=true).
     */
    ConsensusResult vote(const std::string& name, double vote_value);

    /**
     * @brief Advance to the next epoch, clearing all current votes.
     *
     * Call this at the start of each new signal generation cycle.
     */
    void next_epoch();

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /**
     * @brief Most recent consensus result.
     */
    [[nodiscard]] ConsensusResult last_consensus() const;

    /**
     * @brief True when agreement ≥ strong_threshold.
     */
    [[nodiscard]] bool is_strong_consensus() const noexcept;

    /**
     * @brief Number of registered voters.
     */
    [[nodiscard]] int voter_count() const noexcept;

    /**
     * @brief Total consensus events fired.
     */
    [[nodiscard]] uint64_t consensus_count() const noexcept {
        return consensus_count_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /**
     * @brief Clear all votes and reset epoch counter.
     */
    void reset();

    /**
     * @brief Update configuration without clearing voter registrations.
     */
    void update_config(const Config& cfg);

    /**
     * @brief JSON diagnostics snapshot.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    struct VoterState {
        std::string name;
        double      weight{1.0};
        double      current_vote{0.0};
        bool        has_voted{false};
    };

    Config cfg_;
    std::vector<VoterState>                      voters_;
    std::unordered_map<std::string, std::size_t> index_;
    uint64_t                                     epoch_{0};

    ConsensusResult last_result_{};
    std::atomic<uint64_t> consensus_count_{0};

    mutable std::mutex mutex_;

    ConsensusResult compute_consensus_locked() const;
};

} // namespace llmquant
