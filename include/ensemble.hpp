#pragma once

/// @file include/ensemble.hpp
/// @brief Signal ensemble: runs N TradeSignalEngine instances in parallel with
///        different decay parameters and fuses their outputs via weighted
///        average, majority vote, or Kalman filter fusion.
///
/// ## Design goals
///  - Lock-free voting path: `std::atomic<uint64_t>` bit-cast double arrays.
///  - Dynamic engine reweighting based on recent accuracy (EMA of sign hits).
///  - No exceptions in the hot path.
///  - Full compatibility with the existing TradeSignalEngine API.
///
/// ## Feature flag
/// Controlled by `LLMQUANT_ENABLE_ENSEMBLE` (default ON).

#include "LLMAdapter.h"
#include "TradeSignalEngine.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// EnsembleWeight
// ---------------------------------------------------------------------------

/// @brief Per-engine weight record used by the EnsembleVoter.
///
/// Weights are updated dynamically based on each engine's recent accuracy
/// (fraction of emitted signals whose direction matched the next-bar return).
struct EnsembleWeight {
    uint32_t engine_id{0};          ///< Index into SignalEnsemble::engines_.
    double   weight{1.0};           ///< Current vote weight (non-negative).
    double   recent_accuracy{0.5};  ///< EMA of sign(signal) == sign(outcome), ∈ [0,1].
    uint64_t signals_evaluated{0};  ///< Total outcomes this engine has seen.

    /// @brief Normalised weight; set by EnsembleVoter before combining votes.
    double normalised_weight{1.0};
};

// ---------------------------------------------------------------------------
// EnsembleVoter
// ---------------------------------------------------------------------------

/// @brief Fuses signals from N TradeSignalEngine instances.
///
/// Three fusion modes are available:
///
///  - **WeightedAverage**: consensus = Σ(w_i × signal_i) / Σ(w_i).
///    Suitable for correlated engines with different decay timescales.
///
///  - **MajorityVote**: consensus = sign(Σ(w_i × sign(signal_i))).
///    Robust to outlier engines; reduces to simple majority when weights equal.
///
///  - **KalmanFusion**: treats each engine as an independent noisy measurement
///    of the true latent signal.  Measurement noise is estimated as the
///    inverse of recent_accuracy.  Fuses sequentially (scalar Kalman filter).
///
/// ### Lock-free reads
/// The vote array `pending_signals_` is backed by `std::atomic<uint64_t>[]`
/// using bit-cast from double.  EnsembleVoter::vote() is fully lock-free.
/// `fuse()` and `record_outcome()` are called from a single coordinator
/// thread and do not require locking.
class EnsembleVoter {
public:
    /// @brief Signal fusion strategy.
    enum class FusionMode : uint8_t {
        WeightedAverage = 0, ///< Confidence-weighted mean of all engine outputs.
        MajorityVote    = 1, ///< Weighted majority vote (sign aggregation).
        KalmanFusion    = 2, ///< Sequential scalar Kalman filter fusion.
    };

    /// @brief Result produced by fuse().
    struct FusedSignal {
        double   bias;          ///< Fused directional bias ∈ [-1, 1].
        double   volatility;    ///< Fused volatility adjustment ∈ [0, 1].
        double   confidence;    ///< Aggregate confidence ∈ [0, 1].
        double   agreement;     ///< Fraction of engines on the same side, ∈ [0, 1].
        uint32_t engine_count;  ///< Number of engines that contributed.
        FusionMode mode;        ///< Fusion mode used.
    };

    /// @brief Maximum engines supported (compile-time constant).
    static constexpr std::size_t kMaxEngines = 16;

    /// @brief Construct a voter for `n_engines` engines.
    ///
    /// @param n_engines  Number of engines in the ensemble (≤ kMaxEngines).
    /// @param mode       Initial fusion mode.
    /// @throws std::invalid_argument if n_engines > kMaxEngines.
    explicit EnsembleVoter(std::size_t n_engines,
                           FusionMode  mode = FusionMode::WeightedAverage);

    ~EnsembleVoter() = default;
    EnsembleVoter(const EnsembleVoter&) = delete;
    EnsembleVoter& operator=(const EnsembleVoter&) = delete;

    // -----------------------------------------------------------------------
    // Hot path
    // -----------------------------------------------------------------------

    /// @brief Submit a signal vote from engine `engine_id`.
    ///
    /// Thread-safe and lock-free.  Multiple engine threads may call vote()
    /// concurrently without contention.
    ///
    /// @param engine_id  Engine index (0-based, must be < n_engines_).
    /// @param signal     TradeSignal emitted by the engine.
    void vote(uint32_t engine_id, const TradeSignal& signal) noexcept;

    /// @brief Consume all pending votes and produce a fused signal.
    ///
    /// Must be called from a single coordinator thread (not thread-safe with
    /// itself).  Clears all pending vote slots atomically after reading.
    ///
    /// @param weights  Per-engine weights (must have size == n_engines_).
    /// @return Fused signal, or std::nullopt if no votes have been cast.
    [[nodiscard]] std::optional<FusedSignal>
    fuse(const std::vector<EnsembleWeight>& weights) noexcept;

    // -----------------------------------------------------------------------
    // Weight update
    // -----------------------------------------------------------------------

    /// @brief Record a trade outcome to update engine accuracy estimates.
    ///
    /// @param engine_id    Engine whose signal is being evaluated.
    /// @param signal_bias  The bias that was emitted.
    /// @param realised_pnl Observed P&L; positive = correct direction.
    /// @param weights      Weight table to update in-place.
    /// @param alpha        EMA decay for recent_accuracy update (default 0.1).
    void record_outcome(uint32_t                    engine_id,
                        double                      signal_bias,
                        double                      realised_pnl,
                        std::vector<EnsembleWeight>& weights,
                        double                      alpha = 0.1) noexcept;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// @brief Change the fusion mode.  Safe to call between fuse() calls.
    void set_mode(FusionMode mode) noexcept;

    /// @brief Return the current fusion mode.
    [[nodiscard]] FusionMode mode() const noexcept;

    /// @brief Return the number of engines this voter was constructed for.
    [[nodiscard]] std::size_t n_engines() const noexcept;

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// @brief Return the total number of fuse() calls completed.
    [[nodiscard]] uint64_t fuse_count() const noexcept;

    /// @brief Return the total number of vote() calls received.
    [[nodiscard]] uint64_t vote_count() const noexcept;

private:
    // Pending bias votes from each engine (bit-cast double → uint64).
    std::array<std::atomic<uint64_t>, kMaxEngines> pending_bias_{};
    // Pending volatility votes.
    std::array<std::atomic<uint64_t>, kMaxEngines> pending_vol_{};
    // Pending confidence votes.
    std::array<std::atomic<uint64_t>, kMaxEngines> pending_conf_{};
    // Sentinel: non-zero indicates a vote is present.
    std::array<std::atomic<uint8_t>,  kMaxEngines> vote_present_{};

    std::size_t          n_engines_;
    std::atomic<uint8_t> mode_{static_cast<uint8_t>(FusionMode::WeightedAverage)};
    std::atomic<uint64_t> fuse_count_{0};
    std::atomic<uint64_t> vote_count_{0};

    // Kalman filter state (maintained across fuse() calls).
    double kalman_x_{0.0};  ///< State estimate.
    double kalman_p_{1.0};  ///< State covariance.

    /// @brief Weighted-average fusion implementation.
    [[nodiscard]] FusedSignal
    fuse_weighted_average(const std::vector<EnsembleWeight>& weights,
                          const std::vector<double>& biases,
                          const std::vector<double>& vols,
                          const std::vector<double>& confs) const noexcept;

    /// @brief Majority-vote fusion implementation.
    [[nodiscard]] FusedSignal
    fuse_majority(const std::vector<EnsembleWeight>& weights,
                  const std::vector<double>& biases,
                  const std::vector<double>& vols,
                  const std::vector<double>& confs) const noexcept;

    /// @brief Kalman-filter fusion implementation (updates kalman_x_, kalman_p_).
    [[nodiscard]] FusedSignal
    fuse_kalman(const std::vector<EnsembleWeight>& weights,
                const std::vector<double>& biases,
                const std::vector<double>& vols,
                const std::vector<double>& confs) noexcept;
};

// ---------------------------------------------------------------------------
// SignalEnsemble
// ---------------------------------------------------------------------------

/// @brief Runs N TradeSignalEngine instances with different decay parameters
///        in parallel and produces a single fused TradeSignal.
///
/// Each engine operates on the same incoming SemanticWeight stream but with
/// different `signal_decay_rate` and `time_decay_half_life_ms` settings,
/// covering fast (sub-second), medium (few-second), and slow (minute-scale)
/// timescales.  This multi-timescale approach captures both flash-crash and
/// sustained-trend signals simultaneously.
///
/// ### Usage
/// ```cpp
/// SignalEnsemble ensemble;
/// ensemble.process_semantic_weight(weight);
/// if (auto sig = ensemble.last_fused_signal()) {
///     // use sig->bias, sig->confidence, etc.
/// }
/// ```
///
/// ### Thread safety
/// process_semantic_weight() is NOT thread-safe; it must be called from a
/// single producer thread.  last_fused_signal() is always safe (atomic load).
class SignalEnsemble {
public:
    /// @brief Per-engine configuration inserted into TradeSignalEngine::Config.
    struct EngineSpec {
        std::string name;                    ///< Human-readable identifier.
        double      decay_rate{0.95};        ///< signal_decay_rate for this engine.
        double      half_life_ms{0.0};       ///< time_decay_half_life_ms (0 = disabled).
        double      bias_sensitivity{1.0};   ///< Bias scale for this timescale.
        double      volatility_sensitivity{1.0}; ///< Volatility scale.
        double      initial_weight{1.0};     ///< Initial EnsembleWeight::weight.
    };

    /// @brief Full ensemble configuration.
    struct Config {
        /// Per-engine specs.  If empty, a default 3-engine fast/medium/slow
        /// configuration is used.
        std::vector<EngineSpec> engines;

        /// Fusion mode for the EnsembleVoter.
        EnsembleVoter::FusionMode fusion_mode{EnsembleVoter::FusionMode::WeightedAverage};

        /// EMA alpha for dynamic weight updates after outcome recording.
        double weight_update_alpha{0.1};

        /// Base TradeSignalEngine::Config applied to all engines before per-engine overrides.
        TradeSignalEngine::Config base_engine_config{};
    };

    /// @brief Construct with a default 3-engine fast/medium/slow configuration.
    SignalEnsemble();

    /// @brief Construct with a custom configuration.
    explicit SignalEnsemble(Config cfg);

    ~SignalEnsemble() = default;
    SignalEnsemble(const SignalEnsemble&) = delete;
    SignalEnsemble& operator=(const SignalEnsemble&) = delete;

    // -----------------------------------------------------------------------
    // Hot path
    // -----------------------------------------------------------------------

    /// @brief Feed a SemanticWeight to all engines and fuse the results.
    ///
    /// Each engine's process_semantic_weight() is called in sequence on the
    /// calling thread (no background threads — callers own parallelism).
    /// After all engines have been updated, fuse() is called and the result
    /// is stored atomically.
    ///
    /// @param weight Incoming semantic weight to process.
    void process_semantic_weight(const SemanticWeight& weight);

    /// @brief Return the most recent fused signal, or std::nullopt if none yet.
    [[nodiscard]] std::optional<EnsembleVoter::FusedSignal>
    last_fused_signal() const noexcept;

    /// @brief Register a callback fired after every successful fuse().
    /// @param cb Callback receiving the fused signal.  May be null.
    void set_signal_callback(std::function<void(const EnsembleVoter::FusedSignal&)> cb);

    // -----------------------------------------------------------------------
    // Outcome recording (weight update)
    // -----------------------------------------------------------------------

    /// @brief Record a trade outcome for all engines (updates EnsembleWeights).
    ///
    /// @param realised_pnl Observed P&L; positive = correct direction.
    void record_outcome(double realised_pnl) noexcept;

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// @brief Return a copy of the current EnsembleWeight table.
    [[nodiscard]] std::vector<EnsembleWeight> weights() const;

    /// @brief Return the number of engines in this ensemble.
    [[nodiscard]] std::size_t engine_count() const noexcept;

    /// @brief Return the name of engine `i`.
    [[nodiscard]] const std::string& engine_name(std::size_t i) const;

    /// @brief Return read-only access to engine `i`'s stats.
    [[nodiscard]] const TradeSignalEngine& engine(std::size_t i) const;

    /// @brief Return the total number of process_semantic_weight() calls.
    [[nodiscard]] uint64_t tokens_processed() const noexcept;

    /// @brief Return the total number of successful fuse() calls.
    [[nodiscard]] uint64_t fuse_count() const noexcept;

    // -----------------------------------------------------------------------
    // Configuration access
    // -----------------------------------------------------------------------

    /// @brief Return the active configuration.
    [[nodiscard]] const Config& config() const noexcept;

    /// @brief Change the fusion mode at runtime (safe between tokens).
    void set_fusion_mode(EnsembleVoter::FusionMode mode) noexcept;

private:
    Config                          cfg_;
    std::vector<TradeSignalEngine>  engines_;
    std::vector<EnsembleWeight>     weights_;
    EnsembleVoter                   voter_;

    // Last fused signal stored atomically as a shared_ptr for readers.
    std::shared_ptr<EnsembleVoter::FusedSignal> last_signal_;
    mutable std::mutex                           last_signal_mtx_;

    std::function<void(const EnsembleVoter::FusedSignal&)> callback_;
    mutable std::mutex                                      cb_mtx_;

    std::atomic<uint64_t> tokens_processed_{0};
    std::atomic<uint64_t> fuse_count_{0};

    // Most recent per-engine bias for outcome credit assignment.
    std::vector<double> last_engine_bias_;

    // Per-engine last-emitted signal slot (populated by per-engine callbacks).
    std::vector<std::optional<TradeSignal>> engine_last_signals_;

    /// @brief Build the default 3-engine configuration.
    static Config make_default_config();

    /// @brief Initialise engines and weights from cfg_.
    void init_engines();
};

} // namespace llmquant
