#pragma once

/// @file include/dynamic_dict.hpp
/// @brief Dynamic token dictionary with runtime YAML/JSON loading, hot-reload,
///        per-category weight multipliers, and online learning support.
///
/// ## Design goals
///  - Zero allocations on the hot lookup path (read-only after load).
///  - Atomic dictionary swap on hot-reload — readers never block.
///  - Full backward compatibility with the existing LLMAdapter API.
///  - No exceptions in the hot path; all fallible cold-path operations
///    return bool or std::optional.
///
/// ## Feature flag
/// Controlled by `LLMQUANT_ENABLE_DYNAMIC_DICT` (default ON).

#include "LLMAdapter.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// TokenCategory
// ---------------------------------------------------------------------------

/// @brief Semantic category of a token dictionary entry.
///
/// Categories are used by CategoryWeights to apply per-class multipliers
/// before signals reach the accumulator.
enum class TokenCategory : uint8_t {
    Fear        = 0, ///< Fear / uncertainty tokens (e.g. "crash", "panic").
    Bullish     = 1, ///< Bullish / positive market sentiment.
    Bearish     = 2, ///< Bearish / negative market sentiment.
    Volatility  = 3, ///< Volatility-inducing words (e.g. "spike", "whipsaw").
    Corporate   = 4, ///< Corporate action tokens (e.g. "earnings", "dividend").
    Macro       = 5, ///< Macro-economic tokens (e.g. "inflation", "Fed").
    Neutral     = 6, ///< Catch-all; no special multiplier applied.
};

/// @brief Convert a TokenCategory to its canonical lowercase string name.
/// @param cat Category enum value.
/// @return String representation (e.g. "fear", "bullish").
[[nodiscard]] const char* token_category_to_string(TokenCategory cat) noexcept;

/// @brief Parse a category string from a YAML/JSON "category" field.
/// @param s Case-insensitive category name.
/// @return Parsed category, or TokenCategory::Neutral if unrecognised.
[[nodiscard]] TokenCategory token_category_from_string(std::string_view s) noexcept;

// ---------------------------------------------------------------------------
// CategoryWeights
// ---------------------------------------------------------------------------

/// @brief Per-category multiplicative weight applied to base token weights.
///
/// Before a token's bias/volatility is accumulated the engine multiplies it by
/// the matching category multiplier.  Default value for every multiplier is
/// 1.0 (identity — no change to existing behaviour).
///
/// Example: set `fear = 1.5` to amplify fear-category tokens by 50%.
struct CategoryWeights {
    double fear       {1.0}; ///< Multiplier for TokenCategory::Fear.
    double bullish    {1.0}; ///< Multiplier for TokenCategory::Bullish.
    double bearish    {1.0}; ///< Multiplier for TokenCategory::Bearish.
    double volatility {1.0}; ///< Multiplier for TokenCategory::Volatility.
    double corporate  {1.0}; ///< Multiplier for TokenCategory::Corporate.
    double macro      {1.0}; ///< Multiplier for TokenCategory::Macro.
    double neutral    {1.0}; ///< Multiplier for TokenCategory::Neutral.

    /// @brief Return the multiplier for the given category.
    /// @param cat Category to query.
    /// @return Corresponding multiplier (always finite, default 1.0).
    [[nodiscard]] double get(TokenCategory cat) const noexcept;
};

// ---------------------------------------------------------------------------
// TokenEntry
// ---------------------------------------------------------------------------

/// @brief A single entry in the dynamic token dictionary.
///
/// Extends SemanticWeight with provenance metadata used by TokenLearner for
/// online weight adjustment and audit logging.
struct TokenEntry {
    std::string    text;             ///< Raw token text (case-sensitive).
    double         bias_weight;      ///< Directional bias weight ∈ [-1, 1].
    double         volatility_weight;///< Volatility contribution ∈ [0, 1].
    double         sentiment_score;  ///< Overall sentiment ∈ [-1, 1].
    double         confidence;       ///< Base confidence ∈ [0, 1].
    TokenCategory  category;         ///< Semantic category for multiplier lookup.
    std::string    source;           ///< Provenance label (e.g. "yaml", "learned").

    /// @brief Convert to SemanticWeight for consumption by LLMAdapter.
    /// @param cw Category weights to apply.
    /// @return Scaled SemanticWeight with category multiplier applied.
    [[nodiscard]] SemanticWeight to_semantic_weight(const CategoryWeights& cw) const noexcept;

    /// @brief Convert to SemanticWeight with identity category weights.
    [[nodiscard]] SemanticWeight to_semantic_weight() const noexcept;
};

// ---------------------------------------------------------------------------
// DynamicTokenDictionary
// ---------------------------------------------------------------------------

/// @brief Thread-safe, atomically-swappable runtime token dictionary.
///
/// Stores a snapshot of TokenEntry records behind a shared_ptr so that the
/// DictionaryLoader can publish a freshly-built map without acquiring any lock
/// on the hot lookup path.  All const lookup methods take only a relaxed
/// atomic load on the shared_ptr, making them allocation-free and near-zero
/// overhead.
///
/// ### Backward compatibility
/// `map_token_to_weight()` and `map_sequence_to_weight()` mirror the LLMAdapter
/// API exactly, enabling drop-in use wherever an LLMAdapter was used.
///
/// ### Thread safety
/// - Lookup methods (const): fully lock-free via `std::atomic<shared_ptr>`.
/// - Mutation methods: internally synchronised with a mutex; never called
///   from the hot path.
class DynamicTokenDictionary {
public:
    // -----------------------------------------------------------------------
    // Inner snapshot type
    // -----------------------------------------------------------------------

    /// @brief Immutable snapshot of the dictionary held behind a shared_ptr.
    struct Snapshot {
        std::unordered_map<std::string, TokenEntry> entries; ///< Token map.
        CategoryWeights category_weights;                    ///< Per-category multipliers.
        std::string     source_path;                         ///< File loaded from.
        uint64_t        version{0};                          ///< Monotonic version counter.

        /// @brief Construct an empty snapshot.
        Snapshot() = default;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// @brief Construct an empty dictionary (no entries, identity weights).
    DynamicTokenDictionary();

    /// @brief Non-copyable, non-movable (contains atomics and mutex).
    DynamicTokenDictionary(const DynamicTokenDictionary&) = delete;
    DynamicTokenDictionary& operator=(const DynamicTokenDictionary&) = delete;
    DynamicTokenDictionary(DynamicTokenDictionary&&) = delete;
    DynamicTokenDictionary& operator=(DynamicTokenDictionary&&) = delete;

    ~DynamicTokenDictionary() = default;

    // -----------------------------------------------------------------------
    // Hot-path lookup (lock-free)
    // -----------------------------------------------------------------------

    /// @brief Look up a single token.
    ///
    /// Performs a lock-free atomic load of the current snapshot, then an
    /// unordered_map lookup.  Returns a neutral SemanticWeight on miss.
    ///
    /// @param token Token text (case-sensitive).
    /// @return Scaled SemanticWeight, or neutral {0,0.5,0.1,0} on miss.
    [[nodiscard]] SemanticWeight map_token_to_weight(const std::string& token) const noexcept;

    /// @brief Aggregate a token sequence into one confidence-weighted weight.
    ///
    /// Each token is looked up with map_token_to_weight(); results are averaged
    /// weighted by confidence_score.  Semantically identical to
    /// LLMAdapter::map_sequence_to_weight().
    ///
    /// @param tokens Ordered token list.
    /// @return Aggregated SemanticWeight, or zero weight for empty input.
    [[nodiscard]] SemanticWeight map_sequence_to_weight(
        const std::vector<std::string>& tokens) const noexcept;

    // -----------------------------------------------------------------------
    // Snapshot management (cold path)
    // -----------------------------------------------------------------------

    /// @brief Atomically replace the live snapshot.
    ///
    /// Safe to call from any thread while readers are active.  Readers that
    /// started a lookup before the swap finish with the old snapshot; new
    /// readers get the new snapshot.
    ///
    /// @param snap New snapshot to publish.
    void publish_snapshot(std::shared_ptr<Snapshot> snap);

    /// @brief Return the current live snapshot (shared ownership).
    [[nodiscard]] std::shared_ptr<const Snapshot> current_snapshot() const noexcept;

    /// @brief Return the number of entries in the live snapshot.
    [[nodiscard]] std::size_t size() const noexcept;

    /// @brief Return the version number of the live snapshot.
    [[nodiscard]] uint64_t version() const noexcept;

    // -----------------------------------------------------------------------
    // Direct mutation (cold path, internally locked)
    // -----------------------------------------------------------------------

    /// @brief Insert or overwrite a single TokenEntry.
    ///
    /// Builds a new snapshot copying the current one, applies the change, then
    /// atomically publishes it.  O(N) copy — call only during setup or reload.
    ///
    /// @param entry Entry to insert / replace.
    void upsert(const TokenEntry& entry);

    /// @brief Set per-category weight multipliers and republish.
    /// @param weights New CategoryWeights to apply.
    void set_category_weights(const CategoryWeights& weights);

private:
    /// Current live snapshot (atomic shared_ptr, C++20 guaranteed).
    std::shared_ptr<Snapshot> snapshot_;
    mutable std::mutex        mutation_mtx_; ///< Guards cold-path mutations.

    /// @brief Clone the current snapshot for mutation.
    std::shared_ptr<Snapshot> clone_snapshot() const;
};

// ---------------------------------------------------------------------------
// DictionaryLoader
// ---------------------------------------------------------------------------

/// @brief Loads and hot-reloads a DynamicTokenDictionary from a YAML or JSON file.
///
/// On construction the file is loaded immediately.  A background watcher thread
/// polls the file's modification timestamp at a configurable interval and
/// republishes the dictionary atomically whenever the file changes.
///
/// ### Supported file formats
///
/// **JSON** (`*.json`):
/// ```json
/// {
///   "category_weights": { "fear": 1.5, "bullish": 1.2 },
///   "tokens": [
///     { "text": "crash",    "bias": -0.9, "volatility": 0.8,
///       "sentiment": -0.8,  "confidence": 0.95, "category": "fear" },
///     { "text": "rally",    "bias":  0.8, "volatility": 0.3,
///       "sentiment":  0.7,  "confidence": 0.90, "category": "bullish" }
///   ]
/// }
/// ```
///
/// **YAML** (`*.yaml` / `*.yml`):
/// ```yaml
/// category_weights:
///   fear: 1.5
///   bullish: 1.2
/// tokens:
///   - text: crash
///     bias: -0.9
///     volatility: 0.8
///     sentiment: -0.8
///     confidence: 0.95
///     category: fear
/// ```
///
/// **TSV** (legacy fallback, one line per token):
/// ```
/// token  sentiment  confidence  volatility  bias
/// ```
///
/// ### Thread safety
/// All public methods are thread-safe.  The background watcher is a daemon
/// thread that is joined on destruction.
class DictionaryLoader {
public:
    /// @brief Callback fired after every successful hot-reload.
    using ReloadCallback = std::function<void(uint64_t /*new_version*/,
                                              std::size_t /*entry_count*/)>;

    /// @brief Construct and immediately load the dictionary file.
    ///
    /// @param filepath  Path to the YAML/JSON/TSV dictionary file.
    /// @param dict      Dictionary to load into (may be empty initially).
    /// @param poll_ms   File-change poll interval in milliseconds (default 500).
    /// @throws std::runtime_error if the file cannot be opened on first load.
    explicit DictionaryLoader(const std::string&          filepath,
                              DynamicTokenDictionary&     dict,
                              std::chrono::milliseconds   poll_ms =
                                  std::chrono::milliseconds{500});

    /// @brief Stop the watcher thread and release resources.
    ~DictionaryLoader();

    DictionaryLoader(const DictionaryLoader&) = delete;
    DictionaryLoader& operator=(const DictionaryLoader&) = delete;

    /// @brief Register a callback invoked after every successful reload.
    /// @param cb Callback (may be null to clear).
    void set_reload_callback(ReloadCallback cb);

    /// @brief Force an immediate synchronous reload from disk.
    /// @return true on success, false if parsing failed (dictionary unchanged).
    bool reload_now();

    /// @brief Return the number of successful hot-reloads since construction.
    [[nodiscard]] uint64_t reload_count() const noexcept;

    /// @brief Return the path this loader watches.
    [[nodiscard]] const std::string& filepath() const noexcept;

private:
    std::string                 filepath_;
    DynamicTokenDictionary&     dict_;
    std::chrono::milliseconds   poll_ms_;
    ReloadCallback              on_reload_;
    std::atomic<uint64_t>       reload_count_{0};
    std::atomic<bool>           stop_{false};
    std::thread                 watcher_;
    mutable std::mutex          cb_mtx_;

    // Last modification time seen; used to detect changes.
    std::atomic<int64_t>        last_mtime_ns_{0};

    /// @brief Internal: background watcher loop.
    void watch_loop();

    /// @brief Internal: load and parse the file, returning a new Snapshot.
    /// @return Parsed snapshot, or nullptr on failure.
    [[nodiscard]] std::shared_ptr<DynamicTokenDictionary::Snapshot>
    parse_file() const;

    /// @brief Parse JSON format.
    [[nodiscard]] std::shared_ptr<DynamicTokenDictionary::Snapshot>
    parse_json(const std::string& content, const std::string& source) const;

    /// @brief Parse YAML format (simple line-oriented subset).
    [[nodiscard]] std::shared_ptr<DynamicTokenDictionary::Snapshot>
    parse_yaml(const std::string& content, const std::string& source) const;

    /// @brief Parse legacy TSV format.
    [[nodiscard]] std::shared_ptr<DynamicTokenDictionary::Snapshot>
    parse_tsv(const std::string& content, const std::string& source) const;

    /// @brief Get the file's last-write time as nanoseconds since epoch.
    [[nodiscard]] int64_t get_file_mtime_ns() const noexcept;
};

// ---------------------------------------------------------------------------
// TokenLearner
// ---------------------------------------------------------------------------

/// @brief Online learning: adjusts token weights based on signal-to-outcome
///        correlation observed over time (offline analysis mode).
///
/// For each trade outcome supplied via `record_outcome()`, the learner
/// computes a per-token credit/blame score using a simplified Shapley
/// approximation: each token that was active in the signal window that
/// produced the outcome receives a weight adjustment proportional to the
/// outcome return and its original contribution fraction.
///
/// Adjusted weights are accumulated in an exponential moving average with
/// decay factor `alpha` (default 0.05 — slow adaptation).  The learner
/// operates in **offline analysis mode**: it mutates a separate
/// DynamicTokenDictionary rather than the live production dictionary,
/// allowing the pipeline to evaluate learned weights before promoting them.
///
/// ### Thread safety
/// All public methods are internally serialised by a mutex.
/// No lock is held during the DynamicTokenDictionary::publish_snapshot call.
class TokenLearner {
public:
    /// @brief A single trade outcome record for the learner.
    struct Outcome {
        std::vector<std::string> active_tokens; ///< Tokens that contributed to the signal.
        double                   signal_bias;   ///< Emitted directional bias value.
        double                   realised_pnl;  ///< Observed P&L (positive = profit).
        std::chrono::steady_clock::time_point timestamp; ///< When outcome was observed.
    };

    /// @brief Learning configuration.
    struct Config {
        double alpha{0.05};            ///< EMA learning rate (0 < alpha ≤ 1).
        double max_weight_delta{0.1};  ///< Maximum single-update weight change (clamp).
        double min_confidence{0.1};    ///< Minimum confidence a token retains post-update.
        std::size_t min_observations{10}; ///< Minimum observations before weight is updated.
    };

    /// @brief Construct a learner operating on the given shadow dictionary.
    ///
    /// @param shadow_dict Dictionary in which learned weights are accumulated.
    ///                    Should be a copy of the production dictionary.
    /// @param source_dict Production dictionary (read-only reference snapshot).
    /// @param cfg         Learning hyper-parameters.
    explicit TokenLearner(DynamicTokenDictionary&        shadow_dict,
                          const DynamicTokenDictionary&  source_dict,
                          Config                         cfg = {});

    /// @brief Record a trade outcome and update shadow dictionary weights.
    ///
    /// Each active_token in the outcome receives a credit proportional to
    /// its fraction of total confidence-weighted bias contribution and the
    /// outcome sign (positive PnL → increase bias weight; negative → decrease).
    ///
    /// @param outcome Outcome to incorporate.
    void record_outcome(const Outcome& outcome);

    /// @brief Return the number of outcomes recorded so far.
    [[nodiscard]] uint64_t observation_count() const noexcept;

    /// @brief Return the current configuration.
    [[nodiscard]] const Config& config() const noexcept;

    /// @brief Return per-token observation counts for audit.
    [[nodiscard]] std::unordered_map<std::string, uint64_t>
    observation_counts() const;

private:
    DynamicTokenDictionary&       shadow_dict_;
    const DynamicTokenDictionary& source_dict_;
    Config                        cfg_;
    std::atomic<uint64_t>         obs_count_{0};
    mutable std::mutex            mtx_;

    /// Per-token cumulative observation counts (for min_observations guard).
    std::unordered_map<std::string, uint64_t> token_obs_;
};

} // namespace llmquant
