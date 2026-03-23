#pragma once

/**
 * @file include/LanguageRouter.h
 * @brief Detects the language of incoming tokens and routes them to
 *        language-specific semantic weight tables.
 *
 * ## Motivation
 *
 * Financial text in non-English languages carries different signal
 * characteristics:
 *
 *  - Spanish financial terms ("alcista"/"bajista") differ in frequency
 *    distribution from their English counterparts ("bullish"/"bearish").
 *  - Chinese financial text uses characters that require different n-gram
 *    hashing strategies.
 *  - European regulatory language (German, French) tends toward longer
 *    nominal compounds that tokenise differently.
 *
 * Rather than using a single global weight table, this router detects the
 * dominant language of each token (via lightweight Unicode script range
 * checks) and forwards the token to a per-language `ISemanticWeightTable`.
 * This prevents English-calibrated weights from contaminating signals
 * derived from non-English sources.
 *
 * ## Language detection
 *
 * Detection is heuristic and fast (O(n) over token bytes, no ML model):
 *
 *  - ASCII-only printable → English (default).
 *  - Unicode block U+0400–U+04FF (Cyrillic) → Russian.
 *  - Unicode block U+4E00–U+9FFF (CJK Unified Ideographs) → Chinese.
 *  - Unicode block U+0600–U+06FF (Arabic) → Arabic.
 *  - Primarily Latin + Spanish-specific characters (á, é, ñ, ü, ¿, ¡) →
 *    Spanish (distinguished from English after English falls through).
 *  - Other / ambiguous → falls back to the configured default language.
 *
 * Detection is intentionally coarse; the goal is routing, not linguistic
 * accuracy.  Per-language weight tables can be swapped at runtime.
 *
 * ## Thread safety
 *
 * `LanguageRouter` is thread-safe.  All methods acquire an internal reader
 * lock before accessing the weight-table map.
 *
 * ## Feature flag
 * `LLMQUANT_ENABLE_LANGUAGE_ROUTER` (default ON).
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llmquant {

// ---------------------------------------------------------------------------
// Language identifier
// ---------------------------------------------------------------------------

/**
 * @brief Supported language codes.
 *
 * Extend this enum and add detection rules in LanguageRouter::detect() to
 * support additional languages.
 */
enum class Language {
    English, ///< ASCII Latin (default fallback).
    Spanish, ///< Latin with Spanish-specific diacritics.
    Chinese, ///< CJK Unified Ideographs (Simplified / Traditional).
    Arabic,  ///< Arabic script.
    Russian, ///< Cyrillic script.
    Other,   ///< Unrecognised or mixed script.
};

const char* language_name(Language lang) noexcept;

// ---------------------------------------------------------------------------
// SemanticWeightTable interface
// ---------------------------------------------------------------------------

/**
 * @brief Minimal interface for a per-language semantic weight lookup table.
 *
 * Concrete implementations (English, Spanish, etc.) map token strings to
 * directional bias weights in [-1.0, 1.0].
 */
struct ISemanticWeightTable {
    virtual ~ISemanticWeightTable() = default;

    /**
     * @brief Return the semantic weight for `token`.
     *
     * @param token  Lowercase, trimmed token string.
     * @return Weight in [-1.0, 1.0].  Returns 0.0 for unknown tokens.
     */
    [[nodiscard]] virtual double lookup(const std::string& token) const = 0;

    /**
     * @brief Language this table is calibrated for.
     */
    [[nodiscard]] virtual Language language() const = 0;
};

// ---------------------------------------------------------------------------
// Built-in weight tables
// ---------------------------------------------------------------------------

/**
 * @brief Default English financial-sentiment weight table.
 *
 * Covers a curated set of ~60 high-signal financial terms.
 */
std::unique_ptr<ISemanticWeightTable> make_english_weight_table();

/**
 * @brief Spanish financial-sentiment weight table.
 *
 * Covers translations of common bullish/bearish financial terms and
 * Spanish-market-specific terminology.
 */
std::unique_ptr<ISemanticWeightTable> make_spanish_weight_table();

/**
 * @brief Chinese financial-sentiment weight table.
 *
 * Covers common Simplified Chinese financial tokens (Pinyin romanised
 * forms and Unicode character sequences).
 */
std::unique_ptr<ISemanticWeightTable> make_chinese_weight_table();

// ---------------------------------------------------------------------------
// LanguageRouter
// ---------------------------------------------------------------------------

/**
 * @brief Routes tokens to language-specific semantic weight tables.
 *
 * ## Usage
 *
 * ```cpp
 * LanguageRouter router;
 * router.register_table(make_english_weight_table());
 * router.register_table(make_spanish_weight_table());
 *
 * auto [lang, weight] = router.route("alcista");
 * // lang   == Language::Spanish
 * // weight == 0.85  (bullish in Spanish)
 * ```
 */
class LanguageRouter {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Language to use when detection is ambiguous.
         * Default: English.
         */
        Language default_language{Language::English};

        /**
         * @brief If true, emit per-token routing to the log callback.
         * Default: false.
         */
        bool verbose{false};

        /**
         * @brief Optional log callback for debug output.
         */
        std::function<void(const std::string&)> log_fn;
    };

    // -----------------------------------------------------------------------
    // Result
    // -----------------------------------------------------------------------

    /**
     * @brief Result of a route() call.
     */
    struct RouteResult {
        Language detected;   ///< Detected language of the token.
        double   weight;     ///< Semantic weight from the matched table.
        bool     fallback;   ///< True if the default language table was used.
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit LanguageRouter(Config config = Config{});

    // -----------------------------------------------------------------------
    // Table management
    // -----------------------------------------------------------------------

    /**
     * @brief Register a semantic weight table for its declared language.
     *
     * Replaces any previously registered table for the same language.
     * Thread-safe.
     *
     * @param table  Heap-allocated table (ownership transferred).
     */
    void register_table(std::unique_ptr<ISemanticWeightTable> table);

    /**
     * @brief Unregister the table for `lang` (if any).
     *
     * Thread-safe.
     */
    void unregister_table(Language lang);

    // -----------------------------------------------------------------------
    // Routing
    // -----------------------------------------------------------------------

    /**
     * @brief Detect the language of `token` and return the matching weight.
     *
     * If no table is registered for the detected language, falls back to the
     * default language table.  If the default table is also absent, returns
     * weight 0.0.
     *
     * Thread-safe.
     *
     * @param token  Raw token string (not required to be lowercase).
     * @return RouteResult with detected language and semantic weight.
     */
    [[nodiscard]] RouteResult route(const std::string& token) const;

    /**
     * @brief Detect the language of `token` without performing a weight lookup.
     *
     * Thread-safe.
     */
    [[nodiscard]] Language detect(const std::string& token) const noexcept;

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total tokens routed since construction. */
    [[nodiscard]] uint64_t tokens_routed() const noexcept {
        return tokens_routed_.load(std::memory_order_relaxed);
    }

    /** @brief Tokens routed to a fallback table. */
    [[nodiscard]] uint64_t fallback_count() const noexcept {
        return fallback_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Serialise routing statistics to JSON.
     *
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    static std::string to_lower(const std::string& s);

    Config config_;
    mutable std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<ISemanticWeightTable>> tables_;

    mutable std::atomic<uint64_t> tokens_routed_{0};
    mutable std::atomic<uint64_t> fallback_count_{0};
};

} // namespace llmquant
