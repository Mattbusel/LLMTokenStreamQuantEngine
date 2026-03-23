#pragma once

/// @file include/regime_sizer.hpp
/// @brief Regime-Adaptive Position Sizer — Round 7.
///
/// Classifies the current market regime from LLM-derived sentiment features
/// and maps each regime to a position-size fraction of account equity.
///
/// ## Regime Classification
/// The RegimeClassifier applies a deterministic decision tree to a
/// RegimeFeatures snapshot:
///
///   1. HIGH_VOL        — sentiment_vol > 0.4
///   2. TRENDING_BULL   — cross_asset_correlation > 0.7 AND avg_sentiment > 0
///   3. TRENDING_BEAR   — cross_asset_correlation > 0.7 AND avg_sentiment <= 0
///   4. MEAN_REVERTING  — cross_asset_correlation < 0.3
///   5. LOW_VOL         — avg_sentiment >= 0
///   6. UNKNOWN         — fallback
///
/// ## Position Sizing
/// PositionSizer maps (regime, signal_confidence, account_equity) to a
/// concrete PositionSize.  The base_size fraction is multiplied by the
/// regime-specific multiplier and the signal_confidence:
///
///     notional = account_equity * base_size * regime_mult * confidence
///
/// ## Feature flag
/// Controlled by LLMQUANT_ENABLE_REGIME_ADAPTIVE_SIZER (default ON).

#include <map>
#include <string>

namespace llmquant {

// ─── MarketRegime ─────────────────────────────────────────────────────────────

/// Enumeration of market regimes detectable from LLM-derived features.
enum class MarketRegime {
    TRENDING_BULL,   ///< High correlation, bullish sentiment
    TRENDING_BEAR,   ///< High correlation, bearish sentiment
    MEAN_REVERTING,  ///< Low cross-asset correlation
    HIGH_VOL,        ///< Elevated sentiment volatility
    LOW_VOL,         ///< Low volatility, mildly bullish
    UNKNOWN,         ///< Cannot determine regime
};

/// Convert MarketRegime to a human-readable string.
[[nodiscard]] inline const char* regime_name(MarketRegime r) noexcept {
    switch (r) {
        case MarketRegime::TRENDING_BULL:   return "TRENDING_BULL";
        case MarketRegime::TRENDING_BEAR:   return "TRENDING_BEAR";
        case MarketRegime::MEAN_REVERTING:  return "MEAN_REVERTING";
        case MarketRegime::HIGH_VOL:        return "HIGH_VOL";
        case MarketRegime::LOW_VOL:         return "LOW_VOL";
        case MarketRegime::UNKNOWN:         return "UNKNOWN";
        default:                            return "UNKNOWN";
    }
}

// ─── RegimeFeatures ───────────────────────────────────────────────────────────

/// Input features used by RegimeClassifier.
struct RegimeFeatures {
    /// Rolling mean sentiment (bias) across the token stream.
    /// Positive = bullish, negative = bearish.
    double avg_sentiment        = 0.0;

    /// Rolling standard deviation of sentiment; high values indicate
    /// unstable or noisy LLM outputs.
    double sentiment_vol        = 0.0;

    /// Pearson correlation of sentiment across multiple assets.
    /// High correlation → trending regime; low → mean-reverting.
    double cross_asset_correlation = 0.0;

    /// Estimated rate at which signal alpha decays (per unit time).
    /// Not used in the default classifier but available for custom overrides.
    double alpha_decay_rate     = 0.0;
};

// ─── RegimeClassifier ─────────────────────────────────────────────────────────

/// Deterministic decision-tree regime classifier.
///
/// Applies the following priority-ordered rules:
/// 1. HIGH_VOL        if sentiment_vol > 0.4
/// 2. TRENDING_BULL   if cross_asset_correlation > 0.7 AND avg_sentiment > 0
/// 3. TRENDING_BEAR   if cross_asset_correlation > 0.7 AND avg_sentiment <= 0
/// 4. MEAN_REVERTING  if cross_asset_correlation < 0.3
/// 5. LOW_VOL         if avg_sentiment >= 0
/// 6. UNKNOWN         (fallback)
class RegimeClassifier {
public:
    /// Thresholds are configurable for unit testing and regime tuning.
    struct Thresholds {
        double high_vol_sentiment_vol     = 0.4;   ///< Rule 1 threshold
        double high_corr                  = 0.7;   ///< Rules 2 & 3 threshold
        double low_corr                   = 0.3;   ///< Rule 4 threshold
    };

    explicit RegimeClassifier(Thresholds thresholds = Thresholds{});

    /// Classify the current market regime from features.
    [[nodiscard]] MarketRegime classify(const RegimeFeatures& features) const noexcept;

    /// Access the current threshold configuration.
    [[nodiscard]] const Thresholds& thresholds() const noexcept { return thresholds_; }

private:
    Thresholds thresholds_;
};

// ─── SizingConfig ─────────────────────────────────────────────────────────────

/// Configuration for PositionSizer.
struct SizingConfig {
    /// Fraction of account equity to use as the base position size.
    /// Must be in (0, 1].  Default 0.05 (5 % of equity per trade).
    double base_size = 0.05;

    /// Per-regime size multipliers applied on top of base_size.
    /// Missing regimes default to 1.0.
    std::map<MarketRegime, double> regime_multipliers = {
        {MarketRegime::TRENDING_BULL,  1.5},
        {MarketRegime::TRENDING_BEAR,  1.5},
        {MarketRegime::MEAN_REVERTING, 0.8},
        {MarketRegime::HIGH_VOL,       0.3},
        {MarketRegime::LOW_VOL,        1.0},
        {MarketRegime::UNKNOWN,        0.5},
    };
};

// ─── PositionSize ─────────────────────────────────────────────────────────────

/// Output of PositionSizer::size() — concrete trade sizing information.
struct PositionSize {
    double shares_or_contracts = 0.0;  ///< Number of shares or contracts
    double notional_usd        = 0.0;  ///< Dollar notional value
    double risk_usd            = 0.0;  ///< Dollar at risk (notional * base_size)
    MarketRegime regime        = MarketRegime::UNKNOWN;
    double confidence          = 0.0;  ///< Signal confidence used for sizing

    /// Whether the position size is non-trivially non-zero.
    [[nodiscard]] bool is_valid() const noexcept {
        return notional_usd > 0.0;
    }
};

// ─── PositionSizer ────────────────────────────────────────────────────────────

/// Maps a (MarketRegime, signal_confidence, account_equity) triple to a
/// concrete PositionSize.
///
/// Sizing formula:
/// ```
///   multiplier = config.regime_multipliers[regime]  (default 1.0)
///   notional   = account_equity * config.base_size * multiplier * confidence
///   shares     = notional / price_per_unit
///   risk_usd   = notional * config.base_size
/// ```
class PositionSizer {
public:
    explicit PositionSizer(SizingConfig config = SizingConfig{});

    /// Compute position size.
    ///
    /// @param regime            Classified market regime.
    /// @param signal_confidence Signal confidence in [0, 1].  Values outside
    ///                          this range are clamped.
    /// @param account_equity    Current portfolio equity in USD.
    /// @param price_per_unit    Current price per share/contract (for converting
    ///                          notional to units; default 1.0).
    [[nodiscard]] PositionSize size(
        MarketRegime regime,
        double signal_confidence,
        double account_equity,
        double price_per_unit = 1.0) const noexcept;

    /// Access the current configuration.
    [[nodiscard]] const SizingConfig& config() const noexcept { return config_; }

    /// Replace the configuration.
    void update_config(SizingConfig config);

private:
    SizingConfig config_;
};

} // namespace llmquant
