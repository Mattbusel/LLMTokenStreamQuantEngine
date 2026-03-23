/**
 * @file  src/regime_sizer.cpp
 * @brief Implementation of Regime-Adaptive Position Sizer — Round 7.
 *
 * See include/regime_sizer.hpp for the full API contract.
 */

#include "regime_sizer.hpp"

#include <algorithm>
#include <cmath>

namespace llmquant {

// ─── RegimeClassifier ─────────────────────────────────────────────────────────

RegimeClassifier::RegimeClassifier(Thresholds thresholds)
    : thresholds_(thresholds)
{
}

MarketRegime RegimeClassifier::classify(const RegimeFeatures& features) const noexcept {
    const auto& t = thresholds_;

    // Rule 1: Elevated sentiment volatility → HIGH_VOL (highest priority)
    if (features.sentiment_vol > t.high_vol_sentiment_vol) {
        return MarketRegime::HIGH_VOL;
    }

    // Rules 2 & 3: High cross-asset correlation → trending regime
    if (features.cross_asset_correlation > t.high_corr) {
        if (features.avg_sentiment > 0.0) {
            return MarketRegime::TRENDING_BULL;
        } else {
            return MarketRegime::TRENDING_BEAR;
        }
    }

    // Rule 4: Low cross-asset correlation → mean-reverting
    if (features.cross_asset_correlation < t.low_corr) {
        return MarketRegime::MEAN_REVERTING;
    }

    // Rule 5: Moderate correlation + non-negative sentiment → LOW_VOL
    if (features.avg_sentiment >= 0.0) {
        return MarketRegime::LOW_VOL;
    }

    // Fallback
    return MarketRegime::UNKNOWN;
}

// ─── PositionSizer ────────────────────────────────────────────────────────────

PositionSizer::PositionSizer(SizingConfig config)
    : config_(std::move(config))
{
}

PositionSize PositionSizer::size(
    MarketRegime regime,
    double signal_confidence,
    double account_equity,
    double price_per_unit) const noexcept
{
    PositionSize result;
    result.regime     = regime;
    result.confidence = std::clamp(signal_confidence, 0.0, 1.0);

    if (account_equity <= 0.0 || price_per_unit <= 0.0) {
        return result;
    }

    // Look up regime multiplier (default 1.0 for unknown regimes)
    double multiplier = 1.0;
    auto it = config_.regime_multipliers.find(regime);
    if (it != config_.regime_multipliers.end()) {
        multiplier = it->second;
    }

    // Clamp multiplier to a safe range [0, 10]
    multiplier = std::clamp(multiplier, 0.0, 10.0);

    // Notional = equity * base_size * regime_mult * confidence
    const double notional = account_equity
                          * config_.base_size
                          * multiplier
                          * result.confidence;

    result.notional_usd        = notional;
    result.shares_or_contracts = notional / price_per_unit;
    // Risk USD: the fraction of notional that represents the base size exposure
    result.risk_usd            = notional * config_.base_size;

    return result;
}

void PositionSizer::update_config(SizingConfig config) {
    config_ = std::move(config);
}

} // namespace llmquant
