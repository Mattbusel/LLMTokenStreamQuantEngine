/// @file src/signal_combiner.cpp
/// @brief Signal Combiner — Round 6 implementation.
///
/// See include/signal_combiner.hpp for the full API contract.

#include "signal_combiner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llmquant {

// ─── Helpers ──────────────────────────────────────────────────────────────────

static double clamp01(double v) noexcept {
    return std::clamp(v, 0.0, 1.0);
}

static double clamp11(double v) noexcept {
    return std::clamp(v, -1.0, 1.0);
}

// ─── SignalCombiner ───────────────────────────────────────────────────────────

void SignalCombiner::add_signal(const SignalInput& input) {
    SignalInput clamped = input;
    clamped.value      = clamp11(input.value);
    clamped.confidence = clamp01(input.confidence);
    signals_.push_back(std::move(clamped));
}

void SignalCombiner::reset() noexcept {
    signals_.clear();
}

SignalRegime SignalCombiner::classify(double action) noexcept {
    const double abs_action = std::abs(action);
    if (abs_action > kTrendingThreshold) {
        return SignalRegime::TRENDING;
    } else if (abs_action < kMeanRevertingThreshold) {
        return SignalRegime::MEAN_REVERTING;
    }
    return SignalRegime::UNCERTAIN;
}

std::optional<CombinedSignal>
SignalCombiner::combine(CombineMethod method) const {
    if (signals_.empty()) {
        return std::nullopt;
    }

    switch (method) {
        case CombineMethod::WEIGHTED_AVERAGE:
            return _weighted_average();
        case CombineMethod::MAJORITY:
            return _majority();
        case CombineMethod::ENSEMBLE:
            return _ensemble();
    }
    return _weighted_average();  // default
}

CombinedSignal SignalCombiner::_weighted_average() const {
    double sum_wv = 0.0;
    double sum_w  = 0.0;
    std::vector<std::string> names;
    names.reserve(signals_.size());

    for (const auto& s : signals_) {
        sum_wv += s.confidence * s.value;
        sum_w  += s.confidence;
        names.push_back(s.name);
    }

    const double action     = (sum_w > 0.0) ? clamp11(sum_wv / sum_w) : 0.0;
    const double confidence = (sum_w > 0.0) ? sum_w / static_cast<double>(signals_.size()) : 0.0;

    return CombinedSignal{
        action,
        clamp01(confidence),
        std::move(names),
        classify(action),
    };
}

CombinedSignal SignalCombiner::_majority() const {
    double bull_confidence = 0.0;
    double bear_confidence = 0.0;
    std::vector<std::string> names;
    names.reserve(signals_.size());
    double sum_conf = 0.0;

    for (const auto& s : signals_) {
        names.push_back(s.name);
        sum_conf += s.confidence;
        if (s.value > 0.0) {
            bull_confidence += s.confidence;
        } else if (s.value < 0.0) {
            bear_confidence += s.confidence;
        }
    }

    double action = 0.0;
    const double total = bull_confidence + bear_confidence;
    if (total > 0.0) {
        // Net directional confidence normalised to [-1, 1]
        action = clamp11((bull_confidence - bear_confidence) / total);
    }

    const double confidence = (signals_.empty())
        ? 0.0
        : clamp01(sum_conf / static_cast<double>(signals_.size()));

    return CombinedSignal{
        action,
        confidence,
        std::move(names),
        classify(action),
    };
}

CombinedSignal SignalCombiner::_ensemble() const {
    double x = 0.0;            // state estimate
    double sum_conf = 0.0;
    std::vector<std::string> names;
    names.reserve(signals_.size());

    for (const auto& s : signals_) {
        names.push_back(s.name);
        sum_conf += s.confidence;

        const double c = s.confidence;
        const double u = 1.0 - c;                // measurement uncertainty
        const double K = (c + u > 0.0) ? c / (c + u) : 0.0;  // Kalman gain
        x = x + K * (s.value - x);
    }

    const double action     = clamp11(x);
    const double confidence = (signals_.empty())
        ? 0.0
        : clamp01(sum_conf / static_cast<double>(signals_.size()));

    return CombinedSignal{
        action,
        confidence,
        std::move(names),
        classify(action),
    };
}

// ─── SignalHistory ────────────────────────────────────────────────────────────

void SignalHistory::push(const CombinedSignal& signal) {
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();
    }
    buffer_.push_back(signal);
}

std::optional<CombinedSignal> SignalHistory::latest() const {
    if (buffer_.empty()) {
        return std::nullopt;
    }
    return buffer_.back();
}

int SignalHistory::trend_direction() const noexcept {
    if (buffer_.empty()) return 0;

    double mean = 0.0;
    for (const auto& cs : buffer_) {
        mean += cs.action;
    }
    mean /= static_cast<double>(buffer_.size());

    static constexpr double FLAT_THRESHOLD = 0.05;
    if (mean > FLAT_THRESHOLD)  return  1;
    if (mean < -FLAT_THRESHOLD) return -1;
    return 0;
}

double SignalHistory::volatility() const noexcept {
    if (buffer_.size() < 2) return 0.0;

    const std::size_t n = buffer_.size();
    double sum = 0.0;
    for (const auto& cs : buffer_) sum += cs.action;
    const double mean = sum / static_cast<double>(n);

    double sq_sum = 0.0;
    for (const auto& cs : buffer_) {
        const double diff = cs.action - mean;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / static_cast<double>(n - 1));
}

}  // namespace llmquant
