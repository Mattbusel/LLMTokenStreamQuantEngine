/**
 * @file src/ContextWindowTracker.cpp
 * @brief Tracks token position within an LLM context window and decays signal
 *        strength as the context limit approaches.
 *
 * ## Motivation
 *
 * Most transformer-based LLMs exhibit quality degradation near their context
 * limit — the "lost in the middle" effect causes earlier tokens to receive
 * less attention.  A quantitative engine that relies on LLM sentiment scores
 * should therefore discount those scores as the context fills up.
 *
 * ## Signal strength decay model
 *
 * The decay factor d(pos) maps the fill fraction f = pos / max_tokens to a
 * scalar in [0, 1] using one of three decay curves:
 *
 *   - Linear:      d(f) = 1 - f
 *   - Quadratic:   d(f) = (1 - f)^2   (steeper drop near end)
 *   - Cosine:      d(f) = 0.5 * (1 + cos(pi * f))  (smooth S-curve)
 *
 * The choice of curve is configurable.  The default is Cosine, which matches
 * empirically observed attention quality curves better than linear.
 *
 * ## Usage
 *
 * ```cpp
 * ContextWindowTracker tracker;           // default: 128k tokens, cosine
 * tracker.consume(1);                     // one token processed
 * double scale = tracker.signal_scale(); // apply to signal strength
 * ```
 *
 * ## Thread safety
 *
 * All public methods are thread-safe.
 */

#include "ContextWindowTracker.h"

#include <algorithm>
#include <cmath>
#include <numbers>   // std::numbers::pi (C++20)
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ContextWindowTracker::ContextWindowTracker(Config config)
    : config_(std::move(config))
{}

// ---------------------------------------------------------------------------
// consume
// ---------------------------------------------------------------------------

void ContextWindowTracker::consume(uint64_t tokens) {
    uint64_t prev = tokens_used_.fetch_add(tokens, std::memory_order_relaxed);
    uint64_t used = prev + tokens;

    // Fire warning callback once when we cross the warn threshold.
    {
        std::lock_guard<std::mutex> lk(mutex_);
        double frac = fill_fraction_locked(used);

        if (!warned_ && frac >= config_.warn_fraction && config_.on_warn) {
            warned_ = true;
            warn_events_.fetch_add(1, std::memory_order_relaxed);
            config_.on_warn(used, config_.max_tokens);
        }
        if (!critical_ && frac >= config_.critical_fraction && config_.on_critical) {
            critical_ = true;
            critical_events_.fetch_add(1, std::memory_order_relaxed);
            config_.on_critical(used, config_.max_tokens);
        }
        if (!saturated_ && used >= config_.max_tokens && config_.on_saturated) {
            saturated_ = true;
            saturation_events_.fetch_add(1, std::memory_order_relaxed);
            config_.on_saturated(used, config_.max_tokens);
        }
    }
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void ContextWindowTracker::reset() {
    tokens_used_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mutex_);
        warned_    = false;
        critical_  = false;
        saturated_ = false;
    }
    reset_count_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// signal_scale
// ---------------------------------------------------------------------------

double ContextWindowTracker::signal_scale() const {
    double f = fill_fraction();
    f = std::clamp(f, 0.0, 1.0);

    double scale{1.0};
    switch (config_.decay_curve) {
        case DecayCurve::Linear:
            scale = 1.0 - f;
            break;
        case DecayCurve::Quadratic:
            scale = (1.0 - f) * (1.0 - f);
            break;
        case DecayCurve::Cosine:
            scale = 0.5 * (1.0 + std::cos(std::numbers::pi * f));
            break;
    }
    // Apply minimum floor so signal is never fully zeroed out even at 100% fill.
    return std::max(config_.min_scale, scale);
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

double ContextWindowTracker::fill_fraction() const noexcept {
    uint64_t used = tokens_used_.load(std::memory_order_relaxed);
    return fill_fraction_locked(used);
}

double ContextWindowTracker::fill_fraction_locked(uint64_t used) const noexcept {
    if (config_.max_tokens == 0) return 1.0;
    return std::min(1.0, static_cast<double>(used)
                       / static_cast<double>(config_.max_tokens));
}

uint64_t ContextWindowTracker::tokens_remaining() const noexcept {
    uint64_t used = tokens_used_.load(std::memory_order_relaxed);
    if (used >= config_.max_tokens) return 0;
    return config_.max_tokens - used;
}

// ---------------------------------------------------------------------------
// decay_curve_name
// ---------------------------------------------------------------------------

/* static */
const char* ContextWindowTracker::decay_curve_name(DecayCurve c) noexcept {
    switch (c) {
        case DecayCurve::Linear:    return "Linear";
        case DecayCurve::Quadratic: return "Quadratic";
        case DecayCurve::Cosine:    return "Cosine";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

std::string ContextWindowTracker::to_json() const {
    uint64_t used = tokens_used_.load(std::memory_order_relaxed);
    std::ostringstream os;
    os << "{"
       << "\"tokens_used\":"       << used << ","
       << "\"max_tokens\":"        << config_.max_tokens << ","
       << "\"fill_fraction\":"     << fill_fraction() << ","
       << "\"signal_scale\":"      << signal_scale() << ","
       << "\"decay_curve\":\""     << decay_curve_name(config_.decay_curve) << "\","
       << "\"warn_events\":"       << warn_events_.load(std::memory_order_relaxed) << ","
       << "\"critical_events\":"   << critical_events_.load(std::memory_order_relaxed) << ","
       << "\"saturation_events\":" << saturation_events_.load(std::memory_order_relaxed) << ","
       << "\"reset_count\":"       << reset_count_.load(std::memory_order_relaxed)
       << "}";
    return os.str();
}

} // namespace llmquant
