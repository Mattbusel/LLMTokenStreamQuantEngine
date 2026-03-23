#pragma once

/// @file include/alpha_decay.hpp
/// @brief Alpha Decay Model — Round 4 public API.
///
/// # Module: Alpha Decay
///
/// Models how a trading signal's alpha (predictive strength) decays over time.
/// Multiple decay profiles are supported:
///
///   - Exponential:   strength(t) = exp(-λt)   where λ = ln(2) / half_life_ms
///   - Linear:        strength(t) = max(0, 1 - t/duration_ms)
///   - Power Law:     strength(t) = (scale_ms / (scale_ms + t))^exponent
///   - Step Function: strength(t) = levels[i].alpha  for t in [levels[i].ms, levels[i+1].ms)
///
/// ## Usage
///
/// ```cpp
/// #include "alpha_decay.hpp"
/// using namespace llmquant;
///
/// AlphaSignal sig{
///     .strength       = 1.0,
///     .generated_at_ms= 1000,
///     .decay_model    = DecayModel{ Exponential{500.0} },
/// };
///
/// double s = AlphaDecay::current_strength(sig, 1500);  // 0.5 (one half-life)
/// double hl = AlphaDecay::half_life_ms(sig);           // 500.0
///
/// AlphaPortfolio portfolio;
/// portfolio.add_signal("AAPL", sig);
/// double net = portfolio.net_alpha("AAPL", 2000);
/// portfolio.prune(2000, 0.01);
/// ```

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace llmquant {

// ─── Decay model variants ─────────────────────────────────────────────────────

/// Exponential decay: s(t) = exp(-λ·Δt),  λ = ln(2)/half_life_ms.
struct Exponential {
    double half_life_ms = 1000.0;  ///< Time for strength to halve (ms).
};

/// Linear decay: s(t) = max(0, 1 - Δt/duration_ms).
struct Linear {
    double duration_ms = 2000.0;  ///< Time to reach zero strength.
};

/// Power-law decay: s(t) = (scale_ms / (scale_ms + Δt))^exponent.
struct PowerLaw {
    double exponent  = 2.0;
    double scale_ms  = 500.0;
};

/// Step function: returns the alpha for the time bin containing Δt.
/// `levels` is a sorted list of (threshold_ms, alpha) pairs.
/// The alpha of the last entry with threshold_ms <= Δt is used.
/// If Δt < levels[0].first, returns levels[0].second.
/// If Δt >= last threshold, returns 0.0.
struct StepFunction {
    std::vector<std::pair<int64_t, double>> levels;  ///< Sorted by threshold_ms ascending.
};

/// Discriminated union of decay models.
using DecayModel = std::variant<Exponential, Linear, PowerLaw, StepFunction>;

// ─── AlphaSignal ──────────────────────────────────────────────────────────────

/// A trading signal with an associated alpha decay model.
struct AlphaSignal {
    double      strength        = 1.0;  ///< Initial signal strength ∈ [0, 1].
    int64_t     generated_at_ms = 0;    ///< Epoch-ms when the signal was generated.
    DecayModel  decay_model     = Exponential{1000.0};
};

// ─── AlphaDecay ───────────────────────────────────────────────────────────────

/// Stateless utility class for computing alpha decay on AlphaSignal objects.
class AlphaDecay {
public:
    AlphaDecay() = delete;

    // ------------------------------------------------------------------
    // current_strength
    // ------------------------------------------------------------------

    /// Compute the current alpha strength of *signal* at time *now_ms*.
    ///
    /// @param signal   The signal to evaluate.
    /// @param now_ms   Current time in epoch-milliseconds.
    /// @return         Decayed strength in [0, signal.strength].
    ///                 Returns 0 if now_ms < signal.generated_at_ms.
    [[nodiscard]] static double current_strength(
        const AlphaSignal& signal, int64_t now_ms
    ) noexcept {
        double dt = static_cast<double>(now_ms - signal.generated_at_ms);
        if (dt < 0.0) return 0.0;

        double factor = std::visit([dt](const auto& model) -> double {
            return decay_factor(model, dt);
        }, signal.decay_model);

        return signal.strength * factor;
    }

    // ------------------------------------------------------------------
    // half_life_ms
    // ------------------------------------------------------------------

    /// Return the time (in ms) for the signal strength to halve.
    ///
    /// - Exponential: analytical — returns the configured `half_life_ms`.
    /// - Others:      Newton-Raphson search for t s.t. decay_factor(t) == 0.5.
    ///
    /// @param signal  The signal whose half-life to compute.
    /// @return        Half-life in milliseconds, or +∞ if strength never halves.
    [[nodiscard]] static double half_life_ms(const AlphaSignal& signal) noexcept {
        return std::visit([&signal](const auto& model) -> double {
            return compute_half_life(model, signal);
        }, signal.decay_model);
    }

private:
    // ------------------------------------------------------------------
    // Decay factor implementations
    // ------------------------------------------------------------------

    static double decay_factor(const Exponential& m, double dt) noexcept {
        if (m.half_life_ms <= 0.0) return 0.0;
        double lambda = std::log(2.0) / m.half_life_ms;
        return std::exp(-lambda * dt);
    }

    static double decay_factor(const Linear& m, double dt) noexcept {
        if (m.duration_ms <= 0.0) return 0.0;
        return std::max(0.0, 1.0 - dt / m.duration_ms);
    }

    static double decay_factor(const PowerLaw& m, double dt) noexcept {
        if (m.scale_ms <= 0.0 || m.exponent <= 0.0) return 0.0;
        return std::pow(m.scale_ms / (m.scale_ms + dt), m.exponent);
    }

    static double decay_factor(const StepFunction& m, double dt) noexcept {
        if (m.levels.empty()) return 0.0;
        double alpha = m.levels.front().second;
        for (const auto& [threshold_ms, a] : m.levels) {
            if (static_cast<double>(threshold_ms) <= dt) {
                alpha = a;
            } else {
                break;
            }
        }
        return alpha;
    }

    // ------------------------------------------------------------------
    // Half-life computations
    // ------------------------------------------------------------------

    static double compute_half_life(const Exponential& m, const AlphaSignal&) noexcept {
        return m.half_life_ms;  // analytical
    }

    static double compute_half_life(const Linear& m, const AlphaSignal&) noexcept {
        // s(t) = 1 - t/D = 0.5  →  t = D/2
        return m.duration_ms / 2.0;
    }

    static double compute_half_life(const PowerLaw& m, const AlphaSignal&) noexcept {
        // (scale / (scale + t))^exp = 0.5
        // → t = scale * (2^(1/exp) - 1)
        if (m.exponent <= 0.0 || m.scale_ms <= 0.0) return std::numeric_limits<double>::infinity();
        return m.scale_ms * (std::pow(2.0, 1.0 / m.exponent) - 1.0);
    }

    static double compute_half_life(const StepFunction& m, const AlphaSignal& sig) noexcept {
        // Newton-Raphson: find t where decay_factor(t) = 0.5
        // For step functions, use bisection (more robust for discontinuities).
        if (m.levels.empty()) return std::numeric_limits<double>::infinity();

        // Check if 0.5 is achievable
        if (decay_factor(m, 0.0) <= 0.5) return 0.0;
        // Check if we ever reach 0.5
        double large_t = 1e12;
        if (decay_factor(m, large_t) > 0.5) return std::numeric_limits<double>::infinity();

        // Binary search
        double lo = 0.0, hi = large_t;
        for (int i = 0; i < 100; ++i) {
            double mid = 0.5 * (lo + hi);
            if (decay_factor(m, mid) > 0.5) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }
};

// ─── AlphaPortfolio ───────────────────────────────────────────────────────────

/// A collection of AlphaSignals per symbol.
///
/// Accumulates multiple signals for each symbol and provides net alpha queries.
class AlphaPortfolio {
public:
    /// Add a signal for *symbol*.
    void add_signal(const std::string& symbol, const AlphaSignal& signal) {
        _signals[symbol].push_back(signal);
    }

    /// Net alpha for *symbol* at *now_ms*: sum of current_strength for all
    /// active signals (strength >= threshold).
    ///
    /// @param symbol   Ticker / asset symbol.
    /// @param now_ms   Current epoch milliseconds.
    /// @return         Summed alpha, or 0.0 if no signals for *symbol*.
    [[nodiscard]] double net_alpha(const std::string& symbol, int64_t now_ms) const noexcept {
        auto it = _signals.find(symbol);
        if (it == _signals.end()) return 0.0;
        double total = 0.0;
        for (const auto& sig : it->second) {
            total += AlphaDecay::current_strength(sig, now_ms);
        }
        return total;
    }

    /// Remove signals whose current strength is below *threshold*.
    ///
    /// @param now_ms     Current epoch milliseconds.
    /// @param threshold  Minimum strength to retain (default 0.01).
    void prune(int64_t now_ms, double threshold = 0.01) {
        for (auto& [symbol, sigs] : _signals) {
            sigs.erase(
                std::remove_if(sigs.begin(), sigs.end(),
                    [now_ms, threshold](const AlphaSignal& sig) {
                        return AlphaDecay::current_strength(sig, now_ms) < threshold;
                    }),
                sigs.end()
            );
        }
        // Remove empty symbol entries
        for (auto it = _signals.begin(); it != _signals.end(); ) {
            if (it->second.empty()) it = _signals.erase(it);
            else ++it;
        }
    }

    /// All symbols that currently have at least one signal.
    [[nodiscard]] std::vector<std::string> symbols() const {
        std::vector<std::string> out;
        out.reserve(_signals.size());
        for (const auto& [sym, _] : _signals) out.push_back(sym);
        return out;
    }

    /// Number of signals currently tracked for *symbol*.
    [[nodiscard]] std::size_t signal_count(const std::string& symbol) const noexcept {
        auto it = _signals.find(symbol);
        return (it != _signals.end()) ? it->second.size() : 0u;
    }

    /// Clear all signals for all symbols.
    void clear() noexcept { _signals.clear(); }

private:
    std::map<std::string, std::vector<AlphaSignal>> _signals;
};

} // namespace llmquant
