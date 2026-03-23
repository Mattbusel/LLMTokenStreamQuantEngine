#pragma once

/// @file include/execution_timing.hpp
/// @brief Execution Timing Optimizer — Round 3 addition.
///
/// ## Overview
///
/// Three layered components:
///
/// ### 1. MarketImpactModel
/// Estimates market impact of a trade order using the Almgren-Chriss model:
///
///   impact = eta * sigma * sqrt(Q / V)
///
/// where Q = order size, V = average daily volume, sigma = daily volatility.
/// The total impact is decomposed into a linear component (temporary, price
/// moves proportional to participation rate) and a square-root component
/// (permanent, information-driven impact).
///
/// ### 2. ExecutionScheduler
/// Generates TWAP and VWAP execution schedules by slicing a parent order
/// into `n_slices` child orders.
///
/// - **TWAP**: Equal-sized slices, uniform time intervals.
/// - **VWAP**: Slices sized proportional to a historical volume profile so
///   that the portfolio's participation rate stays roughly constant.
///
/// ### 3. OptimalExecution
/// Blends TWAP and VWAP into a single schedule driven by an `urgency`
/// parameter (0–1).  `urgency = 1` → pure TWAP (time-sensitive); `urgency
/// = 0` → pure VWAP (cost-sensitive).
///
/// ## Thread safety
/// All methods are stateless / const and safe to call concurrently.
///
/// ## Guarantees
/// - No dynamic allocation except for the returned schedule vectors
/// - No raw pointers; all ownership by value
/// - Returns std::optional for all fallible operations

#include <optional>
#include <string>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// ImpactEstimate
// ---------------------------------------------------------------------------

/// Market-impact breakdown from the Almgren-Chriss model.
struct ImpactEstimate {
    double linear_impact_bps = 0.0;  ///< Temporary linear component (bps)
    double sqrt_impact_bps   = 0.0;  ///< Permanent sqrt component (bps)
    double total_impact_bps  = 0.0;  ///< Sum of linear and sqrt components
};

// ---------------------------------------------------------------------------
// MarketImpactModel
// ---------------------------------------------------------------------------

/// Almgren-Chriss market-impact estimator.
///
/// Usage:
/// @code
/// MarketImpactModel model(eta=0.1, alpha=0.5);
/// auto est = model.estimate(order_qty=10'000, avg_volume=1'000'000,
///                           volatility=0.015);
/// std::cout << est.total_impact_bps << " bps\n";
/// @endcode
class MarketImpactModel {
public:
    /// @param eta     Permanent impact coefficient (default 0.1).
    /// @param alpha   Linear temporary impact coefficient (default 0.5).
    explicit MarketImpactModel(double eta   = 0.1,
                               double alpha = 0.5) noexcept;

    /// Estimate market impact using the Almgren-Chriss formula.
    ///
    /// @param order_qty   Total shares / contracts to execute (Q).
    /// @param avg_volume  Average daily volume (V).  Must be > 0.
    /// @param volatility  Annualised daily return volatility (sigma).
    ///
    /// @return ImpactEstimate in basis points, or nullopt if inputs are invalid.
    [[nodiscard]] std::optional<ImpactEstimate>
    estimate(double order_qty,
             double avg_volume,
             double volatility) const noexcept;

    double eta()   const noexcept { return eta_;   }
    double alpha() const noexcept { return alpha_; }

private:
    double eta_;    ///< Almgren-Chriss permanent impact coefficient
    double alpha_;  ///< Linear temporary impact coefficient
};

// ---------------------------------------------------------------------------
// TradeSlice
// ---------------------------------------------------------------------------

/// A single child order within an execution schedule.
struct TradeSlice {
    long long start_ms     = 0;    ///< Slice start (Unix epoch milliseconds)
    long long end_ms       = 0;    ///< Slice end (Unix epoch milliseconds)
    double    target_qty   = 0.0;  ///< Target quantity to execute in this slice
    double    expected_price = 0.0;  ///< Expected fill price (includes impact est.)
};

// ---------------------------------------------------------------------------
// TwapSchedule
// ---------------------------------------------------------------------------

/// A TWAP execution schedule.
struct TwapSchedule {
    std::vector<TradeSlice> slices;           ///< Child order slices
    long long               interval_ms  = 0; ///< Time per slice (ms)
    double                  total_qty    = 0.0; ///< Sum of all slice quantities
    double                  avg_price_estimate = 0.0; ///< VWAP of expected prices
};

// ---------------------------------------------------------------------------
// VwapSchedule
// ---------------------------------------------------------------------------

/// A VWAP execution schedule.
struct VwapSchedule {
    std::vector<TradeSlice> slices;           ///< Child order slices
    std::vector<double>     volume_profile;   ///< Historical volume fractions (sum=1)
    double                  total_qty    = 0.0; ///< Sum of all slice quantities
};

// ---------------------------------------------------------------------------
// ExecutionSchedule (blended result)
// ---------------------------------------------------------------------------

/// Output of OptimalExecution::minimize_impact().
struct ExecutionSchedule {
    std::vector<TradeSlice> slices;
    double                  urgency       = 0.5;
    double                  total_qty     = 0.0;
    std::string             method;   ///< "TWAP", "VWAP", or "BLENDED"
};

// ---------------------------------------------------------------------------
// ExecutionScheduler
// ---------------------------------------------------------------------------

/// Generates TWAP and VWAP sliced execution schedules.
class ExecutionScheduler {
public:
    /// @param n_slices      Number of child orders (default 10).
    /// @param start_ms      Execution window start (Unix epoch ms; 0 = now).
    /// @param duration_ms   Total window duration in milliseconds.
    explicit ExecutionScheduler(int       n_slices    = 10,
                                long long start_ms    = 0,
                                long long duration_ms = 3'600'000) noexcept;

    /// Build a TWAP schedule: equal-sized slices at uniform intervals.
    ///
    /// @param total_qty    Total quantity to execute.
    /// @param spot_price   Current spot price (used for expected_price est.).
    ///
    /// @return TwapSchedule, or nullopt if inputs are invalid.
    [[nodiscard]] std::optional<TwapSchedule>
    twap(double total_qty, double spot_price) const noexcept;

    /// Build a VWAP schedule: slices sized proportional to volume_profile.
    ///
    /// @param total_qty      Total quantity to execute.
    /// @param spot_price     Current spot price.
    /// @param volume_profile Historical intraday volume fractions.  Need not
    ///                       sum to 1; the scheduler normalises automatically.
    ///                       If empty, falls back to TWAP (equal weights).
    ///
    /// @return VwapSchedule, or nullopt if inputs are invalid.
    [[nodiscard]] std::optional<VwapSchedule>
    vwap(double total_qty,
         double spot_price,
         const std::vector<double>& volume_profile) const noexcept;

    int       n_slices()    const noexcept { return n_slices_;    }
    long long start_ms()    const noexcept { return start_ms_;    }
    long long duration_ms() const noexcept { return duration_ms_; }

private:
    int       n_slices_;
    long long start_ms_;
    long long duration_ms_;
};

// ---------------------------------------------------------------------------
// OptimalExecution
// ---------------------------------------------------------------------------

/// Blends TWAP and VWAP schedules to minimise market impact at a given urgency.
///
/// `urgency = 1.0` → pure TWAP (time-sensitive; ignores volume profile).
/// `urgency = 0.0` → pure VWAP (cost-sensitive; maximises volume participation).
/// Intermediate values linearly interpolate slice quantities.
class OptimalExecution {
public:
    /// @param scheduler   Pre-configured ExecutionScheduler.
    explicit OptimalExecution(ExecutionScheduler scheduler) noexcept;

    /// Build an optimal execution schedule by blending TWAP and VWAP.
    ///
    /// @param order_qty      Total quantity to execute.
    /// @param urgency        Trade urgency in [0, 1]; 1 = TWAP, 0 = VWAP.
    /// @param volume_profile Historical volume fractions for VWAP weighting.
    /// @param spot_price     Current spot price.
    ///
    /// @return ExecutionSchedule, or nullopt if inputs are invalid.
    [[nodiscard]] std::optional<ExecutionSchedule>
    minimize_impact(double                     order_qty,
                    double                     urgency,
                    const std::vector<double>& volume_profile,
                    double                     spot_price = 100.0) const noexcept;

    const ExecutionScheduler& scheduler() const noexcept { return scheduler_; }

private:
    ExecutionScheduler scheduler_;
};

} // namespace llmquant
