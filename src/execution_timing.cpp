/// @file src/execution_timing.cpp
/// @brief Implementation of MarketImpactModel, ExecutionScheduler, and
///        OptimalExecution.

#include "execution_timing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace llmquant {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static long long now_ms() noexcept {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// MarketImpactModel
// ---------------------------------------------------------------------------

MarketImpactModel::MarketImpactModel(double eta, double alpha) noexcept
    : eta_(eta)
    , alpha_(alpha)
{}

std::optional<ImpactEstimate>
MarketImpactModel::estimate(double order_qty,
                             double avg_volume,
                             double volatility) const noexcept {
    if (order_qty <= 0.0 || avg_volume <= 0.0 || volatility < 0.0) {
        return std::nullopt;
    }

    double participation = order_qty / avg_volume;  // Q / V

    // Almgren-Chriss: permanent impact (sqrt model)
    //   impact_perm = eta * sigma * sqrt(Q/V)
    double sqrt_impact = eta_ * volatility * std::sqrt(participation);

    // Linear temporary impact
    //   impact_temp = alpha * sigma * (Q/V)
    double linear_impact = alpha_ * volatility * participation;

    // Convert to basis points (1 bps = 0.01 %)
    constexpr double BPS = 10'000.0;
    ImpactEstimate est;
    est.sqrt_impact_bps   = sqrt_impact   * BPS;
    est.linear_impact_bps = linear_impact * BPS;
    est.total_impact_bps  = est.sqrt_impact_bps + est.linear_impact_bps;

    return est;
}

// ---------------------------------------------------------------------------
// ExecutionScheduler
// ---------------------------------------------------------------------------

ExecutionScheduler::ExecutionScheduler(int       n_slices,
                                       long long start_ms,
                                       long long duration_ms) noexcept
    : n_slices_(n_slices > 0 ? n_slices : 1)
    , start_ms_(start_ms > 0 ? start_ms : now_ms())
    , duration_ms_(duration_ms > 0 ? duration_ms : 3'600'000LL)
{}

std::optional<TwapSchedule>
ExecutionScheduler::twap(double total_qty, double spot_price) const noexcept {
    if (total_qty <= 0.0 || spot_price <= 0.0) return std::nullopt;

    TwapSchedule sched;
    sched.total_qty   = total_qty;
    sched.interval_ms = duration_ms_ / n_slices_;

    double slice_qty = total_qty / static_cast<double>(n_slices_);
    double price_sum = 0.0;

    sched.slices.reserve(static_cast<std::size_t>(n_slices_));
    for (int i = 0; i < n_slices_; ++i) {
        TradeSlice sl;
        sl.start_ms     = start_ms_ + static_cast<long long>(i)     * sched.interval_ms;
        sl.end_ms       = start_ms_ + static_cast<long long>(i + 1) * sched.interval_ms;
        sl.target_qty   = slice_qty;
        // Assume flat price for TWAP; slight spread per slice position
        sl.expected_price = spot_price;
        price_sum += sl.expected_price * sl.target_qty;
        sched.slices.push_back(sl);
    }

    sched.avg_price_estimate = (total_qty > 0.0) ? price_sum / total_qty : spot_price;
    return sched;
}

std::optional<VwapSchedule>
ExecutionScheduler::vwap(double                     total_qty,
                          double                     spot_price,
                          const std::vector<double>& volume_profile) const noexcept {
    if (total_qty <= 0.0 || spot_price <= 0.0) return std::nullopt;

    // Normalise the volume profile; fall back to equal weights if empty / invalid.
    std::vector<double> weights;
    bool use_profile = !volume_profile.empty();

    if (use_profile) {
        double sum = 0.0;
        for (double v : volume_profile) sum += (v > 0.0 ? v : 0.0);
        if (sum <= 0.0) use_profile = false;

        if (use_profile) {
            for (double v : volume_profile) {
                weights.push_back((v > 0.0 ? v : 0.0) / sum);
            }
        }
    }

    if (!use_profile) {
        // Equal weights (degenerates to TWAP)
        weights.assign(static_cast<std::size_t>(n_slices_),
                       1.0 / static_cast<double>(n_slices_));
    }

    // Resample weights to n_slices_ buckets if necessary
    if (static_cast<int>(weights.size()) != n_slices_) {
        std::vector<double> resampled(static_cast<std::size_t>(n_slices_), 0.0);
        double src_n = static_cast<double>(weights.size());
        double dst_n = static_cast<double>(n_slices_);
        for (int d = 0; d < n_slices_; ++d) {
            double src_idx = d * src_n / dst_n;
            int    si      = static_cast<int>(src_idx);
            if (si >= static_cast<int>(weights.size())) si = static_cast<int>(weights.size()) - 1;
            resampled[static_cast<std::size_t>(d)] = weights[static_cast<std::size_t>(si)];
        }
        // Re-normalise after resampling
        double rs = std::accumulate(resampled.begin(), resampled.end(), 0.0);
        if (rs > 0.0) for (auto& w : resampled) w /= rs;
        weights = std::move(resampled);
    }

    VwapSchedule sched;
    sched.total_qty     = total_qty;
    sched.volume_profile = weights;
    sched.slices.reserve(static_cast<std::size_t>(n_slices_));

    long long interval = duration_ms_ / n_slices_;
    for (int i = 0; i < n_slices_; ++i) {
        TradeSlice sl;
        sl.start_ms      = start_ms_ + static_cast<long long>(i)     * interval;
        sl.end_ms        = start_ms_ + static_cast<long long>(i + 1) * interval;
        sl.target_qty    = total_qty * weights[static_cast<std::size_t>(i)];
        sl.expected_price = spot_price;
        sched.slices.push_back(sl);
    }

    return sched;
}

// ---------------------------------------------------------------------------
// OptimalExecution
// ---------------------------------------------------------------------------

OptimalExecution::OptimalExecution(ExecutionScheduler scheduler) noexcept
    : scheduler_(std::move(scheduler))
{}

std::optional<ExecutionSchedule>
OptimalExecution::minimize_impact(double                     order_qty,
                                   double                     urgency,
                                   const std::vector<double>& volume_profile,
                                   double                     spot_price) const noexcept {
    if (order_qty <= 0.0 || spot_price <= 0.0) return std::nullopt;

    // Clamp urgency to [0, 1]
    double u = std::max(0.0, std::min(1.0, urgency));

    auto twap_opt = scheduler_.twap(order_qty, spot_price);
    auto vwap_opt = scheduler_.vwap(order_qty, spot_price, volume_profile);

    if (!twap_opt.has_value() || !vwap_opt.has_value()) return std::nullopt;

    const auto& tw = *twap_opt;
    const auto& vw = *vwap_opt;

    ExecutionSchedule sched;
    sched.urgency   = u;
    sched.total_qty = order_qty;

    if (u >= 0.999) {
        sched.method = "TWAP";
        sched.slices = tw.slices;
    } else if (u <= 0.001) {
        sched.method = "VWAP";
        sched.slices = vw.slices;
    } else {
        // Blend: slice_qty = u * twap_qty + (1-u) * vwap_qty
        sched.method = "BLENDED";
        int n = scheduler_.n_slices();
        sched.slices.resize(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i) {
            auto& sl = sched.slices[static_cast<std::size_t>(i)];
            const auto& ts = tw.slices[static_cast<std::size_t>(i)];
            const auto& vs = vw.slices[static_cast<std::size_t>(i)];

            sl.start_ms      = ts.start_ms;
            sl.end_ms        = ts.end_ms;
            sl.target_qty    = u * ts.target_qty + (1.0 - u) * vs.target_qty;
            sl.expected_price = spot_price;
        }

        // Re-normalise quantities to exactly order_qty
        double total = 0.0;
        for (auto& sl : sched.slices) total += sl.target_qty;
        if (total > 0.0) {
            for (auto& sl : sched.slices) sl.target_qty *= order_qty / total;
        }
    }

    return sched;
}

} // namespace llmquant
