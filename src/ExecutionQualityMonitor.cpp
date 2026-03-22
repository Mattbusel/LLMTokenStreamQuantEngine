#include "ExecutionQualityMonitor.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

ExecutionQualityMonitor::ExecutionQualityMonitor(Config cfg)
    : cfg_(std::move(cfg))
{
    window_.resize(cfg_.window_size);
}

/*static*/ double ExecutionQualityMonitor::percentile(
    std::vector<double>& v, double pct) noexcept
{
    if (v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(pct * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

void ExecutionQualityMonitor::record(const FillRecord& fill) {
    bool breach_lat = false, breach_slip = false;
    std::function<void(const FillRecord&, bool, bool)> cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        window_[ring_head_] = {fill.latency_us, fill.slippage_bps};
        ring_head_ = (ring_head_ + 1) % static_cast<int>(cfg_.window_size);
        if (ring_fill_ < static_cast<int>(cfg_.window_size)) ++ring_fill_;
        total_fills_.fetch_add(1, std::memory_order_relaxed);
        breach_lat  = (fill.latency_us    > cfg_.latency_sla_us);
        breach_slip = (fill.slippage_bps  > cfg_.slippage_sla_bps);
        if (breach_lat || breach_slip) {
            breach_count_.fetch_add(1, std::memory_order_relaxed);
            cb = cfg_.on_sla_breach;
        }
    }
    if (cb) cb(fill, breach_lat, breach_slip);
}

void ExecutionQualityMonitor::record(uint64_t signal_id,
                                     double latency_us,
                                     double slippage_bps)
{
    record(FillRecord{signal_id, latency_us, slippage_bps});
}

ExecutionQualityMonitor::Stats ExecutionQualityMonitor::compute_stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = ring_fill_;
    if (n == 0) return {};

    std::vector<double> lats(n), slips(n);
    for (int i = 0; i < n; ++i) {
        lats[i]  = window_[i].latency_us;
        slips[i] = window_[i].slippage_bps;
    }
    double sum_lat  = std::accumulate(lats.begin(),  lats.end(),  0.0);
    double sum_slip = std::accumulate(slips.begin(), slips.end(), 0.0);

    std::sort(lats.begin(), lats.end());
    std::sort(slips.begin(), slips.end());

    Stats s;
    s.sample_count       = static_cast<uint64_t>(n);
    s.mean_latency_us    = sum_lat  / n;
    s.mean_slippage_bps  = sum_slip / n;
    s.p50_latency_us     = percentile(lats,  0.50);
    s.p95_latency_us     = percentile(lats,  0.95);
    s.p99_latency_us     = percentile(lats,  0.99);
    s.p95_slippage_bps   = percentile(slips, 0.95);
    return s;
}

void ExecutionQualityMonitor::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ring_head_ = 0;
    ring_fill_ = 0;
    total_fills_.store(0, std::memory_order_relaxed);
    breach_count_.store(0, std::memory_order_relaxed);
}

void ExecutionQualityMonitor::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    window_.assign(cfg_.window_size, {});
    ring_head_ = 0;
    ring_fill_ = 0;
}

std::string ExecutionQualityMonitor::to_stats_json() const {
    auto s = compute_stats();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\"samples\":" << s.sample_count
       << ",\"mean_lat_us\":"   << s.mean_latency_us
       << ",\"p50_lat_us\":"    << s.p50_latency_us
       << ",\"p95_lat_us\":"    << s.p95_latency_us
       << ",\"p99_lat_us\":"    << s.p99_latency_us
       << ",\"mean_slip_bps\":" << s.mean_slippage_bps
       << ",\"p95_slip_bps\":"  << s.p95_slippage_bps
       << ",\"sla_breaches\":"  << breach_count_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

} // namespace llmquant
