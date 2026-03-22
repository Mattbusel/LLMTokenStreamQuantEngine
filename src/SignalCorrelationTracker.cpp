#include "SignalCorrelationTracker.h"

#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SignalCorrelationTracker::SignalCorrelationTracker(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// Pearson correlation (static helper)
// ---------------------------------------------------------------------------

double SignalCorrelationTracker::pearson(const std::deque<double>& a,
                                          const std::deque<double>& b,
                                          int n) noexcept {
    if (n <= 1) return std::numeric_limits<double>::quiet_NaN();

    // Use the last `n` elements from each deque.
    int offset_a = static_cast<int>(a.size()) - n;
    int offset_b = static_cast<int>(b.size()) - n;
    if (offset_a < 0 || offset_b < 0)
        return std::numeric_limits<double>::quiet_NaN();

    double sum_a = 0.0, sum_b = 0.0;
    double sum_aa = 0.0, sum_bb = 0.0, sum_ab = 0.0;
    for (int i = 0; i < n; ++i) {
        double ai = a[static_cast<size_t>(offset_a + i)];
        double bi = b[static_cast<size_t>(offset_b + i)];
        sum_a  += ai;
        sum_b  += bi;
        sum_aa += ai * ai;
        sum_bb += bi * bi;
        sum_ab += ai * bi;
    }

    double dn = static_cast<double>(n);
    double num  = dn * sum_ab - sum_a * sum_b;
    double den2 = (dn * sum_aa - sum_a * sum_a) * (dn * sum_bb - sum_b * sum_b);
    if (den2 <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    double r = num / std::sqrt(den2);
    // Clamp to [-1, 1] to handle floating-point rounding.
    if (r >  1.0) r =  1.0;
    if (r < -1.0) r = -1.0;
    return r;
}

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

void SignalCorrelationTracker::record(const std::string& source, double value) {
    total_samples_.fetch_add(1, std::memory_order_relaxed);

    std::vector<PendingEvent> events;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto& win = windows_[source];
        win.push_back(value);
        while (static_cast<int>(win.size()) > config_.window_size)
            win.pop_front();
        events = recompute_pairs_locked(source);
    }

    // Fire callbacks outside the lock.
    for (const auto& ev : events) {
        if (ev.is_divergence) {
            divergence_events_.fetch_add(1, std::memory_order_relaxed);
            spdlog::debug("[signal_corr] DIVERGE {}<>{} r={:.3f}",
                          ev.src_a, ev.src_b, ev.correlation);
            if (divergence_cb_) divergence_cb_(ev.src_a, ev.src_b, ev.correlation);
        } else {
            convergence_events_.fetch_add(1, std::memory_order_relaxed);
            spdlog::debug("[signal_corr] CONVERGE {}<>{} r={:.3f}",
                          ev.src_a, ev.src_b, ev.correlation);
            if (convergence_cb_) convergence_cb_(ev.src_a, ev.src_b, ev.correlation);
        }
    }
}

// ---------------------------------------------------------------------------
// recompute_pairs_locked
// ---------------------------------------------------------------------------

std::vector<SignalCorrelationTracker::PendingEvent>
SignalCorrelationTracker::recompute_pairs_locked(const std::string& source) {
    std::vector<PendingEvent> events;
    const auto& win_src = windows_.at(source);
    int n_src = static_cast<int>(win_src.size());
    if (n_src < config_.min_samples) return events;

    for (const auto& [other, win_other] : windows_) {
        if (other == source) continue;
        int n_other = static_cast<int>(win_other.size());
        if (n_other < config_.min_samples) continue;

        int n = std::min(n_src, n_other);
        double r = pearson(win_src, win_other, n);
        if (std::isnan(r)) continue;

        if (r < config_.diverge_threshold) {
            PendingEvent ev;
            ev.src_a        = source;
            ev.src_b        = other;
            ev.correlation  = r;
            ev.is_divergence = true;
            events.push_back(std::move(ev));
        } else if (r > config_.converge_threshold) {
            PendingEvent ev;
            ev.src_a        = source;
            ev.src_b        = other;
            ev.correlation  = r;
            ev.is_divergence = false;
            events.push_back(std::move(ev));
        }
    }
    return events;
}

// ---------------------------------------------------------------------------
// get_correlation
// ---------------------------------------------------------------------------

double SignalCorrelationTracker::get_correlation(const std::string& src_a,
                                                  const std::string& src_b) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it_a = windows_.find(src_a);
    auto it_b = windows_.find(src_b);
    if (it_a == windows_.end() || it_b == windows_.end())
        return std::numeric_limits<double>::quiet_NaN();
    int n_a = static_cast<int>(it_a->second.size());
    int n_b = static_cast<int>(it_b->second.size());
    if (n_a < config_.min_samples || n_b < config_.min_samples)
        return std::numeric_limits<double>::quiet_NaN();
    int n = std::min(n_a, n_b);
    return pearson(it_a->second, it_b->second, n);
}

// ---------------------------------------------------------------------------
// source_names / sample_count
// ---------------------------------------------------------------------------

std::vector<std::string> SignalCorrelationTracker::source_names() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::string> names;
    names.reserve(windows_.size());
    for (const auto& [k, _] : windows_) names.push_back(k);
    return names;
}

int SignalCorrelationTracker::sample_count(const std::string& source) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = windows_.find(source);
    if (it == windows_.end()) return 0;
    return static_cast<int>(it->second.size());
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void SignalCorrelationTracker::set_divergence_callback(CorrelationCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    divergence_cb_ = std::move(cb);
}

void SignalCorrelationTracker::set_convergence_callback(CorrelationCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    convergence_cb_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

void SignalCorrelationTracker::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string SignalCorrelationTracker::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream s;
    s << "{\"total_samples\":"    << total_samples_.load(std::memory_order_relaxed)
      << ",\"divergence_events\":" << divergence_events_.load(std::memory_order_relaxed)
      << ",\"convergence_events\":" << convergence_events_.load(std::memory_order_relaxed)
      << ",\"sources\":" << windows_.size()
      << ",\"correlations\":{";

    bool first_pair = true;
    auto names = std::vector<std::string>{};
    for (const auto& [k, _] : windows_) names.push_back(k);
    for (size_t i = 0; i < names.size(); ++i) {
        for (size_t j = i + 1; j < names.size(); ++j) {
            const auto& na = names[i];
            const auto& nb = names[j];
            auto it_a = windows_.find(na);
            auto it_b = windows_.find(nb);
            int na_n = static_cast<int>(it_a->second.size());
            int nb_n = static_cast<int>(it_b->second.size());
            if (na_n < config_.min_samples || nb_n < config_.min_samples) continue;
            double r = pearson(it_a->second, it_b->second, std::min(na_n, nb_n));
            if (std::isnan(r)) continue;
            if (!first_pair) s << ",";
            s << "\"" << na << "<>" << nb << "\":" << r;
            first_pair = false;
        }
    }
    s << "}}";
    return s.str();
}

} // namespace llmquant
