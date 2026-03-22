#include "SignalCalibrationEngine.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SignalCalibrationEngine::SignalCalibrationEngine(Config config)
    : config_(std::move(config)) {
    window_.resize(static_cast<size_t>(config_.window_size));
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

/* static */
double SignalCalibrationEngine::sigmoid(double x) noexcept {
    // Numerically stable sigmoid.
    if (x >= 0.0)
        return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}

// ---------------------------------------------------------------------------
// Online learning
// ---------------------------------------------------------------------------

void SignalCalibrationEngine::gradient_step_locked(double score, double target) {
    // σ = sigmoid(A·score + B)
    double z     = platt_a_ * score + platt_b_;
    double sigma = sigmoid(z);
    double err   = sigma - target;  // ∂L/∂z

    // Gradient of loss w.r.t. A and B (+ L2 regularisation).
    double dA = err * score + config_.l2_reg * platt_a_;
    double dB = err          + config_.l2_reg * platt_b_;

    platt_a_ -= config_.learning_rate * dA;
    platt_b_ -= config_.learning_rate * dB;
}

void SignalCalibrationEngine::record_outcome(double raw_score,
                                             bool positive_outcome) {
    total_recorded_.fetch_add(1, std::memory_order_relaxed);

    double ece = 0.0;
    std::function<void(double)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Evict oldest sample if window is full.
        if (ring_fill_ == config_.window_size) {
            // No correction needed for Platt params — they're updated
            // incrementally; old gradients are naturally "forgotten".
        } else {
            ++ring_fill_;
        }

        // Write into ring buffer.
        window_[static_cast<size_t>(ring_head_)] = {raw_score, positive_outcome};
        ring_head_ = (ring_head_ + 1) % config_.window_size;

        // Gradient step on new sample.
        gradient_step_locked(raw_score, positive_outcome ? 1.0 : 0.0);

        ece = (ring_fill_ >= config_.min_samples)
            ? expected_calibration_error()
            : 0.0;

        cb = config_.on_recalibrated;
    }

    sample_count_.store(static_cast<uint64_t>(ring_fill_),
                        std::memory_order_relaxed);

    if (cb) cb(ece);
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

double SignalCalibrationEngine::calibrate(double raw_score) const noexcept {
    if (static_cast<int>(sample_count_.load(std::memory_order_relaxed))
            < config_.min_samples) {
        // Identity calibration: clamp to (0,1).
        return std::max(1e-6, std::min(1.0 - 1e-6, raw_score));
    }
    std::lock_guard<std::mutex> lk(mutex_);
    double z = platt_a_ * raw_score + platt_b_;
    return sigmoid(z);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

double SignalCalibrationEngine::expected_calibration_error() const {
    // Must be called with mutex_ held (or from within locked section).
    if (ring_fill_ < config_.min_samples) return 0.0;

    int n_bins = config_.n_bins;
    double bin_w = 1.0 / n_bins;

    struct Bin { double sum_pred = 0.0; double sum_actual = 0.0; int cnt = 0; };
    std::vector<Bin> bins(static_cast<size_t>(n_bins));

    int n = ring_fill_;
    for (int i = 0; i < n; ++i) {
        const auto& s = window_[static_cast<size_t>(i)];
        double pred = sigmoid(platt_a_ * s.score + platt_b_);
        int b = static_cast<int>(pred / bin_w);
        b = std::max(0, std::min(n_bins - 1, b));
        bins[static_cast<size_t>(b)].sum_pred   += pred;
        bins[static_cast<size_t>(b)].sum_actual += s.outcome ? 1.0 : 0.0;
        bins[static_cast<size_t>(b)].cnt++;
    }

    double ece = 0.0;
    for (const auto& b : bins) {
        if (b.cnt == 0) continue;
        double mp = b.sum_pred   / b.cnt;
        double ma = b.sum_actual / b.cnt;
        ece += static_cast<double>(b.cnt) / n * std::abs(mp - ma);
    }
    return ece;
}

std::vector<SignalCalibrationEngine::ReliabilityBin>
SignalCalibrationEngine::reliability_diagram_bins() const {
    std::lock_guard<std::mutex> lk(mutex_);

    int n_bins = config_.n_bins;
    double bin_w = 1.0 / n_bins;
    std::vector<ReliabilityBin> result(static_cast<size_t>(n_bins));
    for (int i = 0; i < n_bins; ++i) {
        result[static_cast<size_t>(i)].bin_low  = i       * bin_w;
        result[static_cast<size_t>(i)].bin_high = (i + 1) * bin_w;
    }

    int n = ring_fill_;
    for (int i = 0; i < n; ++i) {
        const auto& s = window_[static_cast<size_t>(i)];
        // Bin by raw score.
        int b = static_cast<int>(s.score / bin_w);
        b = std::max(0, std::min(n_bins - 1, b));
        double pred = sigmoid(platt_a_ * s.score + platt_b_);
        result[static_cast<size_t>(b)].mean_predicted += pred;
        result[static_cast<size_t>(b)].mean_actual    += s.outcome ? 1.0 : 0.0;
        result[static_cast<size_t>(b)].count++;
    }

    for (auto& rb : result) {
        if (rb.count > 0) {
            rb.mean_predicted /= rb.count;
            rb.mean_actual    /= rb.count;
        }
    }
    return result;
}

double SignalCalibrationEngine::platt_a() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return platt_a_;
}

double SignalCalibrationEngine::platt_b() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return platt_b_;
}

// ---------------------------------------------------------------------------
// Config / reset
// ---------------------------------------------------------------------------

void SignalCalibrationEngine::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    window_.assign(static_cast<size_t>(config_.window_size), {0.0, false});
    ring_head_ = 0;
    ring_fill_ = 0;
    platt_a_   = 1.0;
    platt_b_   = 0.0;
    sample_count_.store(0, std::memory_order_relaxed);
    total_recorded_.store(0, std::memory_order_relaxed);
}

SignalCalibrationEngine::Config SignalCalibrationEngine::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void SignalCalibrationEngine::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(window_.begin(), window_.end(), Sample{0.0, false});
    ring_head_ = 0;
    ring_fill_ = 0;
    platt_a_   = 1.0;
    platt_b_   = 0.0;
    sample_count_.store(0, std::memory_order_relaxed);
    total_recorded_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// JSON stats
// ---------------------------------------------------------------------------

std::string SignalCalibrationEngine::to_stats_json() const {
    double ece = 0.0;
    double a = 0.0, b = 0.0;
    uint64_t sc = 0, tr = 0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        a   = platt_a_;
        b   = platt_b_;
        ece = (ring_fill_ >= config_.min_samples) ? expected_calibration_error() : 0.0;
    }
    sc = sample_count_.load(std::memory_order_relaxed);
    tr = total_recorded_.load(std::memory_order_relaxed);

    std::ostringstream os;
    os << std::fixed;
    os.precision(6);
    os << "{\"platt_a\":"  << a
       << ",\"platt_b\":"  << b
       << ",\"ece\":"      << ece
       << ",\"samples\":"  << sc
       << ",\"total_recorded\":" << tr << "}";
    return os.str();
}

} // namespace llmquant
