#include "OnlineGrangerCausality.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace llmquant {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

OnlineGrangerCausality::OnlineGrangerCausality(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(4, cfg_.window);
    cfg_.window = w;
    if (cfg_.min_samples <= 0) cfg_.min_samples = w;
    buf_.resize(static_cast<std::size_t>(w));
}

// Fit restricted OLS: dep_t = a + b*dep_{t-1}, return SS_residual.
double OnlineGrangerCausality::fit_restricted(
        const std::vector<double>& dep, int n,
        double& a_out, double& b_out)
{
    // dep[0..n-2] are lag-1, dep[1..n-1] are current
    int m = n - 1;
    if (m < 2) { a_out = b_out = 0.0; return 0.0; }

    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < m; ++i) {
        double xi = dep[static_cast<std::size_t>(i)];
        double yi = dep[static_cast<std::size_t>(i + 1)];
        sum_x  += xi; sum_y  += yi;
        sum_xx += xi * xi; sum_xy += xi * yi;
    }
    double dm = static_cast<double>(m);
    double denom = dm * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) {
        a_out = sum_y / dm; b_out = 0.0;
    } else {
        b_out = (dm * sum_xy - sum_x * sum_y) / denom;
        a_out = (sum_y - b_out * sum_x) / dm;
    }
    double ss = 0.0;
    for (int i = 0; i < m; ++i) {
        double yi = dep[static_cast<std::size_t>(i + 1)];
        double yh = a_out + b_out * dep[static_cast<std::size_t>(i)];
        double r  = yi - yh;
        ss += r * r;
    }
    return ss;
}

// Fit unrestricted OLS: dep_t = a + b*dep_{t-1} + c*lag_{t-1}, return SS_residual.
double OnlineGrangerCausality::fit_unrestricted(
        const std::vector<double>& dep,
        const std::vector<double>& lag, int n)
{
    int m = n - 1;
    if (m < 3) { double a_tmp = 0.0, b_tmp = 0.0; return fit_restricted(dep, n, a_tmp, b_tmp); }

    // Build normal equations for 3-parameter OLS: [1, dep_lag, other_lag]
    // Solve 3×3 system using Cramer's rule
    double s1 = 0.0, sx = 0.0, sz = 0.0;
    double sxx = 0.0, sxz = 0.0, szz = 0.0;
    double sy = 0.0, sxy = 0.0, szy = 0.0;
    for (int i = 0; i < m; ++i) {
        double xi = dep[static_cast<std::size_t>(i)];
        double zi = lag[static_cast<std::size_t>(i)];
        double yi = dep[static_cast<std::size_t>(i + 1)];
        double dm_one = 1.0;
        s1  += dm_one;
        sx  += xi; sz  += zi;
        sxx += xi * xi; sxz += xi * zi; szz += zi * zi;
        sy  += yi; sxy += xi * yi; szy += zi * yi;
    }

    // Normal equations: [s1 sx sz; sx sxx sxz; sz sxz szz] * [a;b;c] = [sy;sxy;szy]
    double A[3][3] = {{s1,  sx,  sz},
                      {sx,  sxx, sxz},
                      {sz,  sxz, szz}};
    double B[3]    = {sy, sxy, szy};

    // Gaussian elimination with partial pivoting
    double coef[3] = {0.0, 0.0, 0.0};
    for (int col = 0; col < 3; ++col) {
        // Find pivot
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
        if (pivot != col) {
            for (int j = 0; j < 3; ++j) std::swap(A[col][j], A[pivot][j]);
            std::swap(B[col], B[pivot]);
        }
        if (std::abs(A[col][col]) < 1e-12) { coef[col] = 0.0; continue; }
        for (int row = col + 1; row < 3; ++row) {
            double f = A[row][col] / A[col][col];
            for (int j = col; j < 3; ++j) A[row][j] -= f * A[col][j];
            B[row] -= f * B[col];
        }
    }
    // Back-substitution
    for (int row = 2; row >= 0; --row) {
        if (std::abs(A[row][row]) < 1e-12) { coef[row] = 0.0; continue; }
        double sum = B[row];
        for (int j = row + 1; j < 3; ++j) sum -= A[row][j] * coef[j];
        coef[row] = sum / A[row][row];
    }

    double ss = 0.0;
    for (int i = 0; i < m; ++i) {
        double yi = dep[static_cast<std::size_t>(i + 1)];
        double yh = coef[0]
                  + coef[1] * dep[static_cast<std::size_t>(i)]
                  + coef[2] * lag[static_cast<std::size_t>(i)];
        double r = yi - yh;
        ss += r * r;
    }
    return ss;
}

void OnlineGrangerCausality::recompute_locked() {
    int n = fill_;
    if (n < cfg_.min_samples) {
        f_xy_.store(kNaN, std::memory_order_relaxed);
        f_yx_.store(kNaN, std::memory_order_relaxed);
        return;
    }

    int w = cfg_.window;

    // Extract linear sequences from circular buffer
    std::vector<double> xs(static_cast<std::size_t>(n));
    std::vector<double> ys(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = (head_ - n + i + w * 2) % w;
        xs[static_cast<std::size_t>(i)] = buf_[static_cast<std::size_t>(idx)].x;
        ys[static_cast<std::size_t>(i)] = buf_[static_cast<std::size_t>(idx)].y;
    }

    // F(X→Y): restricted=Y~Y_lag, unrestricted=Y~Y_lag+X_lag
    double a_r, b_r;
    double ss_r_xy = fit_restricted(ys, n, a_r, b_r);
    double ss_u_xy = fit_unrestricted(ys, xs, n);

    // F(Y→X): restricted=X~X_lag, unrestricted=X~X_lag+Y_lag
    double ss_r_yx = fit_restricted(xs, n, a_r, b_r);
    double ss_u_yx = fit_unrestricted(xs, ys, n);

    double dof = static_cast<double>(n - 3);  // n observations, 3 params in unres.

    auto compute_f = [&](double ss_r, double ss_u) -> double {
        if (ss_u < 1e-12 || dof <= 0.0) return kNaN;
        double improvement = ss_r - ss_u;
        if (improvement < 0.0) improvement = 0.0;
        return (improvement / ss_u) * dof;
    };

    f_xy_.store(compute_f(ss_r_xy, ss_u_xy), std::memory_order_relaxed);
    f_yx_.store(compute_f(ss_r_yx, ss_u_yx), std::memory_order_relaxed);
}

void OnlineGrangerCausality::record(double x, double y) {
    bool   fire_xy = false, fire_yx = false, fire_bidi = false;
    double fxy_val = kNaN, fyx_val = kNaN;
    std::function<void(double)> xy_cb, yx_cb;
    std::function<void()>       bidi_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        xy_cb   = cfg_.on_x_causes_y;
        yx_cb   = cfg_.on_y_causes_x;
        bidi_cb = cfg_.on_bidirectional;

        total_.fetch_add(1, std::memory_order_relaxed);

        int w = cfg_.window;
        buf_[static_cast<std::size_t>(head_)] = {x, y};
        head_ = (head_ + 1) % w;
        if (fill_ < w) ++fill_;

        recompute_locked();

        fxy_val = f_xy_.load(std::memory_order_relaxed);
        fyx_val = f_yx_.load(std::memory_order_relaxed);

        if (!std::isfinite(fxy_val) || !std::isfinite(fyx_val)) return;

        double thr = cfg_.f_threshold;
        double hys = thr * cfg_.hysteresis;

        bool now_xcay = (fxy_val >= thr);
        bool now_ycax = (fyx_val >= thr);

        // Apply hysteresis for clearing
        if (prev_xcay_ && fxy_val >= hys) now_xcay = true;
        if (prev_ycax_ && fyx_val >= hys) now_ycax = true;

        if (now_xcay && !prev_xcay_) {
            fire_xy = true;
            xy_events_.fetch_add(1, std::memory_order_relaxed);
        }
        if (now_ycax && !prev_ycax_) {
            fire_yx = true;
            yx_events_.fetch_add(1, std::memory_order_relaxed);
        }
        if (now_xcay && now_ycax && !(prev_xcay_ && prev_ycax_)) {
            fire_bidi = true;
        }

        prev_xcay_ = now_xcay;
        prev_ycax_ = now_ycax;
        xcay_.store(now_xcay, std::memory_order_relaxed);
        ycax_.store(now_ycax, std::memory_order_relaxed);
    }

    if (fire_xy   && xy_cb)   try { xy_cb(fxy_val); }   catch (...) {}
    if (fire_yx   && yx_cb)   try { yx_cb(fyx_val); }   catch (...) {}
    if (fire_bidi && bidi_cb) try { bidi_cb(); }         catch (...) {}
}

std::string OnlineGrangerCausality::to_stats_json() const {
    double fxy = f_xy_.load(std::memory_order_relaxed);
    double fyx = f_yx_.load(std::memory_order_relaxed);
    bool   xcy = xcay_.load(std::memory_order_relaxed);
    bool   ycx = ycax_.load(std::memory_order_relaxed);
    uint64_t tot = total_.load(std::memory_order_relaxed);
    uint64_t xye = xy_events_.load(std::memory_order_relaxed);
    uint64_t yxe = yx_events_.load(std::memory_order_relaxed);

    auto fmt = [](double v) -> std::string {
        if (!std::isfinite(v)) return "null";
        char tmp[32]; std::snprintf(tmp, sizeof(tmp), "%.4f", v); return tmp;
    };
    char buf[320];
    std::snprintf(buf, sizeof(buf),
        "{\"f_xy\":%s,\"f_yx\":%s,\"x_causes_y\":%s,\"y_causes_x\":%s,"
        "\"xy_causality_events\":%llu,\"yx_causality_events\":%llu,"
        "\"total_records\":%llu}",
        fmt(fxy).c_str(), fmt(fyx).c_str(),
        xcy ? "true" : "false", ycx ? "true" : "false",
        static_cast<unsigned long long>(xye),
        static_cast<unsigned long long>(yxe),
        static_cast<unsigned long long>(tot));
    return buf;
}

void OnlineGrangerCausality::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), Sample{0.0, 0.0});
    head_ = fill_ = 0;
    prev_xcay_ = prev_ycax_ = false;
    f_xy_.store(kNaN, std::memory_order_relaxed);
    f_yx_.store(kNaN, std::memory_order_relaxed);
    xcay_.store(false, std::memory_order_relaxed);
    ycax_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    xy_events_.store(0, std::memory_order_relaxed);
    yx_events_.store(0, std::memory_order_relaxed);
}

void OnlineGrangerCausality::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(4, cfg_.window);
    cfg_.window = w;
    if (cfg_.min_samples <= 0) cfg_.min_samples = w;
    buf_.assign(static_cast<std::size_t>(w), Sample{0.0, 0.0});
    head_ = fill_ = 0;
    prev_xcay_ = prev_ycax_ = false;
    f_xy_.store(kNaN, std::memory_order_relaxed);
    f_yx_.store(kNaN, std::memory_order_relaxed);
    xcay_.store(false, std::memory_order_relaxed);
    ycax_.store(false, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    xy_events_.store(0, std::memory_order_relaxed);
    yx_events_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
