#include "SignalRegimeHMM.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

namespace {
constexpr double kEps = 1e-300;
constexpr double kTwoPi = 6.283185307179586;

double gaussian_pdf(double x, double mu, double sigma) {
    double z = (x - mu) / (sigma + 1e-10);
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(kTwoPi) + 1e-10);
}
} // namespace

SignalRegimeHMM::SignalRegimeHMM(Config cfg)
    : cfg_(std::move(cfg))
    , alpha_{cfg_.init_prob[0], cfg_.init_prob[1]}
{}

void SignalRegimeHMM::record(double bias) {
    double p_bull{};
    int    new_state{};
    bool   changed = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Forward prediction: α_pred[j] = Σ_i α[i] * A[i][j]
        std::array<double, kStates> alpha_pred{};
        for (int j = 0; j < kStates; ++j) {
            for (int i = 0; i < kStates; ++i) {
                alpha_pred[j] += alpha_[i] * cfg_.transition[i][j];
            }
        }

        // Update with emission: α[j] *= N(x; μ_j, σ_j)
        double Z = 0.0;
        for (int j = 0; j < kStates; ++j) {
            alpha_[j] = alpha_pred[j] * gaussian_pdf(bias,
                            cfg_.state_mean[j], cfg_.state_std[j]);
            alpha_[j] += kEps;
            Z += alpha_[j];
        }
        // Normalise
        for (int j = 0; j < kStates; ++j) alpha_[j] /= Z;

        p_bull = alpha_[1];
        new_state = (p_bull >= cfg_.bullish_threshold) ? 1 : 0;

        uint64_t n = total_.load(std::memory_order_relaxed);
        if (n >= static_cast<uint64_t>(cfg_.min_samples)
                && prev_state_ >= 0 && new_state != prev_state_) {
            changed = true;
            changes_.fetch_add(1, std::memory_order_relaxed);
        }
        int old_state = prev_state_;
        prev_state_   = new_state;
        (void)old_state;

        if (changed && cfg_.on_regime_change) {
            cfg_.on_regime_change(prev_state_ == new_state ? 1 - new_state : prev_state_,
                                  new_state);
        }
    }

    p_bull_.store(p_bull, std::memory_order_relaxed);
    state_.store(new_state, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);
}

std::string SignalRegimeHMM::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"p_bullish\":" << p_bull_.load(std::memory_order_relaxed)
       << ",\"state\":" << state_.load(std::memory_order_relaxed)
       << ",\"regime_changes\":" << changes_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalRegimeHMM::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    alpha_ = {cfg_.init_prob[0], cfg_.init_prob[1]};
    prev_state_ = -1;
    p_bull_.store(0.5, std::memory_order_relaxed);
    state_.store(-1, std::memory_order_relaxed);
    changes_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalRegimeHMM::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
