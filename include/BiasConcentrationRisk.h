#pragma once
/// BiasConcentrationRisk — Herfindahl-Hirschman Index (HHI) over bias magnitudes.
/// Computes HHI = Σ(|x_i| / Σ|x_j|)² over rolling window.
/// HHI near 1 → energy concentrated in single sample (spike).
/// HHI near 1/n → uniformly spread energy (normal noise).
/// Fires on_concentrated when HHI exceeds concentration_threshold.
///
/// Feature flag: LLMQUANT_ENABLE_CONCENTRATION_RISK / LLMQUANT_CONCENTRATION_RISK_ENABLED

#ifdef LLMQUANT_CONCENTRATION_RISK_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class BiasConcentrationRisk {
public:
    struct Config {
        int    window                = 20;   ///< rolling window
        int    min_samples           = 5;    ///< minimum before emitting HHI
        double concentration_threshold = 0.3; ///< HHI above this → on_concentrated

        /// Fires when HHI first exceeds concentration_threshold.
        std::function<void(double hhi)> on_concentrated;
        /// Fires when HHI drops back below concentration_threshold.
        std::function<void(double hhi)> on_dispersed;
    };

    explicit BiasConcentrationRisk(Config cfg = {});

    void record(double bias);

    double hhi() const noexcept; ///< Herfindahl-Hirschman Index ∈ [1/n, 1]

    std::size_t concentration_events() const noexcept;
    std::size_t dispersed_events()     const noexcept;
    std::size_t total_records()        const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> buf_;
    int    head_           = 0;
    int    fill_           = 0;
    bool   prev_concentrated_ = false;

    std::atomic<double>      hhi_{0.0};
    std::atomic<std::size_t> conc_events_{0};
    std::atomic<std::size_t> disp_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_CONCENTRATION_RISK_ENABLED
