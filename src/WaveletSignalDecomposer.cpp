#include "WaveletSignalDecomposer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

WaveletSignalDecomposer::WaveletSignalDecomposer(Config cfg)
    : cfg_(std::move(cfg))
{
    int levels = std::max(1, std::min(8, cfg_.levels));
    int window = std::max(1 << levels, cfg_.window);
    // Round window up to nearest power of two >= 2^levels
    int pw = 1 << levels;
    while (pw < window) pw <<= 1;
    cfg_.window  = pw;
    cfg_.levels  = levels;
    if (cfg_.min_samples <= 0) cfg_.min_samples = cfg_.window;

    buf_.assign(static_cast<std::size_t>(cfg_.window), 0.0);
    level_energy_.assign(static_cast<std::size_t>(cfg_.levels), 0.0);
    energy_atoms_ = std::vector<std::atomic<double>>(static_cast<std::size_t>(cfg_.levels));
    for (auto& a : energy_atoms_) a.store(0.0, std::memory_order_relaxed);
}

void WaveletSignalDecomposer::run_transform_locked() {
    // Build linear buffer in chronological order from circular buf_
    int w = cfg_.window;
    std::vector<double> tmp(static_cast<std::size_t>(w));
    for (int i = 0; i < w; ++i) {
        int idx = (head_ - w + i + w * 2) % w;
        tmp[static_cast<std::size_t>(i)] = buf_[static_cast<std::size_t>(idx)];
    }

    int len = w;
    for (int lvl = 0; lvl < cfg_.levels; ++lvl) {
        int half = len / 2;
        std::vector<double> approx_v(static_cast<std::size_t>(half));
        std::vector<double> detail_v(static_cast<std::size_t>(half));
        for (int i = 0; i < half; ++i) {
            approx_v[static_cast<std::size_t>(i)] =
                (tmp[static_cast<std::size_t>(2*i)] + tmp[static_cast<std::size_t>(2*i+1)]) * 0.5;
            detail_v[static_cast<std::size_t>(i)] =
                (tmp[static_cast<std::size_t>(2*i)] - tmp[static_cast<std::size_t>(2*i+1)]) * 0.5;
        }

        // Compute mean-squared energy of detail coefficients at this level
        double energy = 0.0;
        for (double d : detail_v) energy += d * d;
        energy /= static_cast<double>(half);
        level_energy_[static_cast<std::size_t>(lvl)] = energy;
        energy_atoms_[static_cast<std::size_t>(lvl)].store(energy, std::memory_order_relaxed);

        // Put approx into first half of tmp for next pass
        for (int i = 0; i < half; ++i)
            tmp[static_cast<std::size_t>(i)] = approx_v[static_cast<std::size_t>(i)];
        len = half;
    }

    // After J levels, tmp[0] is the final approximation (mean of means)
    approx_.store(tmp[0], std::memory_order_relaxed);

    // Find max energy across all levels
    double max_e = 0.0;
    for (double e : level_energy_) max_e = std::max(max_e, e);
    max_energy_.store(max_e, std::memory_order_relaxed);
}

void WaveletSignalDecomposer::record(double x) {
    bool   fire_spike = false, fire_clear = false;
    int    spike_lvl  = -1;
    double spike_e    = 0.0;
    std::function<void(int, double)> spike_cb;
    std::function<void()>            clear_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        spike_cb = cfg_.on_high_freq_spike;
        clear_cb = cfg_.on_spike_clear;
        total_.fetch_add(1, std::memory_order_relaxed);

        int w = cfg_.window;
        buf_[static_cast<std::size_t>(head_)] = x;
        head_ = (head_ + 1) % w;
        if (fill_ < w) ++fill_;

        if (fill_ < cfg_.min_samples) return;

        run_transform_locked();

        double max_e = max_energy_.load(std::memory_order_relaxed);
        bool   now_spiking = (max_e > cfg_.spike_threshold);

        if (now_spiking && !prev_spiking_) {
            fire_spike = true;
            spike_events_.fetch_add(1, std::memory_order_relaxed);
            // Find the level with max energy
            spike_lvl = 0;
            for (int i = 1; i < cfg_.levels; ++i) {
                if (level_energy_[static_cast<std::size_t>(i)] >
                    level_energy_[static_cast<std::size_t>(spike_lvl)])
                    spike_lvl = i;
            }
            spike_e = level_energy_[static_cast<std::size_t>(spike_lvl)];
        } else if (!now_spiking && prev_spiking_ &&
                   max_e < cfg_.spike_threshold * cfg_.clear_hysteresis) {
            fire_clear = true;
        }

        spiking_.store(now_spiking, std::memory_order_relaxed);
        prev_spiking_ = now_spiking;
    }

    if (fire_spike && spike_cb) try { spike_cb(spike_lvl, spike_e); } catch (...) {}
    if (fire_clear && clear_cb) try { clear_cb(); } catch (...) {}
}

double WaveletSignalDecomposer::detail_energy(int level) const noexcept {
    if (level < 0 || level >= static_cast<int>(energy_atoms_.size())) return 0.0;
    return energy_atoms_[static_cast<std::size_t>(level)].load(std::memory_order_relaxed);
}

std::string WaveletSignalDecomposer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"approx_mean\":"     << approx_.load(std::memory_order_relaxed) << ","
       << "\"max_detail_energy\":" << max_energy_.load(std::memory_order_relaxed) << ","
       << "\"is_spiking\":"      << (spiking_.load(std::memory_order_relaxed) ? "true" : "false") << ","
       << "\"spike_events\":"    << spike_events_.load(std::memory_order_relaxed) << ","
       << "\"total_records\":"   << total_.load(std::memory_order_relaxed) << ","
       << "\"detail_energies\":[";
    for (int i = 0; i < cfg_.levels; ++i) {
        if (i > 0) ss << ",";
        ss << energy_atoms_[static_cast<std::size_t>(i)].load(std::memory_order_relaxed);
    }
    ss << "]}";
    return ss.str();
}

void WaveletSignalDecomposer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    std::fill(level_energy_.begin(), level_energy_.end(), 0.0);
    for (auto& a : energy_atoms_) a.store(0.0, std::memory_order_relaxed);
    head_  = fill_ = 0;
    prev_spiking_ = false;
    spike_level_  = -1;
    approx_.store(0.0, std::memory_order_relaxed);
    max_energy_.store(0.0, std::memory_order_relaxed);
    spiking_.store(false, std::memory_order_relaxed);
    spike_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void WaveletSignalDecomposer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int levels = std::max(1, std::min(8, cfg_.levels));
    int window = std::max(1 << levels, cfg_.window);
    int pw = 1 << levels;
    while (pw < window) pw <<= 1;
    cfg_.window  = pw;
    cfg_.levels  = levels;
    if (cfg_.min_samples <= 0) cfg_.min_samples = cfg_.window;

    buf_.assign(static_cast<std::size_t>(cfg_.window), 0.0);
    level_energy_.assign(static_cast<std::size_t>(cfg_.levels), 0.0);
    energy_atoms_ = std::vector<std::atomic<double>>(static_cast<std::size_t>(cfg_.levels));
    for (auto& a : energy_atoms_) a.store(0.0, std::memory_order_relaxed);
    head_  = fill_ = 0;
    prev_spiking_ = false;
    spike_level_  = -1;
    approx_.store(0.0, std::memory_order_relaxed);
    max_energy_.store(0.0, std::memory_order_relaxed);
    spiking_.store(false, std::memory_order_relaxed);
    spike_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
