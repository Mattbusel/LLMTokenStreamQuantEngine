#include "SignalTripleEMAOscillator.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalTripleEMAOscillator::SignalTripleEMAOscillator(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalTripleEMAOscillator::record(double bias) {
    double new_trix = 0.0, new_sig = 0.0;
    double new_e1 = 0.0, new_e3 = 0.0;
    bool zero_ev = false, sig_ev = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            ema1_raw_ = ema2_raw_ = ema3_raw_ = bias;
            ema_sig_raw_ = 0.0;
            prev_ema3_   = bias;
            seeded_      = true;
        } else {
            ema1_raw_ += cfg_.alpha * (bias       - ema1_raw_);
            ema2_raw_ += cfg_.alpha * (ema1_raw_  - ema2_raw_);
            ema3_raw_ += cfg_.alpha * (ema2_raw_  - ema3_raw_);

            // TRIX = % change in triple EMA
            if (std::fabs(prev_ema3_) > 1e-12)
                new_trix = (ema3_raw_ - prev_ema3_) / std::fabs(prev_ema3_) * 100.0;
            else
                new_trix = 0.0;

            ema_sig_raw_ += cfg_.alpha_signal * (new_trix - ema_sig_raw_);
            new_sig = ema_sig_raw_;
            prev_ema3_ = ema3_raw_;
        }
        new_e1 = ema1_raw_;
        new_e3 = ema3_raw_;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples)) {
            bool cur_pos   = (new_trix > 0.0);
            bool cur_above = (new_trix > new_sig);

            if (cur_pos != prev_trix_pos_) {
                zero_ev = true;
                zero_cross_.fetch_add(1, std::memory_order_relaxed);
            }
            if (cur_above != prev_above_sig_) {
                sig_ev = true;
                sig_cross_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_trix_pos_  = cur_pos;
            prev_above_sig_ = cur_above;
        }
    }

    ema1_.store(new_e1,   std::memory_order_relaxed);
    ema3_.store(new_e3,   std::memory_order_relaxed);
    trix_.store(new_trix, std::memory_order_relaxed);
    signal_.store(new_sig, std::memory_order_relaxed);
    total_.fetch_add(1,   std::memory_order_relaxed);

    if (zero_ev && cfg_.on_zero_cross)   cfg_.on_zero_cross(new_trix);
    if (sig_ev  && cfg_.on_signal_cross) cfg_.on_signal_cross(new_trix, new_sig);
}

double SignalTripleEMAOscillator::ema1()        const noexcept { return ema1_.load(std::memory_order_relaxed); }
double SignalTripleEMAOscillator::ema3()        const noexcept { return ema3_.load(std::memory_order_relaxed); }
double SignalTripleEMAOscillator::trix()        const noexcept { return trix_.load(std::memory_order_relaxed); }
double SignalTripleEMAOscillator::signal_line() const noexcept { return signal_.load(std::memory_order_relaxed); }

std::size_t SignalTripleEMAOscillator::zero_cross_events()   const noexcept { return zero_cross_.load(std::memory_order_relaxed); }
std::size_t SignalTripleEMAOscillator::signal_cross_events() const noexcept { return sig_cross_.load(std::memory_order_relaxed); }
std::size_t SignalTripleEMAOscillator::total_records()       const noexcept { return total_.load(std::memory_order_relaxed); }

std::string SignalTripleEMAOscillator::to_stats_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\"trix\":"           << trix_.load(std::memory_order_relaxed)
       << ",\"signal_line\":"    << signal_.load(std::memory_order_relaxed)
       << ",\"ema3\":"           << ema3_.load(std::memory_order_relaxed)
       << ",\"zero_crosses\":"   << zero_cross_.load(std::memory_order_relaxed)
       << ",\"signal_crosses\":" << sig_cross_.load(std::memory_order_relaxed)
       << ",\"total_records\":"  << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalTripleEMAOscillator::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    seeded_ = false;
    ema1_raw_ = ema2_raw_ = ema3_raw_ = ema_sig_raw_ = prev_ema3_ = 0.0;
    prev_trix_pos_ = prev_above_sig_ = false;
    ema1_.store(0.0, std::memory_order_relaxed);
    ema3_.store(0.0, std::memory_order_relaxed);
    trix_.store(0.0, std::memory_order_relaxed);
    signal_.store(0.0, std::memory_order_relaxed);
    zero_cross_.store(0, std::memory_order_relaxed);
    sig_cross_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalTripleEMAOscillator::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
