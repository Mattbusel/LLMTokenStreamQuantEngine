#include "SignalParabolicSAR.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalParabolicSAR::SignalParabolicSAR(Config cfg)
    : cfg_(std::move(cfg))
    , af_val_(cfg_.af_start)
{}

void SignalParabolicSAR::record(double bias) {
    double new_sar{};
    double new_af{};
    int    new_dir{};
    bool   reversed = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (!seeded_) {
            // Initialise
            sar_val_ = bias;
            ep_      = bias;
            af_val_  = cfg_.af_start;
            dir_     = 1;  // assume uptrend
            seeded_  = true;
            new_sar  = sar_val_;
            new_af   = af_val_;
            new_dir  = dir_;
        } else {
            // Check for reversal
            bool reversal = (dir_ == 1 && bias < sar_val_) ||
                            (dir_ == -1 && bias > sar_val_);

            if (reversal) {
                // Reverse direction
                dir_     = -dir_;
                sar_val_ = ep_;
                ep_      = bias;
                af_val_  = cfg_.af_start;
                reversed = true;
                reversals_.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Update EP and AF
                if (dir_ == 1 && bias > ep_) {
                    ep_    = bias;
                    af_val_ = std::min(af_val_ + cfg_.af_step, cfg_.af_max);
                } else if (dir_ == -1 && bias < ep_) {
                    ep_    = bias;
                    af_val_ = std::min(af_val_ + cfg_.af_step, cfg_.af_max);
                }
            }

            // Update SAR
            sar_val_ = sar_val_ + af_val_ * (ep_ - sar_val_);

            new_sar = sar_val_;
            new_af  = af_val_;
            new_dir = dir_;
        }
    }

    sar_.store(new_sar, std::memory_order_relaxed);
    af_.store(new_af, std::memory_order_relaxed);
    trend_.store(new_dir, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (reversed && cfg_.on_reversal) cfg_.on_reversal(new_dir, new_sar);
}

std::string SignalParabolicSAR::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"sar\":" << sar_.load(std::memory_order_relaxed)
       << ",\"trend\":" << trend_.load(std::memory_order_relaxed)
       << ",\"af\":"   << af_.load(std::memory_order_relaxed)
       << ",\"reversals\":" << reversals_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void SignalParabolicSAR::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    sar_val_ = 0.0;
    ep_      = 0.0;
    af_val_  = cfg_.af_start;
    dir_     = 1;
    seeded_  = false;
    sar_.store(0.0, std::memory_order_relaxed);
    trend_.store(1, std::memory_order_relaxed);
    af_.store(cfg_.af_start, std::memory_order_relaxed);
    reversals_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalParabolicSAR::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
