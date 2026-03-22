#include "BiasRunLengthEncoder.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace llmquant {

BiasRunLengthEncoder::BiasRunLengthEncoder(Config cfg)
    : cfg_(std::move(cfg)) {}

void BiasRunLengthEncoder::record(double bias) {
    int  fire_dir = 0;
    int  fire_len = 0;
    bool do_fire  = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        obs_count_.fetch_add(1, std::memory_order_relaxed);

        int dir = 0;
        if (bias > 0.0)      dir = +1;
        else if (bias < 0.0) dir = -1;

        if (dir == cur_dir_) {
            ++cur_len_;
        } else {
            // Finalize the previous run (if any).
            if (cur_len_ > 0) {
                run_count_.fetch_add(1, std::memory_order_relaxed);
                if (cur_len_ > max_len_) {
                    max_len_ = cur_len_;
                    max_run_.store(max_len_, std::memory_order_relaxed);
                }
                if (!avg_seeded_) {
                    avg_len_   = static_cast<double>(cur_len_);
                    avg_seeded_ = true;
                } else {
                    avg_len_ = cfg_.avg_alpha * static_cast<double>(cur_len_)
                             + (1.0 - cfg_.avg_alpha) * avg_len_;
                }
                avg_run_.store(avg_len_, std::memory_order_relaxed);

                if (cur_len_ >= cfg_.long_run_threshold && cfg_.on_long_run) {
                    fire_dir = cur_dir_;
                    fire_len = cur_len_;
                    do_fire  = true;
                }
            }
            // Start new run.
            cur_dir_ = dir;
            cur_len_ = 1;
        }

        current_run_.store(cur_len_, std::memory_order_relaxed);
        current_dir_.store(cur_dir_, std::memory_order_relaxed);
    }

    if (do_fire) cfg_.on_long_run(fire_dir, fire_len);
}

std::string BiasRunLengthEncoder::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"current_run\":" << current_run_.load(std::memory_order_relaxed)
        << ",\"current_dir\":" << current_dir_.load(std::memory_order_relaxed)
        << ",\"max_run\":" << max_run_.load(std::memory_order_relaxed)
        << ",\"avg_run\":" << avg_run_.load(std::memory_order_relaxed)
        << ",\"run_count\":" << run_count_.load(std::memory_order_relaxed)
        << ",\"obs_count\":" << obs_count_.load(std::memory_order_relaxed)
        << "}";
    return oss.str();
}

void BiasRunLengthEncoder::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cur_dir_    = 0;
    cur_len_    = 0;
    max_len_    = 0;
    avg_len_    = 0.0;
    avg_seeded_ = false;
    current_run_.store(0, std::memory_order_relaxed);
    current_dir_.store(0, std::memory_order_relaxed);
    max_run_.store(0, std::memory_order_relaxed);
    avg_run_.store(0.0, std::memory_order_relaxed);
    run_count_.store(0, std::memory_order_relaxed);
    obs_count_.store(0, std::memory_order_relaxed);
}

void BiasRunLengthEncoder::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
