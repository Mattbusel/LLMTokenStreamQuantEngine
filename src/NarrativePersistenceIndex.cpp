#include "NarrativePersistenceIndex.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativePersistenceIndex::NarrativePersistenceIndex(Config cfg)
    : cfg_(std::move(cfg))
    , buf_(cfg_.window, 0.0)
{}

void NarrativePersistenceIndex::record(double bias) {
    double new_pi        = 0.5;
    bool persistent_ev   = false;
    bool reverting_ev    = false;
    bool new_persistent  = false;
    bool new_reverting   = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        buf_[head_] = bias;
        head_ = (head_ + 1) % cfg_.window;
        if (fill_ < cfg_.window) ++fill_;

        if (fill_ >= cfg_.min_samples) {
            int n = fill_;
            int w = cfg_.window;
            // Build chronological snapshot (oldest first)
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                int idx = (head_ - n + i + w * 2) % w;
                sum += buf_[static_cast<std::size_t>(idx)];
            }
            double mean = sum / n;

            // Count runs: consecutive values above or below the mean
            // Average run length / window ∈ (0, 1]
            int runs = 1;
            int idx0 = (head_ - n + w * 2) % w;
            bool above = (buf_[static_cast<std::size_t>(idx0)] >= mean);
            for (int i = 1; i < n; ++i) {
                int idx = (head_ - n + i + w * 2) % w;
                bool a = (buf_[static_cast<std::size_t>(idx)] >= mean);
                if (a != above) { ++runs; above = a; }
            }
            // avg run length = n / runs; normalise to [0, 1] via n
            double avg_run = static_cast<double>(n) / runs;
            new_pi = avg_run / n; // ∈ (0, 1]

            new_persistent = (new_pi > cfg_.persistent_threshold);
            new_reverting  = (new_pi < cfg_.reverting_threshold);

            if (new_persistent && !prev_persistent_) {
                persistent_ev = true;
                persistent_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (new_reverting && !prev_reverting_) {
                reverting_ev = true;
                reverting_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_persistent_ = new_persistent;
            prev_reverting_  = new_reverting;
        }
    }

    pi_.store(new_pi, std::memory_order_relaxed);
    is_persistent_.store(new_persistent, std::memory_order_relaxed);
    is_reverting_.store(new_reverting, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (persistent_ev && cfg_.on_persistent)    cfg_.on_persistent(new_pi);
    if (reverting_ev  && cfg_.on_mean_reverting) cfg_.on_mean_reverting(new_pi);
}

double NarrativePersistenceIndex::persistence_index() const noexcept { return pi_.load(std::memory_order_relaxed); }
bool   NarrativePersistenceIndex::is_persistent()     const noexcept { return is_persistent_.load(std::memory_order_relaxed); }
bool   NarrativePersistenceIndex::is_reverting()      const noexcept { return is_reverting_.load(std::memory_order_relaxed); }

std::size_t NarrativePersistenceIndex::persistent_events() const noexcept { return persistent_events_.load(std::memory_order_relaxed); }
std::size_t NarrativePersistenceIndex::reverting_events()  const noexcept { return reverting_events_.load(std::memory_order_relaxed); }
std::size_t NarrativePersistenceIndex::total_records()     const noexcept { return total_.load(std::memory_order_relaxed); }

std::string NarrativePersistenceIndex::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"persistence_index\":" << pi_.load(std::memory_order_relaxed)
       << ",\"is_persistent\":" << (is_persistent_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"is_reverting\":" << (is_reverting_.load(std::memory_order_relaxed) ? "true" : "false")
       << ",\"persistent_events\":" << persistent_events_.load(std::memory_order_relaxed)
       << ",\"reverting_events\":" << reverting_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativePersistenceIndex::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(buf_.begin(), buf_.end(), 0.0);
    head_ = 0; fill_ = 0;
    prev_persistent_ = false; prev_reverting_ = false;
    pi_.store(0.5, std::memory_order_relaxed);
    is_persistent_.store(false, std::memory_order_relaxed);
    is_reverting_.store(false, std::memory_order_relaxed);
    persistent_events_.store(0, std::memory_order_relaxed);
    reverting_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativePersistenceIndex::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    buf_.assign(cfg_.window, 0.0);
    head_ = 0; fill_ = 0;
}

} // namespace llmquant
