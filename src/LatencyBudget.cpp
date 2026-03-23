#include "LatencyBudget.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <numeric>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LatencyBudget::LatencyBudget(Config cfg)
    : cfg_(std::move(cfg)) {
    ring_.resize(cfg_.ring_size, 0);
}

// ---------------------------------------------------------------------------
// start
// ---------------------------------------------------------------------------

LatencyBudget::Token LatencyBudget::start() const noexcept {
    return Token{Clock::now()};
}

// ---------------------------------------------------------------------------
// check_deadline
// ---------------------------------------------------------------------------

bool LatencyBudget::check_deadline(const Token& token) {
    total_calls_.fetch_add(1, std::memory_order_relaxed);

    auto now     = Clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - token.start);

    if (elapsed.count() >= cfg_.budget_us) {
        deadline_exceeded_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// record_completion
// ---------------------------------------------------------------------------

void LatencyBudget::record_completion(const Token& token) {
    auto now     = Clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - token.start);
    int64_t us   = elapsed.count();

    std::lock_guard<std::mutex> lk(mtx_);

    if (!ring_.empty()) {
        ring_[ring_pos_ % ring_.size()] = us;
        ++ring_pos_;
        if (ring_count_ < ring_.size()) ++ring_count_;
    }

    latency_sum_us_ += us;
    ++completion_count_;
}

// ---------------------------------------------------------------------------
// p99_us
// ---------------------------------------------------------------------------

int64_t LatencyBudget::p99_us() const {
    std::lock_guard<std::mutex> lk(mtx_);

    if (ring_count_ == 0) return 0;

    // Copy valid entries, sort, return p99.
    std::vector<int64_t> samples(ring_.begin(),
                                  ring_.begin() + static_cast<std::ptrdiff_t>(ring_count_));
    std::sort(samples.begin(), samples.end());

    size_t idx = static_cast<size_t>(0.99 * static_cast<double>(samples.size() - 1));
    return samples[idx];
}

// ---------------------------------------------------------------------------
// mean_us
// ---------------------------------------------------------------------------

double LatencyBudget::mean_us() const {
    std::lock_guard<std::mutex> lk(mtx_);
    if (completion_count_ == 0) return 0.0;
    return static_cast<double>(latency_sum_us_) / static_cast<double>(completion_count_);
}

// ---------------------------------------------------------------------------
// reset / update_config
// ---------------------------------------------------------------------------

void LatencyBudget::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::fill(ring_.begin(), ring_.end(), 0);
    ring_pos_       = 0;
    ring_count_     = 0;
    latency_sum_us_ = 0;
    completion_count_ = 0;
    total_calls_.store(0, std::memory_order_relaxed);
    deadline_exceeded_.store(0, std::memory_order_relaxed);
}

void LatencyBudget::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_ = cfg;
    ring_.assign(cfg_.ring_size, 0);
    ring_pos_       = 0;
    ring_count_     = 0;
    latency_sum_us_ = 0;
    completion_count_ = 0;
    total_calls_.store(0, std::memory_order_relaxed);
    deadline_exceeded_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string LatencyBudget::to_stats_json() const {
    int64_t p99  = p99_us();
    double  mean = mean_us();
    uint64_t calls    = total_calls_.load(std::memory_order_relaxed);
    uint64_t exceeded = deadline_exceeded_.load(std::memory_order_relaxed);
    double exceed_rate = (calls > 0)
        ? (static_cast<double>(exceeded) / static_cast<double>(calls))
        : 0.0;

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"stage_name\":\"%s\",\"budget_us\":%lld,"
        "\"total_calls\":%llu,\"deadline_exceeded\":%llu,"
        "\"deadline_exceeded_rate\":%.4f,"
        "\"mean_us\":%.2f,\"p99_us\":%lld}",
        cfg_.stage_name.c_str(),
        static_cast<long long>(cfg_.budget_us),
        static_cast<unsigned long long>(calls),
        static_cast<unsigned long long>(exceeded),
        exceed_rate,
        mean,
        static_cast<long long>(p99));
    return buf;
}

} // namespace llmquant
