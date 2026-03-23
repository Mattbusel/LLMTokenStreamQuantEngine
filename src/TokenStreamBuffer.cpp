#include "TokenStreamBuffer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ============================================================================
// Construction
// ============================================================================

TokenStreamBuffer::TokenStreamBuffer(Config cfg)
    : cfg_(std::move(cfg))
    , cap_(std::max(2, cfg_.capacity))
{
    buf_.resize(static_cast<std::size_t>(cap_));
}

// ============================================================================
// Internal ring helpers (called under lock)
// ============================================================================

void TokenStreamBuffer::push_back_locked(BufferedToken tok) {
    buf_[static_cast<std::size_t>(tail_)] = std::move(tok);
    tail_ = (tail_ + 1) % cap_;
    fill_.fetch_add(1, std::memory_order_relaxed);
    update_peak_locked();
}

BufferedToken TokenStreamBuffer::pop_front_locked() {
    BufferedToken t = std::move(buf_[static_cast<std::size_t>(head_)]);
    head_ = (head_ + 1) % cap_;
    fill_.fetch_sub(1, std::memory_order_relaxed);
    return t;
}

void TokenStreamBuffer::update_peak_locked() {
    int f = fill_.load(std::memory_order_relaxed);
    int prev_peak = peak_fill_.load(std::memory_order_relaxed);
    if (f > prev_peak) {
        peak_fill_.store(f, std::memory_order_relaxed);
    }
}

void TokenStreamBuffer::maybe_fire_hwm_locked() {
    if (!cfg_.on_high_watermark) return;
    int f = fill_.load(std::memory_order_relaxed);
    double frac = static_cast<double>(f) / static_cast<double>(cap_);
    if (frac >= cfg_.high_watermark) {
        // Fire outside the lock to avoid deadlock — captured for later.
        // Actual callback fired by caller after releasing lock.
    }
}

// ============================================================================
// Producer
// ============================================================================

bool TokenStreamBuffer::enqueue(BufferedToken token) {
    if (shutdown_.load(std::memory_order_relaxed)) return false;

    std::function<void(int, int)>        hwm_cb;
    std::function<void(const BufferedToken&)> drop_cb;
    bool did_hwm  = false;
    bool result   = false;

    auto mode = cfg_.mode;

    if (mode == BackpressureMode::BLOCK) {
        std::unique_lock<std::mutex> lk(mutex_);
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(cfg_.block_timeout_ms > 0
                          ? cfg_.block_timeout_ms
                          : 24 * 3600 * 1000);

        bool ok = not_full_cv_.wait_until(lk, deadline, [this] {
            return !is_full_locked() || shutdown_.load(std::memory_order_relaxed);
        });

        if (!ok || shutdown_.load(std::memory_order_relaxed)) return false;

        push_back_locked(std::move(token));
        enqueued_.fetch_add(1, std::memory_order_relaxed);

        int f = fill_.load(std::memory_order_relaxed);
        double frac = static_cast<double>(f) / static_cast<double>(cap_);
        if (frac >= cfg_.high_watermark) {
            did_hwm = true;
            hwm_cb  = cfg_.on_high_watermark;
        }
        result = true;
        lk.unlock();
        not_empty_cv_.notify_one();

    } else if (mode == BackpressureMode::DROP_OLDEST) {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (is_full_locked()) {
                // Evict oldest.
                BufferedToken evicted = pop_front_locked();
                dropped_.fetch_add(1, std::memory_order_relaxed);
                drop_cb  = cfg_.on_drop;
                // Fire drop callback outside the lock.
                // Store evicted for callback below.
                push_back_locked(std::move(token));
                enqueued_.fetch_add(1, std::memory_order_relaxed);

                int f = fill_.load(std::memory_order_relaxed);
                double frac = static_cast<double>(f) / static_cast<double>(cap_);
                if (frac >= cfg_.high_watermark) {
                    did_hwm = true;
                    hwm_cb  = cfg_.on_high_watermark;
                }
                result = true;
                // Can't fire callbacks while holding lock.
                // Use a scoped approach: call after lock release.
                // Evicted token is moved — capture it.
                if (drop_cb) drop_cb(evicted);
            } else {
                push_back_locked(std::move(token));
                enqueued_.fetch_add(1, std::memory_order_relaxed);
                result = true;

                int f = fill_.load(std::memory_order_relaxed);
                double frac = static_cast<double>(f) / static_cast<double>(cap_);
                if (frac >= cfg_.high_watermark) {
                    did_hwm = true;
                    hwm_cb  = cfg_.on_high_watermark;
                }
            }
        }
        not_empty_cv_.notify_one();

    } else if (mode == BackpressureMode::DROP_NEWEST) {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (is_full_locked()) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                drop_cb = cfg_.on_drop;
                result  = false;
            } else {
                push_back_locked(std::move(token));
                enqueued_.fetch_add(1, std::memory_order_relaxed);
                result = true;

                int f = fill_.load(std::memory_order_relaxed);
                double frac = static_cast<double>(f) / static_cast<double>(cap_);
                if (frac >= cfg_.high_watermark) {
                    did_hwm = true;
                    hwm_cb  = cfg_.on_high_watermark;
                }
            }
        }
        if (!result && drop_cb) drop_cb(token);
        if (result) not_empty_cv_.notify_one();

    } else { // REJECT
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (is_full_locked()) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                result = false;
            } else {
                push_back_locked(std::move(token));
                enqueued_.fetch_add(1, std::memory_order_relaxed);
                result = true;

                int f = fill_.load(std::memory_order_relaxed);
                double frac = static_cast<double>(f) / static_cast<double>(cap_);
                if (frac >= cfg_.high_watermark) {
                    did_hwm = true;
                    hwm_cb  = cfg_.on_high_watermark;
                }
            }
        }
        if (result) not_empty_cv_.notify_one();
    }

    // Fire high-watermark callback outside all locks.
    if (did_hwm && hwm_cb) {
        hwm_cb(fill_.load(std::memory_order_relaxed), cap_);
    }

    return result;
}

bool TokenStreamBuffer::enqueue(std::string token_str,
                                double weight,
                                double sentiment,
                                int64_t ts_us) {
    BufferedToken t;
    t.token       = std::move(token_str);
    t.weight      = weight;
    t.sentiment   = sentiment;
    t.timestamp_us = ts_us;
    return enqueue(std::move(t));
}

// ============================================================================
// Consumer
// ============================================================================

std::optional<BufferedToken> TokenStreamBuffer::dequeue() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (is_empty_locked()) return std::nullopt;
    BufferedToken t = pop_front_locked();
    dequeued_.fetch_add(1, std::memory_order_relaxed);
    not_full_cv_.notify_one();
    return t;
}

std::optional<BufferedToken> TokenStreamBuffer::dequeue_wait(int timeout_ms) {
    std::unique_lock<std::mutex> lk(mutex_);
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);

    bool ok = not_empty_cv_.wait_until(lk, deadline, [this] {
        return !is_empty_locked() || shutdown_.load(std::memory_order_relaxed);
    });

    if (!ok || (shutdown_.load(std::memory_order_relaxed) && is_empty_locked())) {
        return std::nullopt;
    }
    if (is_empty_locked()) return std::nullopt;

    BufferedToken t = pop_front_locked();
    dequeued_.fetch_add(1, std::memory_order_relaxed);
    lk.unlock();
    not_full_cv_.notify_one();
    return t;
}

int TokenStreamBuffer::drain(std::vector<BufferedToken>& out, int max_tokens) {
    std::lock_guard<std::mutex> lk(mutex_);
    int count = 0;
    while (!is_empty_locked() && count < max_tokens) {
        out.push_back(pop_front_locked());
        dequeued_.fetch_add(1, std::memory_order_relaxed);
        ++count;
    }
    if (count > 0) not_full_cv_.notify_all();
    return count;
}

// ============================================================================
// Control
// ============================================================================

void TokenStreamBuffer::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        shutdown_.store(true, std::memory_order_relaxed);
    }
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
}

void TokenStreamBuffer::clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    head_ = tail_ = 0;
    fill_.store(0, std::memory_order_relaxed);
    peak_fill_.store(0, std::memory_order_relaxed);
    enqueued_.store(0, std::memory_order_relaxed);
    dequeued_.store(0, std::memory_order_relaxed);
    dropped_.store(0, std::memory_order_relaxed);
    shutdown_.store(false, std::memory_order_relaxed);
}

void TokenStreamBuffer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    cap_ = std::max(2, cfg_.capacity);
    buf_.assign(static_cast<std::size_t>(cap_), BufferedToken{});
    head_ = tail_ = 0;
    fill_.store(0, std::memory_order_relaxed);
    peak_fill_.store(0, std::memory_order_relaxed);
    enqueued_.store(0, std::memory_order_relaxed);
    dequeued_.store(0, std::memory_order_relaxed);
    dropped_.store(0, std::memory_order_relaxed);
    shutdown_.store(false, std::memory_order_relaxed);
}

// ============================================================================
// Stats JSON
// ============================================================================

std::string TokenStreamBuffer::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    int    f  = fill_.load(std::memory_order_relaxed);
    double ff = cap_ > 0 ? static_cast<double>(f) / static_cast<double>(cap_) : 0.0;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"capacity\":"      << cap_                                               << ","
       << "\"fill\":"          << f                                                  << ","
       << "\"fill_fraction\":" << ff                                                 << ","
       << "\"peak_fill\":"     << peak_fill_.load(std::memory_order_relaxed)         << ","
       << "\"enqueued\":"      << enqueued_.load(std::memory_order_relaxed)          << ","
       << "\"dequeued\":"      << dequeued_.load(std::memory_order_relaxed)          << ","
       << "\"dropped\":"       << dropped_.load(std::memory_order_relaxed)           << ","
       << "\"shutdown\":"      << (shutdown_.load(std::memory_order_relaxed) ? "true" : "false")
       << "}";
    return ss.str();
}

} // namespace llmquant
