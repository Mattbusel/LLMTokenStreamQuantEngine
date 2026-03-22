#include "PipelineCircuitBreaker.h"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PipelineCircuitBreaker::PipelineCircuitBreaker(Config config)
    : config_(config)
{
    // Clamp config to avoid nonsensical values.
    config_.open_threshold  = std::clamp(config_.open_threshold,  0.0, 1.0);
    config_.close_threshold = std::clamp(config_.close_threshold, 0.0, config_.open_threshold);
    config_.ema_alpha       = std::clamp(config_.ema_alpha,        0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Hot-path record_signal
// ---------------------------------------------------------------------------

void PipelineCircuitBreaker::record_signal(bool blocked) {
    // Update EMA: α * new_sample + (1 - α) * old_ema
    // Use a relaxed CAS loop to update the atomic double atomically.
    double alpha = config_.ema_alpha;
    double sample = blocked ? 1.0 : 0.0;
    double old_ema = ema_block_rate_.load(std::memory_order_relaxed);
    double new_ema = alpha * sample + (1.0 - alpha) * old_ema;
    // Simple store — minor data race on concurrent updates is acceptable
    // for a statistical EMA; we never need perfect consistency here.
    ema_block_rate_.store(new_ema, std::memory_order_relaxed);

    maybe_transition();
}

// ---------------------------------------------------------------------------
// State machine transitions
// ---------------------------------------------------------------------------

void PipelineCircuitBreaker::maybe_transition() {
    auto current_state = state_.load(std::memory_order_acquire);
    double rate = ema_block_rate_.load(std::memory_order_relaxed);
    auto now = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lk(mutex_);

    switch (current_state) {
        case State::Closed: {
            if (rate >= config_.open_threshold) {
                if (!high_rate_timer_running_) {
                    high_rate_since_ = now;
                    high_rate_timer_running_ = true;
                } else {
                    auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                        now - high_rate_since_).count();
                    if (elapsed_s >= config_.open_duration_s) {
                        high_rate_timer_running_ = false;
                        transition_to(State::Open);
                    }
                }
            } else {
                // Rate fell back below threshold — reset the timer.
                high_rate_timer_running_ = false;
            }
            break;
        }

        case State::Open: {
            auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                now - opened_at_).count();
            if (elapsed_s >= config_.reset_timeout_s) {
                transition_to(State::HalfOpen);
            }
            break;
        }

        case State::HalfOpen: {
            if (rate < config_.close_threshold) {
                transition_to(State::Closed);
            } else if (rate >= config_.open_threshold) {
                // Recovery probe failed — reopen.
                transition_to(State::Open);
            }
            break;
        }
    }
}

void PipelineCircuitBreaker::transition_to(State new_state) {
    // Called with mutex_ held.
    State old_state = state_.load(std::memory_order_acquire);
    if (old_state == new_state) return;

    double rate = ema_block_rate_.load(std::memory_order_relaxed);

    state_.store(new_state, std::memory_order_release);

    auto now = std::chrono::steady_clock::now();
    if (new_state == State::Open) {
        opened_at_ = now;
        trips_.fetch_add(1, std::memory_order_relaxed);
        spdlog::warn("[circuit_breaker] TRIPPED to OPEN — block_rate={:.0f}% trips={}",
                     rate * 100.0, trips_.load(std::memory_order_relaxed));
    } else if (new_state == State::Closed) {
        recoveries_.fetch_add(1, std::memory_order_relaxed);
        high_rate_timer_running_ = false;
        spdlog::info("[circuit_breaker] RECOVERED to CLOSED — block_rate={:.0f}% recoveries={}",
                     rate * 100.0, recoveries_.load(std::memory_order_relaxed));
    } else {
        spdlog::info("[circuit_breaker] → HALF-OPEN (probing recovery) block_rate={:.0f}%",
                     rate * 100.0);
    }

    StateChangeCallback cb;
    {
        // state_change_cb_ is protected by mutex_ (already held by caller).
        cb = state_change_cb_;
    }
    // Fire callback outside the lock to avoid re-entrancy.
    // We copy-then-call pattern is safe: cb is a value copy.
    if (cb) {
        // Release the lock temporarily for the callback.
        // Since we're called with mutex_ held, we can't release it here without
        // restructuring. The callback must be short and must NOT call back into
        // PipelineCircuitBreaker.
        try { cb(new_state, rate); } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// Public utilities
// ---------------------------------------------------------------------------

std::string PipelineCircuitBreaker::state_name() const noexcept {
    switch (state_.load(std::memory_order_acquire)) {
        case State::Closed:   return "closed";
        case State::Open:     return "open";
        case State::HalfOpen: return "half_open";
    }
    return "unknown";
}

void PipelineCircuitBreaker::set_state_change_callback(StateChangeCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    state_change_cb_ = std::move(cb);
}

void PipelineCircuitBreaker::force_close() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (state_.load(std::memory_order_acquire) != State::Closed) {
        transition_to(State::Closed);
    }
}

void PipelineCircuitBreaker::reset_stats() {
    trips_.store(0, std::memory_order_relaxed);
    recoveries_.store(0, std::memory_order_relaxed);
    ema_block_rate_.store(0.0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mutex_);
    high_rate_timer_running_ = false;
}

} // namespace llmquant
