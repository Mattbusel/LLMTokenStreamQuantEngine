#pragma once

/// @file SignalReplayBuffer.h
/// @brief Fixed-capacity ring buffer of recent TradeSignals for replay and post-hoc analysis.
///
/// Captures every outgoing TradeSignal in a lock-free ring so that operators
/// can replay the last N signals without replaying the full token stream.
/// Useful for debugging unexpected position changes and for "what-if" analysis.
///
/// Thread-safe: concurrent push from the signal callback and snapshot reads
/// from the stats/replay thread are both safe.

#include "TradeSignalEngine.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalReplayBuffer {
public:
    struct Config {
        /// Maximum signals to retain.  Oldest is overwritten when full.
        /// Default 1024.
        uint32_t capacity{1024};

        /// Optional callback fired on every push (e.g. for streaming to a log).
        std::function<void(const TradeSignal&)> on_push;
    };

    explicit SignalReplayBuffer(Config cfg = Config{});

    // Non-copyable.
    SignalReplayBuffer(const SignalReplayBuffer&)            = delete;
    SignalReplayBuffer& operator=(const SignalReplayBuffer&) = delete;

    /// Push a signal into the ring (overwrites oldest when full).
    void push(const TradeSignal& signal) noexcept;

    /// Snapshot copy of all retained signals in chronological order (oldest first).
    [[nodiscard]] std::vector<TradeSignal> snapshot() const;

    /// Replay retained signals through a callback in chronological order.
    void replay(const std::function<void(const TradeSignal&)>& cb) const;

    /// Number of signals currently retained (≤ capacity).
    [[nodiscard]] uint32_t size() const noexcept;

    /// Total signals pushed since construction (may exceed capacity).
    [[nodiscard]] uint64_t total_pushed() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Clear the ring buffer.
    void reset();

    /// Update config; resets the buffer.
    void update_config(const Config& cfg);

    /// JSON summary (capacity, size, total_pushed, last signal bias/confidence).
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config                   cfg_;
    std::vector<TradeSignal> ring_;
    uint32_t                 capacity_{1024};

    std::atomic<uint64_t> write_pos_{0}; // monotonic write index
    std::atomic<uint64_t> total_{0};

    mutable std::mutex mutex_;
};

} // namespace llmquant
