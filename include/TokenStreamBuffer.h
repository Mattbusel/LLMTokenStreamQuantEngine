#pragma once

/// @file TokenStreamBuffer.h
/// @brief Lock-free-friendly token stream buffer with backpressure management.
///
/// ## Purpose
/// The LLM token stream can burst at highly variable rates.  Downstream
/// components (signal engines, OMS adapters) have bounded processing capacity.
/// Without buffering, bursts either drop tokens or block the producer thread.
///
/// This buffer provides:
/// - A **fixed-capacity ring buffer** that absorbs bursts without allocation.
/// - **Backpressure modes** (BLOCK / DROP_OLDEST / DROP_NEWEST / REJECT) so
///   the caller can choose the trade-off between latency and data loss.
/// - **High-watermark callback** fired when fill exceeds a configurable fraction.
/// - **Drain statistics**: enqueued, dequeued, dropped, and max fill counters.
///
/// ## Backpressure modes
/// | Mode          | When buffer full                          |
/// |---------------|-------------------------------------------|
/// | BLOCK         | Caller blocks until a slot is available.  |
/// | DROP_OLDEST   | Evict oldest token; enqueue new one.      |
/// | DROP_NEWEST   | Silently discard the incoming token.      |
/// | REJECT        | Return false from `enqueue()`.            |
///
/// ## Thread safety
/// All public methods are thread-safe.  The internal ring buffer uses a
/// `std::mutex` + `std::condition_variable` pair.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_TOKEN_STREAM_BUFFER` (default ON).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llmquant {

/// Represents a single token in the stream with its associated metadata.
struct BufferedToken {
    std::string  token;          ///< Raw token string.
    double       weight{0.0};   ///< Model-assigned weight / logprob.
    double       sentiment{0.0};///< Pre-scored sentiment contribution.
    int64_t      timestamp_us{0};///< Producer wall-clock (µs since epoch).
};

// ============================================================================

class TokenStreamBuffer {
public:
    // -----------------------------------------------------------------------
    // Backpressure policy
    // -----------------------------------------------------------------------

    enum class BackpressureMode {
        BLOCK,        ///< Block producer thread until a slot frees.
        DROP_OLDEST,  ///< Evict oldest token; insert new one (never blocks).
        DROP_NEWEST,  ///< Discard the incoming token silently (never blocks).
        REJECT,       ///< Return false from enqueue(); caller handles it.
    };

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /// Ring buffer capacity (number of tokens).  Must be >= 2.  Default 1024.
        int capacity{1024};

        /// Backpressure mode when the buffer is full.  Default DROP_OLDEST.
        BackpressureMode mode{BackpressureMode::DROP_OLDEST};

        /// High-watermark fraction [0,1].  Callback fires when fill/capacity
        /// exceeds this.  Default 0.75.
        double high_watermark{0.75};

        /// Maximum time to block in BLOCK mode before returning false (0 = infinite).
        /// Units: milliseconds.  Default 100 ms.
        int block_timeout_ms{100};

        /// Fired when fill fraction rises above `high_watermark`.
        /// Parameters: (fill_count, capacity).
        std::function<void(int fill, int capacity)> on_high_watermark;

        /// Fired when a token is dropped (DROP_OLDEST or DROP_NEWEST mode).
        /// Parameter: dropped token.
        std::function<void(const BufferedToken&)> on_drop;
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    explicit TokenStreamBuffer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Producer API
    // -----------------------------------------------------------------------

    /// Enqueue a token.
    ///
    /// Behaviour when buffer full is governed by `Config::mode`:
    /// - BLOCK: blocks up to `block_timeout_ms`; returns false on timeout.
    /// - DROP_OLDEST: silently evicts oldest, returns true.
    /// - DROP_NEWEST: discards `token`, returns false.
    /// - REJECT: returns false immediately without modification.
    ///
    /// @return true if token was successfully enqueued; false if dropped/rejected.
    [[nodiscard]] bool enqueue(BufferedToken token);

    /// Convenience overload — constructs BufferedToken in-place.
    [[nodiscard]] bool enqueue(std::string token_str,
                               double weight    = 0.0,
                               double sentiment = 0.0,
                               int64_t ts_us    = 0);

    // -----------------------------------------------------------------------
    // Consumer API
    // -----------------------------------------------------------------------

    /// Dequeue the oldest token.  Returns std::nullopt if buffer is empty.
    [[nodiscard]] std::optional<BufferedToken> dequeue();

    /// Block until a token is available (or until `timeout_ms` milliseconds).
    /// @return The dequeued token, or nullopt on timeout / shutdown.
    [[nodiscard]] std::optional<BufferedToken> dequeue_wait(int timeout_ms = 1000);

    /// Drain up to `max_tokens` tokens into `out`.  Returns count drained.
    int drain(std::vector<BufferedToken>& out, int max_tokens = 64);

    // -----------------------------------------------------------------------
    // Control
    // -----------------------------------------------------------------------

    /// Signal shutdown; unblocks any waiting dequeue_wait() calls.
    void shutdown();

    /// Discard all buffered tokens and reset statistics.
    void clear();

    /// Reconfigure.  Clears the buffer.
    void update_config(const Config& cfg);

    // -----------------------------------------------------------------------
    // Accessors (mostly lock-free)
    // -----------------------------------------------------------------------

    /// Current number of tokens in the buffer.
    [[nodiscard]] int size() const noexcept {
        return fill_.load(std::memory_order_relaxed);
    }

    /// Buffer capacity.
    [[nodiscard]] int capacity() const noexcept {
        return cap_;
    }

    /// Fill fraction [0, 1].
    [[nodiscard]] double fill_fraction() const noexcept {
        if (cap_ == 0) return 0.0;
        return static_cast<double>(fill_.load(std::memory_order_relaxed))
             / static_cast<double>(cap_);
    }

    [[nodiscard]] uint64_t enqueued()    const noexcept { return enqueued_.load(std::memory_order_relaxed); }
    [[nodiscard]] uint64_t dequeued()    const noexcept { return dequeued_.load(std::memory_order_relaxed); }
    [[nodiscard]] uint64_t dropped()     const noexcept { return dropped_.load(std::memory_order_relaxed); }
    [[nodiscard]] int      peak_fill()   const noexcept { return peak_fill_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool     is_shutdown() const noexcept { return shutdown_.load(std::memory_order_relaxed); }

    [[nodiscard]] std::string to_stats_json() const;

private:
    // Ring buffer helpers (called under lock).
    void    push_back_locked(BufferedToken tok);
    BufferedToken pop_front_locked();
    bool    is_full_locked()  const noexcept { return fill_.load(std::memory_order_relaxed) >= cap_; }
    bool    is_empty_locked() const noexcept { return fill_.load(std::memory_order_relaxed) == 0; }
    void    update_peak_locked();
    void    maybe_fire_hwm_locked();

    mutable std::mutex              mutex_;
    std::condition_variable         not_full_cv_;
    std::condition_variable         not_empty_cv_;

    Config                          cfg_;
    int                             cap_{1024};

    // Ring buffer storage.
    std::vector<BufferedToken>      buf_;
    int                             head_{0};  ///< Next dequeue index.
    int                             tail_{0};  ///< Next enqueue index.

    std::atomic<int>                fill_{0};
    std::atomic<int>                peak_fill_{0};
    std::atomic<uint64_t>           enqueued_{0};
    std::atomic<uint64_t>           dequeued_{0};
    std::atomic<uint64_t>           dropped_{0};
    std::atomic<bool>               shutdown_{false};
};

// ============================================================================
// SimpleTokenStreamBuffer — integer circular buffer
// ============================================================================

/// A simple fixed-capacity circular buffer for integer token ids.
///
/// Single-threaded version; see ThreadSafeTokenBuffer for the mutex-guarded
/// variant with blocking push/pop.
class SimpleTokenStreamBuffer {
public:
    explicit SimpleTokenStreamBuffer(std::size_t capacity);

    /// Push a token.  Returns false if buffer is full.
    bool push(int token);

    /// Pop oldest token into out_token.  Returns false if empty.
    bool pop(int& out_token);

    /// Peek at the oldest token without removing.  Returns false if empty.
    bool peek(int& out_token) const;

    std::size_t size()     const;
    std::size_t capacity() const;
    bool        empty()    const;
    bool        full()     const;

    /// Discard all tokens and reset head/tail.
    void clear();

    /// Pop all tokens in FIFO order.
    std::vector<int> drain();

    /// Push multiple tokens.  Returns false if any push fails (buffer full).
    bool push_batch(const std::vector<int>& tokens);

private:
    std::size_t        cap_;
    std::size_t        head_{0};
    std::size_t        tail_{0};
    std::size_t        count_{0};
    std::vector<int>   buf_;
};

// ============================================================================
// ThreadSafeTokenBuffer — mutex-guarded version with blocking ops
// ============================================================================

/// Thread-safe integer token circular buffer with blocking push/pop.
class ThreadSafeTokenBuffer {
public:
    explicit ThreadSafeTokenBuffer(std::size_t capacity);

    bool push(int token);
    bool pop(int& out_token);
    bool peek(int& out_token) const;

    std::size_t size()     const;
    std::size_t capacity() const;
    bool        empty()    const;
    bool        full()     const;

    void clear();
    std::vector<int> drain();
    bool push_batch(const std::vector<int>& tokens);

    /// Block until space is available, then push.
    void wait_and_push(int token);

    /// Block until a token is available or timeout elapses.
    /// Returns false on timeout.
    bool wait_and_pop(int& out, std::chrono::milliseconds timeout);

private:
    std::size_t              cap_;
    std::size_t              head_{0};
    std::size_t              tail_{0};
    std::size_t              count_{0};
    std::vector<int>         buf_;
    mutable std::mutex       mutex_;
    std::condition_variable  not_full_cv_;
    std::condition_variable  not_empty_cv_;
};

} // namespace llmquant
