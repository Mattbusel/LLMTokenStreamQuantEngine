#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>

namespace llmquant {

/// A single token emitted by the simulator.
struct Token {
    /// The raw text of the token.
    std::string text;
    /// Monotonically increasing emission sequence number (starts at 0).
    uint64_t sequence_id;

    Token(std::string t, uint64_t id) : text(std::move(t)), sequence_id(id) {}
};

/// Callback invoked once per emitted token on the simulator worker thread.
using TokenCallback = std::function<void(const Token&)>;

/// Replays a pre-loaded token sequence at a configurable cadence.
///
/// Tokens are emitted on a background worker thread.  The caller registers a
/// TokenCallback via set_token_callback() before calling start().
///
/// Two data sources are supported:
///   - In-memory: call load_tokens_from_memory() with a vector of strings.
///   - File:      call load_tokens_from_file() with a path to a newline /
///                whitespace-delimited token file.
///
/// The simulator loops over the buffer indefinitely until stop() is called.
///
/// Thread safety: start/stop are not thread-safe with respect to each other.
/// get_stats() is always safe; set_token_callback must be called before start().
class TokenStreamSimulator {
public:
    /// Construction-time parameters for the simulator.
    struct Config {
        /// Time between consecutive token emissions.
        std::chrono::microseconds token_interval{std::chrono::microseconds{10000}};
        /// Initial reservation size for the token buffer.
        size_t buffer_size{1024};
        /// When true the simulator expects tokens from load_tokens_from_memory().
        bool use_memory_stream{false};
        /// Path to a token file used when use_memory_stream is false.
        std::string data_file_path{"tokens.txt"};
    };

    /// Live emission statistics updated atomically by the worker thread.
    struct Stats {
        std::atomic<uint64_t> tokens_emitted{0};
        std::atomic<uint64_t> avg_latency_us{0};
        std::atomic<uint64_t> max_latency_us{0};
        std::atomic<uint64_t> ring_buffer_drops{0};   ///< Tokens dropped when ring buffer was full.
    };

    /// Construct a simulator with the given configuration.
    explicit TokenStreamSimulator(const Config& config);

    /// Stop the worker thread (if running) and join it before destruction.
    ~TokenStreamSimulator();

    /// Start the background worker thread.
    ///
    /// If the simulator is already running this is a no-op.
    void start();

    /// Signal the worker thread to stop and block until it exits.
    ///
    /// Safe to call if the simulator was never started.
    void stop();

    /// Register the callback invoked for each emitted token.
    ///
    /// Must be called before start().
    ///
    /// # Arguments
    /// * `callback` — A callable matching the TokenCallback signature.
    void set_token_callback(TokenCallback callback);

    /// Populate the token buffer from a file on disk.
    ///
    /// Each whitespace-separated word becomes a separate token.
    ///
    /// # Arguments
    /// * `filepath` — Path to the token file.
    ///
    /// # Throws
    /// `std::runtime_error` if the file cannot be opened.
    void load_tokens_from_file(const std::string& filepath);

    /// Populate the token buffer from an in-memory vector.
    ///
    /// # Arguments
    /// * `tokens` — Vector of raw token strings to replay.
    void load_tokens_from_memory(const std::vector<std::string>& tokens);

    /// Return a const reference to the live statistics struct.
    ///
    /// Individual fields are updated atomically; the struct as a whole is
    /// not snapshotted atomically.
    const Stats& get_stats() const { return stats_; }

private:
    /// Lock-free SPSC ring buffer for token strings.
    ///
    /// Uses two cache-line-separated atomics (head_ / tail_) to avoid
    /// false sharing.  Capacity is rounded up to the next power of two so
    /// the index mask trick applies.
    struct RingBuffer {
        static constexpr size_t kCacheLineSize = 64;

        explicit RingBuffer(size_t capacity) {
            // Round up to next power of two.
            size_t cap = 1;
            while (cap < capacity) cap <<= 1;
            mask_ = cap - 1;
            slots_.resize(cap);
        }

        /// Try to push a token.  Returns false if the buffer is full.
        bool try_push(std::string token) {
            const size_t t = tail_.load(std::memory_order_relaxed);
            const size_t next = (t + 1) & mask_;
            if (next == head_.load(std::memory_order_acquire)) return false;  // full
            slots_[t] = std::move(token);
            tail_.store(next, std::memory_order_release);
            return true;
        }

        /// Try to pop a token.  Returns false if the buffer is empty.
        bool try_pop(std::string& out) {
            const size_t h = head_.load(std::memory_order_relaxed);
            if (h == tail_.load(std::memory_order_acquire)) return false;  // empty
            out = std::move(slots_[h]);
            head_.store((h + 1) & mask_, std::memory_order_release);
            return true;
        }

        size_t size() const {
            size_t h = head_.load(std::memory_order_acquire);
            size_t t = tail_.load(std::memory_order_acquire);
            return (t - h) & mask_;
        }

        bool empty() const { return size() == 0; }

        void clear() {
            head_.store(0, std::memory_order_relaxed);
            tail_.store(0, std::memory_order_relaxed);
        }

    private:
        alignas(kCacheLineSize) std::atomic<size_t> head_{0};
        alignas(kCacheLineSize) std::atomic<size_t> tail_{0};
        size_t mask_{0};
        std::vector<std::string> slots_;
    };

    void stream_worker();

    Config config_;
    TokenCallback callback_;
    RingBuffer ring_buffer_;
    std::vector<std::string> source_tokens_;   // master token list (filled once, read-only after load)
    std::mutex load_mutex_;                    // protects source_tokens_ during load
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> current_sequence_{0};
    std::thread worker_thread_;
    Stats stats_;
};

} // namespace llmquant
