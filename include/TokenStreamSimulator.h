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
    void stream_worker();

    Config config_;
    TokenCallback callback_;
    std::vector<std::string> token_buffer_;
    std::mutex buffer_mutex_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> current_sequence_{0};
    std::thread worker_thread_;
    Stats stats_;
};

} // namespace llmquant
