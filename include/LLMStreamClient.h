#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

namespace llmquant {

/// Streams tokens from an OpenAI-compatible chat completions endpoint.
///
/// Connects over TCP to `host:port`, sends an HTTP/1.1 POST with
/// `"stream": true`, and forwards each content delta token to the registered
/// callback as it arrives. The connection runs on a background thread; call
/// stop() to terminate it cleanly.
///
/// This is a zero-dependency implementation using POSIX sockets (no libcurl,
/// no Boost.Asio).  It handles chunked Transfer-Encoding by accumulating raw
/// bytes and scanning for SSE `data:` lines.
///
/// Thread safety: connect/stop may be called from any thread. The token
/// callback is invoked from the background reader thread.
class LLMStreamClient {
public:
    /// Called once per decoded token delta.
    using TokenCallback = std::function<void(const std::string& token)>;

    /// Called when the stream ends (EOF or error).
    /// `error` is empty on clean EOF, non-empty on error.
    using DoneCallback = std::function<void(const std::string& error)>;

    /// Connection parameters.
    struct Config {
        std::string host{"api.openai.com"};
        uint16_t    port{443};
        std::string api_key{};
        std::string model{"gpt-4o-mini"};
        std::string system_prompt{
            "You are a financial analyst. Output single tokens representing "
            "market sentiment. Be terse."};
        std::string user_prompt{"Describe current market conditions in one word."};
        std::chrono::seconds connect_timeout{5};
        size_t      max_tokens{256};
        /// When true, use plain HTTP (port 80 or custom); when false, TLS is
        /// assumed but not implemented — set port=80 and use_tls=false for
        /// unencrypted local/mock endpoints.
        bool use_tls{false};
    };

    /// Construct a streaming client with the given connection parameters.
    explicit LLMStreamClient(Config config);

    /// Stop the background reader thread and release the socket.
    ~LLMStreamClient();

    /// Register the token callback (must be set before connect()).
    ///
    /// # Arguments
    /// * `cb` — Callable invoked once per decoded content delta token.
    void set_token_callback(TokenCallback cb);

    /// Register the done callback.
    ///
    /// # Arguments
    /// * `cb` — Callable invoked when the stream ends; `error` is empty on
    ///          clean EOF, non-empty on socket or protocol error.
    void set_done_callback(DoneCallback cb);

    /// Open the TCP connection and start the background reader thread.
    ///
    /// Returns false immediately if already connected or if the socket
    /// cannot be opened (hostname resolution failure, refused connection, etc.).
    ///
    /// # Returns
    /// `true` if the socket was opened and the reader thread was started.
    /// `false` if already running or the connection could not be established.
    bool connect();

    /// Signal the background thread to stop and block until it exits.
    ///
    /// Safe to call multiple times and safe to call before connect().
    void stop();

    /// Returns true if the background reader thread is active.
    bool is_running() const { return running_.load(); }

private:
    void reader_thread();
    bool open_socket();
    void close_socket();

    /// Build the JSON request body for the streaming completions call.
    std::string build_request_body() const;

    /// Build the full HTTP/1.1 POST request string.
    ///
    /// # Arguments
    /// * `body` — JSON request body produced by build_request_body().
    std::string build_http_request(const std::string& body) const;

    /// Parse one SSE `data:` line and extract the token delta.
    ///
    /// Returns empty string if the line is not a content delta or if the
    /// content field is absent or empty in the JSON payload.
    ///
    /// # Arguments
    /// * `data_line` — Raw text following the "data: " SSE prefix.
    static std::string parse_sse_delta(const std::string& data_line);

    Config          config_;
    TokenCallback   token_cb_;
    DoneCallback    done_cb_;
    std::atomic<bool> running_{false};
    std::thread     thread_;
    int             sockfd_{-1};
};

} // namespace llmquant
