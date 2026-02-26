#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

#ifdef _WIN32
#  include <BaseTsd.h>
   using ssize_t = SSIZE_T;
#endif

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
        std::string model{"gpt-4o"};
        std::string system_prompt{
            "You are a financial markets analyst providing real-time commentary "
            "on market conditions, options flow, and sentiment. Be specific, "
            "use tickers, use directional language."};
        std::string user_prompt{
            "Give a fresh real-time market sentiment update with specific "
            "tickers and directional signals."};
        std::chrono::seconds connect_timeout{5};
        size_t      max_tokens{300};
        /// When true, negotiate TLS via OpenSSL (requires LLMQUANT_TLS_ENABLED).
        /// When false, use plain HTTP (suitable for local/mock endpoints).
        bool use_tls{true};
        /// Interval between successive streaming requests in loop mode.
        std::chrono::seconds loop_interval{5};
        /// When true, dump every raw byte received to stderr for 3 seconds then exit.
        bool debug_raw{false};
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

#ifdef LLMQUANT_TLS_ENABLED
    /// Perform the TLS handshake on the already-connected TCP socket.
    ///
    /// # Returns
    /// `true` if the handshake succeeded and the TLS session is ready.
    /// `false` if SSL_connect failed; the socket is left open for the caller
    /// to close.
    bool tls_handshake();

    /// Gracefully shut down and free the active TLS session and SSL_CTX.
    void tls_close();

    /// Send `len` bytes from `buf` over TLS.
    ///
    /// # Returns
    /// Number of bytes written, or a negative value on error.
    ssize_t tls_send(const char* buf, size_t len);

    /// Receive up to `len` bytes from the TLS session into `buf`.
    ///
    /// # Returns
    /// Number of bytes read, or a non-positive value on EOF / error.
    ssize_t tls_recv(char* buf, size_t len);
#endif

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

#ifdef LLMQUANT_TLS_ENABLED
    void* ssl_ctx_{nullptr};   ///< SSL_CTX* — opaque to avoid OpenSSL headers leaking.
    void* ssl_{nullptr};       ///< SSL*     — active TLS session.
#endif
};

} // namespace llmquant
