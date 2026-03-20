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

/**
 * @brief Streams tokens from an OpenAI-compatible chat completions endpoint.
 *
 * Connects over TCP to `host:port`, sends an HTTP/1.1 POST with
 * `"stream": true`, and forwards each content delta token to the registered
 * callback as it arrives.  The connection runs on a background thread; call
 * stop() to terminate it cleanly.
 *
 * This is a zero-dependency implementation using POSIX sockets (no libcurl,
 * no Boost.Asio).  It handles chunked Transfer-Encoding by accumulating raw
 * bytes and scanning for SSE `data:` lines.
 *
 * Thread safety: connect/stop may be called from any thread. The token
 * callback is invoked from the background reader thread.
 */
class LLMStreamClient {
public:
    /// Called once per decoded token delta.
    using TokenCallback = std::function<void(const std::string& token)>;

    /**
     * @brief Called when the stream ends (EOF or error).
     *
     * `error` is empty on clean EOF, non-empty on socket or protocol error.
     */
    using DoneCallback = std::function<void(const std::string& error)>;

    /**
     * @brief Connection parameters for LLMStreamClient.
     */
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
        /// When true, attempt to reuse the TCP connection across requests.
        /// Only effective for servers that support persistent SSE streams.
        /// Set to false for OpenAI which closes after [DONE].
        bool use_keep_alive{false};
    };

    /**
     * @brief Construct a streaming client with the given connection parameters.
     *
     * @param config Connection, authentication, and behaviour configuration.
     * @throws std::runtime_error on Windows if WSAStartup fails, or if
     *         SSL_CTX_new fails when TLS support is compiled in.
     */
    explicit LLMStreamClient(Config config);

    /**
     * @brief Stop the background reader thread and release the socket.
     */
    ~LLMStreamClient();

    /**
     * @brief Register the token callback (must be set before connect()).
     *
     * @param cb Callable invoked once per decoded content delta token.
     */
    void set_token_callback(TokenCallback cb);

    /**
     * @brief Register the done callback.
     *
     * @param cb Callable invoked when the stream ends; `error` is empty on
     *           clean EOF, non-empty on socket or protocol error.
     */
    void set_done_callback(DoneCallback cb);

    /**
     * @brief Open the TCP connection and start the background reader thread.
     *
     * Returns false immediately if already connected or if the socket
     * cannot be opened (hostname resolution failure, refused connection, etc.).
     *
     * @return true if the socket was opened and the reader thread was started;
     *         false if already running or the connection could not be established.
     */
    bool connect();

    /**
     * @brief Signal the background thread to stop and block until it exits.
     *
     * Safe to call multiple times and safe to call before connect().
     */
    void stop();

    /**
     * @brief Returns true if the background reader thread is active.
     *
     * @return true while the reader thread is running.
     */
    bool is_running() const { return running_.load(); }

    /**
     * @brief Return the total number of non-empty tokens received since connect().
     *
     * Thread-safe (atomic read).
     *
     * @return Total tokens emitted via the token callback.
     */
    uint64_t tokens_received() const noexcept {
        return tokens_received_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the number of times the reader thread has reconnected.
     *
     * Thread-safe (atomic read).
     *
     * @return Reconnect attempt count since connect().
     */
    uint64_t reconnect_count() const noexcept {
        return reconnect_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Parse one SSE `data:` line and extract the token delta.
     *
     * Exposed as public static for unit testing of the parsing logic without
     * a live server.  Returns empty string if the line is not a content delta
     * or if the content field is absent or empty in the JSON payload.
     *
     * @param data_line Raw text following the "data: " SSE prefix.
     * @return The extracted token string, or empty if no delta is present.
     */
    static std::string parse_sse_delta(const std::string& data_line);

private:
    void reader_thread();
    bool open_socket();
    void close_socket();

#ifdef LLMQUANT_TLS_ENABLED
    /**
     * @brief Perform the TLS handshake on the already-connected TCP socket.
     *
     * @return true if the handshake succeeded and the TLS session is ready;
     *         false if SSL_connect failed (socket is left open for the caller to close).
     */
    bool tls_handshake();

    /**
     * @brief Gracefully shut down and free the active TLS session and SSL_CTX.
     */
    void tls_close();

    /**
     * @brief Send `len` bytes from `buf` over TLS.
     *
     * @param buf Pointer to the data to send.
     * @param len Number of bytes to send.
     * @return Number of bytes written, or a negative value on error.
     */
    ssize_t tls_send(const char* buf, size_t len);

    /**
     * @brief Receive up to `len` bytes from the TLS session into `buf`.
     *
     * @param buf Destination buffer.
     * @param len Maximum number of bytes to receive.
     * @return Number of bytes read, or a non-positive value on EOF / error.
     */
    ssize_t tls_recv(char* buf, size_t len);
#endif

    /**
     * @brief Build the JSON request body for the streaming completions call.
     *
     * @return Serialised JSON string suitable for the HTTP POST body.
     */
    std::string build_request_body() const;

    /**
     * @brief Build the full HTTP/1.1 POST request string.
     *
     * @param body JSON request body produced by build_request_body().
     * @return Complete HTTP request string ready for transmission.
     */
    std::string build_http_request(const std::string& body) const;

    Config          config_;
    TokenCallback   token_cb_;
    DoneCallback    done_cb_;
    std::atomic<bool> running_{false};
    /// Set to true before close_socket() in stop() so the reader thread can
    /// distinguish an intentional socket close from an unexpected recv() error.
    std::atomic<bool> shutdown_requested_{false};
    std::thread     thread_;
    int             sockfd_{-1};
    // On Windows, WSAGetLastError() is thread-local and only valid immediately
    // after the failing call.  When stop() closes the socket from the main
    // thread while the reader thread is in recv(), the reader thread captures
    // the WSA error here atomically so it can be logged after the join.
    std::atomic<int>      last_socket_error_{0};
    std::atomic<uint64_t> tokens_received_{0};
    std::atomic<uint64_t> reconnect_count_{0};

#ifdef LLMQUANT_TLS_ENABLED
    void* ssl_ctx_{nullptr};   ///< SSL_CTX* — opaque to avoid OpenSSL headers leaking.
    void* ssl_{nullptr};       ///< SSL*     — active TLS session.
#endif
};

} // namespace llmquant
