#pragma once

/**
 * @file include/ApiServer.h
 * @brief Minimal HTTP REST server exposing the LLMTokenStreamQuantEngine
 *        signal pipeline over a simple JSON API.
 *
 * ## Endpoints
 *
 * | Method | Path                    | Description                                      |
 * |--------|-------------------------|--------------------------------------------------|
 * | GET    | /signal/latest          | Return the most recently generated signal as JSON |
 * | GET    | /signal/history?n=100   | Return the last N signals (default 50, max 1000) |
 * | POST   | /token                  | Process a token and return the resulting signal   |
 * | GET    | /health                 | Simple liveness check (returns `{"status":"ok"}`) |
 * | GET    | /stats                  | Signal store statistics                           |
 *
 * ## Request / Response format
 *
 * ### POST /token
 * ```
 * Content-Type: text/plain
 * Body: <token string>
 * ```
 * Response (200 OK):
 * ```json
 * {"timestamp_ns":..., "delta_bias":..., "confidence":..., "latency_us":...}
 * ```
 *
 * ### GET /signal/latest
 * Returns a single JSON object identical to the POST /token response,
 * or `{"error":"no signals yet"}` if the store is empty.
 *
 * ### GET /signal/history?n=100
 * Returns a JSON array of the last N signals, newest first.
 *
 * ## Implementation
 *
 * The server uses a minimal hand-written HTTP/1.1 parser that handles
 * GET and POST requests.  No third-party HTTP library is required.
 * Connection handling is single-threaded (one request at a time) to keep
 * the implementation simple and avoid introducing thread-safety concerns
 * in the signal pipeline.  For production use, wrap behind nginx or haproxy.
 *
 * ## Thread safety
 *
 * `ApiServer::start()` launches a background thread that owns the listening
 * socket.  The `token_handler` callback is invoked from that thread; callers
 * must ensure their `TradeSignalEngine` is thread-safe (it is, by design).
 *
 * ## Feature flag
 * `LLMQUANT_ENABLE_API_SERVER` (default ON).
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "SignalStore.h"
#include "TradeSignalEngine.h"

namespace llmquant {

/**
 * @brief Configuration for ApiServer.
 */
struct ApiServerConfig {
    /// Bind host.  Default "127.0.0.1".
    std::string host{"127.0.0.1"};

    /// Bind port.  Default 8765.
    uint16_t port{8765};

    /// Maximum pending connections in the accept queue.  Default 8.
    int backlog{8};

    /// Read/write timeout per connection in seconds.  Default 5.
    int timeout_secs{5};

    /// Default number of history signals returned when ?n is omitted.
    uint64_t default_history_n{50};

    /// Maximum number of history signals that can be requested.
    uint64_t max_history_n{1000};
};

/**
 * @brief Minimal HTTP REST server for the signal pipeline.
 *
 * ## Lifecycle
 *
 * ```cpp
 * SignalStore store;
 * ApiServer   server(config, store);
 *
 * // Register a callback that processes a token string and returns a signal.
 * server.set_token_handler([&](const std::string& token) -> TradeSignal {
 *     return engine.process_token(token);
 * });
 *
 * server.start();          // launches background thread
 * // ... application runs ...
 * server.stop();           // graceful shutdown
 * ```
 */
class ApiServer {
public:
    using TokenHandler = std::function<TradeSignal(const std::string& token)>;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @brief Construct the server (does not bind the socket yet).
     *
     * @param config  Server configuration.
     * @param store   Signal store used for /signal/latest and /signal/history.
     */
    explicit ApiServer(ApiServerConfig config, SignalStore& store);

    ~ApiServer();

    ApiServer(const ApiServer&)            = delete;
    ApiServer& operator=(const ApiServer&) = delete;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /**
     * @brief Register the callback invoked by POST /token.
     *
     * Must be called before `start()`.
     *
     * @param handler  Function that accepts a token string and returns a signal.
     */
    void set_token_handler(TokenHandler handler);

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /**
     * @brief Bind the socket and start the background accept loop.
     *
     * @throws std::runtime_error if the socket cannot be bound.
     */
    void start();

    /**
     * @brief Signal the server to stop accepting new connections and join the
     *        background thread.
     *
     * Idempotent.
     */
    void stop();

    /**
     * @brief Returns true if the server is currently running.
     */
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Port the server is listening on (valid after start()).
     */
    [[nodiscard]] uint16_t bound_port() const noexcept { return config_.port; }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /** @brief Total HTTP requests handled since start(). */
    [[nodiscard]] uint64_t requests_handled() const noexcept {
        return requests_handled_.load(std::memory_order_relaxed);
    }

    /** @brief Total POST /token requests handled. */
    [[nodiscard]] uint64_t token_requests() const noexcept {
        return token_requests_.load(std::memory_order_relaxed);
    }

private:
    // Internal types for the minimal HTTP parser.
    struct ParsedRequest {
        std::string method;   // "GET" or "POST"
        std::string path;     // e.g. "/signal/latest"
        std::string query;    // e.g. "n=100"
        std::string body;     // POST body
        bool        valid{false};
    };

    // Socket helpers (POSIX).
    int  create_server_socket();
    void accept_loop();
    void handle_connection(int client_fd);

    static ParsedRequest parse_request(const std::string& raw);

    std::string handle_get_latest();
    std::string handle_get_history(const std::string& query);
    std::string handle_post_token(const std::string& body);
    std::string handle_get_health();
    std::string handle_get_stats();

    static std::string http_response(int status, const std::string& body,
                                     const std::string& content_type = "application/json");
    static std::string signal_to_json(const TradeSignal& sig);
    static std::string stored_signal_to_json(const StoredSignal& sig);

    ApiServerConfig config_;
    SignalStore&    store_;
    TokenHandler    token_handler_;

    int             server_fd_{-1};
    std::thread     accept_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    std::atomic<uint64_t> requests_handled_{0};
    std::atomic<uint64_t> token_requests_{0};
};

} // namespace llmquant
