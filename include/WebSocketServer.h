#pragma once

/**
 * @file include/WebSocketServer.h
 * @brief WebSocket live feed server for LLMTokenStreamQuantEngine.
 *
 * Accepts LLM token streams over WebSocket, processes each token through the
 * pipeline, and emits quantitative signals as JSON frames in real time.
 *
 * Compatible with SSE streams from OpenAI/Anthropic APIs.
 *
 * ## Protocol
 *
 * ### Input (client -> server)
 * ```json
 * {"token": "bullish"}
 * {"token": "recession", "timestamp_us": 1704067200000000}
 * ```
 * Also accepts bare SSE lines:
 * ```
 * data: {"choices":[{"delta":{"content":"bull"}}]}
 * data: [DONE]
 * ```
 *
 * ### Output (server -> client)
 * ```json
 * {"timestamp_ns":1704067200000000000,"direction":1,"strength":0.742,
 *  "confidence":0.831,"volatility_adjustment":0.12,"latency_us":4.7}
 * ```
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace llmquant {

struct WebSocketSession; // forward declaration

/**
 * @brief Configuration for the WebSocket server.
 */
struct WebSocketServerConfig {
    /// Bind host (default: 0.0.0.0).
    std::string host{"0.0.0.0"};
    /// Bind port (default: 9200).
    uint16_t port{9200};
    /// Maximum concurrent client sessions (default: 64).
    std::size_t max_sessions{64};
    /// Heartbeat ping interval in seconds (default: 30).
    int heartbeat_interval_secs{30};
    /// If true, log each incoming frame to the log callback.
    bool verbose{false};
};

/**
 * @brief WebSocket server that processes LLM token streams and emits signals.
 *
 * Thread safety: `on_open`, `on_message`, and `on_close` are thread-safe
 * (protected by an internal mutex). `start()` and `stop()` must be called
 * from the same thread.
 *
 * ## Transport integration
 * The transport layer (libwebsockets, Boost.Beast, uWebSockets, etc.) calls:
 *   - `on_open(send_fn)`  → returns session_id
 *   - `on_message(session_id, frame)`
 *   - `on_close(session_id)`
 *
 * The `send_fn` callback is invoked synchronously on the calling thread
 * when the engine emits a signal for that session.
 *
 * ## Usage
 * ```cpp
 * WebSocketServerConfig cfg;
 * cfg.port = 9200;
 * WebSocketServer server(cfg);
 * server.set_log_callback([](const std::string& msg) {
 *     std::cout << "[WS] " << msg << "\n";
 * });
 * server.start();
 * // ... (transport integration sets up accept loop) ...
 * server.stop();
 * ```
 */
class WebSocketServer {
public:
    explicit WebSocketServer(WebSocketServerConfig config = WebSocketServerConfig{});
    ~WebSocketServer();

    // Non-copyable, non-movable.
    WebSocketServer(const WebSocketServer&) = delete;
    WebSocketServer& operator=(const WebSocketServer&) = delete;

    /**
     * @brief Start the server (launches heartbeat thread).
     *
     * Does not block. The transport-level accept loop must be managed by the caller.
     */
    void start();

    /**
     * @brief Stop the server and close all sessions.
     */
    void stop();

    /**
     * @brief Returns true if the server is currently running.
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * @brief Register a new client session.
     *
     * @param send_fn Callback invoked with each outbound JSON frame.
     * @return Opaque session ID to pass to on_message / on_close, or 0 on error.
     */
    uint64_t on_open(std::function<void(const std::string&)> send_fn);

    /**
     * @brief Process an incoming text frame from a client.
     *
     * @param session_id Session ID returned by on_open.
     * @param frame      Raw WebSocket text frame (JSON or SSE line).
     */
    void on_message(uint64_t session_id, const std::string& frame);

    /**
     * @brief Close and remove a client session.
     *
     * @param session_id Session ID returned by on_open.
     */
    void on_close(uint64_t session_id);

    /**
     * @brief Set a log callback for server events and errors.
     */
    void set_log_callback(std::function<void(const std::string&)> cb);

    /**
     * @brief Return the number of currently active sessions.
     */
    [[nodiscard]] std::size_t session_count() const noexcept;

private:
    WebSocketServerConfig config_;
    std::atomic<bool>     running_;
    std::atomic<uint64_t> next_session_id_;
    std::thread           heartbeat_thread_;
    mutable std::mutex    sessions_mutex_;

    std::unordered_map<uint64_t, std::unique_ptr<WebSocketSession>> sessions_;

    std::function<void(const std::string&)> on_log_{
        [](const std::string& msg) { /* no-op default */ (void)msg; }
    };
};

} // namespace llmquant
