#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace llmquant {

/**
 * @brief Lightweight HTTP server exposing a /health JSON endpoint.
 *
 * Designed for Kubernetes liveness/readiness probes and operator dashboards.
 * Serves a single JSON object at GET /health; all other paths return 404.
 *
 * The health payload is built on each request by a registered callback.
 * Example response body:
 * @code{.json}
 * {
 *   "ok": true,
 *   "uptime_s": 123,
 *   "circuit_breaker": "closed",
 *   "block_rate": 0.05,
 *   "oms_connected": true,
 *   "p99_latency_us": 8,
 *   "signals_generated": 1234,
 *   "signals_blocked": 11,
 *   "redis_connected": false,
 *   "version": "1.2.0"
 * }
 * @endcode
 *
 * HTTP status code:
 *   200 when ok == true
 *   503 when ok == false (pipeline is degraded)
 *
 * Thread safety: start/stop may be called from any thread.
 * The callback is invoked from the server thread on each request.
 *
 * Compile-time feature flag: LLMQUANT_HEALTH_SERVER_ENABLED
 * (controlled by CMake option LLMQUANT_ENABLE_HEALTH_SERVER).
 */
class HealthServer {
public:
    /**
     * @brief Server configuration.
     */
    struct Config {
        /** @brief TCP port to listen on (default: 8080). */
        uint16_t    port{8080};
        /** @brief Bind address (default: all interfaces). */
        std::string bind_address{"0.0.0.0"};
    };

    /**
     * @brief Health check callback.
     *
     * Must return a pair of {ok, json_body} where:
     *   ok        — true if the pipeline is healthy; false if degraded.
     *   json_body — The JSON string to include in the response body.
     *               Must be a valid JSON object (no trailing newline needed).
     */
    using HealthCallback = std::function<std::pair<bool, std::string>()>;

    /**
     * @brief Construct the health server with the given configuration.
     *
     * @param config Port and bind address.
     */
    explicit HealthServer(Config config);

    /** @brief Destructor calls stop() and cleans up Winsock on Windows. */
    ~HealthServer();

    /**
     * @brief Register the callback invoked on each /health request.
     *
     * @param cb Callable returning {ok, json_body}.
     */
    void set_health_callback(HealthCallback cb);

    /**
     * @brief Bind the listen socket and start the server thread.
     *
     * @return true on success; false if the socket could not be bound.
     */
    [[nodiscard]] bool start();

    /** @brief Signal the server thread to stop and block until it exits. */
    void stop();

    /**
     * @brief Returns true while the server thread is active.
     *
     * @return Running state.
     */
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total HTTP requests served since start().
     *
     * @return Request count.
     */
    [[nodiscard]] uint64_t requests_served() const noexcept {
        return requests_served_.load(std::memory_order_relaxed);
    }

private:
    void server_thread();
    std::string build_response(bool ok, const std::string& body) const;

    Config         config_;
    HealthCallback health_cb_;

    std::atomic<bool>     running_{false};
    std::atomic<bool>     stop_requested_{false};
    std::thread           thread_;
    int                   listen_fd_{-1};
    bool                  wsa_initialized_{false};

    std::atomic<uint64_t> requests_served_{0};
};

} // namespace llmquant
