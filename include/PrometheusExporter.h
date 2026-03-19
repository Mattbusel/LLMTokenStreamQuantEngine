#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace llmquant {

/**
 * @brief Lightweight HTTP server that exposes a Prometheus /metrics scrape endpoint.
 *
 * Starts a background thread that accepts TCP connections on the configured
 * port, serves one HTTP/1.0 response per connection, and repeats.
 * The metrics snapshot is obtained by calling the registered MetricsCallback
 * at scrape time.
 *
 * Thread safety: start/stop may be called from any thread. The metrics
 * callback is invoked from the background server thread.
 */
class PrometheusExporter {
public:
    /**
     * @brief Server configuration.
     */
    struct Config {
        /** @brief TCP port to listen on (default Prometheus node exporter port). */
        uint16_t    port{9100};
        /** @brief Address to bind (default all interfaces). */
        std::string bind_address{"0.0.0.0"};
    };

    /**
     * @brief Called by the background thread at each scrape to build the metrics body.
     *
     * Must return a valid Prometheus text-format string.
     */
    using MetricsCallback = std::function<std::string()>;

    /**
     * @brief Construct the exporter with the given configuration.
     *
     * @param config Server bind address and port configuration.
     */
    explicit PrometheusExporter(Config config);

    /**
     * @brief Destructor calls stop() and cleans up Winsock on Windows.
     */
    ~PrometheusExporter();

    /**
     * @brief Register the callback invoked at each scrape to produce the metrics body.
     *
     * @param cb Callable returning a Prometheus text-format string.
     */
    void set_metrics_callback(MetricsCallback cb);

    /**
     * @brief Bind the listen socket and start the scrape-server background thread.
     *
     * @return true on success; false if the socket could not be bound.
     */
    bool start();

    /**
     * @brief Signal the server thread to stop and block until it exits.
     */
    void stop();

    /**
     * @brief Returns true while the server thread is active.
     *
     * @return Running state.
     */
    bool is_running() const { return running_.load(); }

private:
    void server_thread();
    bool bind_socket();
    void close_listen_socket();

    Config            config_;
    MetricsCallback   metrics_cb_;
    std::atomic<bool> running_{false};
    /// Guards metrics_cb_ against re-entrancy if a slow scrape triggers another scrape.
    std::atomic<bool> callback_in_progress_{false};
    std::thread       thread_;
    int               listen_fd_{-1};
};

} // namespace llmquant
