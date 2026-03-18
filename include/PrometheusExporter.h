#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace llmquant {

/// Lightweight HTTP server that exposes a Prometheus /metrics scrape endpoint.
///
/// Starts a background thread that accepts TCP connections on the configured
/// port, serves one HTTP/1.0 response per connection, and repeats.
/// The metrics snapshot is obtained by calling the registered MetricsCallback
/// at scrape time.
///
/// Thread safety: start/stop may be called from any thread.  The metrics
/// callback is invoked from the background server thread.
class PrometheusExporter {
public:
    /// Server configuration.
    struct Config {
        /// TCP port to listen on (default Prometheus node exporter port).
        uint16_t    port{9100};
        /// Address to bind (default all interfaces).
        std::string bind_address{"0.0.0.0"};
    };

    /// Called by the background thread at each scrape to build the metrics body.
    /// Must return a valid Prometheus text-format string.
    using MetricsCallback = std::function<std::string()>;

    /// Construct the exporter with the given configuration.
    explicit PrometheusExporter(Config config);

    /// Destructor: calls stop() and cleans up Winsock on Windows.
    ~PrometheusExporter();

    /// Register the callback invoked at each scrape to produce the metrics body.
    ///
    /// # Arguments
    /// * `cb` — Callable returning a Prometheus text-format string.
    void set_metrics_callback(MetricsCallback cb);

    /// Bind the listen socket and start the scrape-server background thread.
    ///
    /// # Returns
    /// `true` on success, `false` if the socket could not be bound.
    bool start();

    /// Signal the server thread to stop and block until it exits.
    void stop();

    /// Returns true while the server thread is active.
    bool is_running() const { return running_.load(); }

private:
    void server_thread();
    bool bind_socket();
    void close_listen_socket();

    Config            config_;
    MetricsCallback   metrics_cb_;
    std::atomic<bool> running_{false};
    std::thread       thread_;
    int               listen_fd_{-1};
};

} // namespace llmquant
