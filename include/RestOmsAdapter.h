#pragma once

#include "OmsAdapter.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace llmquant {

/// Polls a REST endpoint for position data on a fixed interval.
///
/// ## Expected Response Body (JSON, one flat object)
/// ```json
/// {
///   "net_position": 0.35,
///   "position_limit": 1.0,
///   "pnl": -1.23,
///   "pnl_limit": -10.0
/// }
/// ```
///
/// ## Transport
/// Uses plain HTTP/1.1 GET over a POSIX TCP socket. For HTTPS, place behind a
/// local TLS-terminating proxy (nginx, stunnel) or build with OpenSSL using the
/// same pattern as LLMStreamClient.
///
/// ## Thread Safety
/// start/stop are safe from any thread. The position callback is invoked from
/// the poller thread.
class RestOmsAdapter : public OmsAdapter {
public:
    /// Connection and polling configuration.
    struct Config {
        /// Target host name or IP address.
        std::string host{"127.0.0.1"};
        /// TCP port to connect on.
        uint16_t    port{8080};
        /// URL path for the position endpoint.
        std::string path{"/positions"};
        /// Optional Bearer token sent in the Authorization header.
        std::string api_key{};
        /// How often to poll the endpoint.
        std::chrono::milliseconds poll_interval{500};
        /// TCP connect/recv timeout in seconds.
        int timeout_s{3};
    };

    /// Construct the adapter with the given configuration.
    ///
    /// # Arguments
    /// * `config` — Connection and polling parameters.
    explicit RestOmsAdapter(Config config);

    /// Destructor: calls stop() and cleans up Winsock on Windows.
    ~RestOmsAdapter() override;

    /// Register the callback that receives position snapshots.
    ///
    /// # Arguments
    /// * `cb` — Called on the poller thread for each successful parse.
    void set_position_callback(PositionCallback cb) override;

    /// Open the background poller thread.
    ///
    /// # Returns
    /// `false` if already running.
    bool start() override;

    /// Signal the poller to stop and block until the thread exits.
    void stop() override;

    /// True while the background poller thread is active.
    bool is_running() const override { return running_.load(); }

    /// Returns a description string containing host, port, path and interval.
    std::string description() const override;

    /// Return the number of successful position updates received since start().
    uint64_t update_count() const { return update_count_.load(); }

    /// Return the number of failed HTTP requests since start().
    uint64_t error_count() const { return error_count_.load(); }

private:
    /// Main loop executed on the poller thread.
    void poller_thread();

    /// Open a fresh TCP connection to config_.host:config_.port.
    /// Returns true on success; sets sockfd_ to the connected descriptor.
    bool open_socket();

    /// Close the current socket descriptor if open.
    void close_socket();

    /// Build the HTTP GET request string.
    std::string build_request() const;

    /// Parse a minimal JSON position object from an HTTP response.
    ///
    /// # Arguments
    /// * `body` — Full HTTP response (including headers) or bare JSON.
    /// * `out`  — Output parameter populated on success.
    ///
    /// # Returns
    /// `false` if any required field is missing or malformed.
    static bool parse_position(const std::string& body,
                               RiskManager::PositionState& out);

    Config            config_;
    PositionCallback  callback_;
    std::atomic<bool> running_{false};
    std::thread       thread_;
    int               sockfd_{-1};
    std::atomic<uint64_t> update_count_{0};
    std::atomic<uint64_t> error_count_{0};
};

} // namespace llmquant
