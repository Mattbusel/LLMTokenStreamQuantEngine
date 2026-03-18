#pragma once

#include "OmsAdapter.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace llmquant {

/**
 * @brief Polls a REST endpoint for position data on a fixed interval.
 *
 * Expected response body (JSON, one flat object):
 * @code{.json}
 * {
 *   "net_position": 0.35,
 *   "position_limit": 1.0,
 *   "pnl": -1.23,
 *   "pnl_limit": -10.0
 * }
 * @endcode
 *
 * Transport: uses plain HTTP/1.1 GET over a POSIX TCP socket. For HTTPS,
 * place behind a local TLS-terminating proxy (nginx, stunnel) or build
 * with OpenSSL using the same pattern as LLMStreamClient.
 *
 * Thread safety: start/stop are safe from any thread. The position callback
 * is invoked from the poller thread.
 */
class RestOmsAdapter : public OmsAdapter {
public:
    /**
     * @brief Connection and polling configuration.
     */
    struct Config {
        /** @brief Target host name or IP address. */
        std::string host{"127.0.0.1"};
        /** @brief TCP port to connect on. */
        uint16_t    port{8080};
        /** @brief URL path for the position endpoint. */
        std::string path{"/positions"};
        /** @brief Optional Bearer token sent in the Authorization header. */
        std::string api_key{};
        /** @brief How often to poll the endpoint. */
        std::chrono::milliseconds poll_interval{500};
        /** @brief TCP connect/recv timeout in seconds. */
        int timeout_s{3};
    };

    /**
     * @brief Construct the adapter with the given configuration.
     *
     * @param config Connection and polling parameters.
     */
    explicit RestOmsAdapter(Config config);

    /**
     * @brief Destructor calls stop() and cleans up Winsock on Windows.
     */
    ~RestOmsAdapter() override;

    /**
     * @brief Register the callback that receives position snapshots.
     *
     * @param cb Called on the poller thread for each successful parse.
     */
    void set_position_callback(PositionCallback cb) override;

    /**
     * @brief Open the background poller thread.
     *
     * @return false if already running.
     */
    bool start() override;

    /**
     * @brief Signal the poller to stop and block until the thread exits.
     */
    void stop() override;

    /**
     * @brief Returns true while the background poller thread is active.
     *
     * @return Running state.
     */
    bool is_running() const override { return running_.load(); }

    /**
     * @brief Returns a description string containing host, port, path and interval.
     *
     * @return Human-readable adapter description.
     */
    std::string description() const override;

    /**
     * @brief Return the number of successful position updates received since start().
     *
     * @return Update count.
     */
    uint64_t update_count() const { return update_count_.load(); }

    /**
     * @brief Return the number of failed HTTP requests since start().
     *
     * @return Error count.
     */
    uint64_t error_count() const { return error_count_.load(); }

private:
    /** @brief Main loop executed on the poller thread. */
    void poller_thread();

    /**
     * @brief Open a fresh TCP connection to config_.host:config_.port.
     *
     * @return true on success; sets sockfd_ to the connected descriptor.
     */
    bool open_socket();

    /** @brief Close the current socket descriptor if open. */
    void close_socket();

    /**
     * @brief Build the HTTP GET request string.
     *
     * @return Formatted HTTP/1.1 GET request.
     */
    std::string build_request() const;

    /**
     * @brief Parse a minimal JSON position object from an HTTP response.
     *
     * @param body Full HTTP response (including headers) or bare JSON.
     * @param out  Output parameter populated on success.
     * @return false if any required field is missing or malformed.
     */
    static bool parse_position(const std::string& body,
                               RiskManager::PositionState& out);

    Config            config_;
    PositionCallback  callback_;
    std::atomic<bool> running_{false};
    std::thread       thread_;
    int               sockfd_{-1};
    bool              wsa_initialized_{false};  ///< True iff WSAStartup succeeded (Windows only).
    std::atomic<uint64_t> update_count_{0};
    std::atomic<uint64_t> error_count_{0};
};

} // namespace llmquant
