#pragma once

#include "OmsAdapter.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace llmquant {

/**
 * @brief Reads FIX 4.2 messages from a FIX acceptor and extracts position updates.
 *
 * Supported inbound message types:
 *   - 35=8  ExecutionReport: tag 54 (side) + tag 32 (LastQty) -> net_position.
 *   - 35=AP PositionReport:  tag 702 (LongQty) - tag 703 (ShortQty) -> net_position.
 *
 * Thread safety: start/stop are safe from any thread. The position callback
 * is invoked from the reader thread.
 */
class FixOmsAdapter : public OmsAdapter {
public:
    /** @brief Connection and session configuration. */
    struct Config {
        /** @brief FIX acceptor host name or IP address. */
        std::string host{"127.0.0.1"};
        /** @brief FIX acceptor TCP port. */
        uint16_t    port{9878};
        /** @brief SenderCompID for all outbound FIX messages. */
        std::string sender_comp_id{"QUANT_ENGINE"};
        /** @brief TargetCompID for all outbound FIX messages. */
        std::string target_comp_id{"OMS"};
        /** @brief HeartBtInt value sent in Logon; also the heartbeat send interval (seconds). */
        int heartbeat_interval_s{30};
        /** @brief position_limit injected into the PositionState emitted to callers. */
        double position_limit{1.0};
        /** @brief pnl_limit injected into the PositionState emitted to callers. */
        double pnl_limit{-10.0};
    };

    /**
     * @brief Construct the adapter with the given session configuration.
     * @param config Session parameters including host, port, and comp IDs.
     */
    explicit FixOmsAdapter(Config config);

    /** @brief Destructor calls stop() and cleans up Winsock on Windows. */
    ~FixOmsAdapter() override;

    /**
     * @brief Register the callback invoked on each position update.
     * @param cb Called on the reader thread for each ExecutionReport or PositionReport.
     */
    void set_position_callback(PositionCallback cb) override;

    /**
     * @brief Connect to the FIX acceptor and start the reader thread.
     * @return false if already running or the TCP connection fails.
     */
    bool start() override;

    /** @brief Signal the reader thread to stop and block until it exits. */
    void stop() override;

    /**
     * @brief Returns true while the reader thread is active.
     * @return Running state.
     */
    bool is_running() const override { return running_.load(); }

    /**
     * @brief Returns a description string containing host, port, and comp IDs.
     * @return Human-readable adapter description.
     */
    std::string description() const override;

    /**
     * @brief Return total FIX messages parsed since start().
     * @return Parsed message count.
     */
    uint64_t messages_parsed() const { return messages_parsed_.load(); }

    /**
     * @brief Return the number of times the FIX session has reconnected since start().
     *
     * Incremented each time reconnect_with_backoff() is called.
     * Thread-safe (atomic read).
     *
     * @return Reconnect attempt count.
     */
    uint64_t get_reconnect_count() const noexcept {
        return reconnect_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Return the current outbound FIX message sequence number.
     *
     * Returns the next sequence number that will be assigned to an outgoing
     * message.  Starts at 1 (per FIX 4.2 spec) and increments per message.
     * Not thread-safe — must be called before start() or after stop().
     *
     * @return Current outbound sequence number.
     */
    uint32_t get_seq_num() const noexcept { return seq_num_; }

protected:
    /**
     * @brief Compute the FIX 4.2 checksum for a message body string.
     *
     * Exposed as protected so test subclasses can verify checksum logic
     * without a live FIX session.
     *
     * @param msg The raw FIX message body (all tags before tag 10).
     * @return Three-character zero-padded checksum string (e.g. "042").
     */
    static std::string fix_checksum(const std::string& msg);

    /**
     * @brief Wrap a partial FIX message body with BeginString/BodyLength/CheckSum.
     *
     * Exposed as protected so test subclasses can verify message framing.
     *
     * @param body Partial FIX message body (tags 35 onward, without tags 8/9/10).
     * @return Complete framed FIX message string.
     */
    std::string fix_message(const std::string& body) const;

private:
    void reader_thread();
    bool reconnect_with_backoff();
    void send_sequence_reset();
    void send_resend_request(int begin_seq, int end_seq);
    bool open_socket();
    void close_socket();

    std::string build_logon(bool reset_seq = false) const;
    std::string build_heartbeat() const;
    std::string build_resend_request(int begin_seq) const;
    std::string build_sequence_reset(int new_seq_num) const;

    using FixFields = std::map<int, std::string>;
    static FixFields parse_fix(const std::string& raw);
    void handle_message(const FixFields& fields);
    void apply_execution_report(const FixFields& fields);
    void apply_position_report(const FixFields& fields);
    void emit_position();

    Config            config_;
    PositionCallback  callback_;
    std::atomic<bool> running_{false};
    /// Set to true before close_socket() in stop() so the reader thread can
    /// distinguish an intentional socket close from an unexpected recv() error.
    std::atomic<bool> shutdown_requested_{false};
    std::thread       thread_;
    int               sockfd_{-1};
    // FIX 4.2 seq nums are unsigned 32-bit per spec
    uint32_t          seq_num_{1};              ///< Next outbound sequence number.
    uint32_t          expected_inbound_seq_{1}; ///< Next expected inbound MsgSeqNum.

    mutable std::mutex pos_mutex_;
    double net_position_{0.0};
    double pnl_{0.0};

    std::atomic<uint64_t> messages_parsed_{0};
    std::atomic<uint64_t> reconnect_count_{0};
    int  reconnect_attempts_{0};
    bool wsa_initialized_{false}; ///< True iff WSAStartup succeeded (Windows only).
    static constexpr int kMaxReconnectBackoffSeconds = 60;
};

} // namespace llmquant
