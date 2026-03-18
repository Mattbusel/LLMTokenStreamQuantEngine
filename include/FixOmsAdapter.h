#pragma once

#include "OmsAdapter.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace llmquant {

/// Reads FIX 4.2 messages from a FIX acceptor and extracts position updates.
///
/// ## Supported Inbound Message Types
/// - `35=8`  ExecutionReport — extracts tag 54 (side: 1=Buy, 2=Sell) and
///           tag 32 (LastQty) to accumulate net_position.
/// - `35=AP` PositionReport  — extracts tag 702 (LongQty) and tag 703
///           (ShortQty) to set net_position = LongQty - ShortQty.
///
/// ## Session Behaviour
/// Sends a FIX Logon (35=A) on connect and a Heartbeat (35=0) every
/// HeartBtInt seconds.  Sequence gaps are detected via tag 34 (MsgSeqNum); on
/// a gap a ResendRequest (35=2) is sent and a SequenceReset (35=4) is handled
/// to advance the expected inbound sequence.  On reconnect the Logon carries
/// ResetSeqNumFlag (141=Y) to reset both sides to seq 1.
///
/// ## Thread Safety
/// start/stop are safe from any thread.  The position callback is invoked from
/// the reader thread.
class FixOmsAdapter : public OmsAdapter {
public:
    /// Connection and session configuration.
    struct Config {
        /// FIX acceptor host name or IP address.
        std::string host{"127.0.0.1"};
        /// FIX acceptor TCP port.
        uint16_t    port{9878};
        /// SenderCompID for all outbound FIX messages.
        std::string sender_comp_id{"QUANT_ENGINE"};
        /// TargetCompID for all outbound FIX messages.
        std::string target_comp_id{"OMS"};
        /// HeartBtInt value sent in Logon; also the heartbeat send interval.
        int heartbeat_interval_s{30};
        /// position_limit injected into the PositionState emitted to callers.
        double position_limit{1.0};
        /// pnl_limit injected into the PositionState emitted to callers.
        double pnl_limit{-10.0};
    };

    /// Construct the adapter with the given session configuration.
    ///
    /// # Arguments
    /// * `config` — Session parameters including host, port, and comp IDs.
    explicit FixOmsAdapter(Config config);

    /// Destructor: calls stop() and cleans up Winsock on Windows.
    ~FixOmsAdapter() override;

    /// Register the callback invoked on each position update.
    ///
    /// # Arguments
    /// * `cb` — Called on the reader thread for each ExecutionReport or
    ///          PositionReport that changes the accumulated position.
    void set_position_callback(PositionCallback cb) override;

    /// Connect to the FIX acceptor and start the reader thread.
    ///
    /// # Returns
    /// `false` if already running or the TCP connection fails.
    bool start() override;

    /// Signal the reader thread to stop and block until it exits.
    void stop() override;

    /// True while the reader thread is active.
    bool is_running() const override { return running_.load(); }

    /// Returns a description string containing host, port, and comp IDs.
    std::string description() const override;

    /// Return total FIX messages parsed since start().
    uint64_t messages_parsed() const { return messages_parsed_.load(); }

private:
    /// Main loop executed on the reader thread.
    void reader_thread();

    /// Close socket, wait with exponential backoff, reopen, re-logon.
    /// Returns true on successful reconnect, false if stopped.
    bool reconnect_with_backoff();
    /// Send SequenceReset-Reset (35=4, GapFillFlag=N, NewSeqNo=1).
    void send_sequence_reset();
    /// Send ResendRequest (35=2) for range [begin_seq, end_seq].
    void send_resend_request(int begin_seq, int end_seq);

    /// Open a TCP connection to config_.host:config_.port.
    bool open_socket();

    /// Close the current socket descriptor if open.
    void close_socket();

    // -----------------------------------------------------------------------
    // FIX message construction
    // -----------------------------------------------------------------------

    /// Build a complete FIX Logon (35=A) message.
    /// When `reset_seq` is true the message includes ResetSeqNumFlag (141=Y).
    std::string build_logon(bool reset_seq = false) const;

    /// Build a complete FIX Heartbeat (35=0) message.
    std::string build_heartbeat() const;

    /// Build a FIX ResendRequest (35=2) asking the acceptor to retransmit
    /// messages from `begin_seq` to end-of-session (tag 16=0).
    std::string build_resend_request(int begin_seq) const;

    /// Build a FIX SequenceReset-Reset (35=4, GapFillFlag=N) advancing the
    /// sender's sequence number to `new_seq_num`.
    std::string build_sequence_reset(int new_seq_num) const;

    /// Wrap a pre-assembled FIX body with BeginString, BodyLength, and
    /// Checksum fields and return the complete transmittable message.
    std::string fix_message(const std::string& body) const;

    /// Compute the FIX checksum (sum of all byte values mod 256, 3 digits).
    static std::string fix_checksum(const std::string& msg);

    // -----------------------------------------------------------------------
    // FIX message parsing
    // -----------------------------------------------------------------------

    /// Tag-value map for a single parsed FIX message.
    using FixFields = std::map<int, std::string>;

    /// Parse a raw SOH-delimited FIX message into a tag-value map.
    static FixFields parse_fix(const std::string& raw);

    /// Dispatch a parsed message to the appropriate handler.
    void handle_message(const FixFields& fields);

    // -----------------------------------------------------------------------
    // Position accumulation
    // -----------------------------------------------------------------------

    /// Handle an ExecutionReport (35=8): update net_position from fills.
    void apply_execution_report(const FixFields& fields);

    /// Handle a PositionReport (35=AP): set net_position = LongQty - ShortQty.
    void apply_position_report(const FixFields& fields);

    /// Construct a PositionState from the current internal state and invoke
    /// the registered callback.  Must be called with pos_mutex_ held.
    void emit_position();

    Config            config_;
    PositionCallback  callback_;
    std::atomic<bool> running_{false};
    std::thread       thread_;
    int               sockfd_{-1};
    // seq_num_ and expected_inbound_seq_ are accessed exclusively from reader_thread()
    // and the helpers it calls (build_*, send_*, handle_message, reconnect_with_backoff).
    // They must NOT be accessed from stop(), the constructor (after start()), or any
    // other thread without adding synchronisation.
    int               seq_num_{1};            ///< Next outbound sequence number.
    int               expected_inbound_seq_{1}; ///< Next expected inbound MsgSeqNum.

    // Accumulated position protected by pos_mutex_.
    mutable std::mutex pos_mutex_;
    double net_position_{0.0};
    double pnl_{0.0};

    std::atomic<uint64_t> messages_parsed_{0};

    int  reconnect_attempts_{0};
    bool wsa_initialized_{false};  ///< True iff WSAStartup succeeded (Windows only).
    static constexpr int kMaxReconnectBackoffSeconds = 60;
};

} // namespace llmquant
