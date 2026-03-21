#include "FixOmsAdapter.h"

#ifdef _WIN32
  #define NOMINMAX
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using ssize_t = int;
#else
  #include <sys/socket.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <netinet/tcp.h>
  #include <cerrno>
#endif

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

static constexpr char SOH = '\x01';

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

FixOmsAdapter::FixOmsAdapter(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    wsa_initialized_ = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#endif
}

FixOmsAdapter::~FixOmsAdapter() {
    stop();
#ifdef _WIN32
    if (wsa_initialized_) WSACleanup();
#endif
}

// ---------------------------------------------------------------------------
// OmsAdapter interface
// ---------------------------------------------------------------------------

void FixOmsAdapter::set_position_callback(PositionCallback cb) {
    callback_ = std::move(cb);
}

bool FixOmsAdapter::start() {
    if (running_.load()) return false;
    shutdown_requested_ = false;
    running_ = true;
    thread_ = std::thread(&FixOmsAdapter::reader_thread, this);
    return true;
}

void FixOmsAdapter::stop() {
    // Closing the socket from another thread is the only portable way to unblock recv();
    // EBADF/WSAENOTSOCK after close_socket() is expected and not an error.
    shutdown_requested_ = true;
    running_ = false;
    // close_socket() must come before thread_.join(): the background thread may
    // be blocked in recv() and will not observe running_==false until the socket
    // is closed and recv() returns an error.  Closing the socket here is
    // intentional and is the only way to unblock the blocking recv() call.
    // The formal race on sockfd_ is benign in practice because close_socket()
    // runs on the calling thread while the reader_thread only writes sockfd_ in
    // open_socket() — which it will not enter again once running_ is false.
    close_socket();
    if (thread_.joinable()) thread_.join();
}

std::string FixOmsAdapter::description() const {
    return "FIX 4.2 OMS adapter: " + config_.host + ":" +
           std::to_string(config_.port) +
           " [" + config_.sender_comp_id + "->" + config_.target_comp_id + "]";
}

// ---------------------------------------------------------------------------
// Socket helpers
// ---------------------------------------------------------------------------

bool FixOmsAdapter::open_socket() {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string ps = std::to_string(config_.port);
    addrinfo* res = nullptr;
    if (getaddrinfo(config_.host.c_str(), ps.c_str(), &hints, &res) != 0) {
        return false;
    }

    sockfd_ = -1;
    for (auto* rp = res; rp != nullptr; rp = rp->ai_next) {
        int fd = static_cast<int>(
            socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol));
        if (fd < 0) continue;
        if (::connect(fd, rp->ai_addr,
                      static_cast<int>(rp->ai_addrlen)) == 0) {
            sockfd_ = fd;
            // Disable Nagle's algorithm for low-latency FIX message delivery.
            int flag = 1;
            if (setsockopt(sockfd_, IPPROTO_TCP, TCP_NODELAY,
                           reinterpret_cast<const char*>(&flag), sizeof(flag)) != 0) {
                spdlog::debug("[FixOmsAdapter] TCP_NODELAY setsockopt failed (non-fatal)");
            }
            // Set a 30-second receive timeout so the reader thread doesn't block forever.
#ifdef _WIN32
            DWORD rcv_timeout_ms = 30000;
            setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rcv_timeout_ms), sizeof(rcv_timeout_ms));
#else
            struct timeval rcv_tv{30, 0};
            setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rcv_tv), sizeof(rcv_tv));
#endif
            break;
        }
#ifdef _WIN32
        closesocket(fd);
#else
        ::close(fd);
#endif
    }
    freeaddrinfo(res);
    return sockfd_ >= 0;
}

void FixOmsAdapter::close_socket() {
    if (sockfd_ >= 0) {
#ifdef _WIN32
        closesocket(sockfd_);
#else
        ::close(sockfd_);
#endif
        sockfd_ = -1;
    }
}

// ---------------------------------------------------------------------------
// FIX message construction
// ---------------------------------------------------------------------------

namespace {

/// UTC timestamp in FIX format: YYYYMMDD-HH:MM:SS
std::string fix_utctime() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d-%H:%M:%S");
    return oss.str();
}

} // anonymous namespace

std::string FixOmsAdapter::fix_checksum(const std::string& msg) {
    unsigned int sum = 0;
    for (unsigned char c : msg) sum += c;
    sum %= 256;
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << sum;
    return oss.str();
}

std::string FixOmsAdapter::fix_message(const std::string& body) const {
    // BodyLength (tag 9) = bytes from start of body through end of checksum tag.
    // Checksum tag "10=XXX\x01" is always 7 bytes.
    const size_t body_len = body.size() + 7;
    std::ostringstream header;
    header << "8=FIX.4.2" << SOH << "9=" << body_len << SOH;
    std::string full = header.str() + body;
    full += "10=" + fix_checksum(full) + SOH;
    return full;
}

std::string FixOmsAdapter::build_logon(bool reset_seq) const {
    std::ostringstream body;
    body << "35=A" << SOH
         << "49=" << config_.sender_comp_id << SOH
         << "56=" << config_.target_comp_id << SOH
         << "34=" << seq_num_ << SOH
         << "52=" << fix_utctime() << SOH
         << "98=0" << SOH   // EncryptMethod=None
         << "108=" << config_.heartbeat_interval_s << SOH;
    if (reset_seq) body << "141=Y" << SOH;  // ResetSeqNumFlag
    return fix_message(body.str());
}

std::string FixOmsAdapter::build_resend_request(int begin_seq) const {
    std::ostringstream body;
    body << "35=2" << SOH
         << "49=" << config_.sender_comp_id << SOH
         << "56=" << config_.target_comp_id << SOH
         << "34=" << seq_num_ << SOH
         << "52=" << fix_utctime() << SOH
         << "7=" << begin_seq << SOH   // BeginSeqNo
         << "16=0" << SOH;             // EndSeqNo=0 means "to present"
    return fix_message(body.str());
}

std::string FixOmsAdapter::build_sequence_reset(int new_seq_num) const {
    std::ostringstream body;
    body << "35=4" << SOH
         << "49=" << config_.sender_comp_id << SOH
         << "56=" << config_.target_comp_id << SOH
         << "34=" << seq_num_ << SOH
         << "52=" << fix_utctime() << SOH
         << "36=" << new_seq_num << SOH  // NewSeqNo
         << "123=N" << SOH;             // GapFillFlag=N (hard reset)
    return fix_message(body.str());
}

std::string FixOmsAdapter::build_heartbeat() const {
    std::ostringstream body;
    body << "35=0" << SOH
         << "49=" << config_.sender_comp_id << SOH
         << "56=" << config_.target_comp_id << SOH
         << "34=" << seq_num_ << SOH
         << "52=" << fix_utctime() << SOH;
    return fix_message(body.str());
}

// ---------------------------------------------------------------------------
// FIX message parsing
// ---------------------------------------------------------------------------

FixOmsAdapter::FixFields FixOmsAdapter::parse_fix(const std::string& raw) {
    FixFields fields;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t eq = raw.find('=', pos);
        if (eq == std::string::npos) break;

        size_t soh = raw.find(SOH, eq + 1);
        if (soh == std::string::npos) soh = raw.size();

        try {
            int tag = std::stoi(raw.substr(pos, eq - pos));
            // Reject out-of-range FIX tags (valid range: 1–9999).
            if (tag >= 1 && tag <= 9999)
                fields[tag] = raw.substr(eq + 1, soh - eq - 1);
        } catch (const std::exception& ex) {
            spdlog::debug("[fix_oms] failed to parse tag value '{}': {}",
                          raw.substr(pos, eq - pos), ex.what());
        } catch (...) {}

        pos = soh + 1;
    }
    return fields;
}

void FixOmsAdapter::handle_message(const FixFields& fields) {
    messages_parsed_++;

    // --- Sequence number tracking ---
    auto seq_it = fields.find(34);  // MsgSeqNum
    if (seq_it != fields.end()) {
        try {
            // FIX 4.2 seq nums are unsigned 32-bit per spec
            uint32_t received_seq = static_cast<uint32_t>(std::stoul(seq_it->second));
            if (received_seq > expected_inbound_seq_) {
                // Gap detected: request retransmission of missing messages.
                send_resend_request(static_cast<int>(expected_inbound_seq_),
                                    static_cast<int>(received_seq - 1));
                // Advance expected so we don't re-request on the next message.
                expected_inbound_seq_ = received_seq + 1;
            } else if (received_seq == expected_inbound_seq_) {
                expected_inbound_seq_++;
            }
            // received_seq < expected_inbound_seq_: duplicate — skip content processing below.
            else { return; }
        } catch (...) {}
    }

    auto it = fields.find(35);
    if (it == fields.end()) return;

    const std::string& msg_type = it->second;
    if (msg_type == "8") {
        apply_execution_report(fields);
    } else if (msg_type == "AP") {
        apply_position_report(fields);
    } else if (msg_type == "4") {
        // SequenceReset: advance expected inbound sequence to tag 36 (NewSeqNo).
        auto ns_it = fields.find(36);
        if (ns_it != fields.end()) {
            // FIX 4.2 seq nums are unsigned 32-bit per spec; use stoul to avoid signed overflow
            try { expected_inbound_seq_ = static_cast<uint32_t>(std::stoul(ns_it->second)); } catch (...) {}
        }
    } else if (msg_type == "2") {
        // ResendRequest: the counterparty wants messages we cannot replay.
        // Respond with a SequenceReset-GapFill (35=4, 123=Y) covering the
        // requested range so the session can resume without a full disconnect.
        auto begin_it = fields.find(7);   // BeginSeqNo
        auto end_it   = fields.find(16);  // EndSeqNo
        if (begin_it != fields.end()) {
            try {
                // FIX 4.2 seq nums are unsigned 32-bit per spec; use stoul to avoid signed overflow
                uint32_t begin = static_cast<uint32_t>(std::stoul(begin_it->second));
                // EndSeqNo=0 means "to current" — use our next outbound seq.
                uint32_t new_seq = (end_it != fields.end() && end_it->second != "0")
                              ? static_cast<uint32_t>(std::stoul(end_it->second)) + 1u
                              : seq_num_;
                std::ostringstream body;
                body << "35=4"  << SOH
                     << "49=" << config_.sender_comp_id << SOH
                     << "56=" << config_.target_comp_id << SOH
                     << "34=" << begin << SOH
                     << "52=" << fix_utctime() << SOH
                     << "36=" << new_seq << SOH
                     << "123=Y" << SOH;  // GapFillFlag=Y
                std::string msg = fix_message(body.str());
                auto sent = ::send(sockfd_, msg.c_str(), static_cast<int>(msg.size()), 0);
                if (sent < 0) {
                    spdlog::error("[fix_oms] send() failed in handle_message");
                } else if (static_cast<size_t>(sent) != msg.size()) {
                    spdlog::warn("[fix_oms] handle_message: inline send partial ({}/{})", sent, msg.size());
                }
            } catch (...) {}
        }
    }
    // Logon (A), Heartbeat (0), and other types accepted silently.
}

// ---------------------------------------------------------------------------
// Position accumulation
// ---------------------------------------------------------------------------

void FixOmsAdapter::apply_execution_report(const FixFields& fields) {
    // Tag 54: Side (1=Buy, 2=Sell).  Tag 32: LastQty.
    auto side_it = fields.find(54);
    auto qty_it  = fields.find(32);
    if (side_it == fields.end() || qty_it == fields.end()) return;

    try {
        double qty  = std::stod(qty_it->second);
        double sign = (side_it->second == "1") ? 1.0 : -1.0;
        {
            std::lock_guard<std::mutex> lock(pos_mutex_);
            net_position_ += sign * qty;
        }
        emit_position();
    } catch (...) {}
}

void FixOmsAdapter::apply_position_report(const FixFields& fields) {
    // Tag 702: LongQty.  Tag 703: ShortQty.
    auto long_it  = fields.find(702);
    auto short_it = fields.find(703);
    if (long_it == fields.end() && short_it == fields.end()) return;

    try {
        double lq = (long_it  != fields.end()) ? std::stod(long_it->second)  : 0.0;
        double sq = (short_it != fields.end()) ? std::stod(short_it->second) : 0.0;
        {
            std::lock_guard<std::mutex> lock(pos_mutex_);
            net_position_ = lq - sq;
        }
        emit_position();
    } catch (...) {}
}

void FixOmsAdapter::emit_position() {
    // Copy state while holding the mutex, then call the callback outside it
    // to avoid deadlocks if the callback itself touches the adapter.
    RiskManager::PositionState state;
    {
        std::lock_guard<std::mutex> lock(pos_mutex_);
        state.net_position   = net_position_;
        state.pnl            = pnl_;
    }
    state.position_limit = config_.position_limit;
    state.pnl_limit      = config_.pnl_limit;

    if (callback_) {
        callback_(state);
        update_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Session recovery helpers
// ---------------------------------------------------------------------------

bool FixOmsAdapter::reconnect_with_backoff() {
    reconnect_count_.fetch_add(1, std::memory_order_relaxed);
    close_socket();
    int backoff_s = std::min(1 << reconnect_attempts_, kMaxReconnectBackoffSeconds);
    spdlog::info("[FixOmsAdapter] reconnecting in {} seconds", backoff_s);
    for (int i = 0; i < backoff_s * 10 && running_.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!running_.load()) return false;
    if (!open_socket()) {
        ++reconnect_attempts_;
        return false;
    }
    seq_num_ = 1;
    std::string logon = build_logon(/*reset_seq=*/true);
    ssize_t sent = ::send(sockfd_, logon.c_str(), static_cast<int>(logon.size()), 0);
    if (sent < 0 || static_cast<size_t>(sent) != logon.size()) {
        close_socket();
        ++reconnect_attempts_;
        return false;
    }
    seq_num_++;
    reconnect_attempts_ = 0;
    spdlog::info("[FixOmsAdapter] reconnected successfully");
    return true;
}

void FixOmsAdapter::send_sequence_reset() {
    std::string msg = build_sequence_reset(1);
    auto sent = ::send(sockfd_, msg.c_str(), static_cast<int>(msg.size()), 0);
    if (sent < 0) {
        spdlog::error("[fix_oms] send() failed in send_sequence_reset");
    } else if (static_cast<size_t>(sent) != msg.size()) {
        spdlog::warn("[fix_oms] send_sequence_reset: partial send ({}/{})", sent, msg.size());
    } else {
        seq_num_ = 1;
    }
}

void FixOmsAdapter::send_resend_request(int begin_seq, int end_seq) {
    std::ostringstream body;
    body << "35=2" << SOH
         << "49=" << config_.sender_comp_id << SOH
         << "56=" << config_.target_comp_id << SOH
         << "34=" << seq_num_ << SOH
         << "52=" << fix_utctime() << SOH
         << "7="  << begin_seq << SOH
         << "16=" << end_seq   << SOH;
    std::string msg = fix_message(body.str());
    auto sent = ::send(sockfd_, msg.c_str(), static_cast<int>(msg.size()), 0);
    if (sent < 0) {
        spdlog::error("[fix_oms] send() failed in send_resend_request");
    } else if (static_cast<size_t>(sent) != msg.size()) {
        spdlog::warn("[fix_oms] send_resend_request: partial send ({}/{})", sent, msg.size());
    } else {
        seq_num_++;
    }
}

// ---------------------------------------------------------------------------
// Reader thread
// ---------------------------------------------------------------------------

void FixOmsAdapter::reader_thread() {
    if (!open_socket()) {
        running_ = false;
        return;
    }

    // Send FIX Logon.
    std::string logon = build_logon();
    ssize_t sent = send(sockfd_, logon.c_str(), static_cast<int>(logon.size()), 0);
    if (sent < 0 || static_cast<size_t>(sent) != logon.size()) {
        close_socket();
        running_ = false;
        return;
    }
    seq_num_++;

    auto last_heartbeat = std::chrono::steady_clock::now();
    std::string buf;
    buf.reserve(8192);

    char chunk[4096];
    while (running_.load()) {
        ssize_t n = recv(sockfd_, chunk,
                         static_cast<int>(sizeof(chunk) - 1), 0);
        if (n <= 0) {
#ifdef _WIN32
            int recv_err = WSAGetLastError();
            // WSAENOTSOCK/WSAEBADF after close_socket() from stop() is expected — not an error.
            if (shutdown_requested_.load() &&
                (recv_err == WSAENOTSOCK || recv_err == WSAEBADF || recv_err == WSAEINTR)) {
                break;
            }
            // Receive timeout — log warning and attempt reconnect.
            if (recv_err == WSAETIMEDOUT) {
                spdlog::warn("[FixOmsAdapter] no data received in 30s — possible dead connection");
                if (running_.load() && reconnect_with_backoff()) {
                    last_heartbeat = std::chrono::steady_clock::now();
                    buf.clear();
                    continue;
                }
                break;
            }
#else
            int recv_err = errno;
            // EBADF after close_socket() from stop() is expected — not an error.
            if (shutdown_requested_.load() && (recv_err == EBADF || recv_err == EINTR)) {
                break;
            }
            // Receive timeout — log warning and attempt reconnect.
            if (recv_err == EAGAIN || recv_err == EWOULDBLOCK) {
                spdlog::warn("[FixOmsAdapter] no data received in 30s — possible dead connection");
                if (running_.load() && reconnect_with_backoff()) {
                    last_heartbeat = std::chrono::steady_clock::now();
                    buf.clear();
                    continue;
                }
                break;
            }
#endif
            error_count_.fetch_add(1, std::memory_order_relaxed);
            spdlog::warn("[FixOmsAdapter] recv failed, reconnecting");
            if (reconnect_with_backoff()) {
                last_heartbeat = std::chrono::steady_clock::now();
                buf.clear();
                continue;
            }
            break;  // stop() was called or reconnect failed — exit thread
        }
        chunk[n] = '\0';
        buf.append(chunk, static_cast<size_t>(n));

        // A FIX message ends with the checksum tag (10=NNN\x01).
        // Scan the accumulated buffer for complete messages.
        size_t start = 0;
        while (true) {
            size_t cs = buf.find("10=", start);
            if (cs == std::string::npos) break;
            size_t end_soh = buf.find(SOH, cs);
            if (end_soh == std::string::npos) break;

            std::string msg = buf.substr(start, end_soh - start + 1);
            try {
                handle_message(parse_fix(msg));
            } catch (const std::exception& ex) {
                spdlog::debug("[fix_oms] skipping malformed message (first 256 bytes): {}",
                              msg.substr(0, 256));
                spdlog::debug("[fix_oms] parse error: {}", ex.what());
            } catch (...) {
                spdlog::debug("[fix_oms] skipping malformed message (first 256 bytes): {}",
                              msg.substr(0, 256));
            }
            start = end_soh + 1;
        }
        // Keep any incomplete trailing fragment.
        buf = buf.substr(start);

        // Send heartbeat on the configured interval.
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_heartbeat);
        if (elapsed.count() >= config_.heartbeat_interval_s) {
            std::string hb = build_heartbeat();
            ssize_t hb_sent = ::send(sockfd_, hb.c_str(),
                                     static_cast<int>(hb.size()), 0);
            if (hb_sent < 0 || static_cast<size_t>(hb_sent) != hb.size()) {
                error_count_.fetch_add(1, std::memory_order_relaxed);
                spdlog::warn("[FixOmsAdapter] heartbeat send failed, reconnecting");
                if (reconnect_with_backoff()) {
                    last_heartbeat = std::chrono::steady_clock::now();
                    buf.clear();
                    continue;
                }
                break;
            }
            seq_num_++;
            last_heartbeat = now;
        }
    }

    close_socket();
    running_ = false;
}

} // namespace llmquant
