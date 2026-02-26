#include "FixOmsAdapter.h"

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using ssize_t = int;
#else
  #include <sys/socket.h>
  #include <netdb.h>
  #include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace llmquant {

static constexpr char SOH = '\x01';

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

FixOmsAdapter::FixOmsAdapter(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

FixOmsAdapter::~FixOmsAdapter() {
    stop();
#ifdef _WIN32
    WSACleanup();
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
    running_ = true;
    thread_ = std::thread(&FixOmsAdapter::reader_thread, this);
    return true;
}

void FixOmsAdapter::stop() {
    running_ = false;
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
    std::ostringstream header;
    header << "8=FIX.4.2" << SOH
           << "9=" << body.size() << SOH;
    std::string full = header.str() + body;
    full += "10=" + fix_checksum(full) + SOH;
    return full;
}

std::string FixOmsAdapter::build_logon() const {
    std::ostringstream body;
    body << "35=A" << SOH
         << "49=" << config_.sender_comp_id << SOH
         << "56=" << config_.target_comp_id << SOH
         << "34=" << seq_num_ << SOH
         << "52=" << fix_utctime() << SOH
         << "98=0" << SOH   // EncryptMethod=None
         << "108=" << config_.heartbeat_interval_s << SOH;
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
            fields[tag] = raw.substr(eq + 1, soh - eq - 1);
        } catch (...) {}

        pos = soh + 1;
    }
    return fields;
}

void FixOmsAdapter::handle_message(const FixFields& fields) {
    messages_parsed_++;
    auto it = fields.find(35);
    if (it == fields.end()) return;

    const std::string& msg_type = it->second;
    if (msg_type == "8") {
        apply_execution_report(fields);
    } else if (msg_type == "AP") {
        apply_position_report(fields);
    }
    // Logon (A), Heartbeat (0), and other message types are silently accepted.
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

    if (callback_) callback_(state);
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
    send(sockfd_, logon.c_str(), static_cast<int>(logon.size()), 0);
    seq_num_++;

    auto last_heartbeat = std::chrono::steady_clock::now();
    std::string buf;
    buf.reserve(8192);

    char chunk[4096];
    while (running_.load()) {
        ssize_t n = recv(sockfd_, chunk,
                         static_cast<int>(sizeof(chunk) - 1), 0);
        if (n <= 0) break;
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
            handle_message(parse_fix(msg));
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
            send(sockfd_, hb.c_str(), static_cast<int>(hb.size()), 0);
            seq_num_++;
            last_heartbeat = now;
        }
    }

    close_socket();
    running_ = false;
}

} // namespace llmquant
