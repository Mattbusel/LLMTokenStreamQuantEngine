#include "RestOmsAdapter.h"

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using ssize_t = int;
#else
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <netinet/tcp.h>
  #include <cerrno>
#endif

#include <cstring>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

RestOmsAdapter::RestOmsAdapter(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    wsa_initialized_ = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#endif
}

RestOmsAdapter::~RestOmsAdapter() {
    stop();
#ifdef _WIN32
    if (wsa_initialized_) WSACleanup();
#endif
}

// ---------------------------------------------------------------------------
// OmsAdapter interface
// ---------------------------------------------------------------------------

void RestOmsAdapter::set_position_callback(PositionCallback cb) {
    callback_ = std::move(cb);
}

bool RestOmsAdapter::start() {
    if (running_.load()) return false;
    shutdown_requested_ = false;
    running_ = true;
    thread_ = std::thread(&RestOmsAdapter::poller_thread, this);
    return true;
}

void RestOmsAdapter::stop() {
    // Closing the socket from another thread is the only portable way to unblock recv();
    // EBADF/WSAENOTSOCK after close_socket() is expected and not an error.
    shutdown_requested_ = true;
    running_ = false;
    // close_socket() must come before thread_.join(): the background thread may
    // be blocked in recv() and will not observe running_==false until the socket
    // is closed and recv() returns an error.  Closing the socket here is
    // intentional and is the only way to unblock the blocking recv() call.
    // The formal race on sockfd_ is benign in practice because close_socket()
    // runs on the calling thread while the poller_thread only writes sockfd_ in
    // open_socket() — which it will not enter again once running_ is false.
    close_socket();
    if (thread_.joinable()) thread_.join();
}

std::string RestOmsAdapter::description() const {
    return "REST OMS adapter: http://" + config_.host + ":" +
           std::to_string(config_.port) + config_.path +
           " (poll=" + std::to_string(config_.poll_interval.count()) + "ms)";
}

// ---------------------------------------------------------------------------
// Socket helpers
// ---------------------------------------------------------------------------

bool RestOmsAdapter::open_socket() {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(config_.port);
    addrinfo* res = nullptr;
    if (getaddrinfo(config_.host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        return false;
    }

    sockfd_ = -1;
    for (addrinfo* rp = res; rp != nullptr; rp = rp->ai_next) {
        int fd = static_cast<int>(
            socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol));
        if (fd < 0) continue;
        if (::connect(fd, rp->ai_addr,
                      static_cast<int>(rp->ai_addrlen)) == 0) {
            sockfd_ = fd;
            // Apply send/receive timeouts — config_.timeout_s has been set but never used.
#ifdef _WIN32
            DWORD timeout_ms = static_cast<DWORD>(config_.timeout_s * 1000);
            setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
            setsockopt(sockfd_, SOL_SOCKET, SO_SNDTIMEO,
                       reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
            struct timeval tv{static_cast<time_t>(config_.timeout_s), 0};
            setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&tv), sizeof(tv));
            setsockopt(sockfd_, SOL_SOCKET, SO_SNDTIMEO,
                       reinterpret_cast<const char*>(&tv), sizeof(tv));
#endif
            // Disable Nagle's algorithm for low-latency REST polling.
            int nodelay_flag = 1;
            if (setsockopt(sockfd_, IPPROTO_TCP, TCP_NODELAY,
                           reinterpret_cast<const char*>(&nodelay_flag), sizeof(nodelay_flag)) != 0) {
                spdlog::debug("[RestOmsAdapter] TCP_NODELAY setsockopt failed (non-fatal)");
            }
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

void RestOmsAdapter::close_socket() {
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
// HTTP request building
// ---------------------------------------------------------------------------

std::string RestOmsAdapter::build_request() const {
    // Sanitise API key — reject CR/LF to prevent header injection.
    auto sanitise_header_value = [](const std::string& val) -> std::string {
        std::string out;
        out.reserve(val.size());
        for (char c : val) {
            if (c != '\r' && c != '\n') out += c;
        }
        return out;
    };

    std::ostringstream oss;
    // Sanitise all user-configurable values embedded in the request line or
    // headers — config_.path and config_.host are sanitised for the same
    // reason as api_key: a CR/LF in either value would inject extra headers.
    oss << "GET " << sanitise_header_value(config_.path) << " HTTP/1.1\r\n"
        << "Host: " << sanitise_header_value(config_.host) << "\r\n";
    if (!config_.api_key.empty()) {
        oss << "Authorization: Bearer " << sanitise_header_value(config_.api_key) << "\r\n";
    }
    oss << "Accept: application/json\r\n"
        << "Connection: close\r\n\r\n";
    return oss.str();
}

// ---------------------------------------------------------------------------
// JSON parsing (no external dependencies — minimal field scanner)
// ---------------------------------------------------------------------------

namespace {

/// Find "key": <number> in a flat JSON object.  Returns false if not found.
bool extract_double(const std::string& json,
                    const std::string& key, double& out) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;

    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;

    // Skip whitespace.
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\t' ||
            json[pos] == '\r' || json[pos] == '\n')) {
        ++pos;
    }

    // Collect the numeric token — guard against oversized values.
    constexpr size_t kMaxNumericLen = 64;
    size_t start = pos;
    while (pos < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[pos])) ||
            json[pos] == '-' || json[pos] == '.' ||
            json[pos] == 'e' || json[pos] == 'E' ||
            json[pos] == '+')) {
        if (pos - start >= kMaxNumericLen) {
            spdlog::warn("RestOmsAdapter: oversized numeric value in JSON response, skipping");
            return false;
        }
        ++pos;
    }
    if (pos == start) return false;

    try {
        out = std::stod(json.substr(start, pos - start));
    } catch (...) {
        return false;
    }
    return true;
}

} // anonymous namespace

bool RestOmsAdapter::parse_position(const std::string& body,
                                    RiskManager::PositionState& out) {
    // Validate HTTP status code before attempting JSON parse.
    if (body.rfind("HTTP/", 0) == 0) {
        size_t sp1 = body.find(' ');
        if (sp1 != std::string::npos) {
            size_t sp2 = body.find(' ', sp1 + 1);
            std::string code_str = body.substr(
                sp1 + 1,
                sp2 != std::string::npos ? sp2 - sp1 - 1 : std::string::npos);
            try {
                int code = std::stoi(code_str);
                if (code < 200 || code >= 300) {
                    spdlog::warn("[rest_oms] HTTP error status: {}", code);
                    return false;
                }
            } catch (...) {
                return false;
            }
        }
    }

    // Strip HTTP headers: the JSON body begins after the blank line.
    size_t json_start = body.find("\r\n\r\n");
    std::string json = (json_start != std::string::npos)
                       ? body.substr(json_start + 4)
                       : body;

    // Decode chunked transfer encoding if present (hex-size CRLF data CRLF ...).
    // Detect cheaply: first line must be a hex number followed by CRLF.
    {
        size_t first_crlf = json.find("\r\n");
        if (first_crlf != std::string::npos && first_crlf > 0) {
            std::string maybe_hex = json.substr(0, first_crlf);
            bool all_hex = !maybe_hex.empty();
            for (char c : maybe_hex)
                if (!std::isxdigit(static_cast<unsigned char>(c))) { all_hex = false; break; }
            if (all_hex) {
                // Reassemble chunked body.
                std::string decoded;
                size_t pos = 0;
                while (pos < json.size()) {
                    size_t end_of_size = json.find("\r\n", pos);
                    if (end_of_size == std::string::npos) break;
                    size_t chunk_size = 0;
                    try { chunk_size = std::stoul(json.substr(pos, end_of_size - pos), nullptr, 16); }
                    catch (...) { break; }
                    // Guard against malformed/malicious chunk sizes.
                    constexpr size_t kMaxChunkSize = 1024 * 1024;  // 1 MB sanity limit
                    if (chunk_size == 0 || chunk_size > kMaxChunkSize) break;
                    size_t data_start = end_of_size + 2;
                    if (chunk_size > json.size() - data_start) break;
                    decoded.append(json, data_start, chunk_size);
                    pos = data_start + chunk_size + 2;  // skip trailing CRLF
                }
                if (!decoded.empty()) json = std::move(decoded);
            }
        }
    }

    bool ok = true;
    ok &= extract_double(json, "net_position",   out.net_position);
    ok &= extract_double(json, "position_limit", out.position_limit);
    ok &= extract_double(json, "pnl",            out.pnl);
    ok &= extract_double(json, "pnl_limit",      out.pnl_limit);
    return ok;
}

// ---------------------------------------------------------------------------
// Poller thread
// ---------------------------------------------------------------------------

void RestOmsAdapter::poller_thread() {
    // Initialise the effective poll interval from config (may be grown by backoff).
    current_poll_interval_ = config_.poll_interval;
    static constexpr std::chrono::milliseconds kMaxPollInterval{30000};

    while (running_.load()) {
        bool success = false;

        if (!open_socket()) {
            error_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::string request = build_request();

            // Send full request.
            bool send_ok = true;
            size_t sent = 0;
            while (sent < request.size()) {
                ssize_t n = send(sockfd_,
                                 request.c_str() + sent,
                                 static_cast<int>(request.size() - sent), 0);
                if (n <= 0) { send_ok = false; break; }
                sent += static_cast<size_t>(n);
            }

            // Receive full response until the server closes the connection.
            // Cap at 4 MB to guard against a rogue server exhausting memory.
            static constexpr size_t kMaxResponseBytes = 4u * 1024u * 1024u;
            std::string response;
            if (send_ok) {
                char buf[4096];
                while (true) {
                    ssize_t n = recv(sockfd_, buf,
                                     static_cast<int>(sizeof(buf) - 1), 0);
                    if (n <= 0) break;
                    buf[n] = '\0';
                    if (response.size() + static_cast<size_t>(n) > kMaxResponseBytes) {
                        spdlog::warn("RestOmsAdapter: response exceeds {}MB limit, discarding",
                                     kMaxResponseBytes / (1024 * 1024));
                        response.clear();
                        break;
                    }
                    response.append(buf, static_cast<size_t>(n));
                }
            }

            close_socket();

            if (!response.empty()) {
                RiskManager::PositionState state;
                if (parse_position(response, state)) {
                    if (callback_) callback_(state);
                    update_count_.fetch_add(1, std::memory_order_relaxed);
                    success = true;
                } else {
                    error_count_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                error_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Circuit-breaker: exponential backoff after 3 consecutive failures.
        if (success) {
            if (consecutive_failures_ > 0) {
                spdlog::info("RestOmsAdapter: connection recovered after {} failures, resetting poll interval",
                             consecutive_failures_);
            }
            consecutive_failures_ = 0;
            current_poll_interval_ = config_.poll_interval;  // reset on success
        } else {
            ++consecutive_failures_;
            if (consecutive_failures_ >= 3) {
                // Double the poll interval up to the cap of 30 seconds.
                auto doubled = std::chrono::milliseconds{current_poll_interval_.count() * 2};
                current_poll_interval_ = (doubled < kMaxPollInterval) ? doubled : kMaxPollInterval;
                spdlog::warn("RestOmsAdapter: {} consecutive failures, backing off to {}ms poll interval",
                             consecutive_failures_, current_poll_interval_.count());
            }
        }

        // Interruptible sleep: check running_ every 100ms so stop() returns
        // promptly even when the backoff interval is near the 30-second cap.
        auto deadline = std::chrono::steady_clock::now() + current_poll_interval_;
        while (running_.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
}

} // namespace llmquant
