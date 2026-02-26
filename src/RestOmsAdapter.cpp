#include "RestOmsAdapter.h"

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using ssize_t = int;
#else
  #include <sys/socket.h>
  #include <netdb.h>
  #include <unistd.h>
#endif

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

RestOmsAdapter::RestOmsAdapter(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

RestOmsAdapter::~RestOmsAdapter() {
    stop();
#ifdef _WIN32
    WSACleanup();
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
    running_ = true;
    thread_ = std::thread(&RestOmsAdapter::poller_thread, this);
    return true;
}

void RestOmsAdapter::stop() {
    running_ = false;
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
    std::ostringstream oss;
    oss << "GET " << config_.path << " HTTP/1.1\r\n"
        << "Host: " << config_.host << "\r\n";
    if (!config_.api_key.empty()) {
        oss << "Authorization: Bearer " << config_.api_key << "\r\n";
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

    // Collect the numeric token.
    size_t start = pos;
    while (pos < json.size() &&
           (std::isdigit(static_cast<unsigned char>(json[pos])) ||
            json[pos] == '-' || json[pos] == '.' ||
            json[pos] == 'e' || json[pos] == 'E' ||
            json[pos] == '+')) {
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
    // Strip HTTP headers: the JSON body begins after the blank line.
    size_t json_start = body.find("\r\n\r\n");
    std::string json = (json_start != std::string::npos)
                       ? body.substr(json_start + 4)
                       : body;

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
    while (running_.load()) {
        if (!open_socket()) {
            error_count_++;
            std::this_thread::sleep_for(config_.poll_interval);
            continue;
        }

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
        std::string response;
        if (send_ok) {
            char buf[4096];
            while (true) {
                ssize_t n = recv(sockfd_, buf,
                                 static_cast<int>(sizeof(buf) - 1), 0);
                if (n <= 0) break;
                buf[n] = '\0';
                response.append(buf, static_cast<size_t>(n));
            }
        }

        close_socket();

        if (!response.empty()) {
            RiskManager::PositionState state;
            if (parse_position(response, state)) {
                if (callback_) callback_(state);
                update_count_++;
            } else {
                error_count_++;
            }
        } else {
            error_count_++;
        }

        std::this_thread::sleep_for(config_.poll_interval);
    }
}

} // namespace llmquant
