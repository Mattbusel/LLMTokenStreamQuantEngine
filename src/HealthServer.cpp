#include "HealthServer.h"

#include <cerrno>
#include <cstring>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <utility>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   using socklen_t = int;
#  define CLOSE_SOCKET(fd) closesocket(fd)
#  define SOCKET_WOULD_BLOCK (WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == WSAEINTR)
#  define SOCKET_ERRNO WSAGetLastError()
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define CLOSE_SOCKET(fd) ::close(fd)
#  define INVALID_SOCKET (-1)
#  define SOCKET_WOULD_BLOCK (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
#  define SOCKET_ERRNO errno
   using SOCKET = int;
#endif

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

HealthServer::HealthServer(Config config)
    : config_(std::move(config))
{
#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0) wsa_initialized_ = true;
#endif
}

HealthServer::~HealthServer() {
    stop();
#ifdef _WIN32
    if (wsa_initialized_) WSACleanup();
#endif
}

void HealthServer::set_health_callback(HealthCallback cb) {
    health_cb_ = std::move(cb);
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

bool HealthServer::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return false;

    stop_requested_.store(false, std::memory_order_relaxed);

    SOCKET fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == INVALID_SOCKET) {
        spdlog::error("[health_server] socket() failed: {}", SOCKET_ERRNO);
        running_.store(false, std::memory_order_relaxed);
        return false;
    }

    int opt = 1;
#ifdef _WIN32
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(config_.port);
    if (config_.bind_address == "0.0.0.0" || config_.bind_address.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, config_.bind_address.c_str(), &addr.sin_addr);
    }

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        spdlog::error("[health_server] bind(port={}) failed: {}", config_.port, SOCKET_ERRNO);
        CLOSE_SOCKET(fd);
        running_.store(false, std::memory_order_relaxed);
        return false;
    }

    if (::listen(fd, 8) < 0) {
        spdlog::error("[health_server] listen() failed: {}", SOCKET_ERRNO);
        CLOSE_SOCKET(fd);
        running_.store(false, std::memory_order_relaxed);
        return false;
    }

    // Set non-blocking so we can check stop_requested_ periodically.
#ifdef _WIN32
    u_long nb = 1;
    ioctlsocket(fd, FIONBIO, &nb);
#else
    int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
#endif

    listen_fd_ = static_cast<int>(fd);
    thread_ = std::thread(&HealthServer::server_thread, this);
    spdlog::info("[health_server] listening on port {}", config_.port);
    return true;
}

void HealthServer::stop() {
    stop_requested_.store(true, std::memory_order_relaxed);
    if (listen_fd_ >= 0) {
        CLOSE_SOCKET(static_cast<SOCKET>(listen_fd_));
        listen_fd_ = -1;
    }
    if (thread_.joinable()) thread_.join();
    running_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Server loop
// ---------------------------------------------------------------------------

void HealthServer::server_thread() {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        SOCKET client = ::accept(static_cast<SOCKET>(listen_fd_), nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            if (SOCKET_WOULD_BLOCK) {
                // No connection pending — sleep briefly and retry.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            if (!stop_requested_.load(std::memory_order_relaxed)) {
                spdlog::warn("[health_server] accept() error: {}", SOCKET_ERRNO);
            }
            break;
        }

        // Accepted sockets inherit the listen socket's non-blocking flag on
        // Windows. Reset to blocking so recv waits for the full request before
        // we decide which response to send.
#ifdef _WIN32
        {
            u_long blocking = 0;
            ioctlsocket(client, FIONBIO, &blocking);
        }
#endif

        // Read the HTTP request (we only care about the first line).
        char buf[1024] = {};
        int  n = static_cast<int>(::recv(client, buf, sizeof(buf) - 1, 0));
        std::string request(n > 0 ? buf : "");

        // Build the response.
        bool ok = true;
        std::string body = "{\"ok\":true}";
        if (health_cb_) {
            try {
                auto [cb_ok, cb_body] = health_cb_();
                ok   = cb_ok;
                body = cb_body;
            } catch (...) {
                ok   = false;
                body = "{\"ok\":false,\"error\":\"callback threw\"}";
            }
        }

        // Only respond to GET /health; return 404 for other paths.
        bool is_health = (request.find("GET /health") != std::string::npos);
        std::string response;
        if (is_health) {
            response = build_response(ok, body);
            requests_served_.fetch_add(1, std::memory_order_relaxed);
        } else if (request.find("GET / ") != std::string::npos
                   || request.find("GET /\r") != std::string::npos) {
            // 302 redirect to /health only for bare root path.
            response =
                "HTTP/1.0 302 Found\r\n"
                "Location: /health\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
        } else {
            response =
                "HTTP/1.0 404 Not Found\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
        }

        ::send(client, response.c_str(), static_cast<int>(response.size()), 0);
        CLOSE_SOCKET(client);
    }

    running_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// HTTP response builder
// ---------------------------------------------------------------------------

std::string HealthServer::build_response(bool ok, const std::string& body) const {
    std::ostringstream r;
    int status_code = ok ? 200 : 503;
    const char* status_text = ok ? "OK" : "Service Unavailable";
    r << "HTTP/1.0 " << status_code << " " << status_text << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
    return r.str();
}

} // namespace llmquant
