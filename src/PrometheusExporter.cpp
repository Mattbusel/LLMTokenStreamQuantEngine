#include "PrometheusExporter.h"

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using ssize_t = int;
#else
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
#endif

#include <iostream>
#include <sstream>
#include <cstring>

namespace llmquant {

PrometheusExporter::PrometheusExporter(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

PrometheusExporter::~PrometheusExporter() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

void PrometheusExporter::set_metrics_callback(MetricsCallback cb) {
    metrics_cb_ = std::move(cb);
}

bool PrometheusExporter::bind_socket() {
    listen_fd_ = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (listen_fd_ < 0) return false;

    // SO_REUSEADDR so tests can re-bind quickly on the same port.
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(config_.port);
    addr.sin_addr.s_addr = inet_addr(config_.bind_address.c_str());
    if (addr.sin_addr.s_addr == INADDR_NONE)
        addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_listen_socket();
        return false;
    }
    if (::listen(listen_fd_, 4) != 0) {
        close_listen_socket();
        return false;
    }
    return true;
}

void PrometheusExporter::close_listen_socket() {
    if (listen_fd_ >= 0) {
#ifdef _WIN32
        closesocket(listen_fd_);
#else
        ::close(listen_fd_);
#endif
        listen_fd_ = -1;
    }
}

bool PrometheusExporter::start() {
    if (running_.load()) return false;
    if (!bind_socket()) return false;
    running_ = true;
    thread_ = std::thread(&PrometheusExporter::server_thread, this);
    return true;
}

void PrometheusExporter::stop() {
    running_ = false;
    // Close the listen socket so accept() unblocks.
    close_listen_socket();
    if (thread_.joinable()) thread_.join();
}

void PrometheusExporter::server_thread() {
    while (running_.load() && listen_fd_ >= 0) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int addrlen = sizeof(client_addr);
#else
        socklen_t addrlen = sizeof(client_addr);
#endif
        int client_fd = static_cast<int>(
            ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &addrlen));
        if (client_fd < 0) break;  // listen socket closed by stop()

        // Set a receive timeout so a slow/misbehaving client cannot hang the server.
#ifdef _WIN32
        DWORD timeout_ms = 5000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
        struct timeval tv{5, 0};  // 5 second timeout
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&tv), sizeof(tv));
#endif

        // Read and discard HTTP request headers (look for blank line).
        char buf[1024];
        std::string headers;
        bool done = false;
        while (!done) {
            ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            headers.append(buf, static_cast<size_t>(n));
            if (headers.find("\r\n\r\n") != std::string::npos ||
                headers.find("\n\n")     != std::string::npos)
                done = true;
        }

        // Build and send the response.
        std::string body = metrics_cb_ ? metrics_cb_() : "";
        std::ostringstream resp;
        resp << "HTTP/1.0 200 OK\r\n"
             << "Content-Type: text/plain; version=0.0.4\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "\r\n"
             << body;
        std::string resp_str = resp.str();
        size_t sent = 0;
        while (sent < resp_str.size()) {
            ssize_t n = ::send(client_fd, resp_str.c_str() + sent,
                               static_cast<int>(resp_str.size() - sent), 0);
            if (n <= 0) break;
            sent += static_cast<size_t>(n);
        }

#ifdef _WIN32
        closesocket(client_fd);
#else
        ::close(client_fd);
#endif
    }

    running_ = false;
}

} // namespace llmquant
