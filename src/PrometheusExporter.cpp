#include "PrometheusExporter.h"
#include <cmath>
#include <sstream>

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
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace llmquant {

PrometheusExporter::PrometheusExporter(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        throw std::runtime_error("PrometheusExporter: WSAStartup failed");
    }
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

    // SO_REUSEADDR: On Linux this allows binding while the previous socket is in
    // TIME_WAIT (useful for fast test re-binding); on Windows SO_REUSEADDR has
    // SO_REUSEPORT semantics (allows multiple sockets to bind the same port),
    // which is broader than the POSIX meaning — be aware of this difference.
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));
#ifndef _WIN32
    // On Linux, also set SO_REUSEPORT for proper port reuse semantics (allows
    // multiple processes/sockets to bind the same port with load distribution).
    int reuseport = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT,
               reinterpret_cast<const char*>(&reuseport), sizeof(reuseport));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(config_.port);
    // Use inet_pton (POSIX/Winsock2) instead of deprecated inet_addr.
    if (inet_pton(AF_INET, config_.bind_address.c_str(), &addr.sin_addr) != 1) {
        spdlog::error("PrometheusExporter: invalid bind_address '{}', refusing to default to 0.0.0.0",
                      config_.bind_address);
        close_listen_socket();
        return false;
    }

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_listen_socket();
        return false;
    }
    if (::listen(listen_fd_, SOMAXCONN) < 0) {
#ifdef _WIN32
        spdlog::error("PrometheusExporter: listen() failed: WSA error {}", WSAGetLastError());
#else
        spdlog::error("PrometheusExporter: listen() failed: {}", strerror(errno));
#endif
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
        constexpr size_t kMaxHeaderBytes = 65536; // 64 KB
        char buf[1024];
        std::string headers;
        bool done = false;
        bool header_overflow = false;
        while (!done) {
            ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            headers.append(buf, static_cast<size_t>(n));
            if (headers.size() > kMaxHeaderBytes) {
                // send 431 Request Header Fields Too Large and close
                const char* resp431 = "HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n";
                ::send(client_fd, resp431, static_cast<int>(strlen(resp431)), 0);
                header_overflow = true;
                break; // exit recv loop
            }
            if (headers.find("\r\n\r\n") != std::string::npos ||
                headers.find("\n\n")     != std::string::npos)
                done = true;
        }
        if (header_overflow) {
#ifdef _WIN32
            closesocket(client_fd);
#else
            ::close(client_fd);
#endif
            continue;
        }

        // Build and send the response — re-entrancy guard prevents concurrent scrapes
        // from calling metrics_cb_ simultaneously.
        std::string body;
        bool expected_cb = false;
        if (metrics_cb_ && callback_in_progress_.compare_exchange_strong(expected_cb, true)) {
            try { body = metrics_cb_(); } catch (...) {}
            callback_in_progress_.store(false);
        } else if (metrics_cb_) {
            spdlog::warn("PrometheusExporter: metrics callback re-entered, skipping scrape");
            // Return empty response for this scrape.
        }
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
        if (sent == resp_str.size())
            scrape_count_.fetch_add(1, std::memory_order_relaxed);

#ifdef _WIN32
        closesocket(client_fd);
#else
        ::close(client_fd);
#endif
    }

    running_ = false;
}

// ---------------------------------------------------------------------------
// Static text-format helpers
// ---------------------------------------------------------------------------

std::string PrometheusExporter::format_gauge(const std::string& name, double value,
                                              const std::string& help) {
    std::ostringstream ss;
    if (!help.empty()) {
        ss << "# HELP " << name << " " << help << "\n";
        ss << "# TYPE " << name << " gauge\n";
    }
    // Prometheus text format does not accept NaN or Inf — replace with 0.
    const double safe_value = std::isfinite(value) ? value : 0.0;
    ss << name << " " << safe_value << "\n";
    return ss.str();
}

std::string PrometheusExporter::format_counter(const std::string& name, uint64_t value,
                                                const std::string& help) {
    std::ostringstream ss;
    if (!help.empty()) {
        ss << "# HELP " << name << " " << help << "\n";
        ss << "# TYPE " << name << " counter\n";
    }
    ss << name << " " << value << "\n";
    return ss.str();
}

std::string PrometheusExporter::format_info(const std::string& name,
                                             const std::map<std::string, std::string>& labels,
                                             const std::string& help) {
    std::ostringstream ss;
    if (!help.empty()) {
        ss << "# HELP " << name << " " << help << "\n";
        ss << "# TYPE " << name << " gauge\n";
    }
    ss << name << "{";
    bool first = true;
    for (const auto& [k, v] : labels) {
        if (!first) ss << ",";
        ss << k << "=\"" << v << "\"";
        first = false;
    }
    ss << "} 1\n";
    return ss.str();
}

std::string PrometheusExporter::format_histogram(const std::string& name,
                                                   const std::vector<HistogramBucket>& buckets,
                                                   double sum_us, uint64_t count,
                                                   const std::string& help) {
    std::ostringstream ss;
    if (!help.empty()) {
        ss << "# HELP " << name << " " << help << "\n";
        ss << "# TYPE " << name << " histogram\n";
    }
    for (const auto& b : buckets) {
        if (std::isinf(b.upper_bound_us)) continue;  // +Inf is emitted explicitly below
        ss << name << "_bucket{le=\"" << b.upper_bound_us << "\"} " << b.count << "\n";
    }
    ss << name << "_bucket{le=\"+Inf\"} " << count << "\n";
    const double safe_sum = std::isfinite(sum_us) ? sum_us : 0.0;
    ss << name << "_sum " << safe_sum << "\n";
    ss << name << "_count " << count << "\n";
    return ss.str();
}

} // namespace llmquant
