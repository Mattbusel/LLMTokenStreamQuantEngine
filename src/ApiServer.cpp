/**
 * @file src/ApiServer.cpp
 * @brief Minimal HTTP/1.1 REST server for the LLMTokenStreamQuantEngine.
 *
 * Uses only POSIX sockets (sys/socket.h, netinet/in.h, unistd.h).
 * No Boost, no libmicrohttpd, no asio — intentionally minimal.
 *
 * The accept loop runs in a background std::thread.  Each accepted
 * connection is handled synchronously (read request, write response, close)
 * before accepting the next one.  This single-connection-at-a-time model is
 * intentional: the API is for monitoring and occasional token injection, not
 * for high-throughput ingestion (use WebSocketServer for that).
 */

#include "ApiServer.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

// POSIX socket headers — gracefully degraded on non-POSIX targets.
#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
   using socket_t = SOCKET;
#  define INVALID_SOCKET_VAL INVALID_SOCKET
#  define CLOSE_SOCKET(s) closesocket(s)
#  define SOCK_NONBLOCK 0
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
   using socket_t = int;
#  define INVALID_SOCKET_VAL (-1)
#  define CLOSE_SOCKET(s) ::close(s)
#endif

namespace llmquant {

// ---------------------------------------------------------------------------
// Platform init (Windows WSA)
// ---------------------------------------------------------------------------

namespace {
#if defined(_WIN32)
struct WsaInit {
    WsaInit() {
        WSADATA wd{};
        WSAStartup(MAKEWORD(2, 2), &wd);
    }
    ~WsaInit() { WSACleanup(); }
};
static WsaInit wsa_init_; // NOLINT(cert-err58-cpp)
#endif
} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

ApiServer::ApiServer(ApiServerConfig config, SignalStore& store)
    : config_(std::move(config))
    , store_(store)
{}

ApiServer::~ApiServer() {
    stop();
}

void ApiServer::set_token_handler(TokenHandler handler) {
    token_handler_ = std::move(handler);
}

// ---------------------------------------------------------------------------
// start / stop
// ---------------------------------------------------------------------------

void ApiServer::start() {
    if (running_.load(std::memory_order_relaxed)) return;

    server_fd_ = create_server_socket();
    if (server_fd_ == INVALID_SOCKET_VAL) {
        throw std::runtime_error("ApiServer: failed to create server socket");
    }

    running_.store(true, std::memory_order_relaxed);
    stop_requested_.store(false, std::memory_order_relaxed);
    accept_thread_ = std::thread(&ApiServer::accept_loop, this);
}

void ApiServer::stop() {
    if (!running_.load(std::memory_order_relaxed)) return;
    stop_requested_.store(true, std::memory_order_relaxed);
    running_.store(false, std::memory_order_relaxed);

    if (server_fd_ != INVALID_SOCKET_VAL) {
        CLOSE_SOCKET(server_fd_);
        server_fd_ = INVALID_SOCKET_VAL;
    }
    if (accept_thread_.joinable()) accept_thread_.join();
}

// ---------------------------------------------------------------------------
// create_server_socket
// ---------------------------------------------------------------------------

int ApiServer::create_server_socket() {
    socket_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == INVALID_SOCKET_VAL) return INVALID_SOCKET_VAL;

    int opt = 1;
#if defined(_WIN32)
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Set receive timeout.
    struct timeval tv{};
    tv.tv_sec  = config_.timeout_secs;
    tv.tv_usec = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(config_.port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        CLOSE_SOCKET(fd);
        return INVALID_SOCKET_VAL;
    }

    if (::listen(fd, config_.backlog) < 0) {
        CLOSE_SOCKET(fd);
        return INVALID_SOCKET_VAL;
    }

    return static_cast<int>(fd);
}

// ---------------------------------------------------------------------------
// accept_loop
// ---------------------------------------------------------------------------

void ApiServer::accept_loop() {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        struct sockaddr_in client_addr{};
        socklen_t addrlen = sizeof(client_addr);

        int client_fd = static_cast<int>(
            ::accept(server_fd_,
                     reinterpret_cast<struct sockaddr*>(&client_addr),
                     &addrlen));

        if (client_fd < 0) {
            // accept() interrupted by stop() closing the socket.
            if (stop_requested_.load(std::memory_order_relaxed)) break;
            continue;
        }

#if !defined(_WIN32)
        // Set timeout on client socket.
        struct timeval tv{};
        tv.tv_sec  = config_.timeout_secs;
        tv.tv_usec = 0;
        ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

        handle_connection(client_fd);
    }
}

// ---------------------------------------------------------------------------
// handle_connection
// ---------------------------------------------------------------------------

void ApiServer::handle_connection(int client_fd) {
    // Read the full request (up to 64 KiB).
    constexpr std::size_t BUF_SIZE = 65536;
    std::string raw;
    raw.resize(BUF_SIZE);
    ssize_t n = ::recv(client_fd, &raw[0], BUF_SIZE - 1, 0);
    if (n <= 0) { CLOSE_SOCKET(client_fd); return; }
    raw.resize(static_cast<std::size_t>(n));

    ParsedRequest req = parse_request(raw);
    std::string response;

    if (!req.valid) {
        response = http_response(400, R"({"error":"bad request"})");
    } else if (req.method == "GET") {
        if (req.path == "/signal/latest") {
            response = handle_get_latest();
        } else if (req.path == "/signal/history") {
            response = handle_get_history(req.query);
        } else if (req.path == "/health") {
            response = handle_get_health();
        } else if (req.path == "/stats") {
            response = handle_get_stats();
        } else {
            response = http_response(404, R"({"error":"not found"})");
        }
    } else if (req.method == "POST" && req.path == "/token") {
        response = handle_post_token(req.body);
    } else {
        response = http_response(405, R"({"error":"method not allowed"})");
    }

    requests_handled_.fetch_add(1, std::memory_order_relaxed);
    ::send(client_fd, response.data(), static_cast<int>(response.size()), 0);
    CLOSE_SOCKET(client_fd);
}

// ---------------------------------------------------------------------------
// parse_request
// ---------------------------------------------------------------------------

/* static */
ApiServer::ParsedRequest ApiServer::parse_request(const std::string& raw) {
    ParsedRequest req;
    // Find the request line end.
    auto line_end = raw.find("\r\n");
    if (line_end == std::string::npos) line_end = raw.find('\n');
    if (line_end == std::string::npos) return req;

    std::string req_line = raw.substr(0, line_end);
    // Parse "METHOD /path?query HTTP/1.x"
    std::size_t sp1 = req_line.find(' ');
    if (sp1 == std::string::npos) return req;
    std::size_t sp2 = req_line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return req;

    req.method = req_line.substr(0, sp1);
    std::string target = req_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Split path and query string.
    auto q = target.find('?');
    if (q != std::string::npos) {
        req.path  = target.substr(0, q);
        req.query = target.substr(q + 1);
    } else {
        req.path = target;
    }

    // Extract body for POST requests.
    const std::string sep = "\r\n\r\n";
    auto body_start = raw.find(sep);
    if (body_start != std::string::npos) {
        req.body = raw.substr(body_start + sep.size());
        // Trim trailing whitespace / null bytes.
        while (!req.body.empty() && (req.body.back() == '\0'
                                  || req.body.back() == '\r'
                                  || req.body.back() == '\n'))
            req.body.pop_back();
    }

    req.valid = true;
    return req;
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

std::string ApiServer::handle_get_latest() {
    auto sig = store_.latest();
    if (sig.timestamp_ns == 0) {
        return http_response(404, R"({"error":"no signals yet"})");
    }
    return http_response(200, stored_signal_to_json(sig));
}

std::string ApiServer::handle_get_history(const std::string& query) {
    uint64_t n = config_.default_history_n;

    // Parse ?n=<value>
    auto npos = query.find("n=");
    if (npos != std::string::npos) {
        std::string val = query.substr(npos + 2);
        auto eq = val.find('&');
        if (eq != std::string::npos) val = val.substr(0, eq);
        uint64_t parsed = 0;
        auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), parsed);
        if (ec == std::errc{}) n = parsed;
    }
    n = std::clamp(n, uint64_t{1}, config_.max_history_n);

    auto signals = store_.last_n(n);
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < signals.size(); ++i) {
        if (i > 0) os << ",";
        os << stored_signal_to_json(signals[i]);
    }
    os << "]";
    return http_response(200, os.str());
}

std::string ApiServer::handle_post_token(const std::string& body) {
    if (body.empty()) {
        return http_response(400, R"({"error":"empty token"})");
    }
    if (!token_handler_) {
        return http_response(503, R"({"error":"token handler not registered"})");
    }

    token_requests_.fetch_add(1, std::memory_order_relaxed);
    TradeSignal sig = token_handler_(body);
    store_.insert(sig);
    return http_response(200, signal_to_json(sig));
}

std::string ApiServer::handle_get_health() {
    return http_response(200, R"({"status":"ok"})");
}

std::string ApiServer::handle_get_stats() {
    return http_response(200, store_.to_stats_json());
}

// ---------------------------------------------------------------------------
// Serialisation helpers
// ---------------------------------------------------------------------------

/* static */
std::string ApiServer::signal_to_json(const TradeSignal& sig) {
    std::ostringstream os;
    os << "{"
       << "\"timestamp_ns\":"        << sig.timestamp_ns          << ","
       << "\"delta_bias\":"          << sig.delta_bias_shift       << ","
       << "\"volatility_adjustment\":"<< sig.volatility_adjustment << ","
       << "\"spread_modifier\":"     << sig.spread_modifier        << ","
       << "\"confidence\":"          << sig.confidence             << ","
       << "\"latency_us\":"          << sig.latency_us             << ","
       << "\"strategy_toggle\":"     << sig.strategy_toggle        << ","
       << "\"strategy_weight\":"     << sig.strategy_weight        << ","
       << "\"signal_quality\":"      << sig.signal_quality
       << "}";
    return os.str();
}

/* static */
std::string ApiServer::stored_signal_to_json(const StoredSignal& sig) {
    std::ostringstream os;
    os << "{"
       << "\"id\":"               << sig.id              << ","
       << "\"timestamp_ns\":"     << sig.timestamp_ns    << ","
       << "\"delta_bias\":"       << sig.delta_bias      << ","
       << "\"volatility_adj\":"   << sig.volatility_adj  << ","
       << "\"spread_modifier\":"  << sig.spread_modifier << ","
       << "\"confidence\":"       << sig.confidence      << ","
       << "\"latency_us\":"       << sig.latency_us      << ","
       << "\"strategy_toggle\":"  << sig.strategy_toggle << ","
       << "\"strategy_weight\":"  << sig.strategy_weight << ","
       << "\"signal_quality\":"   << sig.signal_quality
       << "}";
    return os.str();
}

/* static */
std::string ApiServer::http_response(int status,
                                     const std::string& body,
                                     const std::string& content_type) {
    std::string status_text;
    switch (status) {
        case 200: status_text = "OK";                    break;
        case 400: status_text = "Bad Request";           break;
        case 404: status_text = "Not Found";             break;
        case 405: status_text = "Method Not Allowed";    break;
        case 503: status_text = "Service Unavailable";   break;
        default:  status_text = "Internal Server Error"; break;
    }

    std::ostringstream os;
    os << "HTTP/1.1 " << status << " " << status_text << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "\r\n"
       << body;
    return os.str();
}

} // namespace llmquant
