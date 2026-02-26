#include "LLMStreamClient.h"

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
  using ssize_t = int;
#else
  #include <sys/socket.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <arpa/inet.h>
#endif

#include <cstring>
#include <iostream>
#include <sstream>

namespace llmquant {

LLMStreamClient::LLMStreamClient(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

LLMStreamClient::~LLMStreamClient() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

void LLMStreamClient::set_token_callback(TokenCallback cb) { token_cb_ = std::move(cb); }
void LLMStreamClient::set_done_callback(DoneCallback cb)   { done_cb_  = std::move(cb); }

bool LLMStreamClient::connect() {
    if (running_.load()) return false;
    if (!open_socket()) return false;
    running_ = true;
    thread_ = std::thread(&LLMStreamClient::reader_thread, this);
    return true;
}

void LLMStreamClient::stop() {
    running_ = false;
    close_socket();
    if (thread_.joinable()) thread_.join();
}

bool LLMStreamClient::open_socket() {
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
        int fd = static_cast<int>(socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol));
        if (fd < 0) continue;
        if (::connect(fd, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0) {
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

void LLMStreamClient::close_socket() {
    if (sockfd_ >= 0) {
#ifdef _WIN32
        closesocket(sockfd_);
#else
        ::close(sockfd_);
#endif
        sockfd_ = -1;
    }
}

std::string LLMStreamClient::build_request_body() const {
    // Minimal JSON construction — no external JSON library required.
    std::ostringstream oss;
    oss << "{"
        << "\"model\":\"" << config_.model << "\","
        << "\"stream\":true,"
        << "\"max_tokens\":" << config_.max_tokens << ","
        << "\"messages\":["
        <<   "{\"role\":\"system\",\"content\":\"" << config_.system_prompt << "\"},"
        <<   "{\"role\":\"user\",\"content\":\"" << config_.user_prompt << "\"}"
        << "]}";
    return oss.str();
}

std::string LLMStreamClient::build_http_request(const std::string& body) const {
    std::ostringstream oss;
    oss << "POST /v1/chat/completions HTTP/1.1\r\n"
        << "Host: " << config_.host << "\r\n"
        << "Authorization: Bearer " << config_.api_key << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Accept: text/event-stream\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << body;
    return oss.str();
}

std::string LLMStreamClient::parse_sse_delta(const std::string& data_line) {
    // data_line is the raw text after "data: ".
    // Look for "\"content\":\"<token>\"" — minimal parser, no regex.
    const std::string needle = "\"content\":\"";
    size_t pos = data_line.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    std::string token;
    while (pos < data_line.size() && data_line[pos] != '"') {
        if (data_line[pos] == '\\' && pos + 1 < data_line.size()) {
            ++pos;  // advance past backslash
            switch (data_line[pos]) {
                case 'n': token += '\n'; break;
                case 't': token += '\t'; break;
                default:  token += data_line[pos]; break;
            }
        } else {
            token += data_line[pos];
        }
        ++pos;
    }
    return token;
}

void LLMStreamClient::reader_thread() {
    std::string request = build_http_request(build_request_body());

    // Send the full HTTP request.
    size_t sent = 0;
    while (sent < request.size() && running_.load()) {
        ssize_t n = send(sockfd_,
                         request.c_str() + sent,
                         static_cast<int>(request.size() - sent), 0);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }

    // Read response: skip HTTP headers, then process SSE body line by line.
    std::string buf;
    buf.reserve(4096);
    bool headers_done = false;

    char chunk[4096];
    while (running_.load()) {
        ssize_t n = recv(sockfd_, chunk, sizeof(chunk) - 1, 0);
        if (n <= 0) break;
        chunk[n] = '\0';
        buf.append(chunk, static_cast<size_t>(n));

        if (!headers_done) {
            size_t hdr_end = buf.find("\r\n\r\n");
            if (hdr_end == std::string::npos) continue;
            buf = buf.substr(hdr_end + 4);
            headers_done = true;
        }

        // Scan for complete SSE lines terminated by '\n'.
        size_t start = 0;
        while (true) {
            size_t nl = buf.find('\n', start);
            if (nl == std::string::npos) break;

            std::string line = buf.substr(start, nl - start);
            // Strip trailing CR for CRLF line endings.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            start = nl + 1;

            if (line.rfind("data: ", 0) == 0) {
                std::string payload = line.substr(6);
                if (payload == "[DONE]") {
                    running_ = false;
                    break;
                }
                std::string token = parse_sse_delta(payload);
                if (!token.empty() && token_cb_) token_cb_(token);
            }
        }
        buf = buf.substr(start);
    }

    close_socket();
    running_ = false;
    if (done_cb_) done_cb_("");
}

} // namespace llmquant
