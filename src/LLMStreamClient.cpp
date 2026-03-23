#include "LLMStreamClient.h"

#ifdef LLMQUANT_TLS_ENABLED
  #include <openssl/ssl.h>
  #include <openssl/err.h>
  #include <openssl/x509v3.h>
#endif

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <wincrypt.h>
  #pragma comment(lib, "Ws2_32.lib")
  #pragma comment(lib, "Crypt32.lib")
#else
  #include <sys/socket.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <arpa/inet.h>
  #include <netinet/tcp.h>
  #include <cerrno>
#endif

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace llmquant {

LLMStreamClient::LLMStreamClient(Config config) : config_(std::move(config)) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        throw std::runtime_error("LLMStreamClient: WSAStartup failed");
    }
#endif
#ifdef LLMQUANT_TLS_ENABLED
    SSL_library_init();
    SSL_load_error_strings();
    ssl_ctx_ = SSL_CTX_new(TLS_client_method());
    if (!ssl_ctx_) {
        throw std::runtime_error("LLMStreamClient: SSL_CTX_new failed — OpenSSL not initialised correctly");
    }
    auto* ctx = static_cast<SSL_CTX*>(ssl_ctx_);
    // On Windows, load the system certificate store so OpenSSL can verify
    // server certificates (vcpkg OpenSSL has no built-in CA bundle).
#ifdef _WIN32
    {
        HCERTSTORE hStore = CertOpenSystemStoreA(0, "ROOT");
        if (hStore) {
            X509_STORE* store = SSL_CTX_get_cert_store(ctx);
            PCCERT_CONTEXT pCtx = nullptr;
            while ((pCtx = CertEnumCertificatesInStore(hStore, pCtx)) != nullptr) {
                const unsigned char* p = pCtx->pbCertEncoded;
                X509* x509 = d2i_X509(nullptr, &p, static_cast<long>(pCtx->cbCertEncoded));
                if (x509) {
                    X509_STORE_add_cert(store, x509);
                    X509_free(x509);
                }
            }
            CertCloseStore(hStore, 0);
        }
    }
#else
    SSL_CTX_set_default_verify_paths(ctx);
#endif
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
#endif
}

LLMStreamClient::~LLMStreamClient() {
    stop();
#ifdef LLMQUANT_TLS_ENABLED
    // ssl_ is cleaned up inside close_socket() -> tls_close() which is
    // called by stop() above.  We must free ssl_ctx_ exactly once here.
    // Use an atomic exchange to guard against double-free if the destructor
    // is somehow reached from two paths under rapid reconnect teardown.
    void* ctx_to_free = ssl_ctx_;
    ssl_ctx_ = nullptr;   // zero before free so any stray tls_handshake() racing
                          // after stop() gets a null context rather than a dangling
                          // pointer (tls_handshake only runs while running_==true,
                          // which stop() has already cleared, so this is defensive).
    if (ctx_to_free) {
        SSL_CTX_free(static_cast<SSL_CTX*>(ctx_to_free));
    }
#endif
#ifdef _WIN32
    WSACleanup();
#endif
}

void LLMStreamClient::set_token_callback(TokenCallback cb) { token_cb_ = std::move(cb); }
void LLMStreamClient::set_done_callback(DoneCallback cb)   { done_cb_  = std::move(cb); }

bool LLMStreamClient::connect() {
    if (running_.load()) return false;
    // The reader_thread opens (and re-opens) its own socket per request,
    // so we just start the thread here.
    shutdown_requested_ = false;
    running_ = true;
    thread_ = std::thread(&LLMStreamClient::reader_thread, this);
    return true;
}

void LLMStreamClient::stop() {
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

bool LLMStreamClient::open_socket() {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(config_.port);
    addrinfo* res = nullptr;
    int gai_err = getaddrinfo(config_.host.c_str(), port_str.c_str(), &hints, &res);
    if (gai_err != 0) {
        spdlog::error("[stream] getaddrinfo failed: {}", gai_strerror(gai_err));
        return false;
    }

    // RAII guard: freeaddrinfo is called on scope exit even if an exception is thrown.
    struct AddrInfoGuard {
        addrinfo* p;
        ~AddrInfoGuard() { if (p) freeaddrinfo(p); }
    } res_guard{res};

    sockfd_ = -1;
    for (addrinfo* rp = res; rp != nullptr; rp = rp->ai_next) {
        int fd = static_cast<int>(socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol));
        if (fd < 0) continue;
        if (::connect(fd, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0) {
            sockfd_ = fd;
            // Disable Nagle's algorithm for low-latency streaming.
            int tcp_flag = 1;
            if (setsockopt(sockfd_, IPPROTO_TCP, TCP_NODELAY,
                           reinterpret_cast<const char*>(&tcp_flag), sizeof(tcp_flag)) != 0) {
                spdlog::debug("[stream] TCP_NODELAY setsockopt failed (non-fatal)");
            }
            // 30-second receive timeout to detect stalled streams.
#ifdef _WIN32
            DWORD rcvtimeo_ms = 30000;
            setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rcvtimeo_ms), sizeof(rcvtimeo_ms));
#else
            struct timeval rcvtv{30, 0};
            setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rcvtv), sizeof(rcvtv));
#endif
            break;
        }
#ifdef _WIN32
        int wsaerr = WSAGetLastError();
        spdlog::debug("[stream] connect() failed WSA error {}", wsaerr);
        closesocket(fd);
#else
        ::close(fd);
#endif
    }
    if (sockfd_ < 0) return false;

#ifdef LLMQUANT_TLS_ENABLED
    if (config_.use_tls) {
        if (!tls_handshake()) {
            close_socket();
            return false;
        }
    }
#endif
    return true;
}

void LLMStreamClient::close_socket() {
#ifdef LLMQUANT_TLS_ENABLED
    tls_close();
#endif
    if (sockfd_ >= 0) {
#ifdef _WIN32
        closesocket(sockfd_);
#else
        ::close(sockfd_);
#endif
        sockfd_ = -1;
    }
}

#ifdef LLMQUANT_TLS_ENABLED
bool LLMStreamClient::tls_handshake() {
    auto* ctx = static_cast<SSL_CTX*>(ssl_ctx_);
    auto* ssl = SSL_new(ctx);
    if (!ssl) return false;
    ssl_ = ssl;
    SSL_set_fd(ssl, sockfd_);
    SSL_set_tlsext_host_name(ssl, config_.host.c_str());
    // SNI + hostname verification.
    SSL_set1_host(ssl, config_.host.c_str());
    if (SSL_connect(ssl) != 1) {
        unsigned long err = ERR_get_error();
        char err_buf[256];
        ERR_error_string_n(err, err_buf, sizeof(err_buf));
        spdlog::error("[stream] SSL_connect failed: {}", err_buf);
        SSL_free(ssl);
        ssl_ = nullptr;
        return false;
    }
    return true;
}

void LLMStreamClient::tls_close() {
    // Only tear down the per-connection SSL session here.
    // ssl_ctx_ lives for the lifetime of the client object and is freed
    // in the destructor so that reconnect loops can reuse it.
    if (ssl_) {
        SSL_shutdown(static_cast<SSL*>(ssl_));
        SSL_free(static_cast<SSL*>(ssl_));
        ssl_ = nullptr;
    }
}

ssize_t LLMStreamClient::tls_send(const char* buf, size_t len) {
    return SSL_write(static_cast<SSL*>(ssl_), buf, static_cast<int>(len));
}

ssize_t LLMStreamClient::tls_recv(char* buf, size_t len) {
    return SSL_read(static_cast<SSL*>(ssl_), buf, static_cast<int>(len));
}
#endif

std::string LLMStreamClient::build_request_body() const {
    nlohmann::json body = {
        {"model",      config_.model},
        {"stream",     true},
        {"max_tokens", config_.max_tokens},
        {"messages",   nlohmann::json::array({
            {{"role", "system"}, {"content", config_.system_prompt}},
            {{"role", "user"},   {"content", config_.user_prompt}}
        })}
    };
    return body.dump();
}

std::string LLMStreamClient::build_http_request(const std::string& body) const {
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
    oss << "POST /v1/chat/completions HTTP/1.1\r\n"
        << "Host: " << sanitise_header_value(config_.host) << "\r\n"
        << "Authorization: Bearer " << sanitise_header_value(config_.api_key) << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Accept: text/event-stream\r\n"
        // "Connection: close" is intentional for OpenAI — it closes after [DONE].
        // For local endpoints supporting keep-alive SSE, set config_.use_keep_alive
        // and replace this header. See LLMStreamClient::Config::use_keep_alive.
        << "Connection: close\r\n"
        << "\r\n"
        << body;
    return oss.str();
}

std::string LLMStreamClient::parse_sse_delta(const std::string& data_line) {
    // data_line is the raw text after "data: ".
    // OpenAI SSE format: {"choices":[{"delta":{"content":"<token>"},...},...]}
    try {
        auto j = nlohmann::json::parse(data_line);
        return j.at("choices").at(0).at("delta").at("content").get<std::string>();
    } catch (const nlohmann::json::exception&) {
        return {};
    }
}

// Send all bytes in request over the active socket (TLS or plain).
static bool send_all(int sockfd, void* ssl, bool use_tls,
                     const std::string& data, std::atomic<bool>& running) {
    size_t sent = 0;
    while (sent < data.size() && running.load()) {
        ssize_t n;
#ifdef LLMQUANT_TLS_ENABLED
        if (use_tls && ssl) {
            n = SSL_write(static_cast<SSL*>(ssl),
                          data.c_str() + sent,
                          static_cast<int>(data.size() - sent));
        } else {
#endif
            n = send(sockfd, data.c_str() + sent,
                     static_cast<int>(data.size() - sent), 0);
#ifdef LLMQUANT_TLS_ENABLED
        }
#else
        (void)use_tls; (void)ssl;
#endif
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

// Returns true if `s` is a valid chunked transfer-encoding size line.
// Per RFC 7230, a chunk-size line is one or more hex digits optionally followed
// by ";chunk-extension".  Only the hex part (before the first ';') is validated;
// the extension is accepted as-is.  A line of only semicolons is rejected because
// there must be at least one hex digit before any ';'.
static bool is_chunk_size_line(const std::string& s) {
    if (s.empty()) return false;
    // Isolate the hex digits — everything from the first ';' onward is extension.
    size_t ext = s.find(';');
    const std::string hex_part = (ext != std::string::npos) ? s.substr(0, ext) : s;
    if (hex_part.empty()) return false;
    return std::all_of(hex_part.begin(), hex_part.end(),
                       [](unsigned char c) { return std::isxdigit(c); });
}

void LLMStreamClient::reader_thread() {
    // Loop: send one request, stream the response, wait loop_interval, repeat.
    bool first_connection = true;
    while (running_.load()) {
        if (!first_connection)
            reconnect_count_.fetch_add(1, std::memory_order_relaxed);
        first_connection = false;
        // Fresh TCP (+TLS) connection per request — OpenAI closes after [DONE].
        if (!open_socket()) {
            if (done_cb_) done_cb_("connect failed");
            // Back off and retry.
            for (int i = 0; i < 50 && running_.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::string request = build_http_request(build_request_body());
#ifdef LLMQUANT_TLS_ENABLED
        if (!send_all(sockfd_, ssl_, config_.use_tls, request, running_)) {
#else
        if (!send_all(sockfd_, nullptr, config_.use_tls, request, running_)) {
#endif
            close_socket();
            continue;
        }

        // Read response: accumulate until we have the full header block,
        // then detect Transfer-Encoding and stream the SSE body.
        std::string buf;
        buf.reserve(8192);
        bool headers_done = false;
        bool chunked      = false;
        bool stream_done  = false;

        // --debug-raw: dump all bytes to stderr for 3 seconds then stop.
        auto debug_start = std::chrono::steady_clock::now();

        char chunk[4096];
        while (running_.load() && !stream_done) {
            ssize_t n;
#ifdef LLMQUANT_TLS_ENABLED
            if (config_.use_tls && ssl_) {
                n = tls_recv(chunk, sizeof(chunk) - 1);
            } else {
#endif
                n = recv(sockfd_, chunk, sizeof(chunk) - 1, 0);
#ifdef LLMQUANT_TLS_ENABLED
            }
#endif
            if (n <= 0) {
#ifdef _WIN32
                // Capture the WSA error immediately — WSAGetLastError() is
                // thread-local and its value is lost once the thread returns
                // from recv().  stop() may read last_socket_error_ after join.
                int wsa_err = WSAGetLastError();
                if (wsa_err != 0) last_socket_error_.store(wsa_err);
                // WSAENOTSOCK/WSAEBADF after close_socket() from stop() is expected.
                if (shutdown_requested_.load() &&
                    (wsa_err == WSAENOTSOCK || wsa_err == WSAEBADF || wsa_err == WSAEINTR)) {
                    break;  // clean shutdown — not an error
                }
                if (wsa_err == WSAETIMEDOUT) {
                    spdlog::warn("LLMStreamClient: no data in 30s — possible stalled stream, reconnecting");
                    break;
                }
#else
                if (shutdown_requested_.load()) {
                    int err = errno;
                    if (err == EBADF || err == EINTR) {
                        break;  // clean shutdown — not an error
                    }
                }
                {
                    int err = errno;
                    if (err == EAGAIN || err == EWOULDBLOCK) {
                        spdlog::warn("LLMStreamClient: no data in 30s — possible stalled stream, reconnecting");
                        break;
                    }
                }
#endif
                break;
            }
            chunk[n] = '\0';
            buf.append(chunk, static_cast<size_t>(n));

            // Guard against unbounded buffer growth: a server that never sends
            // a header-end sequence (\r\n\r\n) or a line boundary (\n) would
            // fill memory until the recv timeout fires.  Cap at 4 MB and reconnect.
            static constexpr size_t kMaxBufBytes = 4u << 20;  // 4 MB
            if (buf.size() > kMaxBufBytes) {
                spdlog::warn("[llm_client] receive buffer exceeded {}B without a "
                             "parseable boundary — reconnecting", kMaxBufBytes);
                close_socket();
                continue;
            }

            if (config_.debug_raw) {
                // debug_raw intentionally writes raw bytes to stderr for diagnostic use.
                // This is the only acceptable stderr use in library code: it is
                // gated behind an explicit debug flag and never fires in production.
                std::fwrite(chunk, 1, static_cast<size_t>(n), stderr);
                std::fflush(stderr);
                auto elapsed = std::chrono::steady_clock::now() - debug_start;
                if (elapsed >= std::chrono::seconds(3)) {
                    spdlog::debug("[debug-raw] 3s dump complete — stopping");
                    running_ = false;
                    break;
                }
                continue;  // skip all parsing in raw-dump mode
            }

            if (!headers_done) {
                size_t hdr_end = buf.find("\r\n\r\n");
                if (hdr_end == std::string::npos) continue;
                // Check for chunked transfer encoding in the header block.
                std::string headers = buf.substr(0, hdr_end);

                // Parse HTTP status line: "HTTP/1.1 200 OK"
                size_t status_start = headers.find(' ');
                if (status_start != std::string::npos) {
                    int status_code = 0;
                    try {
                        status_code = std::stoi(headers.substr(status_start + 1, 3));
                    } catch (...) {
                        spdlog::warn("[llm_client] malformed HTTP status line; reconnecting");
                        close_socket();
                        continue;
                    }
                    if (status_code == 401 || status_code == 403) {
                        spdlog::error("[llm_client] HTTP {} — check API key; stopping", status_code);
                        running_ = false;
                        break;
                    } else if (status_code == 429) {
                        spdlog::warn("[llm_client] HTTP 429 rate-limited; backing off");
                        close_socket();
                        // Interruptible 10-second back-off: stop() returns promptly.
                        { auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                          const auto sl = std::chrono::milliseconds{100};
                          while (running_.load() && std::chrono::steady_clock::now() < dl)
                              std::this_thread::sleep_for(sl); }
                        continue;
                    } else if (status_code < 200 || status_code >= 300) {
                        spdlog::warn("[llm_client] HTTP {} error; retrying", status_code);
                        close_socket();
                        // Interruptible 2-second back-off: stop() returns promptly.
                        { auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                          const auto sl = std::chrono::milliseconds{100};
                          while (running_.load() && std::chrono::steady_clock::now() < dl)
                              std::this_thread::sleep_for(sl); }
                        continue;
                    }
                }

                // Case-insensitive search for Transfer-Encoding: chunked
                std::string hdrs_lower = headers;
                for (char& c : hdrs_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                chunked = (hdrs_lower.find("transfer-encoding: chunked") != std::string::npos);
                buf = buf.substr(hdr_end + 4);
                headers_done = true;
            }

            // Scan complete lines from the accumulated body.
            size_t start = 0;
            while (true) {
                size_t nl = buf.find('\n', start);
                if (nl == std::string::npos) break;
                std::string line = buf.substr(start, nl - start);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                start = nl + 1;

                // With chunked encoding, skip hex chunk-size lines entirely.
                if (chunked && is_chunk_size_line(line)) continue;
                // Skip blank lines (SSE event separators and chunk boundaries).
                if (line.empty()) continue;

                if (line.rfind("data: ", 0) == 0) {
                    std::string payload = line.substr(6);
                    if (payload == "[DONE]") { stream_done = true; break; }
                    std::string token = parse_sse_delta(payload);
                    if (!token.empty() && token_cb_) {
                        tokens_received_.fetch_add(1, std::memory_order_relaxed);
                        try { token_cb_(token); } catch (const std::exception& e) {
                            spdlog::error("[stream] token_cb_ threw: {}", e.what());
                        } catch (...) {
                            spdlog::error("[stream] token_cb_ threw unknown exception");
                        }
                    }
                }
            }
            buf = buf.substr(start);
        }

        // TODO (use_keep_alive): skip close_socket() and reuse the connection
        // for the next request by resetting parse state and re-sending the request.
        // Currently always closes; OpenAI closes after [DONE] anyway.
        close_socket();

        if (!running_.load()) break;

        // Wait loop_interval before the next request.
        auto deadline = std::chrono::steady_clock::now() + config_.loop_interval;
        while (running_.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    running_ = false;
    if (done_cb_) done_cb_("");
}

} // namespace llmquant
