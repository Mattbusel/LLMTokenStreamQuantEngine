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
#ifdef LLMQUANT_TLS_ENABLED
    SSL_library_init();
    SSL_load_error_strings();
    ssl_ctx_ = SSL_CTX_new(TLS_client_method());
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
    // ssl_ is cleaned up inside close_socket() -> tls_close().
    // If ssl_ctx_ survived (e.g. never connected), free it here.
    if (ssl_ctx_) {
        SSL_CTX_free(static_cast<SSL_CTX*>(ssl_ctx_));
        ssl_ctx_ = nullptr;
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
    int gai_err = getaddrinfo(config_.host.c_str(), port_str.c_str(), &hints, &res);
    if (gai_err != 0) {
        std::cerr << "[stream] getaddrinfo failed: " << gai_strerror(gai_err) << "\n";
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
        int wsaerr = WSAGetLastError();
        std::cerr << "[stream] connect() failed WSA error " << wsaerr << "\n";
        closesocket(fd);
#else
        ::close(fd);
#endif
    }
    freeaddrinfo(res);
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
        std::cerr << "[stream] SSL_connect failed: " << err_buf << "\n";
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
    // OpenAI SSE format: {"choices":[{"delta":{"content":"<token>"},...},...]}
    // Scope the search to after "delta": to avoid matching top-level "content".
    const std::string delta_key = "\"delta\":";
    size_t delta_pos = data_line.find(delta_key);
    const std::string search_region = (delta_pos != std::string::npos)
                                          ? data_line.substr(delta_pos)
                                          : data_line;

    const std::string needle = "\"content\":\"";
    size_t pos = search_region.find(needle);
    if (pos == std::string::npos) return {};
    // Remap pos back to data_line base (pos is relative to search_region).
    pos += (delta_pos != std::string::npos ? delta_pos : 0);
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

// Returns true if `s` is a valid hex chunk-size token (possibly with extensions).
static bool is_chunk_size_line(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c)) && c != ';') return false;
    }
    return true;
}

void LLMStreamClient::reader_thread() {
    // Loop: send one request, stream the response, wait loop_interval, repeat.
    while (running_.load()) {
        // Fresh TCP (+TLS) connection per request — OpenAI closes after [DONE].
        if (!open_socket()) {
            if (done_cb_) done_cb_("connect failed");
            // Back off and retry.
            for (int i = 0; i < 50 && running_.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::string request = build_http_request(build_request_body());
        if (!send_all(sockfd_, ssl_, config_.use_tls, request, running_)) {
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
            if (n <= 0) break;
            chunk[n] = '\0';
            buf.append(chunk, static_cast<size_t>(n));

            if (config_.debug_raw) {
                std::cerr.write(chunk, n);
                std::cerr.flush();
                auto elapsed = std::chrono::steady_clock::now() - debug_start;
                if (elapsed >= std::chrono::seconds(3)) {
                    std::cerr << "\n[debug-raw] 3s dump complete — exiting\n";
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
                    if (!token.empty() && token_cb_) token_cb_(token);
                }
            }
            buf = buf.substr(start);
        }

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
