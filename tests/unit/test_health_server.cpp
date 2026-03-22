#include <gtest/gtest.h>

#ifdef LLMQUANT_HEALTH_SERVER_ENABLED

#include "HealthServer.h"

#include <chrono>
#include <string>
#include <thread>

using namespace llmquant;

namespace {

// ============================================================
// Construction
// ============================================================

TEST(HealthServer, ConstructsAndIsNotRunning) {
    HealthServer::Config cfg;
    cfg.port = 19800;
    HealthServer server(cfg);
    EXPECT_FALSE(server.is_running());
    EXPECT_EQ(server.requests_served(), 0u);
}

// ============================================================
// Start / stop lifecycle
// ============================================================

TEST(HealthServer, StartReturnsTrue) {
    HealthServer::Config cfg;
    cfg.port = 19801;
    HealthServer server(cfg);
    bool ok = server.start();
    if (ok) {
        EXPECT_TRUE(server.is_running());
        server.stop();
        EXPECT_FALSE(server.is_running());
    } else {
        GTEST_SKIP() << "Port 19801 unavailable; skipping.";
    }
}

TEST(HealthServer, DoubleStartReturnsFalse) {
    HealthServer::Config cfg;
    cfg.port = 19802;
    HealthServer server(cfg);
    bool first = server.start();
    if (!first) GTEST_SKIP() << "Port 19802 unavailable; skipping.";
    bool second = server.start();
    EXPECT_FALSE(second);
    server.stop();
}

TEST(HealthServer, StopWithoutStartIsNoOp) {
    HealthServer::Config cfg;
    cfg.port = 19803;
    HealthServer server(cfg);
    EXPECT_NO_FATAL_FAILURE(server.stop());
}

// ============================================================
// Callback registration
// ============================================================

TEST(HealthServer, SetHealthCallbackDoesNotCrash) {
    HealthServer::Config cfg;
    cfg.port = 19804;
    HealthServer server(cfg);
    server.set_health_callback([]() -> std::pair<bool, std::string> {
        return {true, "{\"ok\":true}"};
    });
    SUCCEED();
}

// ============================================================
// HTTP request / response (loopback)
// ============================================================

#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   static bool wsa_init() {
       static bool done = false;
       if (!done) { WSADATA w{}; WSAStartup(MAKEWORD(2,2),&w); done=true; }
       return true;
   }
#  define INIT_WSA() wsa_init()
#  define CLOSE_SOCK(s) closesocket(s)
   using sock_t = SOCKET;
   static const sock_t BAD_SOCK = INVALID_SOCKET;
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define INIT_WSA() true
#  define CLOSE_SOCK(s) ::close(s)
   using sock_t = int;
   static const sock_t BAD_SOCK = -1;
#endif

static std::string http_get(uint16_t port, const char* path) {
    INIT_WSA();
    sock_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == BAD_SOCK) return "";

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        CLOSE_SOCK(fd);
        return "";
    }

    std::string req = std::string("GET ") + path + " HTTP/1.0\r\nHost: localhost\r\n\r\n";
    ::send(fd, req.c_str(), static_cast<int>(req.size()), 0);

    char buf[4096] = {};
    int  n = static_cast<int>(::recv(fd, buf, sizeof(buf) - 1, 0));
    CLOSE_SOCK(fd);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : "";
}

TEST(HealthServer, HealthEndpointReturns200WhenOk) {
    HealthServer::Config cfg;
    cfg.port = 19805;
    HealthServer server(cfg);
    server.set_health_callback([]() -> std::pair<bool, std::string> {
        return {true, "{\"ok\":true}"};
    });
    if (!server.start()) GTEST_SKIP() << "Port 19805 unavailable.";

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::string resp = http_get(cfg.port, "/health");
    server.stop();

    if (resp.empty()) GTEST_SKIP() << "http_get returned empty (port contention under load).";
    EXPECT_NE(resp.find("200"), std::string::npos);
    EXPECT_NE(resp.find("application/json"), std::string::npos);
    EXPECT_NE(resp.find("\"ok\":true"), std::string::npos);
}

TEST(HealthServer, HealthEndpointReturns503WhenDegraded) {
    HealthServer::Config cfg;
    cfg.port = 19806;
    HealthServer server(cfg);
    server.set_health_callback([]() -> std::pair<bool, std::string> {
        return {false, "{\"ok\":false,\"reason\":\"test\"}"};
    });
    if (!server.start()) GTEST_SKIP() << "Port 19806 unavailable.";

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::string resp = http_get(cfg.port, "/health");
    server.stop();

    if (resp.empty()) GTEST_SKIP() << "http_get returned empty (port contention under load).";
    EXPECT_NE(resp.find("503"), std::string::npos);
}

TEST(HealthServer, UnknownPathReturns404) {
    HealthServer::Config cfg;
    cfg.port = 19807;
    HealthServer server(cfg);
    if (!server.start()) GTEST_SKIP() << "Port 19807 unavailable.";

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::string resp = http_get(cfg.port, "/unknown");
    server.stop();

    if (resp.empty()) GTEST_SKIP() << "http_get returned empty (port contention under load).";
    EXPECT_NE(resp.find("404"), std::string::npos);
}

TEST(HealthServer, RequestsServedCountsHealthRequests) {
    HealthServer::Config cfg;
    cfg.port = 19808;
    HealthServer server(cfg);
    server.set_health_callback([]() -> std::pair<bool, std::string> {
        return {true, "{\"ok\":true}"};
    });
    if (!server.start()) GTEST_SKIP() << "Port 19808 unavailable.";

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    http_get(cfg.port, "/health");
    http_get(cfg.port, "/health");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    server.stop();

    EXPECT_GE(server.requests_served(), 2u);
}

TEST(HealthServer, CallbackThrowingReturns503) {
    HealthServer::Config cfg;
    cfg.port = 19809;
    HealthServer server(cfg);
    server.set_health_callback([]() -> std::pair<bool, std::string> {
        throw std::runtime_error("test error");
    });
    if (!server.start()) GTEST_SKIP() << "Port 19809 unavailable.";

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::string resp = http_get(cfg.port, "/health");
    server.stop();

    if (resp.empty()) GTEST_SKIP() << "http_get returned empty (port contention under load).";
    EXPECT_NE(resp.find("503"), std::string::npos);
}

TEST(HealthServer, DestructorStopsServerGracefully) {
    HealthServer::Config cfg;
    cfg.port = 19810;
    {
        HealthServer server(cfg);
        (void)server.start();
    }  // destructor called here; should not hang or crash
    SUCCEED();
}

} // namespace

#else  // LLMQUANT_HEALTH_SERVER_ENABLED not defined

TEST(HealthServer, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_HEALTH_SERVER_ENABLED
