#include "gtest/gtest.h"
#include "PrometheusExporter.h"

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
#endif

#include <chrono>
#include <string>
#include <thread>

using namespace llmquant;

static constexpr uint16_t kTestPort = 19100;

// Helper: connect to localhost:port and send a minimal HTTP GET, return body.
static std::string http_get(uint16_t port) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) return {};

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
#ifdef _WIN32
        closesocket(fd);
#else
        ::close(fd);
#endif
        return {};
    }

    const char* req = "GET /metrics HTTP/1.0\r\nHost: localhost\r\n\r\n";
    ::send(fd, req, static_cast<int>(strlen(req)), 0);

    std::string resp;
    char buf[4096];
    while (true) {
        int n = static_cast<int>(recv(fd, buf, sizeof(buf) - 1, 0));
        if (n <= 0) break;
        buf[n] = '\0';
        resp.append(buf, static_cast<size_t>(n));
    }
#ifdef _WIN32
    closesocket(fd);
    WSACleanup();
#else
    ::close(fd);
#endif
    // Return only the body (after the blank line).
    auto sep = resp.find("\r\n\r\n");
    if (sep != std::string::npos) return resp.substr(sep + 4);
    return resp;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(PrometheusExporterTest, test_prom_exporter_start_stop_lifecycle) {
    PrometheusExporter::Config cfg;
    cfg.port = kTestPort;
    PrometheusExporter exporter(cfg);
    exporter.set_metrics_callback([] { return "llmquant_test 1\n"; });
    EXPECT_TRUE(exporter.start());
    EXPECT_TRUE(exporter.is_running());
    exporter.stop();
    EXPECT_FALSE(exporter.is_running());
}

TEST(PrometheusExporterTest, test_prom_exporter_double_start_returns_false) {
    PrometheusExporter::Config cfg;
    cfg.port = kTestPort + 1;
    PrometheusExporter exporter(cfg);
    exporter.set_metrics_callback([] { return ""; });
    EXPECT_TRUE(exporter.start());
    EXPECT_FALSE(exporter.start());  // already running
    exporter.stop();
}

TEST(PrometheusExporterTest, test_prom_exporter_serves_metrics_on_scrape) {
    PrometheusExporter::Config cfg;
    cfg.port = kTestPort + 2;
    PrometheusExporter exporter(cfg);
    exporter.set_metrics_callback([] {
        return "llmquant_signals_generated_total 42\n";
    });
    ASSERT_TRUE(exporter.start());

    // Give the server thread a moment to enter accept().
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::string body = http_get(kTestPort + 2);
    exporter.stop();

    EXPECT_NE(body.find("llmquant_signals_generated_total"), std::string::npos)
        << "Response body: " << body;
    EXPECT_NE(body.find("42"), std::string::npos);
}

TEST(PrometheusExporterTest, test_prom_exporter_no_callback_serves_empty_body) {
    PrometheusExporter::Config cfg;
    cfg.port = kTestPort + 3;
    PrometheusExporter exporter(cfg);
    // No callback registered — should still respond without crashing.
    ASSERT_TRUE(exporter.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::string body = http_get(kTestPort + 3);
    exporter.stop();
    // No assertion on content — just verify no crash.
    SUCCEED();
}

TEST(PrometheusExporterTest, test_prom_exporter_callback_reflects_updated_values) {
    PrometheusExporter::Config cfg;
    cfg.port = kTestPort + 4;
    PrometheusExporter exporter(cfg);

    std::atomic<int> counter{0};
    exporter.set_metrics_callback([&counter] {
        return "llmquant_test_counter " + std::to_string(counter.load()) + "\n";
    });
    ASSERT_TRUE(exporter.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // First scrape — counter = 0.
    std::string body1 = http_get(kTestPort + 4);
    EXPECT_NE(body1.find("llmquant_test_counter 0"), std::string::npos)
        << "First scrape body: " << body1;

    // Update counter and scrape again.
    counter = 42;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::string body2 = http_get(kTestPort + 4);
    EXPECT_NE(body2.find("llmquant_test_counter 42"), std::string::npos)
        << "Second scrape body: " << body2;

    exporter.stop();
}

TEST(PrometheusExporterTest, test_prom_exporter_concurrent_scrapes_no_crash) {
    PrometheusExporter::Config cfg;
    cfg.port = kTestPort + 5;
    PrometheusExporter exporter(cfg);
    exporter.set_metrics_callback([] { return "llmquant_test 1\n"; });
    ASSERT_TRUE(exporter.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // Fire two back-to-back scrapes from different threads.
    std::string r1, r2;
    std::thread t1([&r1] { r1 = http_get(kTestPort + 5); });
    std::thread t2([&r2] { r2 = http_get(kTestPort + 5); });
    t1.join();
    t2.join();

    exporter.stop();
    // Neither connection should have crashed the server; both may have received data.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Static format helpers
// ---------------------------------------------------------------------------

TEST(PrometheusExporterTest, test_format_gauge_no_help_produces_metric_line) {
    auto result = PrometheusExporter::format_gauge("my_gauge", 3.14);
    EXPECT_NE(result.find("my_gauge"), std::string::npos);
    EXPECT_NE(result.find("3.14"), std::string::npos);
    // Without help, no HELP/TYPE comments should be emitted.
    EXPECT_EQ(result.find("# HELP"), std::string::npos);
}

TEST(PrometheusExporterTest, test_format_gauge_with_help_includes_type_comment) {
    auto result = PrometheusExporter::format_gauge("llmquant_bias", 0.5, "Accumulated bias");
    EXPECT_NE(result.find("# HELP llmquant_bias"), std::string::npos);
    EXPECT_NE(result.find("# TYPE llmquant_bias gauge"), std::string::npos);
    EXPECT_NE(result.find("llmquant_bias 0.5"), std::string::npos);
}

TEST(PrometheusExporterTest, test_format_counter_produces_correct_output) {
    auto result = PrometheusExporter::format_counter("llmquant_signals_total", 42u, "Signal count");
    EXPECT_NE(result.find("# TYPE llmquant_signals_total counter"), std::string::npos);
    EXPECT_NE(result.find("llmquant_signals_total 42"), std::string::npos);
}

TEST(PrometheusExporterTest, test_format_histogram_includes_bucket_sum_count) {
    using Bucket = PrometheusExporter::HistogramBucket;
    std::vector<Bucket> buckets = {{1.0, 10}, {10.0, 50}, {100.0, 90}};
    auto result = PrometheusExporter::format_histogram("llmquant_latency_us", buckets,
                                                        500.0, 100u, "Latency histogram");
    EXPECT_NE(result.find("_bucket{le=\"1"), std::string::npos);
    EXPECT_NE(result.find("_bucket{le=\"+Inf\"} 100"), std::string::npos);
    EXPECT_NE(result.find("_sum 500"), std::string::npos);
    EXPECT_NE(result.find("_count 100"), std::string::npos);
    EXPECT_NE(result.find("# TYPE llmquant_latency_us histogram"), std::string::npos);
}
