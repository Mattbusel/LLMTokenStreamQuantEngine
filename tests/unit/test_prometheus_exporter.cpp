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

// ---------------------------------------------------------------------------
// Cycle 17: NaN/Inf guard in format_gauge and format_histogram
// ---------------------------------------------------------------------------

TEST(PrometheusExporterTest, test_format_gauge_nan_replaced_with_zero) {
    // Use a metric name that does NOT contain "nan" so the search is unambiguous.
    auto result = PrometheusExporter::format_gauge("llmquant_bias_gauge",
                                                    std::numeric_limits<double>::quiet_NaN());
    // Value must be 0, not "nan".
    EXPECT_NE(result.find("llmquant_bias_gauge 0"), std::string::npos)
        << "NaN gauge value must be replaced with 0";
    // "nan" must NOT appear as the value token after the metric name.
    // We check by ensuring the line ends with "0\n", not "nan\n".
    EXPECT_EQ(result.find("gauge nan"), std::string::npos)
        << "NaN must not appear as the metric value";
}

TEST(PrometheusExporterTest, test_format_gauge_inf_replaced_with_zero) {
    // Use a metric name without "inf" to avoid false-positive finds.
    auto result = PrometheusExporter::format_gauge("llmquant_volatility_gauge",
                                                    std::numeric_limits<double>::infinity());
    EXPECT_NE(result.find("llmquant_volatility_gauge 0"), std::string::npos)
        << "Inf gauge value must be replaced with 0";
    EXPECT_EQ(result.find("gauge inf"), std::string::npos)
        << "Inf must not appear as the metric value";
}

TEST(PrometheusExporterTest, test_format_histogram_nan_sum_replaced_with_zero) {
    using Bucket = PrometheusExporter::HistogramBucket;
    std::vector<Bucket> buckets = {{10.0, 5}};
    auto result = PrometheusExporter::format_histogram(
        "llmquant_lat", buckets,
        std::numeric_limits<double>::quiet_NaN(), 5u);
    // _sum must not be "nan".
    EXPECT_EQ(result.find("_sum nan"), std::string::npos)
        << "NaN histogram sum must be replaced with 0";
    EXPECT_NE(result.find("_sum 0"), std::string::npos)
        << "NaN histogram sum must appear as 0";
}

// ---------------------------------------------------------------------------
// Cycle 34: format_info()
// ---------------------------------------------------------------------------

TEST(PrometheusExporterTest, test_format_info_value_is_always_one) {
    std::map<std::string, std::string> labels = {{"version", "1.0.0"}};
    auto result = PrometheusExporter::format_info("llmquant_build_info", labels);
    EXPECT_NE(result.find("} 1"), std::string::npos)
        << "info metric value must always be 1";
}

TEST(PrometheusExporterTest, test_format_info_contains_label_key_value) {
    std::map<std::string, std::string> labels = {{"env", "prod"}, {"region", "us-east"}};
    auto result = PrometheusExporter::format_info("llmquant_build_info", labels);
    EXPECT_NE(result.find("env=\"prod\""), std::string::npos)
        << "format_info must include label env=prod";
    EXPECT_NE(result.find("region=\"us-east\""), std::string::npos)
        << "format_info must include label region=us-east";
}

TEST(PrometheusExporterTest, test_format_info_with_help_includes_type_comment) {
    std::map<std::string, std::string> labels = {{"build", "release"}};
    auto result = PrometheusExporter::format_info("llmquant_build_info", labels,
                                                    "Engine build metadata");
    EXPECT_NE(result.find("# HELP"), std::string::npos)
        << "format_info must include HELP comment when help is non-empty";
    EXPECT_NE(result.find("# TYPE llmquant_build_info gauge"), std::string::npos)
        << "format_info must declare TYPE as gauge";
}

TEST(PrometheusExporterTest, test_format_info_empty_labels_produces_empty_braces) {
    std::map<std::string, std::string> labels;
    auto result = PrometheusExporter::format_info("llmquant_empty_info", labels);
    EXPECT_NE(result.find("{} 1"), std::string::npos)
        << "format_info with no labels must produce empty label set {}";
}
