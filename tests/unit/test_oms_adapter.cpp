#include "gtest/gtest.h"
#include "MockOmsAdapter.h"
#include "RestOmsAdapter.h"
#include "RiskManager.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static RiskManager::PositionState make_pos(double net, double limit,
                                           double pnl, double pnl_limit) {
    RiskManager::PositionState s;
    s.net_position   = net;
    s.position_limit = limit;
    s.pnl            = pnl;
    s.pnl_limit      = pnl_limit;
    return s;
}

// ---------------------------------------------------------------------------
// Test 1: MockOmsAdapter emits exactly the loaded number of states.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_emits_all_states) {
    MockOmsAdapter::Config cfg;
    cfg.emit_interval = std::chrono::milliseconds{5};
    MockOmsAdapter adapter(cfg);

    adapter.load_states({
        make_pos(0.1, 1.0,  0.5, -10.0),
        make_pos(0.2, 1.0,  0.4, -10.0),
        make_pos(0.3, 1.0,  0.3, -10.0),
    });
    adapter.set_position_callback([](const RiskManager::PositionState&) {});
    adapter.start();

    // Wait long enough for all 3 states to be emitted (3 * 5 ms + margin).
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    EXPECT_EQ(adapter.emitted_count(), 3u)
        << "All loaded states must be emitted";
}

// ---------------------------------------------------------------------------
// Test 2: Callback receives the correct net_position values in order.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_callback_receives_correct_values) {
    MockOmsAdapter::Config cfg;
    cfg.emit_interval = std::chrono::milliseconds{5};
    MockOmsAdapter adapter(cfg);

    adapter.load_states({
        make_pos(0.10, 1.0, 0.0, -10.0),
        make_pos(0.25, 1.0, 0.0, -10.0),
        make_pos(-0.10, 1.0, 0.0, -10.0),
    });

    std::vector<double> received;
    std::mutex mu;
    adapter.set_position_callback([&](const RiskManager::PositionState& s) {
        std::lock_guard<std::mutex> lock(mu);
        received.push_back(s.net_position);
    });
    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    std::lock_guard<std::mutex> lock(mu);
    ASSERT_EQ(received.size(), 3u);
    EXPECT_DOUBLE_EQ(received[0],  0.10);
    EXPECT_DOUBLE_EQ(received[1],  0.25);
    EXPECT_DOUBLE_EQ(received[2], -0.10);
}

// ---------------------------------------------------------------------------
// Test 3: stop() mid-sequence halts emission before all states are sent.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_stop_before_all_states_emitted) {
    MockOmsAdapter::Config cfg;
    cfg.emit_interval = std::chrono::milliseconds{50};  // slow enough to stop mid-way
    MockOmsAdapter adapter(cfg);

    std::vector<RiskManager::PositionState> states;
    for (int i = 0; i < 10; ++i) {
        states.push_back(make_pos(static_cast<double>(i) * 0.05, 1.0, 0.0, -10.0));
    }
    adapter.load_states(states);
    adapter.set_position_callback([](const RiskManager::PositionState&) {});

    adapter.start();
    // Stop after ~80 ms: enough for ~1 emission but not all 10 (10 * 50 ms = 500 ms).
    std::this_thread::sleep_for(std::chrono::milliseconds{80});
    adapter.stop();

    EXPECT_LT(adapter.emitted_count(), 10u)
        << "stop() mid-sequence must leave some states unemitted";
}

// ---------------------------------------------------------------------------
// Test 4: is_running() returns false after the sequence is exhausted.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_is_running_false_after_sequence_exhausted) {
    MockOmsAdapter::Config cfg;
    cfg.emit_interval = std::chrono::milliseconds{5};
    MockOmsAdapter adapter(cfg);

    adapter.load_states({
        make_pos(0.1, 1.0, 0.0, -10.0),
        make_pos(0.2, 1.0, 0.0, -10.0),
    });
    adapter.set_position_callback([](const RiskManager::PositionState&) {});
    adapter.start();

    // Wait for both states + one full interval as headroom.
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    EXPECT_FALSE(adapter.is_running())
        << "Adapter must self-stop after exhausting the state sequence";
}

// ---------------------------------------------------------------------------
// Test 5: A second call to start() on an already-running adapter returns false.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_start_twice_returns_false) {
    MockOmsAdapter::Config cfg;
    cfg.emit_interval = std::chrono::milliseconds{50};
    MockOmsAdapter adapter(cfg);

    adapter.load_states({make_pos(0.1, 1.0, 0.0, -10.0)});
    adapter.set_position_callback([](const RiskManager::PositionState&) {});

    bool first  = adapter.start();
    bool second = adapter.start();
    adapter.stop();

    EXPECT_TRUE(first);
    EXPECT_FALSE(second) << "Second start() on running adapter must return false";
}

// ---------------------------------------------------------------------------
// Test 6: stop() before start() is a safe no-op.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_stop_before_start_is_safe) {
    MockOmsAdapter adapter;
    // Must not throw, deadlock, or crash.
    adapter.stop();
    EXPECT_FALSE(adapter.is_running());
}

// ---------------------------------------------------------------------------
// Test 7: RestOmsAdapter on a refused port does not hang indefinitely.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_rest_oms_connect_refused_does_not_hang) {
    RestOmsAdapter::Config cfg;
    cfg.host          = "127.0.0.1";
    cfg.port          = 1;                           // port 1 should be refused
    cfg.poll_interval = std::chrono::milliseconds{20};
    cfg.timeout_s     = 1;

    RestOmsAdapter adapter(cfg);
    adapter.set_position_callback([](const RiskManager::PositionState&) {});

    auto t0 = std::chrono::steady_clock::now();
    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{60});
    adapter.stop();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    EXPECT_LT(elapsed, 5000)
        << "stop() must return in < 5 s on a refused endpoint";
}

// ---------------------------------------------------------------------------
// Test 8: error_count increments when the endpoint is unreachable.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_rest_oms_error_count_increments_on_bad_host) {
    RestOmsAdapter::Config cfg;
    cfg.host          = "127.0.0.1";
    cfg.port          = 1;
    cfg.poll_interval = std::chrono::milliseconds{20};

    RestOmsAdapter adapter(cfg);
    adapter.set_position_callback([](const RiskManager::PositionState&) {});
    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{80});
    adapter.stop();

    EXPECT_GT(adapter.error_count(), 0u)
        << "error_count must be > 0 after polling an unreachable endpoint";
}

// ---------------------------------------------------------------------------
// Test 9: description() contains the configured host and port.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_rest_oms_description_contains_host_and_port) {
    RestOmsAdapter::Config cfg;
    cfg.host = "192.168.1.42";
    cfg.port = 9090;

    RestOmsAdapter adapter(cfg);
    std::string desc = adapter.description();

    EXPECT_NE(desc.find("192.168.1.42"), std::string::npos)
        << "description() must contain the configured host";
    EXPECT_NE(desc.find("9090"), std::string::npos)
        << "description() must contain the configured port";
}

// ---------------------------------------------------------------------------
// Test 10: Position from MockOmsAdapter correctly feeds into RiskManager.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_position_feeds_into_risk_manager) {
    RiskManager::Config rm_cfg;
    rm_cfg.max_bias_magnitude       = 1.0;
    rm_cfg.max_volatility_magnitude = 1.0;
    rm_cfg.max_signals_per_second   = 1000;
    rm_cfg.max_drawdown             = 100.0;
    RiskManager risk_mgr(rm_cfg);

    MockOmsAdapter::Config oms_cfg;
    oms_cfg.emit_interval = std::chrono::milliseconds{5};
    MockOmsAdapter adapter(oms_cfg);

    RiskManager::PositionState target = make_pos(0.42, 1.0, 1.23, -10.0);
    adapter.load_states({target});

    adapter.set_position_callback([&](const RiskManager::PositionState& s) {
        risk_mgr.update_position(s);
    });
    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    adapter.stop();

    auto pos = risk_mgr.get_position();
    EXPECT_DOUBLE_EQ(pos.net_position,   0.42);
    EXPECT_DOUBLE_EQ(pos.position_limit, 1.0);
    EXPECT_DOUBLE_EQ(pos.pnl,            1.23);
    EXPECT_DOUBLE_EQ(pos.pnl_limit,     -10.0);
}

// ---------------------------------------------------------------------------
// Test 11: An empty state list: adapter stops immediately without emitting.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_mock_oms_empty_state_list_stops_immediately) {
    MockOmsAdapter::Config cfg;
    cfg.emit_interval = std::chrono::milliseconds{5};
    MockOmsAdapter adapter(cfg);

    adapter.load_states({});
    adapter.set_position_callback([](const RiskManager::PositionState&) {});
    adapter.start();

    std::this_thread::sleep_for(std::chrono::milliseconds{50});

    EXPECT_EQ(adapter.emitted_count(), 0u)
        << "Empty state list must result in zero emissions";
    EXPECT_FALSE(adapter.is_running())
        << "Adapter must stop itself when state list is empty";
}

// ---------------------------------------------------------------------------
// Test 12: update_count stays zero on a permanently unreachable endpoint.
// ---------------------------------------------------------------------------
TEST(OmsAdapterTest, test_rest_oms_update_count_zero_on_bad_endpoint) {
    RestOmsAdapter::Config cfg;
    cfg.host          = "127.0.0.1";
    cfg.port          = 1;
    cfg.poll_interval = std::chrono::milliseconds{20};

    RestOmsAdapter adapter(cfg);
    adapter.set_position_callback([](const RiskManager::PositionState&) {});
    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{80});
    adapter.stop();

    EXPECT_EQ(adapter.update_count(), 0u)
        << "update_count must remain 0 when no valid responses are received";
}

// ===========================================================================
// parse_position() tests — accessed via a testable subclass that promotes
// the private static method to public.
// ===========================================================================

namespace {

class TestableRestOmsAdapter : public RestOmsAdapter {
public:
    explicit TestableRestOmsAdapter(RestOmsAdapter::Config cfg)
        : RestOmsAdapter(std::move(cfg)) {}

    bool test_parse_position(const std::string& body,
                              RiskManager::PositionState& out) {
        return parse_position(body, out);
    }
};

static RestOmsAdapter::Config make_rest_config() {
    RestOmsAdapter::Config cfg;
    cfg.host          = "127.0.0.1";
    cfg.port          = 19998;   // nothing listening here
    cfg.poll_interval = std::chrono::milliseconds{500};
    return cfg;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test 13: Valid HTTP 200 response with all required JSON fields succeeds.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_valid_http200) {
    TestableRestOmsAdapter adapter(make_rest_config());

    const std::string response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "\r\n"
        "{\"net_position\":0.35,\"position_limit\":1.0,"
        "\"pnl\":-1.23,\"pnl_limit\":-10.0}";

    RiskManager::PositionState out{};
    bool ok = adapter.test_parse_position(response, out);

    EXPECT_TRUE(ok) << "Valid HTTP 200 with all fields must parse successfully";
    EXPECT_DOUBLE_EQ(out.net_position,   0.35);
    EXPECT_DOUBLE_EQ(out.position_limit, 1.0);
    EXPECT_DOUBLE_EQ(out.pnl,           -1.23);
    EXPECT_DOUBLE_EQ(out.pnl_limit,    -10.0);
}

// ---------------------------------------------------------------------------
// Test 14: HTTP 404 response returns false.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_http404_returns_false) {
    TestableRestOmsAdapter adapter(make_rest_config());

    const std::string response =
        "HTTP/1.1 404 Not Found\r\n"
        "Content-Length: 0\r\n"
        "\r\n";

    RiskManager::PositionState out{};
    EXPECT_FALSE(adapter.test_parse_position(response, out))
        << "HTTP 404 must cause parse_position() to return false";
}

// ---------------------------------------------------------------------------
// Test 15: HTTP 503 response returns false.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_http503_returns_false) {
    TestableRestOmsAdapter adapter(make_rest_config());

    const std::string response =
        "HTTP/1.1 503 Service Unavailable\r\n"
        "\r\n";

    RiskManager::PositionState out{};
    EXPECT_FALSE(adapter.test_parse_position(response, out))
        << "HTTP 503 must cause parse_position() to return false";
}

// ---------------------------------------------------------------------------
// Test 16: Malformed JSON (not valid JSON at all) returns false.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_malformed_json_returns_false) {
    TestableRestOmsAdapter adapter(make_rest_config());

    const std::string response =
        "HTTP/1.1 200 OK\r\n"
        "\r\n"
        "this is not json {{{{";

    RiskManager::PositionState out{};
    EXPECT_FALSE(adapter.test_parse_position(response, out))
        << "Malformed JSON body must cause parse_position() to return false";
}

// ---------------------------------------------------------------------------
// Test 17: JSON that is otherwise valid but missing one required field
//          ("pnl_limit") returns false.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_missing_field_returns_false) {
    TestableRestOmsAdapter adapter(make_rest_config());

    // "pnl_limit" is omitted.
    const std::string response =
        "HTTP/1.1 200 OK\r\n"
        "\r\n"
        "{\"net_position\":0.1,\"position_limit\":1.0,\"pnl\":0.5}";

    RiskManager::PositionState out{};
    EXPECT_FALSE(adapter.test_parse_position(response, out))
        << "JSON missing pnl_limit must cause parse_position() to return false";
}

// ---------------------------------------------------------------------------
// Test 18: Bare JSON body (no HTTP headers at all) is accepted when all
//          required fields are present.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_bare_json_no_headers) {
    TestableRestOmsAdapter adapter(make_rest_config());

    const std::string body =
        "{\"net_position\":-0.5,\"position_limit\":2.0,"
        "\"pnl\":3.14,\"pnl_limit\":-50.0}";

    RiskManager::PositionState out{};
    bool ok = adapter.test_parse_position(body, out);

    EXPECT_TRUE(ok) << "Bare JSON without HTTP headers must parse successfully";
    EXPECT_DOUBLE_EQ(out.net_position,   -0.5);
    EXPECT_DOUBLE_EQ(out.position_limit,  2.0);
    EXPECT_DOUBLE_EQ(out.pnl,             3.14);
    EXPECT_DOUBLE_EQ(out.pnl_limit,     -50.0);
}

// ---------------------------------------------------------------------------
// Test 19: Chunked transfer encoding is decoded before parsing.
//          A single chunk containing the JSON payload must be parsed correctly.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_parse_position_chunked_body_decoded) {
    TestableRestOmsAdapter adapter(make_rest_config());

    // The JSON body we want to decode.
    const std::string json_body =
        "{\"net_position\":0.9,\"position_limit\":1.0,"
        "\"pnl\":0.0,\"pnl_limit\":-10.0}";

    // Encode as a single chunked block: hex size CRLF data CRLF 0 CRLF CRLF
    std::ostringstream chunked;
    chunked << std::hex << json_body.size() << "\r\n"
            << json_body << "\r\n"
            << "0\r\n\r\n";

    const std::string response =
        "HTTP/1.1 200 OK\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n"
        + chunked.str();

    RiskManager::PositionState out{};
    bool ok = adapter.test_parse_position(response, out);

    EXPECT_TRUE(ok) << "Chunked-encoded response with valid JSON must parse successfully";
    EXPECT_DOUBLE_EQ(out.net_position,   0.9);
    EXPECT_DOUBLE_EQ(out.position_limit, 1.0);
    EXPECT_DOUBLE_EQ(out.pnl,           0.0);
    EXPECT_DOUBLE_EQ(out.pnl_limit,    -10.0);
}

// ---------------------------------------------------------------------------
// Test 20: description() for RestOmsAdapter contains the path and poll
//          interval in addition to host and port.
// ---------------------------------------------------------------------------
TEST(RestOmsAdapterParsingTest, test_rest_description_contains_path_and_interval) {
    RestOmsAdapter::Config cfg;
    cfg.host          = "10.0.0.1";
    cfg.port          = 8080;
    cfg.path          = "/api/v1/positions";
    cfg.poll_interval = std::chrono::milliseconds{250};

    RestOmsAdapter adapter(cfg);
    std::string desc = adapter.description();

    EXPECT_NE(desc.find("/api/v1/positions"), std::string::npos)
        << "description() must contain the configured path";
    EXPECT_NE(desc.find("250"), std::string::npos)
        << "description() must contain the poll interval in milliseconds";
}
