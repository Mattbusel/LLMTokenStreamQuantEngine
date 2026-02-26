#include "gtest/gtest.h"
#include "MockOmsAdapter.h"
#include "RestOmsAdapter.h"
#include "RiskManager.h"

#include <atomic>
#include <chrono>
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
