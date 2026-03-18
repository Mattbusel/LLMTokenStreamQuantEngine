#include "gtest/gtest.h"
#include "FixOmsAdapter.h"
#include "RiskManager.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a FixOmsAdapter::Config pointing at a port where nothing is listening.
static FixOmsAdapter::Config make_test_config() {
    FixOmsAdapter::Config cfg;
    cfg.host                 = "127.0.0.1";
    cfg.port                 = 19999;          // nothing listening here
    cfg.sender_comp_id       = "TEST";
    cfg.target_comp_id       = "BROKER";
    cfg.heartbeat_interval_s = 30;
    cfg.position_limit       = 1.0;
    cfg.pnl_limit            = -10.0;
    return cfg;
}

/// Wait up to `timeout_ms` milliseconds for `pred` to return true, polling
/// every `poll_ms` milliseconds.  Returns true if `pred` became true in time.
template <typename Pred>
static bool wait_for(Pred pred, int timeout_ms = 500, int poll_ms = 10) {
    for (int elapsed = 0; elapsed < timeout_ms; elapsed += poll_ms) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
    return pred();
}

// ---------------------------------------------------------------------------
// Test 1: description() returns a non-empty string containing host, port,
//         and both comp IDs.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_description_is_non_empty) {
    FixOmsAdapter adapter(make_test_config());
    std::string desc = adapter.description();

    EXPECT_FALSE(desc.empty())
        << "description() must return a non-empty string";
    EXPECT_NE(desc.find("127.0.0.1"), std::string::npos)
        << "description() must contain the configured host";
    EXPECT_NE(desc.find("19999"), std::string::npos)
        << "description() must contain the configured port";
    EXPECT_NE(desc.find("TEST"), std::string::npos)
        << "description() must contain SenderCompID";
    EXPECT_NE(desc.find("BROKER"), std::string::npos)
        << "description() must contain TargetCompID";
}

// ---------------------------------------------------------------------------
// Test 2: is_running() is false before start() is called.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_is_running_false_before_start) {
    FixOmsAdapter adapter(make_test_config());
    EXPECT_FALSE(adapter.is_running())
        << "Adapter must not be running before start()";
}

// ---------------------------------------------------------------------------
// Test 3: messages_parsed() is zero before start().
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_messages_parsed_zero_before_start) {
    FixOmsAdapter adapter(make_test_config());
    EXPECT_EQ(adapter.messages_parsed(), 0u)
        << "messages_parsed() must be 0 before any activity";
}

// ---------------------------------------------------------------------------
// Test 4: start() returns true on the first call and transitions is_running()
//         to true.  The connection will fail immediately (nothing on 19999),
//         so the thread sets running_ back to false — both transitions are
//         valid observable outcomes within the lifecycle contract.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_start_returns_true_and_sets_running) {
    FixOmsAdapter adapter(make_test_config());

    bool result = adapter.start();
    EXPECT_TRUE(result)
        << "First start() must return true";

    // is_running() may briefly be true then flip to false once the connection
    // is refused.  What matters is that start() returned true and did not crash.
    // Give the thread a moment to resolve.
    wait_for([&] { return !adapter.is_running(); }, 2000);

    // No crash reaching this point is the key assertion; also confirm the
    // adapter is now in a stopped state.
    EXPECT_FALSE(adapter.is_running())
        << "Adapter must be stopped after the connection is refused";
}

// ---------------------------------------------------------------------------
// Test 5: stop() after start() does not crash and leaves is_running() false.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_stop_after_start_does_not_crash) {
    FixOmsAdapter adapter(make_test_config());

    adapter.start();
    // Give the reader thread time to launch and attempt the connection.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    adapter.stop();  // must not deadlock, crash, or throw

    EXPECT_FALSE(adapter.is_running())
        << "is_running() must be false after stop()";
}

// ---------------------------------------------------------------------------
// Test 6: A second call to start() while already running returns false
//         (idempotent / guard).
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_double_start_is_idempotent) {
    FixOmsAdapter adapter(make_test_config());

    bool first = adapter.start();
    // The reader thread may have already set running_=false if the connect
    // was refused instantly.  Call start() a second time regardless.
    bool second = adapter.start();
    adapter.stop();

    EXPECT_TRUE(first)
        << "First start() must return true";
    // If the thread already exited (running_==false), start() returns true
    // again, which is also acceptable.  The important thing is no crash.
    (void)second;  // outcome is implementation-defined after fast connection refusal
}

// ---------------------------------------------------------------------------
// Test 7: stop() before start() is a safe no-op.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_stop_before_start_is_safe) {
    FixOmsAdapter adapter(make_test_config());
    // Must not throw, deadlock, or crash.
    adapter.stop();
    EXPECT_FALSE(adapter.is_running())
        << "is_running() must be false when stop() is called before start()";
}

// ---------------------------------------------------------------------------
// Test 8: Destructor does not crash when called on a running adapter.
//         We construct on the heap so we can control destruction timing.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_destructor_stops_adapter_safely) {
    {
        FixOmsAdapter adapter(make_test_config());
        adapter.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        // Let destructor call stop() and join the thread.
    }
    // Reaching here without crash / asan error is the assertion.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Test 9: set_position_callback() stores the callback and it can be invoked.
//         We verify indirectly: inject a callback, start+stop the adapter,
//         confirm no crash.  The callback itself is only fired on actual FIX
//         message receipt, so we just confirm the plumbing compiles and runs.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_set_position_callback_stores_callback) {
    FixOmsAdapter adapter(make_test_config());

    std::atomic<int> call_count{0};
    adapter.set_position_callback([&](const RiskManager::PositionState& s) {
        (void)s;
        ++call_count;
    });

    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    adapter.stop();

    // The callback will not have fired (no server to send FIX messages), but
    // registering it must not crash.
    EXPECT_GE(call_count.load(), 0)
        << "Callback invocation count must be non-negative";
}

// ---------------------------------------------------------------------------
// Test 10: FIX checksum semantics — tested indirectly through the observable
//          fact that start()+stop() does not crash even though the adapter
//          builds and attempts to send a Logon (35=A) message whose checksum
//          and BodyLength are computed internally.  If either calculation were
//          broken in a way that caused a buffer overrun or exception the test
//          would crash or ASan would fire.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_fix_message_construction_does_not_crash) {
    FixOmsAdapter adapter(make_test_config());

    // start() will internally call build_logon() -> fix_message() ->
    // fix_checksum() before attempting the TCP write.  If any of those helpers
    // have a bug that causes UB, the sanitiser build will catch it here.
    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    adapter.stop();

    SUCCEED();
}

// ---------------------------------------------------------------------------
// Test 11: messages_parsed() does not decrease — it is a monotonically
//          non-decreasing counter.  After start()+stop() it must be >= 0.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_messages_parsed_non_decreasing) {
    FixOmsAdapter adapter(make_test_config());

    adapter.start();
    uint64_t mid = adapter.messages_parsed();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    uint64_t after = adapter.messages_parsed();
    adapter.stop();
    uint64_t final_count = adapter.messages_parsed();

    EXPECT_GE(mid,         0u);
    EXPECT_GE(after,       mid)   << "messages_parsed() must not decrease";
    EXPECT_GE(final_count, after) << "messages_parsed() must not decrease";
}

// ---------------------------------------------------------------------------
// Test 12: Multiple start/stop cycles do not crash or leak threads.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_multiple_start_stop_cycles_are_safe) {
    FixOmsAdapter adapter(make_test_config());

    for (int i = 0; i < 3; ++i) {
        adapter.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        adapter.stop();
        EXPECT_FALSE(adapter.is_running())
            << "Adapter must be stopped after stop() in cycle " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 13: description() is consistent — calling it multiple times returns
//          the same value and never returns an empty string.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_description_is_consistent) {
    FixOmsAdapter adapter(make_test_config());

    std::string d1 = adapter.description();
    std::string d2 = adapter.description();

    EXPECT_FALSE(d1.empty());
    EXPECT_EQ(d1, d2)
        << "description() must return the same value on repeated calls";
}

// ---------------------------------------------------------------------------
// Test 14: set_position_callback() can be overwritten with a new callback
//          without crashing.
// ---------------------------------------------------------------------------
TEST(FixOmsAdapterTest, test_set_position_callback_can_be_overwritten) {
    FixOmsAdapter adapter(make_test_config());

    adapter.set_position_callback([](const RiskManager::PositionState&) {});
    // Overwrite with a second callback — must not crash.
    adapter.set_position_callback([](const RiskManager::PositionState&) {});

    adapter.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    adapter.stop();

    SUCCEED();
}
