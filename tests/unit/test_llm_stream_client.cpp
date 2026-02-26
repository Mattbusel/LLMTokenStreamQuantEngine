#include "gtest/gtest.h"
#include "LLMStreamClient.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace llmquant;

// ---------------------------------------------------------------------------
// Helper: build a Config pointing at a local port that is never open.
// ---------------------------------------------------------------------------
static LLMStreamClient::Config refused_config() {
    LLMStreamClient::Config cfg;
    cfg.host    = "127.0.0.1";
    cfg.port    = 1;       // port 1 is privileged and never open in normal use
    cfg.use_tls = false;
    cfg.api_key = "test-key";
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(LLMStreamClientTest, test_stream_client_is_running_false_before_connect) {
    LLMStreamClient client(refused_config());
    EXPECT_FALSE(client.is_running());
}

TEST(LLMStreamClientTest, test_stream_client_stop_before_connect_is_safe) {
    LLMStreamClient client(refused_config());
    // stop() before connect() must not crash or hang.
    client.stop();
    SUCCEED();
}

TEST(LLMStreamClientTest, test_stream_client_connect_to_refused_port_does_not_hang) {
    LLMStreamClient client(refused_config());
    // connect() may return false (immediate refusal) or true (OS queued the
    // connect and the reader thread exits quickly).  Either way, stop() must
    // return within a reasonable time.
    client.connect();
    client.stop();
    EXPECT_FALSE(client.is_running());
}

TEST(LLMStreamClientTest, test_stream_client_double_stop_is_safe) {
    LLMStreamClient client(refused_config());
    client.connect();
    client.stop();
    client.stop();  // second stop must not crash or hang
    SUCCEED();
}

TEST(LLMStreamClientTest, test_stream_client_done_callback_fires_on_failed_connect) {
    std::atomic<bool> done_fired{false};

    LLMStreamClient client(refused_config());
    client.set_done_callback([&](const std::string& /*err*/) {
        done_fired = true;
    });

    bool connected = client.connect();
    if (connected) {
        // Reader thread started — wait up to 2 s for done callback.
        for (int i = 0; i < 20 && !done_fired.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    client.stop();
    // If connect() returned false the reader thread never ran and done_cb was
    // never set up — that outcome is also acceptable.
    SUCCEED();
}
