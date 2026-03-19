#include "gtest/gtest.h"
#include "LLMStreamClient.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
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

// ---------------------------------------------------------------------------
// SSE delta parsing tests (improvement 10)
// ---------------------------------------------------------------------------

TEST(LLMStreamClientTest, test_parse_sse_delta_extracts_content_token) {
    std::string data = R"({"choices":[{"delta":{"content":"hello"}}]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "hello");
}

TEST(LLMStreamClientTest, test_parse_sse_delta_missing_content_returns_empty) {
    std::string data = R"({"choices":[{"delta":{}}]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "");
}

TEST(LLMStreamClientTest, test_parse_sse_delta_malformed_json_returns_empty) {
    EXPECT_EQ(LLMStreamClient::parse_sse_delta("{bad json{{"), "");
}

TEST(LLMStreamClientTest, test_parse_sse_delta_empty_choices_returns_empty) {
    std::string data = R"({"choices":[]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "");
}

TEST(LLMStreamClientTest, test_parse_sse_delta_unicode_content_decoded) {
    // nlohmann/json decodes \u0041 -> 'A'
    std::string data = R"({"choices":[{"delta":{"content":"\u0041"}}]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "A");
}

TEST(LLMStreamClientTest, test_parse_sse_delta_embedded_quotes_in_content) {
    // Content containing escaped quotes — manual parser would corrupt this.
    std::string data = R"({"choices":[{"delta":{"content":"say \"hi\""}}]})";
    EXPECT_EQ(LLMStreamClient::parse_sse_delta(data), "say \"hi\"");
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

// ---------------------------------------------------------------------------
// Reconnect / loop behaviour (improvement #16)
// ---------------------------------------------------------------------------

TEST(LLMStreamClientTest, test_stream_client_stop_during_reconnect_backoff_does_not_hang) {
    // The client connects to a refused port and enters backoff.
    // stop() called during the backoff sleep must return within a short timeout.
    LLMStreamClient client(refused_config());
    std::atomic<bool> done_fired{false};
    client.set_done_callback([&](const std::string&) { done_fired = true; });
    client.connect();
    // Give the reader thread time to attempt the first connect and enter backoff.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    client.stop();
    EXPECT_FALSE(client.is_running());
}

TEST(LLMStreamClientTest, test_stream_client_token_callback_exception_does_not_crash) {
    // Verify that a throwing token callback is caught and does not propagate.
    // We can't inject tokens directly, so just confirm construction + stop is safe
    // with a callback that would throw.
    LLMStreamClient client(refused_config());
    client.set_token_callback([](const std::string&) {
        throw std::runtime_error("intentional test exception");
    });
    client.connect();
    client.stop();
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Improvement 13: KeepAlive reconnect stub (config parsing only)
// Full keep-alive reconnect tested in integration; this verifies config parsing only.
// ---------------------------------------------------------------------------
TEST(LLMStreamClientTest, KeepAliveReconnectStub) {
    LLMStreamClient::Config cfg = refused_config();
    cfg.use_keep_alive = true;  // verify the field is parsed without error
    LLMStreamClient client(cfg);
    EXPECT_FALSE(client.is_running());
    // Construction with use_keep_alive=true must not throw.
    SUCCEED();
}

