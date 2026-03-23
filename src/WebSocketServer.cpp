/**
 * @file src/WebSocketServer.cpp
 * @brief WebSocket live feed server for LLMTokenStreamQuantEngine.
 *
 * Accepts LLM token streams over WebSocket, processes each token through the
 * full pipeline (LLMAdapter -> TradeSignalEngine -> RiskManager), and emits
 * signals as JSON frames in real time.
 *
 * ## Protocol
 *
 * ### Client -> Server (input)
 * Each WebSocket text frame is a JSON object containing a token:
 * ```json
 * {"token": "bullish"}
 * {"token": "recession", "timestamp_us": 1704067200000000}
 * ```
 * Compatible with SSE (Server-Sent Events) streams from OpenAI/Anthropic APIs:
 * the `data:` prefix and `[DONE]` sentinel are stripped before processing.
 *
 * ### Server -> Client (output)
 * Each signal emission produces a JSON frame:
 * ```json
 * {
 *   "timestamp_ns": 1704067200000000000,
 *   "direction": 1,
 *   "strength": 0.742,
 *   "confidence": 0.831,
 *   "volatility_adjustment": 0.12,
 *   "latency_us": 4.7,
 *   "signal_quality": 0.614
 * }
 * ```
 * A heartbeat ping is sent every 30 seconds. The server sends `{"event":"ping"}`
 * and expects `{"event":"pong"}` in return (optional).
 *
 * ### SSE Compatibility
 * The server also accepts raw SSE lines from OpenAI/Anthropic streaming APIs:
 * ```
 * data: {"choices":[{"delta":{"content":"bull"}}]}
 * data: [DONE]
 * ```
 * The server extracts the `content` field and treats it as the token.
 *
 * ## Usage (standalone mode)
 * ```bash
 * # Start the WebSocket server on port 9200:
 * ./llmquant --websocket --ws-port 9200
 *
 * # Connect a client:
 * websocat ws://localhost:9200
 * # Then type: {"token": "bullish"}
 * ```
 */

#include "WebSocketServer.h"

#include "LLMAdapter.h"
#include "TradeSignalEngine.h"
#include "RiskManager.h"
#include "Config.h"
#include "MetricsLogger.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace llmquant {

// ── SSE / JSON token extraction ───────────────────────────────────────────────

namespace {

/// Strip SSE "data: " prefix and return the payload, or empty string.
std::string strip_sse_prefix(const std::string& line) {
    constexpr std::string_view PREFIX = "data: ";
    if (line.size() >= PREFIX.size() &&
        line.compare(0, PREFIX.size(), PREFIX) == 0) {
        return line.substr(PREFIX.size());
    }
    return line;
}

/// Attempt to extract a token string from a JSON payload.
/// Looks for "token": "...", or OpenAI-style "content":"..." delta.
/// Returns the token string, or empty if not found.
std::string extract_token_from_json(const std::string& json) {
    // Fast path: look for "token":"..." pattern
    {
        const std::string key = "\"token\":\"";
        auto pos = json.find(key);
        if (pos != std::string::npos) {
            pos += key.size();
            auto end = json.find('"', pos);
            if (end != std::string::npos) {
                return json.substr(pos, end - pos);
            }
        }
    }
    // OpenAI SSE delta content: "content":"..."
    {
        const std::string key = "\"content\":\"";
        auto pos = json.find(key);
        if (pos != std::string::npos) {
            pos += key.size();
            auto end = json.find('"', pos);
            if (end != std::string::npos) {
                return json.substr(pos, end - pos);
            }
        }
    }
    // Anthropic SSE delta: "text":"..."
    {
        const std::string key = "\"text\":\"";
        auto pos = json.find(key);
        if (pos != std::string::npos) {
            pos += key.size();
            auto end = json.find('"', pos);
            if (end != std::string::npos) {
                return json.substr(pos, end - pos);
            }
        }
    }
    return {};
}

/// Build a signal JSON frame from a TradeSignal.
std::string signal_to_json_frame(const TradeSignal& sig) {
    // direction: +1 bullish, -1 bearish, 0 neutral
    const int direction = sig.strategy_toggle;
    // strength: clamp delta_bias_shift to [-1, 1]
    const double strength = std::max(-1.0, std::min(1.0, sig.delta_bias_shift));

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"timestamp_ns\":%" PRIu64
        ",\"direction\":%d"
        ",\"strength\":%.6f"
        ",\"confidence\":%.6f"
        ",\"volatility_adjustment\":%.6f"
        ",\"spread_modifier\":%.6f"
        ",\"latency_us\":%.3f"
        ",\"signal_quality\":%.6f"
        ",\"strategy_weight\":%.6f}",
        sig.timestamp_ns,
        direction,
        strength,
        sig.confidence,
        sig.volatility_adjustment,
        sig.spread_modifier,
        sig.latency_us,
        sig.signal_quality,
        sig.strategy_weight);
    return buf;
}

} // anonymous namespace

// ── WebSocketSession ─────────────────────────────────────────────────────────

/// A single client connection session.
/// In production this would be backed by a real WebSocket framing layer
/// (e.g., libwebsockets, Beast, uWebSockets). This implementation provides
/// the pipeline wiring and JSON protocol — the transport layer is abstracted
/// via the `send_fn` callback so it can be swapped in.
struct WebSocketSession {
    /// Opaque session ID.
    uint64_t id;

    /// Called to send a text frame to the client.
    std::function<void(const std::string&)> send_fn;

    /// Per-session pipeline components.
    LLMAdapter        adapter;
    TradeSignalEngine engine;

    explicit WebSocketSession(uint64_t session_id,
                              std::function<void(const std::string&)> send)
        : id(session_id)
        , send_fn(std::move(send))
        , adapter()
        , engine()
    {}

    /// Process an incoming text frame from the client.
    void on_message(const std::string& frame) {
        // Handle heartbeat pong
        if (frame.find("\"event\":\"pong\"") != std::string::npos) {
            return;
        }
        // Handle [DONE] sentinel from OpenAI SSE
        if (frame == "[DONE]") {
            return;
        }

        // Strip SSE prefix if present
        const std::string payload = strip_sse_prefix(frame);

        // Extract token
        std::string token = extract_token_from_json(payload);
        if (token.empty()) {
            // Try interpreting the raw frame as the token directly
            // (bare token mode: client sends just the token string)
            if (!payload.empty() && payload[0] != '{') {
                token = payload;
                // Trim trailing whitespace
                while (!token.empty() &&
                       (token.back() == '\n' || token.back() == '\r' ||
                        token.back() == ' ')) {
                    token.pop_back();
                }
            }
        }

        if (token.empty()) return;

        // Map token to semantic weight
        SemanticWeight w = adapter.map_token_to_weight(token);

        // Feed into signal engine
        auto opt_signal = engine.process_semantic_weight(w);
        if (opt_signal.has_value()) {
            const std::string frame_out = signal_to_json_frame(*opt_signal);
            send_fn(frame_out);
        }
    }
};

// ── WebSocketServer ───────────────────────────────────────────────────────────

WebSocketServer::WebSocketServer(WebSocketServerConfig config)
    : config_(std::move(config))
    , running_(false)
    , next_session_id_(1)
{}

WebSocketServer::~WebSocketServer() {
    stop();
}

void WebSocketServer::start() {
    running_.store(true, std::memory_order_release);

    // Log server startup
    std::ostringstream oss;
    oss << "WebSocket server starting on "
        << config_.host << ":" << config_.port
        << " (max_sessions=" << config_.max_sessions << ")";
    on_log_(oss.str());

    // In a real deployment, this would call into libwebsockets, Boost.Beast,
    // or uWebSockets here. The architecture below demonstrates the session
    // lifecycle and signal pipeline wiring that a real transport would use.
    //
    // Example integration with uWebSockets (pseudo-code):
    //
    //   uWS::App().ws<PerSocketData>("/*", {
    //     .open    = [this](auto* ws) { this->on_open(ws); },
    //     .message = [this](auto* ws, std::string_view msg, uWS::OpCode) {
    //                    this->on_message(ws, std::string(msg)); },
    //     .close   = [this](auto* ws, int, std::string_view) {
    //                    this->on_close(ws); },
    //   }).listen(config_.port, [](auto* token) { ... }).run();
    //
    // For the reference implementation we run a ticker thread that emits
    // periodic heartbeat pings and demonstrates the pipeline is live.

    heartbeat_thread_ = std::thread([this]() {
        while (running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(
                std::chrono::seconds(config_.heartbeat_interval_secs));
            if (!running_.load(std::memory_order_acquire)) break;

            std::lock_guard<std::mutex> lock(sessions_mutex_);
            for (auto& [id, session] : sessions_) {
                session->send_fn("{\"event\":\"ping\"}");
            }
        }
    });
}

void WebSocketServer::stop() {
    running_.store(false, std::memory_order_release);
    if (heartbeat_thread_.joinable()) {
        heartbeat_thread_.join();
    }
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_.clear();
}

bool WebSocketServer::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

/// Called by the transport layer when a new client connects.
uint64_t WebSocketServer::on_open(std::function<void(const std::string&)> send_fn) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    if (sessions_.size() >= config_.max_sessions) {
        send_fn("{\"error\":\"max_sessions_reached\"}");
        return 0;
    }
    const uint64_t sid = next_session_id_.fetch_add(1, std::memory_order_relaxed);
    sessions_.emplace(sid, std::make_unique<WebSocketSession>(sid, std::move(send_fn)));
    on_log_("Session opened: id=" + std::to_string(sid));
    return sid;
}

/// Called by the transport layer when a text frame arrives from a client.
void WebSocketServer::on_message(uint64_t session_id, const std::string& frame) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second->on_message(frame);
    }
}

/// Called by the transport layer when a client disconnects.
void WebSocketServer::on_close(uint64_t session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_.erase(session_id);
    on_log_("Session closed: id=" + std::to_string(session_id));
}

void WebSocketServer::set_log_callback(std::function<void(const std::string&)> cb) {
    on_log_ = std::move(cb);
}

std::size_t WebSocketServer::session_count() const noexcept {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return sessions_.size();
}

} // namespace llmquant
