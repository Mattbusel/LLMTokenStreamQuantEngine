// InferenceSession.cpp — LLM inference session lifecycle management.

#include "InferenceSession.h"

#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const char* session_state_name(SessionState state) noexcept {
    switch (state) {
        case SessionState::Created:   return "Created";
        case SessionState::Running:   return "Running";
        case SessionState::Paused:    return "Paused";
        case SessionState::Completed: return "Completed";
        case SessionState::Failed:    return "Failed";
    }
    return "Unknown";
}

static bool is_terminal(SessionState s) noexcept {
    return s == SessionState::Completed || s == SessionState::Failed;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

InferenceSession::InferenceSession(
    std::string session_id,
    std::string model_id,
    GenerationConfig config)
    : session_id_(std::move(session_id))
    , model_id_(std::move(model_id))
    , config_(config)
    , state_(SessionState::Created)
{}

// ---------------------------------------------------------------------------
// State transitions
// ---------------------------------------------------------------------------

void InferenceSession::start() {
    if (state_ == SessionState::Running) return;
    if (is_terminal(state_)) {
        throw std::logic_error(
            "Cannot start a session in terminal state: " +
            std::string(session_state_name(state_)));
    }
    state_ = SessionState::Running;
}

void InferenceSession::pause() {
    if (state_ != SessionState::Running) return;
    state_ = SessionState::Paused;
}

void InferenceSession::resume() {
    if (state_ != SessionState::Paused) return;
    state_ = SessionState::Running;
}

void InferenceSession::complete() {
    if (is_terminal(state_)) return;
    state_ = SessionState::Completed;
}

void InferenceSession::fail(const std::string& reason) {
    if (is_terminal(state_)) return;
    state_          = SessionState::Failed;
    failure_reason_ = reason;
}

// ---------------------------------------------------------------------------
// Token management
// ---------------------------------------------------------------------------

void InferenceSession::append_token(const std::string& token) {
    if (state_ != SessionState::Running && state_ != SessionState::Paused) return;
    output_tokens_.push_back(token);
}

const std::vector<std::string>& InferenceSession::output_tokens() const noexcept {
    return output_tokens_;
}

std::size_t InferenceSession::token_count() const noexcept {
    return output_tokens_.size();
}

// ---------------------------------------------------------------------------
// Timing & throughput
// ---------------------------------------------------------------------------

uint64_t InferenceSession::elapsed_ms() const noexcept {
    return static_cast<uint64_t>(token_count()) * 10ULL;
}

double InferenceSession::throughput_tps() const noexcept {
    uint64_t ms = elapsed_ms();
    if (ms == 0) return 0.0;
    return static_cast<double>(token_count()) / (static_cast<double>(ms) / 1000.0);
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

SessionState InferenceSession::state() const noexcept {
    return state_;
}

const std::string& InferenceSession::session_id() const noexcept {
    return session_id_;
}

const std::string& InferenceSession::model_id() const noexcept {
    return model_id_;
}

const GenerationConfig& InferenceSession::config() const noexcept {
    return config_;
}

const std::string& InferenceSession::failure_reason() const noexcept {
    return failure_reason_;
}

std::string InferenceSession::summary() const {
    std::ostringstream oss;
    oss << "InferenceSession{"
        << "id=" << session_id_
        << ", model=" << model_id_
        << ", state=" << session_state_name(state_)
        << ", tokens=" << token_count()
        << ", elapsed_ms=" << elapsed_ms()
        << ", tps=" << throughput_tps();
    if (!failure_reason_.empty()) {
        oss << ", failure=" << failure_reason_;
    }
    oss << "}";
    return oss.str();
}

}  // namespace llmquant
