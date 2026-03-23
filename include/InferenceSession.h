#pragma once
// InferenceSession.h — Manage the full lifecycle of an LLM inference session.

#include "GenerationConfig.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llmquant {

/// State machine states for an InferenceSession.
enum class SessionState {
    Created,    ///< Session constructed but not yet started.
    Running,    ///< Actively generating tokens.
    Paused,     ///< Temporarily suspended; can be resumed.
    Completed,  ///< Generation finished successfully.
    Failed,     ///< Generation terminated with an error.
};

/// Convert a SessionState to a human-readable string.
const char* session_state_name(SessionState state) noexcept;

/// Manages the lifecycle of a single LLM inference request.
///
/// Tokens are appended incrementally; timing statistics are derived from
/// token count (mock: each token takes 10 ms).
///
/// Thread-safety: this class is NOT thread-safe. External synchronisation
/// is required if multiple threads access the same session.
class InferenceSession {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Construct a session with identifiers and generation configuration.
    InferenceSession(std::string session_id,
                     std::string model_id,
                     GenerationConfig config = GenerationConfig{});

    // Non-copyable (each session has unique identity).
    InferenceSession(const InferenceSession&)            = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    // Movable.
    InferenceSession(InferenceSession&&)            = default;
    InferenceSession& operator=(InferenceSession&&) = default;

    ~InferenceSession() = default;

    // ------------------------------------------------------------------
    // State transitions
    // ------------------------------------------------------------------

    /// Transition Created → Running.
    /// No-op if already Running; throws if terminal state.
    void start();

    /// Transition Running → Paused.
    /// No-op if already Paused; no-op if not Running.
    void pause();

    /// Transition Paused → Running.
    /// No-op if already Running; no-op if not Paused.
    void resume();

    /// Transition Running/Paused → Completed.
    void complete();

    /// Transition any live state → Failed with a reason string.
    void fail(const std::string& reason);

    // ------------------------------------------------------------------
    // Token management
    // ------------------------------------------------------------------

    /// Append a token to the output sequence.
    /// Only valid in Running or Paused state; ignored otherwise.
    void append_token(const std::string& token);

    /// Read-only view of all output tokens generated so far.
    const std::vector<std::string>& output_tokens() const noexcept;

    /// Number of output tokens generated.
    std::size_t token_count() const noexcept;

    // ------------------------------------------------------------------
    // Timing & throughput (mock)
    // ------------------------------------------------------------------

    /// Mock elapsed time in milliseconds: token_count * 10.
    uint64_t elapsed_ms() const noexcept;

    /// Tokens per second: token_count / (elapsed_ms / 1000.0).
    /// Returns 0.0 if elapsed_ms == 0.
    double throughput_tps() const noexcept;

    // ------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------

    /// Current state of the session.
    SessionState state() const noexcept;

    /// Session identifier string.
    const std::string& session_id() const noexcept;

    /// Model identifier string.
    const std::string& model_id() const noexcept;

    /// Generation configuration.
    const GenerationConfig& config() const noexcept;

    /// Failure reason (non-empty only when state == Failed).
    const std::string& failure_reason() const noexcept;

    /// Human-readable summary of the session.
    std::string summary() const;

private:
    std::string        session_id_;
    std::string        model_id_;
    GenerationConfig   config_;
    SessionState       state_{SessionState::Created};
    std::vector<std::string> output_tokens_;
    std::string        failure_reason_;
};

}  // namespace llmquant
