#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

/// State of the streaming decoder.
enum class StreamState {
    Active,
    Paused,
    Finished,
    Error,
};

/// A partially decoded token byte sequence.
struct PartialToken {
    std::vector<uint8_t> bytes;
    bool is_complete{false};
};

/// Incremental / streaming token decoder.
///
/// Tokens are mapped to bytes via (token_id % 256) as ASCII for testing,
/// with token_id == eos_token_id signalling end-of-stream.
class StreamingDecoder {
public:
    StreamingDecoder(int vocab_size, int eos_token_id);

    /// Decode a single token and return any text ready to emit.
    std::string feed(int token_id);

    /// Decode a batch of tokens and return accumulated text.
    std::string feed_batch(const std::vector<int>& token_ids);

    /// Flush any buffered partial bytes; return remaining text.
    std::string flush();

    /// Reset decoder to initial state.
    void reset();

    /// Current stream state.
    StreamState state() const;

    /// Number of tokens decoded so far.
    int tokens_decoded() const;

    /// Total bytes emitted so far.
    std::size_t bytes_emitted() const;

    /// Set stop sequences; decoding halts when any stop string appears in output.
    void set_stop_sequences(const std::vector<std::string>& stops);

    /// Check if text contains any stop sequence; returns true and updates state.
    bool check_stop_sequences(const std::string& text);

    /// Validate and convert a byte vector to a UTF-8 string.
    static std::string bytes_to_utf8(const std::vector<uint8_t>& bytes);

private:
    int vocab_size_;
    int eos_token_id_;
    StreamState state_{StreamState::Active};
    int tokens_decoded_{0};
    std::size_t bytes_emitted_{0};
    std::vector<uint8_t> partial_buffer_;
    std::vector<std::string> stop_sequences_;

    /// Map a token id to its byte representation.
    std::vector<uint8_t> token_to_bytes(int token_id) const;

    /// Emit buffered bytes as text, respecting UTF-8 continuation bytes.
    std::string emit_buffer();
};
