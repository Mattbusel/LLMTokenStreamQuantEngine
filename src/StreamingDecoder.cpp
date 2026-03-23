#include "StreamingDecoder.h"
#include <algorithm>
#include <stdexcept>

StreamingDecoder::StreamingDecoder(int vocab_size, int eos_token_id)
    : vocab_size_(vocab_size), eos_token_id_(eos_token_id) {}

std::string StreamingDecoder::feed(int token_id) {
    if (state_ != StreamState::Active) {
        return {};
    }

    if (token_id == eos_token_id_) {
        std::string remaining = flush();
        state_ = StreamState::Finished;
        return remaining;
    }

    ++tokens_decoded_;
    auto bytes = token_to_bytes(token_id);
    for (uint8_t b : bytes) {
        partial_buffer_.push_back(b);
    }

    std::string emitted = emit_buffer();
    if (!emitted.empty()) {
        bytes_emitted_ += emitted.size();
        if (check_stop_sequences(emitted)) {
            state_ = StreamState::Finished;
        }
    }
    return emitted;
}

std::string StreamingDecoder::feed_batch(const std::vector<int>& token_ids) {
    std::string result;
    for (int tid : token_ids) {
        result += feed(tid);
        if (state_ != StreamState::Active) {
            break;
        }
    }
    return result;
}

std::string StreamingDecoder::flush() {
    if (partial_buffer_.empty()) {
        return {};
    }
    // Emit remaining bytes as-is, replacing invalid UTF-8 with replacement char
    std::string out = bytes_to_utf8(partial_buffer_);
    bytes_emitted_ += out.size();
    partial_buffer_.clear();
    return out;
}

void StreamingDecoder::reset() {
    state_ = StreamState::Active;
    tokens_decoded_ = 0;
    bytes_emitted_ = 0;
    partial_buffer_.clear();
}

StreamState StreamingDecoder::state() const {
    return state_;
}

int StreamingDecoder::tokens_decoded() const {
    return tokens_decoded_;
}

std::size_t StreamingDecoder::bytes_emitted() const {
    return bytes_emitted_;
}

void StreamingDecoder::set_stop_sequences(const std::vector<std::string>& stops) {
    stop_sequences_ = stops;
}

bool StreamingDecoder::check_stop_sequences(const std::string& text) {
    for (const auto& stop : stop_sequences_) {
        if (text.find(stop) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string StreamingDecoder::bytes_to_utf8(const std::vector<uint8_t>& bytes) {
    // Validate UTF-8: emit valid sequences, replace invalid bytes with U+FFFD
    std::string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    while (i < bytes.size()) {
        uint8_t b = bytes[i];
        int seqlen = 0;
        if (b <= 0x7F) {
            seqlen = 1;
        } else if ((b & 0xE0) == 0xC0) {
            seqlen = 2;
        } else if ((b & 0xF0) == 0xE0) {
            seqlen = 3;
        } else if ((b & 0xF8) == 0xF0) {
            seqlen = 4;
        } else {
            // Invalid leading byte — emit replacement character U+FFFD (UTF-8: EF BF BD)
            out += "\xEF\xBF\xBD";
            ++i;
            continue;
        }

        if (i + static_cast<std::size_t>(seqlen) > bytes.size()) {
            // Incomplete sequence at end — emit replacement
            out += "\xEF\xBF\xBD";
            i = bytes.size();
            break;
        }

        // Validate continuation bytes
        bool valid = true;
        for (int j = 1; j < seqlen; ++j) {
            if ((bytes[i + j] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (valid) {
            for (int j = 0; j < seqlen; ++j) {
                out += static_cast<char>(bytes[i + j]);
            }
        } else {
            out += "\xEF\xBF\xBD";
        }
        i += static_cast<std::size_t>(seqlen);
    }
    return out;
}

std::vector<uint8_t> StreamingDecoder::token_to_bytes(int token_id) const {
    // Special tokens (id < 3) return empty
    if (token_id < 3) {
        return {};
    }
    // Map to printable ASCII: use (token_id % 95) + 32 to get printable chars
    uint8_t byte_val = static_cast<uint8_t>((token_id % 95) + 32);
    return {byte_val};
}

std::string StreamingDecoder::emit_buffer() {
    if (partial_buffer_.empty()) {
        return {};
    }

    // Find the last complete UTF-8 character boundary
    std::size_t safe_end = 0;
    std::size_t i = 0;
    while (i < partial_buffer_.size()) {
        uint8_t b = partial_buffer_[i];
        int seqlen = 0;
        if (b <= 0x7F) {
            seqlen = 1;
        } else if ((b & 0xE0) == 0xC0) {
            seqlen = 2;
        } else if ((b & 0xF0) == 0xE0) {
            seqlen = 3;
        } else if ((b & 0xF8) == 0xF0) {
            seqlen = 4;
        } else {
            // Invalid byte — mark as single-byte replacement
            seqlen = 1;
        }

        if (i + static_cast<std::size_t>(seqlen) <= partial_buffer_.size()) {
            safe_end = i + static_cast<std::size_t>(seqlen);
            i = safe_end;
        } else {
            // Incomplete sequence remains in buffer
            break;
        }
    }

    if (safe_end == 0) {
        return {};
    }

    std::vector<uint8_t> to_emit(partial_buffer_.begin(), partial_buffer_.begin() + static_cast<std::ptrdiff_t>(safe_end));
    partial_buffer_.erase(partial_buffer_.begin(), partial_buffer_.begin() + static_cast<std::ptrdiff_t>(safe_end));
    return bytes_to_utf8(to_emit);
}
