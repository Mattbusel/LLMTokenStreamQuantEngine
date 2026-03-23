#pragma once
// BatchProcessor.h — Batch inference processing for LLM token generation.
// Manages a batch of generation requests, applies sampling, pads sequences,
// and builds attention masks.

#include <string>
#include <vector>

namespace llmquant {

/// A single inference request with generation parameters.
struct BatchRequest {
    int                request_id{0};
    std::vector<int>   token_ids;
    int                max_new_tokens{64};
    float              temperature{1.0f};
    float              top_p{0.9f};
};

/// The result of processing one request.
struct BatchResponse {
    int                request_id{0};
    std::vector<int>   generated_ids;
    std::string        finish_reason;  // "length", "eos", "stop"
};

/// Manages batched LLM inference with sampling and padding utilities.
class BatchProcessor {
public:
    /// @param max_batch_size  Maximum number of concurrent requests.
    /// @param max_seq_len     Maximum total sequence length (prompt + generated).
    /// @param eos_token_id    Token ID that signals end-of-sequence.
    BatchProcessor(int max_batch_size, int max_seq_len, int eos_token_id);

    /// Add a request to the current batch.
    /// @return false if the batch is already full.
    bool add_request(BatchRequest req);

    /// Process the current batch given per-request logit vectors.
    /// @param logits_per_request  One logit vector per queued request (vocab-sized).
    /// @return                    One BatchResponse per request in insertion order.
    std::vector<BatchResponse> process_batch(
        const std::vector<std::vector<float>>& logits_per_request);

    /// Pad all sequences to the same length using pad_id.
    /// @return Padded sequences (new allocation; originals unchanged).
    std::vector<std::vector<int>> pad_sequences(
        std::vector<std::vector<int>>& sequences, int pad_id);

    /// Build a binary attention mask: 1 for real tokens, 0 for padding.
    std::vector<std::vector<int>> create_attention_mask(
        const std::vector<std::vector<int>>& padded);

    int  current_batch_size() const;
    bool is_full()            const;
    void clear();

    /// JSON string with batch statistics.
    std::string stats() const;

private:
    int max_batch_size_;
    int max_seq_len_;
    int eos_token_id_;

    std::vector<BatchRequest>  requests_;
    std::vector<BatchResponse> responses_;

    // Statistics
    long long total_requests_processed_{0};
    long long total_tokens_generated_{0};
    int       eos_stops_{0};
    int       length_stops_{0};

    /// Apply temperature + top-p sampling to a logit vector; return sampled token ID.
    int sample_token(const std::vector<float>& logits, float temperature, float top_p) const;

    /// Compute softmax in-place on a vector.
    static std::vector<float> softmax(const std::vector<float>& logits);
};

} // namespace llmquant
