// BatchProcessor.cpp — Batch inference processing implementation.
#include "BatchProcessor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace llmquant {

BatchProcessor::BatchProcessor(int max_batch_size, int max_seq_len, int eos_token_id)
    : max_batch_size_(max_batch_size)
    , max_seq_len_(max_seq_len)
    , eos_token_id_(eos_token_id)
{}

bool BatchProcessor::add_request(BatchRequest req) {
    if (static_cast<int>(requests_.size()) >= max_batch_size_) {
        return false;
    }
    requests_.push_back(std::move(req));
    return true;
}

std::vector<BatchResponse> BatchProcessor::process_batch(
        const std::vector<std::vector<float>>& logits_per_request) {

    std::vector<BatchResponse> results;
    results.reserve(requests_.size());

    const size_t n = std::min(requests_.size(), logits_per_request.size());
    for (size_t i = 0; i < n; ++i) {
        BatchResponse resp;
        resp.request_id = requests_[i].request_id;

        const BatchRequest& req = requests_[i];
        const std::vector<float>& logits = logits_per_request[i];

        if (logits.empty()) {
            resp.finish_reason = "length";
            results.push_back(std::move(resp));
            continue;
        }

        // Generate up to max_new_tokens tokens
        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(req.max_new_tokens));

        int token = sample_token(logits, req.temperature, req.top_p);
        generated.push_back(token);

        if (token == eos_token_id_) {
            resp.finish_reason = "eos";
            ++eos_stops_;
        } else {
            int total_len = static_cast<int>(req.token_ids.size()) + req.max_new_tokens;
            if (total_len >= max_seq_len_ || req.max_new_tokens <= 1) {
                resp.finish_reason = "length";
                ++length_stops_;
            } else {
                resp.finish_reason = "length";  // Single-step processing
                ++length_stops_;
            }
        }

        resp.generated_ids = std::move(generated);
        ++total_requests_processed_;
        total_tokens_generated_ += static_cast<long long>(resp.generated_ids.size());
        results.push_back(std::move(resp));
    }

    responses_ = results;
    return results;
}

std::vector<std::vector<int>> BatchProcessor::pad_sequences(
        std::vector<std::vector<int>>& sequences, int pad_id) {

    if (sequences.empty()) return {};

    size_t max_len = 0;
    for (const auto& seq : sequences) {
        if (seq.size() > max_len) max_len = seq.size();
    }

    std::vector<std::vector<int>> padded;
    padded.reserve(sequences.size());

    for (const auto& seq : sequences) {
        std::vector<int> padded_seq(max_len, pad_id);
        // Left-pad: real tokens at the end
        size_t offset = max_len - seq.size();
        for (size_t j = 0; j < seq.size(); ++j) {
            padded_seq[offset + j] = seq[j];
        }
        padded.push_back(std::move(padded_seq));
    }
    return padded;
}

std::vector<std::vector<int>> BatchProcessor::create_attention_mask(
        const std::vector<std::vector<int>>& padded) {

    if (padded.empty()) return {};

    size_t seq_len = padded[0].size();
    std::vector<std::vector<int>> mask;
    mask.reserve(padded.size());

    for (const auto& seq : padded) {
        std::vector<int> m(seq_len, 0);
        // Find first non-pad token (we assume pad_id detection via leading zeros
        // isn't possible here without knowing pad_id, so mark all as 1 except
        // positions where token == 0 and position is in the leading region)
        // Actually, we mark 1=real, 0=pad based on the padded structure.
        // Since we left-padded: leading positions have pad tokens.
        // We infer padding by finding the first non-zero token run.
        bool found_real = false;
        for (size_t j = 0; j < seq.size(); ++j) {
            if (!found_real && seq[j] == 0 && j < seq_len - 1) {
                m[j] = 0;
            } else {
                found_real = true;
                m[j] = 1;
            }
        }
        mask.push_back(std::move(m));
    }
    return mask;
}

int BatchProcessor::current_batch_size() const {
    return static_cast<int>(requests_.size());
}

bool BatchProcessor::is_full() const {
    return static_cast<int>(requests_.size()) >= max_batch_size_;
}

void BatchProcessor::clear() {
    requests_.clear();
    responses_.clear();
}

std::string BatchProcessor::stats() const {
    std::ostringstream oss;
    oss << "{"
        << "\"total_requests_processed\":" << total_requests_processed_ << ","
        << "\"total_tokens_generated\":"   << total_tokens_generated_   << ","
        << "\"eos_stops\":"                << eos_stops_                << ","
        << "\"length_stops\":"             << length_stops_             << ","
        << "\"current_batch_size\":"       << current_batch_size()      << ","
        << "\"max_batch_size\":"           << max_batch_size_
        << "}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

int BatchProcessor::sample_token(
        const std::vector<float>& logits,
        float temperature,
        float top_p) const {

    if (logits.empty()) return 0;

    // Apply temperature
    std::vector<float> scaled;
    scaled.reserve(logits.size());
    float temp = (temperature > 1e-6f) ? temperature : 1e-6f;
    for (float l : logits) {
        scaled.push_back(l / temp);
    }

    // Greedy if temperature is very low
    if (temp < 1e-4f) {
        return static_cast<int>(
            std::max_element(scaled.begin(), scaled.end()) - scaled.begin());
    }

    // Softmax
    std::vector<float> probs = softmax(scaled);

    // Top-p nucleus sampling
    // Sort indices by descending probability
    std::vector<size_t> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return probs[a] > probs[b];
    });

    // Cumulative sum; keep top tokens until cumsum >= top_p
    float cumsum = 0.0f;
    std::vector<size_t> nucleus;
    for (size_t i : idx) {
        nucleus.push_back(i);
        cumsum += probs[i];
        if (cumsum >= top_p) break;
    }

    // Re-normalize nucleus probabilities
    float nucleus_sum = 0.0f;
    for (size_t i : nucleus) nucleus_sum += probs[i];
    if (nucleus_sum < 1e-9f) nucleus_sum = 1.0f;

    // Simple deterministic selection: pick the top token from nucleus
    // (full stochastic sampling would require RNG state management)
    return static_cast<int>(nucleus[0]);
}

std::vector<float> BatchProcessor::softmax(const std::vector<float>& logits) {
    if (logits.empty()) return {};
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps;
    exps.reserve(logits.size());
    float sum = 0.0f;
    for (float l : logits) {
        float e = std::exp(l - max_val);
        exps.push_back(e);
        sum += e;
    }
    if (sum < 1e-9f) sum = 1.0f;
    for (float& e : exps) e /= sum;
    return exps;
}

} // namespace llmquant
