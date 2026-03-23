#pragma once
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <optional>

namespace llmquant {

/// Attention sink: keep first N tokens' KV cache fixed (they attract attention)
/// Sliding window: keep last W tokens active
/// Total active = sink_size + window_size
struct KvSlot {
    uint32_t token_id;
    std::vector<float> key;    // [n_heads * head_dim]
    std::vector<float> value;  // [n_heads * head_dim]
    uint32_t position;         // original sequence position
};

class AttentionSink {
public:
    struct Config {
        size_t sink_size = 4;        // number of initial tokens to keep as sinks
        size_t window_size = 512;    // sliding window of recent tokens
        size_t n_heads = 32;
        size_t head_dim = 128;
        bool normalize_sink = true;  // apply RoPE correction to sink positions
    };

    explicit AttentionSink(Config config = {});
    ~AttentionSink() = default;

    /// Add a new token's KV to the cache
    void push_kv(uint32_t token_id, std::vector<float> key, std::vector<float> value);

    /// Get all active KV slots (sinks + window)
    std::vector<const KvSlot*> active_slots() const;

    /// Total active sequence length
    size_t active_length() const;

    /// Number of evicted tokens (cannot be recalled)
    size_t evicted_count() const;

    /// Current attention mask: which positions attend to which
    /// Returns [active_length] bitmask of valid positions
    std::vector<bool> attention_mask() const;

    /// Reset to empty state
    void reset();

    /// Statistics
    struct Stats {
        size_t total_pushed;
        size_t current_active;
        size_t evicted;
        double eviction_rate;
    };
    Stats stats() const;

private:
    Config config_;
    std::vector<KvSlot> sink_slots_;     // fixed sink tokens
    std::deque<KvSlot> window_slots_;    // sliding window
    mutable std::mutex mutex_;
    std::atomic<size_t> total_pushed_{0};
    std::atomic<size_t> total_evicted_{0};
    uint32_t next_position_{0};
};

} // namespace llmquant
