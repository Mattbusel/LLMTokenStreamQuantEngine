#include "AttentionSink.h"
#include <algorithm>

namespace llmquant {

AttentionSink::AttentionSink(Config config) : config_(std::move(config)) {
    sink_slots_.reserve(config_.sink_size);
}

void AttentionSink::push_kv(uint32_t token_id, std::vector<float> key, std::vector<float> value) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_pushed_++;

    KvSlot slot{token_id, std::move(key), std::move(value), next_position_++};

    if (sink_slots_.size() < config_.sink_size) {
        // Still filling sink
        sink_slots_.push_back(std::move(slot));
    } else {
        // Add to sliding window
        if (window_slots_.size() >= config_.window_size) {
            window_slots_.pop_front();
            total_evicted_++;
        }
        window_slots_.push_back(std::move(slot));
    }
}

std::vector<const KvSlot*> AttentionSink::active_slots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const KvSlot*> result;
    result.reserve(sink_slots_.size() + window_slots_.size());
    for (const auto& s : sink_slots_) result.push_back(&s);
    for (const auto& s : window_slots_) result.push_back(&s);
    return result;
}

size_t AttentionSink::active_length() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sink_slots_.size() + window_slots_.size();
}

size_t AttentionSink::evicted_count() const {
    return total_evicted_.load();
}

std::vector<bool> AttentionSink::attention_mask() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<bool> mask(sink_slots_.size() + window_slots_.size(), true);
    return mask;  // all active slots can attend to each other
}

void AttentionSink::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    sink_slots_.clear();
    window_slots_.clear();
    next_position_ = 0;
}

AttentionSink::Stats AttentionSink::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = total_pushed_.load();
    size_t evicted = total_evicted_.load();
    size_t active = sink_slots_.size() + window_slots_.size();
    double rate = total > 0 ? static_cast<double>(evicted) / total : 0.0;
    return {total, active, evicted, rate};
}

} // namespace llmquant
