#include "SignalReplayBuffer.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <iomanip>

namespace llmquant {

SignalReplayBuffer::SignalReplayBuffer(Config cfg)
    : cfg_(std::move(cfg)), capacity_(cfg_.capacity)
{
    ring_.resize(capacity_);
}

void SignalReplayBuffer::push(const TradeSignal& signal) noexcept {
    std::function<void(const TradeSignal&)> cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        uint64_t pos = write_pos_.fetch_add(1, std::memory_order_relaxed);
        ring_[pos % capacity_] = signal;
        total_.fetch_add(1, std::memory_order_relaxed);
        cb = cfg_.on_push;
    }
    if (cb) cb(signal);
}

std::vector<TradeSignal> SignalReplayBuffer::snapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t pos   = write_pos_.load(std::memory_order_relaxed);
    uint64_t total = total_.load(std::memory_order_relaxed);
    uint32_t count = static_cast<uint32_t>(std::min(total, static_cast<uint64_t>(capacity_)));
    if (count == 0) return {};

    std::vector<TradeSignal> out;
    out.reserve(count);
    // Walk from oldest to newest.
    uint64_t oldest = (total > capacity_) ? pos : 0;
    for (uint32_t i = 0; i < count; ++i)
        out.push_back(ring_[(oldest + i) % capacity_]);
    return out;
}

void SignalReplayBuffer::replay(const std::function<void(const TradeSignal&)>& cb) const {
    if (!cb) return;
    for (const auto& s : snapshot()) cb(s);
}

uint32_t SignalReplayBuffer::size() const noexcept {
    uint64_t t = total_.load(std::memory_order_relaxed);
    return static_cast<uint32_t>(std::min(t, static_cast<uint64_t>(capacity_)));
}

void SignalReplayBuffer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    write_pos_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalReplayBuffer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_      = cfg;
    capacity_ = cfg_.capacity;
    ring_.assign(capacity_, TradeSignal{});
    write_pos_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

std::string SignalReplayBuffer::to_stats_json() const {
    uint32_t sz = size();
    uint64_t tot = total_.load(std::memory_order_relaxed);
    double last_bias = 0.0, last_conf = 0.0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (sz > 0) {
            uint64_t pos = write_pos_.load(std::memory_order_relaxed);
            const auto& last = ring_[(pos - 1 + capacity_) % capacity_];
            last_bias = last.delta_bias_shift;
            last_conf = last.confidence;
        }
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"capacity\":" << capacity_
       << ",\"size\":"         << sz
       << ",\"total_pushed\":" << tot
       << ",\"last_bias\":"    << last_bias
       << ",\"last_conf\":"    << last_conf
       << "}";
    return ss.str();
}

} // namespace llmquant
