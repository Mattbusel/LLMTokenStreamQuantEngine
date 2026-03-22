#include "NarrativeChangeDetector.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

NarrativeChangeDetector::NarrativeChangeDetector(Config config)
    : config_(config) {
    int min_fill = (config_.min_fill <= 0) ? config_.window_size : config_.min_fill;
    config_.min_fill = min_fill;
    ring_b_.assign(config_.window_size, 0);
    config_break_threshold_.store(config_.break_threshold, std::memory_order_relaxed);
    // Start counter at window_size-1 so the first fill triggers an immediate
    // promotion — the first full window becomes window_a_ without delay.
    tokens_in_window_a_ = config_.window_size - 1;
}

// Cosine similarity between window_a_ and window_b_.
// Caller must hold mutex_.
double NarrativeChangeDetector::compute_cosine_locked() const {
    if (window_a_.empty() || window_b_.empty()) return 1.0;

    double dot  = 0.0;
    double na2  = 0.0;
    double nb2  = 0.0;

    // Σ a_i² and Σ a_i * b_i (only for tokens in a).
    for (auto& [tok, cnt_a] : window_a_) {
        na2 += static_cast<double>(cnt_a) * cnt_a;
        auto it = window_b_.find(tok);
        if (it != window_b_.end())
            dot += static_cast<double>(cnt_a) * it->second;
    }
    // Σ b_i²
    for (auto& [tok, cnt_b] : window_b_)
        nb2 += static_cast<double>(cnt_b) * cnt_b;

    double denom = std::sqrt(na2) * std::sqrt(nb2);
    return (denom < 1e-12) ? 1.0 : dot / denom;
}

void NarrativeChangeDetector::update_similarity_locked(double sim) {
    // EMA update.
    double alpha = config_.ema_alpha;
    double prev  = similarity_ema_.load(std::memory_order_relaxed);
    similarity_ema_.store(prev + alpha * (sim - prev), std::memory_order_relaxed);
    similarity_.store(sim, std::memory_order_relaxed);

    if (sim < config_break_threshold_.load(std::memory_order_relaxed)) {
        breaks_.fetch_add(1, std::memory_order_relaxed);
        // Invoke callback outside lock to avoid re-entrancy.
        // We'll store a local copy and call after the lock section.
    }
}

void NarrativeChangeDetector::record(uint64_t token_hash) noexcept {
    total_.fetch_add(1, std::memory_order_relaxed);

    bool is_break = false;
    double sim    = 1.0;
    BreakCallback cb_copy;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // --- Slide window_b_ ---
        if (ring_fill_ == config_.window_size) {
            // Evict oldest token from window_b_.
            uint64_t old = ring_b_[ring_head_];
            auto it = window_b_.find(old);
            if (it != window_b_.end()) {
                if (--(it->second) == 0) window_b_.erase(it);
            }
        }
        ring_b_[ring_head_] = token_hash;
        ring_head_ = (ring_head_ + 1) % config_.window_size;
        if (ring_fill_ < config_.window_size) ++ring_fill_;
        ++window_b_[token_hash];

        // --- Promote window_b_ to window_a_ once per full window of new tokens ---
        // tokens_in_window_a_ starts at window_size-1 so the FIRST time ring_fill_
        // reaches window_size, the increment brings it to window_size and we
        // promote immediately.  After the first promotion it resets to 0 and we
        // wait another full window before the next promotion.
        if (ring_fill_ == config_.window_size) {
            if (++tokens_in_window_a_ >= config_.window_size) {
                window_a_ = window_b_;
                tokens_in_window_a_ = 0;
            }
        }

        // --- Compute similarity when both windows have enough data ---
        if (ring_fill_ >= config_.min_fill && !window_a_.empty()) {
            sim = compute_cosine_locked();
            update_similarity_locked(sim);
            is_break = (sim < config_break_threshold_.load(std::memory_order_relaxed));
            if (is_break) cb_copy = break_cb_;
        }
    }

    if (is_break && cb_copy) cb_copy(sim);
}

void NarrativeChangeDetector::set_break_callback(BreakCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    break_cb_ = std::move(cb);
}

void NarrativeChangeDetector::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    config_ = config;
    int min_fill = (config_.min_fill <= 0) ? config_.window_size : config_.min_fill;
    config_.min_fill = min_fill;
    config_break_threshold_.store(config_.break_threshold, std::memory_order_relaxed);
    ring_b_.assign(config_.window_size, 0);
    ring_head_ = 0;
    ring_fill_ = 0;
    window_a_.clear();
    window_b_.clear();
    tokens_in_window_a_ = 0;
    similarity_.store(1.0, std::memory_order_relaxed);
    similarity_ema_.store(1.0, std::memory_order_relaxed);
    breaks_.store(0, std::memory_order_relaxed);
}

NarrativeChangeDetector::Config NarrativeChangeDetector::get_config() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return config_;
}

void NarrativeChangeDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(ring_b_.begin(), ring_b_.end(), 0);
    ring_head_ = 0;
    ring_fill_ = 0;
    window_a_.clear();
    window_b_.clear();
    tokens_in_window_a_ = 0;
    total_.store(0, std::memory_order_relaxed);
    breaks_.store(0, std::memory_order_relaxed);
    similarity_.store(1.0, std::memory_order_relaxed);
    similarity_ema_.store(1.0, std::memory_order_relaxed);
}

std::string NarrativeChangeDetector::to_stats_json() const {
    double sim     = similarity_.load(std::memory_order_relaxed);
    double sim_ema = similarity_ema_.load(std::memory_order_relaxed);
    uint64_t brks  = breaks_.load(std::memory_order_relaxed);
    uint64_t tot   = total_.load(std::memory_order_relaxed);

    std::ostringstream o;
    o << std::fixed;
    o.precision(4);
    o << "{\"similarity\":"     << sim
      << ",\"similarity_ema\":" << sim_ema
      << ",\"is_break\":"       << (is_narrative_break() ? "true" : "false")
      << ",\"total_breaks\":"   << brks
      << ",\"total_recorded\":" << tot << "}";
    return o.str();
}

} // namespace llmquant
