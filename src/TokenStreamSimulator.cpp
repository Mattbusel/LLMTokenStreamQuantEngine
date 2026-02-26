#include "TokenStreamSimulator.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace llmquant {

TokenStreamSimulator::TokenStreamSimulator(const Config& config)
    : config_(config), ring_buffer_(config_.buffer_size) {
}

TokenStreamSimulator::~TokenStreamSimulator() {
    stop();
}

void TokenStreamSimulator::start() {
    if (running_.load()) return;
    running_ = true;
    worker_thread_ = std::thread(&TokenStreamSimulator::stream_worker, this);
}

void TokenStreamSimulator::stop() {
    running_ = false;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void TokenStreamSimulator::set_token_callback(TokenCallback callback) {
    callback_ = std::move(callback);
}

void TokenStreamSimulator::load_tokens_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open token file: " + filepath);
    }
    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) tokens.push_back(token);
    }
    load_tokens_from_memory(tokens);
}

void TokenStreamSimulator::load_tokens_from_memory(const std::vector<std::string>& tokens) {
    {
        std::lock_guard<std::mutex> lock(load_mutex_);
        source_tokens_ = tokens;
    }
    // Pre-fill the ring buffer.
    ring_buffer_.clear();
    for (const auto& t : tokens) {
        if (!ring_buffer_.try_push(t)) break;   // buffer full — stream_worker will refill
    }
}

void TokenStreamSimulator::stream_worker() {
    size_t source_idx = 0;

    while (running_.load()) {
        std::string token_text;

        if (!ring_buffer_.try_pop(token_text)) {
            // Ring empty: refill from source_tokens_ in a tight loop, then retry.
            {
                std::lock_guard<std::mutex> lock(load_mutex_);
                if (!source_tokens_.empty()) {
                    for (size_t k = 0; k < config_.buffer_size / 2; ++k) {
                        const std::string& t = source_tokens_[source_idx % source_tokens_.size()];
                        if (!ring_buffer_.try_push(t)) {
                            stats_.ring_buffer_drops++;
                            break;
                        }
                        ++source_idx;
                    }
                }
            }
            // Still nothing — interval sleep and retry.
            if (!ring_buffer_.try_pop(token_text)) {
                std::this_thread::sleep_for(config_.token_interval);
                continue;
            }
        }

        uint64_t seq = current_sequence_.fetch_add(1);
        Token token(std::move(token_text), seq);

        if (callback_) {
            auto start = std::chrono::high_resolution_clock::now();
            callback_(token);
            auto end   = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            stats_.avg_latency_us = latency.count();
            stats_.max_latency_us = std::max(stats_.max_latency_us.load(),
                                             static_cast<uint64_t>(latency.count()));
        }

        stats_.tokens_emitted++;
        std::this_thread::sleep_for(config_.token_interval);
    }
}

} // namespace llmquant
