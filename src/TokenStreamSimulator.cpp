#include "TokenStreamSimulator.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace llmquant {

TokenStreamSimulator::TokenStreamSimulator(const Config& config) 
    : config_(config) {
    token_buffer_.reserve(config_.buffer_size);
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
    
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    token_buffer_.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            token_buffer_.push_back(token);
        }
    }
}

void TokenStreamSimulator::load_tokens_from_memory(const std::vector<std::string>& tokens) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    token_buffer_ = tokens;
}

void TokenStreamSimulator::stream_worker() {
    while (running_.load()) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        if (token_buffer_.empty()) {
            std::this_thread::sleep_for(config_.token_interval);
            continue;
        }
        
        // Emit token
        size_t index = current_sequence_.load() % token_buffer_.size();
        Token token(token_buffer_[index], current_sequence_.fetch_add(1));
        
        if (callback_) {
            auto start = std::chrono::high_resolution_clock::now();
            callback_(token);
            auto end = std::chrono::high_resolution_clock::now();
            
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

