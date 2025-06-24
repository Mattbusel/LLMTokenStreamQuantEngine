#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>

namespace llmquant {

struct Token {
    std::string text;
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_id;
    
    Token(const std::string& t, uint64_t seq_id) 
        : text(t), timestamp(std::chrono::high_resolution_clock::now()), sequence_id(seq_id) {}
};

using TokenCallback = std::function<void(const Token&)>;

class TokenStreamSimulator {
public:
    struct Config {
        std::chrono::microseconds token_interval{10000}; // 10ms
        size_t buffer_size{1024};
        bool use_memory_stream{true};
        std::string data_file_path;
    };

    explicit TokenStreamSimulator(const Config& config);
    ~TokenStreamSimulator();

    void start();
    void stop();
    void set_token_callback(TokenCallback callback);
    void load_tokens_from_file(const std::string& filepath);
    void load_tokens_from_memory(const std::vector<std::string>& tokens);
    
    struct Stats {
        std::atomic<uint64_t> tokens_emitted{0};
        std::atomic<uint64_t> avg_latency_us{0};
        std::atomic<uint64_t> max_latency_us{0};
    };
    
    const Stats& get_stats() const { return stats_; }

private:
    void stream_worker();
    
    Config config_;
    std::vector<std::string> token_buffer_;
    TokenCallback callback_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> current_sequence_{0};
    Stats stats_;
    mutable std::mutex buffer_mutex_;
};

} // namespace llmquant
