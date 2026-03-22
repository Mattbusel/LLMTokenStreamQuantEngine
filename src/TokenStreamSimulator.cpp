#include "TokenStreamSimulator.h"
#include <fstream>
#include <iomanip>
#include <sstream>

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

        // 1. Try to pop a token from the ring buffer.
        if (!ring_buffer_.try_pop(token_text)) {
            // 2. Ring empty: refill from source_tokens_ in a tight loop, then retry.
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
            // 3. Still empty after refill: interruptible back-off sleep, then retry loop.
            if (!ring_buffer_.try_pop(token_text)) {
                auto deadline = std::chrono::steady_clock::now() + config_.token_interval;
                const auto slice = std::chrono::milliseconds{50};
                while (running_.load() && std::chrono::steady_clock::now() < deadline) {
                    auto remaining = deadline - std::chrono::steady_clock::now();
                    std::this_thread::sleep_for(remaining < slice ? remaining : slice);
                }
                continue;  // Do NOT fall through to the dispatch sleep.
            }
        }

        // 4. We have a token: dispatch via callback, update stats, THEN sleep (normal cadence).
        uint64_t seq = current_sequence_.fetch_add(1, std::memory_order_relaxed);
        Token token(std::move(token_text), seq);

        if (callback_) {
            auto start = std::chrono::high_resolution_clock::now();
            callback_(token);
            auto end   = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            // Welford-style running average for callback latency.
            uint64_t count = stats_.tokens_emitted.load(std::memory_order_relaxed) + 1;  // +1 because tokens_emitted increments after this
            double old_avg = static_cast<double>(stats_.avg_latency_us.load(std::memory_order_relaxed));
            double new_avg = old_avg + (static_cast<double>(latency.count()) - old_avg)
                             / static_cast<double>(count);
            stats_.avg_latency_us.store(static_cast<uint64_t>(new_avg), std::memory_order_relaxed);
            uint64_t cur_max = stats_.max_latency_us.load(std::memory_order_relaxed);
            uint64_t new_max = std::max(cur_max, static_cast<uint64_t>(latency.count()));
            stats_.max_latency_us.store(new_max, std::memory_order_relaxed);
        }

        stats_.tokens_emitted.fetch_add(1, std::memory_order_relaxed);
        // Interruptible cadence sleep: break into 50ms slices so stop() returns
        // promptly even when token_interval is in the seconds range.
        // Clamp each slice to the remaining time so short intervals (< 50ms)
        // are honoured accurately.
        {
            auto deadline = std::chrono::steady_clock::now() + config_.token_interval;
            const auto slice = std::chrono::milliseconds{50};
            while (running_.load() && std::chrono::steady_clock::now() < deadline) {
                auto remaining = deadline - std::chrono::steady_clock::now();
                std::this_thread::sleep_for(remaining < slice ? remaining : slice);
            }
        }
    }
}

std::string TokenStreamSimulator::format_stats() const {
    uint64_t emitted = stats_.tokens_emitted.load(std::memory_order_relaxed);
    uint64_t drops   = stats_.ring_buffer_drops.load(std::memory_order_relaxed);
    uint64_t avg_lat = stats_.avg_latency_us.load(std::memory_order_relaxed);
    uint64_t max_lat = stats_.max_latency_us.load(std::memory_order_relaxed);
    double drop_rate = get_drop_rate();

    std::ostringstream oss;
    oss << "emitted=" << emitted
        << " drops=" << drops
        << " avg_lat=" << avg_lat << "\xc2\xb5s"
        << " max_lat=" << max_lat << "\xc2\xb5s"
        << " drop_rate=" << std::fixed << std::setprecision(3) << drop_rate;
    return oss.str();
}

} // namespace llmquant
