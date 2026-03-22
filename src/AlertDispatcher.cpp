#include "AlertDispatcher.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

AlertDispatcher::AlertDispatcher(Config config)
    : config_(std::move(config)) {}

AlertDispatcher::~AlertDispatcher() {
    stop();
}

bool AlertDispatcher::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return false;
    stop_requested_.store(false, std::memory_order_relaxed);
    thread_ = std::thread(&AlertDispatcher::dispatch_loop, this);
    spdlog::debug("[alert_dispatcher] started");
    return true;
}

void AlertDispatcher::stop() {
    if (!running_.load(std::memory_order_relaxed)) return;
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stop_requested_.store(true, std::memory_order_relaxed);
    }
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
    running_.store(false, std::memory_order_relaxed);
}

void AlertDispatcher::post(Severity severity,
                           const std::string& category,
                           const std::string& message) {
    total_posted_.fetch_add(1, std::memory_order_relaxed);
    if (static_cast<int>(severity) < static_cast<int>(config_.min_severity)) return;
    if (!running_.load(std::memory_order_relaxed)) return;

    Alert a;
    a.severity = severity;
    a.category = category;
    a.message  = message;

    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        if (is_duplicate_locked(a)) {
            dedup_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (static_cast<int>(queue_.size()) >= config_.max_queue_depth) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        queue_.push(std::move(a));
    }
    cv_.notify_one();
}

bool AlertDispatcher::is_duplicate_locked(const Alert& a) {
    if (config_.dedup_window_ms <= 0) return false;
    std::string key = a.category + "\x1F" + a.message;
    auto now = std::chrono::steady_clock::now();
    auto window = std::chrono::milliseconds(config_.dedup_window_ms);
    for (auto& entry : dedup_cache_) {
        if (entry.key == key) {
            if (now - entry.last_seen < window) return true;
            entry.last_seen = now;
            return false;
        }
    }
    dedup_cache_.push_back({key, now});
    if (dedup_cache_.size() > 512) {
        dedup_cache_.erase(
            std::remove_if(dedup_cache_.begin(), dedup_cache_.end(),
                [&](const DedupEntry& e) { return now - e.last_seen > window * 2; }),
            dedup_cache_.end());
    }
    return false;
}

void AlertDispatcher::dispatch_loop() {
    while (true) {
        Alert a;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            cv_.wait(lk, [this]() {
                return !queue_.empty() || stop_requested_.load(std::memory_order_relaxed);
            });
            if (queue_.empty()) break;
            a = std::move(queue_.front());
            queue_.pop();
        }
        std::vector<AlertSink> sinks_copy;
        {
            std::lock_guard<std::mutex> lk(sinks_mutex_);
            sinks_copy = sinks_;
        }
        for (auto& sink : sinks_copy) {
            try { sink(a); } catch (...) {}
        }
        dispatched_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

void AlertDispatcher::add_sink(AlertSink sink) {
    std::lock_guard<std::mutex> lk(sinks_mutex_);
    sinks_.push_back(std::move(sink));
}

void AlertDispatcher::remove_all_sinks() {
    std::lock_guard<std::mutex> lk(sinks_mutex_);
    sinks_.clear();
}

bool AlertDispatcher::add_file_sink(const std::string& filepath) {
    auto f = std::make_shared<std::ofstream>(filepath, std::ios::app);
    if (!f->is_open()) {
        spdlog::error("[alert_dispatcher] cannot open sink file '{}'", filepath);
        return false;
    }
    add_sink([f](const Alert& a) {
        auto t = std::chrono::system_clock::to_time_t(a.timestamp);
        std::tm tm_buf{};
#ifdef _WIN32
        gmtime_s(&tm_buf, &t);
#else
        gmtime_r(&t, &tm_buf);
#endif
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
        std::string escaped;
        for (char c : a.message) {
            if (c == '"')  escaped += "\\\"";
            else if (c == '\\') escaped += "\\\\";
            else escaped += c;
        }
        *f << "{\"ts\":\"" << ts
           << "\",\"sev\":\"" << severity_name(a.severity)
           << "\",\"cat\":\""  << a.category
           << "\",\"msg\":\""  << escaped << "\"}\n";
        f->flush();
    });
    return true;
}

void AlertDispatcher::add_spdlog_sink() {
    add_sink([](const Alert& a) {
        switch (a.severity) {
            case Severity::Debug:    spdlog::debug("[{}] {}",    a.category, a.message); break;
            case Severity::Info:     spdlog::info("[{}] {}",     a.category, a.message); break;
            case Severity::Warn:     spdlog::warn("[{}] {}",     a.category, a.message); break;
            case Severity::Critical: spdlog::critical("[{}] {}", a.category, a.message); break;
        }
    });
}

const char* AlertDispatcher::severity_name(Severity s) noexcept {
    switch (s) {
        case Severity::Debug:    return "DEBUG";
        case Severity::Info:     return "INFO";
        case Severity::Warn:     return "WARN";
        case Severity::Critical: return "CRITICAL";
    }
    return "UNKNOWN";
}

std::string AlertDispatcher::to_stats_json() const {
    std::ostringstream o;
    o << "{\"running\":"      << (is_running() ? "true" : "false")
      << ",\"total_posted\":" << total_posted_.load(std::memory_order_relaxed)
      << ",\"dispatched\":"   << dispatched_count_.load(std::memory_order_relaxed)
      << ",\"dropped\":"      << dropped_count_.load(std::memory_order_relaxed)
      << ",\"dedup\":"        << dedup_count_.load(std::memory_order_relaxed)
      << "}";
    return o.str();
}

void AlertDispatcher::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    config_ = config;
}

} // namespace llmquant
