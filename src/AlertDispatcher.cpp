#include "AlertDispatcher.h"

#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

AlertDispatcher::AlertDispatcher(Config config)
    : config_(std::move(config)) {}

AlertDispatcher::~AlertDispatcher() {
    stop();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool AlertDispatcher::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return false;

    stop_requested_.store(false, std::memory_order_relaxed);
    thread_ = std::thread(&AlertDispatcher::dispatch_loop, this);
    return true;
}

void AlertDispatcher::stop() {
    stop_requested_.store(true, std::memory_order_relaxed);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    running_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Posting
// ---------------------------------------------------------------------------

void AlertDispatcher::post(Severity severity, const std::string& category,
                            const std::string& message) {
    total_posted_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> cfg_lk(sinks_mutex_);
        if (severity < config_.min_severity) return;
    }

    Alert a;
    a.severity  = severity;
    a.category  = category;
    a.message   = message;
    a.timestamp = std::chrono::system_clock::now();

    {
        std::lock_guard<std::mutex> lk(queue_mutex_);

        if (!running_.load(std::memory_order_relaxed)) return;

        if (static_cast<int>(queue_.size()) >= config_.max_queue_depth) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        if (config_.dedup_window_ms > 0 && is_duplicate_locked(a)) {
            dedup_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        queue_.push(std::move(a));
    }
    cv_.notify_one();
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

void AlertDispatcher::add_sink(AlertSink sink) {
    std::lock_guard<std::mutex> lk(sinks_mutex_);
    sinks_.push_back(std::move(sink));
}

void AlertDispatcher::remove_all_sinks() {
    std::lock_guard<std::mutex> lk(sinks_mutex_);
    sinks_.clear();
}

bool AlertDispatcher::add_file_sink(const std::string& filepath) {
    auto file = std::make_shared<std::ofstream>(filepath, std::ios::app);
    if (!file->is_open()) return false;

    add_sink([file](const Alert& a) {
        // ISO-8601 timestamp via time_t
        auto tt = std::chrono::system_clock::to_time_t(a.timestamp);
        char tsbuf[32] = {};
#ifdef _WIN32
        struct tm tminfo{};
        gmtime_s(&tminfo, &tt);
        std::strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%dT%H:%M:%SZ", &tminfo);
#else
        struct tm tminfo{};
        gmtime_r(&tt, &tminfo);
        std::strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%dT%H:%M:%SZ", &tminfo);
#endif

        // Minimal JSON escaping for message string
        auto escape = [](const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                if (c == '"')       { out += "\\\""; }
                else if (c == '\\') { out += "\\\\"; }
                else if (c == '\n') { out += "\\n";  }
                else if (c == '\r') { out += "\\r";  }
                else                { out += c;       }
            }
            return out;
        };

        *file << "{\"ts\":\"" << tsbuf
              << "\",\"sev\":\"" << AlertDispatcher::severity_name(a.severity)
              << "\",\"cat\":\"" << escape(a.category)
              << "\",\"msg\":\"" << escape(a.message) << "\"}\n";
        file->flush();
    });
    return true;
}

void AlertDispatcher::add_spdlog_sink() {
    add_sink([](const Alert& a) {
        const auto& msg = "[" + a.category + "] " + a.message;
        switch (a.severity) {
            case Severity::Debug:    spdlog::debug("{}", msg);    break;
            case Severity::Info:     spdlog::info("{}", msg);     break;
            case Severity::Warn:     spdlog::warn("{}", msg);     break;
            case Severity::Critical: spdlog::critical("{}", msg); break;
        }
    });
}

// ---------------------------------------------------------------------------
// Background dispatch loop
// ---------------------------------------------------------------------------

void AlertDispatcher::dispatch_loop() {
    while (true) {
        Alert a;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            cv_.wait(lk, [this] {
                return !queue_.empty() || stop_requested_.load(std::memory_order_relaxed);
            });

            if (queue_.empty()) break;  // stop requested and queue drained
            a = std::move(queue_.front());
            queue_.pop();
        }

        // Dispatch to all sinks (outside queue lock).
        std::lock_guard<std::mutex> slk(sinks_mutex_);
        for (auto& sink : sinks_) {
            try { sink(a); } catch (...) { /* sinks must not throw */ }
        }
        dispatched_count_.fetch_add(1, std::memory_order_relaxed);
    }
    running_.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

bool AlertDispatcher::is_duplicate_locked(const Alert& a) {
    const std::string key = a.category + '\0' + a.message;
    const auto now = std::chrono::steady_clock::now();
    const auto window = std::chrono::milliseconds(config_.dedup_window_ms);

    for (auto& entry : dedup_cache_) {
        if (entry.key == key && (now - entry.last_seen) < window) {
            return true;
        }
    }

    // Evict stale entries and add this one.
    dedup_cache_.erase(
        std::remove_if(dedup_cache_.begin(), dedup_cache_.end(),
                       [&](const DedupEntry& e) {
                           return (now - e.last_seen) >= window;
                       }),
        dedup_cache_.end());
    dedup_cache_.push_back({key, now});
    return false;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

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
    o << "{\"total_posted\":"    << total_posted_.load(std::memory_order_relaxed)
      << ",\"dropped\":"         << dropped_count_.load(std::memory_order_relaxed)
      << ",\"deduped\":"         << dedup_count_.load(std::memory_order_relaxed)
      << ",\"dispatched\":"      << dispatched_count_.load(std::memory_order_relaxed)
      << ",\"running\":"         << (running_.load(std::memory_order_relaxed) ? "true" : "false")
      << "}";
    return o.str();
}

void AlertDispatcher::update_config(const Config& config) {
    std::lock_guard<std::mutex> lk(sinks_mutex_);
    config_ = config;
}

} // namespace llmquant
