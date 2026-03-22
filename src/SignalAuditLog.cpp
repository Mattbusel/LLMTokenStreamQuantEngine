#include "SignalAuditLog.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <spdlog/spdlog.h>
#include <sstream>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace llmquant {

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

static std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// SignalAuditLog — construction / destruction
// ---------------------------------------------------------------------------

SignalAuditLog::SignalAuditLog(Config config)
    : config_(std::move(config))
{
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&SignalAuditLog::writer_thread, this);
}

SignalAuditLog::~SignalAuditLog() {
    // Signal the writer thread to drain the queue and exit.
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stop_requested_.store(true, std::memory_order_relaxed);
    }
    queue_cv_.notify_one();
    if (thread_.joinable()) thread_.join();

    if (file_) {
        std::fflush(file_);
        std::fclose(file_);
        file_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void SignalAuditLog::log_signal(const TradeSignal& signal,
                                bool               passed,
                                const std::string& reason)
{
    if (!passed && !config_.log_blocked) return;
    if ( passed && !config_.log_passed)  return;

    std::string json = build_json(signal, passed, reason);
    records_enqueued_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        if (config_.max_queue_depth > 0 && queue_.size() >= config_.max_queue_depth) {
            records_dropped_.fetch_add(1, std::memory_order_relaxed);
            // Log at most once per second to avoid spamming spdlog from the hot path.
            static std::atomic<int64_t> last_warn_ns{0};
            auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            int64_t expected = last_warn_ns.load(std::memory_order_relaxed);
            if (now_ns - expected > 1'000'000'000LL) {
                if (last_warn_ns.compare_exchange_weak(expected, now_ns,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    spdlog::warn("[audit_log] queue full (depth={}); dropping record",
                                 config_.max_queue_depth);
                }
            }
            return;
        }
        queue_.push(std::move(json));
    }
    queue_cv_.notify_one();
}

// ---------------------------------------------------------------------------
// Background writer thread
// ---------------------------------------------------------------------------

void SignalAuditLog::writer_thread() {
    // Open or create the output file.
    file_ = std::fopen(config_.filepath.c_str(), "a");
    if (!file_) {
        spdlog::error("[audit_log] cannot open '{}': {}", config_.filepath, std::strerror(errno));
        running_.store(false, std::memory_order_relaxed);
        return;
    }
    // Determine current file size for rotation tracking.
    if (std::fseek(file_, 0, SEEK_END) == 0) {
        file_bytes_ = static_cast<uint64_t>(std::ftell(file_));
    }
    spdlog::info("[audit_log] opened '{}' ({} bytes existing)", config_.filepath, file_bytes_);

    while (true) {
        std::string record;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this] {
                return !queue_.empty() || stop_requested_.load(std::memory_order_relaxed);
            });
            if (queue_.empty()) break;  // stop_requested and queue drained
            record = std::move(queue_.front());
            queue_.pop();
        }
        write_record(record);
    }

    // Flush remaining records after stop signal.
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        while (!queue_.empty()) {
            write_record(queue_.front());
            queue_.pop();
        }
    }

    if (file_) {
        std::fflush(file_);
        std::fclose(file_);
        file_ = nullptr;
    }
    running_.store(false, std::memory_order_relaxed);
}

void SignalAuditLog::write_record(const std::string& json_line) {
    if (!file_) return;

    rotate_if_needed();

    if (std::fputs(json_line.c_str(), file_) < 0 ||
        std::fputc('\n', file_) == EOF) {
        spdlog::warn("[audit_log] write error: {}", std::strerror(errno));
        return;
    }
    // Flush after each record so the file is readable mid-run.
    std::fflush(file_);

    file_bytes_ += static_cast<uint64_t>(json_line.size()) + 1U;
    records_written_.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Log rotation
// ---------------------------------------------------------------------------

void SignalAuditLog::rotate_if_needed() {
    if (config_.rotate_bytes == 0 || file_bytes_ < config_.rotate_bytes) return;

    std::fflush(file_);
    std::fclose(file_);
    file_ = nullptr;
    file_bytes_ = 0;

    // Rename: base -> base.1, base.1 -> base.2, etc.
    // Shift existing rotated files up.
    if (config_.keep_files > 0) {
        for (int i = config_.keep_files - 1; i >= 1; --i) {
            std::string src = config_.filepath + "." + std::to_string(i);
            std::string dst = config_.filepath + "." + std::to_string(i + 1);
            // Remove destination if it exists, then rename.
            std::remove(dst.c_str());
            std::rename(src.c_str(), dst.c_str());
        }
    }
    std::string rotated = config_.filepath + ".1";
    std::remove(rotated.c_str());
    if (std::rename(config_.filepath.c_str(), rotated.c_str()) != 0) {
        spdlog::warn("[audit_log] rotate rename failed: {}", std::strerror(errno));
    }

    prune_old_files();

    file_ = std::fopen(config_.filepath.c_str(), "a");
    if (!file_) {
        spdlog::error("[audit_log] cannot reopen '{}' after rotate: {}",
                      config_.filepath, std::strerror(errno));
        return;
    }
    rotations_.fetch_add(1, std::memory_order_relaxed);
    spdlog::info("[audit_log] rotated to '{}' (rotation #{})", rotated, rotations_.load());
}

void SignalAuditLog::prune_old_files() {
    if (config_.keep_files <= 0) return;
    // Delete files beyond keep_files (numbered keep_files+1, keep_files+2, …).
    for (int i = config_.keep_files + 1; i <= config_.keep_files + 10; ++i) {
        std::string path = config_.filepath + "." + std::to_string(i);
        if (std::remove(path.c_str()) != 0) break;  // stop on first missing file
    }
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

std::string SignalAuditLog::build_json(const TradeSignal& signal,
                                        bool               passed,
                                        const std::string& reason)
{
    // Serialise directly using snprintf for consistent float formatting and
    // to avoid locale-dependent std::ostringstream decimal separators.
    char buf[512];
    // reason is JSON-escaped separately to handle arbitrary strings.
    std::string reason_esc = escape_json_string(reason);
    std::snprintf(buf, sizeof(buf),
        "{\"ts_ns\":%" PRIu64
        ",\"passed\":%s"
        ",\"reason\":\"%s\""
        ",\"bias\":%.6f"
        ",\"vol\":%.6f"
        ",\"conf\":%.6f"
        ",\"spread\":%.6f"
        ",\"quality\":%.6f"
        ",\"lat_us\":%.2f"
        ",\"strategy\":%d}",
        signal.timestamp_ns,
        passed ? "true" : "false",
        reason_esc.c_str(),
        signal.delta_bias_shift,
        signal.volatility_adjustment,
        signal.confidence,
        signal.spread_modifier,
        signal.signal_quality,
        signal.latency_us,
        signal.strategy_toggle);
    return buf;
}

} // namespace llmquant
