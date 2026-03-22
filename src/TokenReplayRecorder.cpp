#include "TokenReplayRecorder.h"

#include <cerrno>
#include <cstring>
#include <spdlog/spdlog.h>
#include <thread>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

TokenReplayRecorder::TokenReplayRecorder(std::string filepath)
    : filepath_(std::move(filepath)) {}

TokenReplayRecorder::~TokenReplayRecorder() {
    close();
}

// ---------------------------------------------------------------------------
// Recording lifecycle
// ---------------------------------------------------------------------------

bool TokenReplayRecorder::start_recording() {
    std::lock_guard<std::mutex> lk(write_mutex_);

    if (recording_.load(std::memory_order_relaxed)) {
        spdlog::warn("[replay_recorder] start_recording() called while already recording");
        return false;
    }

    file_ = std::fopen(filepath_.c_str(), "wb");
    if (!file_) {
        spdlog::error("[replay_recorder] cannot open '{}': {}", filepath_, std::strerror(errno));
        return false;
    }

    // Write placeholder file header (frame_count filled in on close).
    FileHeader hdr{};
    hdr.start_ns = 0;  // filled on first record()
    if (std::fwrite(&hdr, sizeof(hdr), 1, file_) != 1) {
        spdlog::error("[replay_recorder] failed to write file header");
        std::fclose(file_);
        file_ = nullptr;
        return false;
    }
    std::fflush(file_);

    frames_written_.store(0, std::memory_order_relaxed);
    start_ns_ = 0;
    recording_.store(true, std::memory_order_release);
    spdlog::info("[replay_recorder] recording started → '{}'", filepath_);
    return true;
}

void TokenReplayRecorder::record(const std::string& token,
                                 double             sentiment_score,
                                 uint64_t           timestamp_ns)
{
    if (!recording_.load(std::memory_order_relaxed)) return;

    if (timestamp_ns == 0) {
        auto now = std::chrono::system_clock::now();
        timestamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count());
    }

    std::lock_guard<std::mutex> lk(write_mutex_);
    if (!recording_.load(std::memory_order_relaxed) || !file_) return;

    // Capture start_ns on first frame.
    if (start_ns_ == 0) start_ns_ = timestamp_ns;

    FrameHeader frame{};
    frame.timestamp_ns    = timestamp_ns;
    frame.sentiment_score = sentiment_score;
    frame.token_len       = static_cast<uint32_t>(token.size());
    frame.flags           = 0;

    if (std::fwrite(&frame, sizeof(frame), 1, file_) != 1) {
        spdlog::error("[replay_recorder] write error (frame header)");
        return;
    }
    if (!token.empty()) {
        if (std::fwrite(token.data(), 1, token.size(), file_) != token.size()) {
            spdlog::error("[replay_recorder] write error (token bytes)");
            return;
        }
    }

    frames_written_.fetch_add(1, std::memory_order_relaxed);
}

void TokenReplayRecorder::stop_recording() {
    std::lock_guard<std::mutex> lk(write_mutex_);
    if (!recording_.load(std::memory_order_relaxed)) return;
    recording_.store(false, std::memory_order_release);

    if (!file_) return;

    // Update frame_count and start_ns in the file header.
    uint64_t count = frames_written_.load(std::memory_order_relaxed);
    std::fseek(file_, 0, SEEK_SET);
    FileHeader hdr{};
    hdr.frame_count = count;
    hdr.start_ns    = start_ns_;
    if (std::fwrite(&hdr, sizeof(hdr), 1, file_) != 1) {
        spdlog::error("[replay_recorder] failed to finalise file header");
    }
    std::fflush(file_);
    spdlog::info("[replay_recorder] recording stopped — {} frames written", count);
}

void TokenReplayRecorder::close() {
    stop_recording();
    std::lock_guard<std::mutex> lk(write_mutex_);
    if (file_) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Static playback API
// ---------------------------------------------------------------------------

bool TokenReplayRecorder::read_file_header(const std::string& filepath,
                                           FileHeader&        hdr) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;
    bool ok = (std::fread(&hdr, sizeof(hdr), 1, f) == 1) &&
              (hdr.magic == kFileMagic);
    std::fclose(f);
    return ok;
}

bool TokenReplayRecorder::load_recording(const std::string&     filepath,
                                         std::vector<TokenEvent>& events) {
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) {
        spdlog::error("[replay_recorder] load: cannot open '{}'", filepath);
        return false;
    }

    FileHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != kFileMagic) {
        spdlog::error("[replay_recorder] load: invalid magic in '{}'", filepath);
        std::fclose(f);
        return false;
    }

    events.clear();
    events.reserve(static_cast<size_t>(hdr.frame_count));

    FrameHeader frame{};
    while (std::fread(&frame, sizeof(frame), 1, f) == 1) {
        TokenEvent ev;
        ev.timestamp_ns    = frame.timestamp_ns;
        ev.sentiment_score = frame.sentiment_score;
        ev.flags           = frame.flags;

        if (frame.token_len > 0) {
            ev.token.resize(frame.token_len);
            if (std::fread(ev.token.data(), 1, frame.token_len, f) != frame.token_len) {
                spdlog::error("[replay_recorder] load: truncated token data");
                std::fclose(f);
                return false;
            }
        }
        events.push_back(std::move(ev));
    }

    std::fclose(f);
    spdlog::info("[replay_recorder] loaded {} frames from '{}'", events.size(), filepath);
    return true;
}

int64_t TokenReplayRecorder::replay(const std::string&      filepath,
                                    const PlaybackCallback& cb,
                                    double                  speed_multiplier)
{
    std::vector<TokenEvent> events;
    if (!load_recording(filepath, events)) return -1;

    uint64_t prev_ts = 0;
    uint64_t idx     = 0;
    for (const auto& ev : events) {
        if (speed_multiplier > 0.0 && prev_ts > 0 && ev.timestamp_ns > prev_ts) {
            auto delay_ns = static_cast<uint64_t>(
                static_cast<double>(ev.timestamp_ns - prev_ts) / speed_multiplier);
            if (delay_ns > 0)
                std::this_thread::sleep_for(std::chrono::nanoseconds(delay_ns));
        }
        try { cb(ev, idx); } catch (...) {}
        prev_ts = ev.timestamp_ns;
        ++idx;
    }
    return static_cast<int64_t>(events.size());
}

} // namespace llmquant
