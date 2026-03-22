#include "TokenStreamRecorder.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace llmquant {

namespace {

// Castagnoli CRC-32 (polynomial 0xEDB88320).
uint32_t crc32_table[256] = {};
bool crc32_inited = false;

void init_crc32_table() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320u : 0u);
        crc32_table[i] = crc;
    }
    crc32_inited = true;
}

uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(
            system_clock::now().time_since_epoch()).count());
}

} // namespace

// Fixed-size binary header for each record.
#pragma pack(push, 1)
struct RecordHeader {
    uint64_t sequence;
    uint64_t timestamp_ns;
    double   weight;
    uint32_t token_len;
    uint32_t crc32;
};
#pragma pack(pop)
static_assert(sizeof(RecordHeader) == 32, "RecordHeader must be 32 bytes");

/*static*/ uint32_t TokenStreamRecorder::crc32_compute(const void* data,
                                                         std::size_t len) noexcept
{
    if (!crc32_inited) init_crc32_table();
    uint32_t crc = 0xFFFFFFFFu;
    const auto* p = static_cast<const uint8_t*>(data);
    for (std::size_t i = 0; i < len; ++i)
        crc = (crc >> 8) ^ crc32_table[(crc ^ p[i]) & 0xFF];
    return crc ^ 0xFFFFFFFFu;
}

TokenStreamRecorder::TokenStreamRecorder(Config cfg)
    : cfg_(std::move(cfg))
{
    if (!cfg_.path.empty())
        open(cfg_.path);
}

TokenStreamRecorder::~TokenStreamRecorder() {
    close();
}

bool TokenStreamRecorder::open(const std::string& path) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (fp_) std::fclose(static_cast<FILE*>(fp_));

    FILE* f = std::fopen(path.c_str(), "ab");
    if (!f) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_.on_error) cfg_.on_error("fopen failed: " + path);
        return false;
    }
    fp_           = f;
    current_path_ = path;
    is_open_.store(true, std::memory_order_relaxed);
    records_.store(0, std::memory_order_relaxed);
    bytes_.store(0, std::memory_order_relaxed);
    return true;
}

void TokenStreamRecorder::close() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (fp_) {
        std::fflush(static_cast<FILE*>(fp_));
        std::fclose(static_cast<FILE*>(fp_));
        fp_ = nullptr;
    }
    is_open_.store(false, std::memory_order_relaxed);
}

bool TokenStreamRecorder::rotate(const std::string& new_path) {
    uint64_t rec_count;
    std::string old_path;
    std::function<void(const std::string&, const std::string&, uint64_t)> cb;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (fp_) {
            std::fflush(static_cast<FILE*>(fp_));
            std::fclose(static_cast<FILE*>(fp_));
            fp_ = nullptr;
        }
        old_path  = current_path_;
        rec_count = records_.load(std::memory_order_relaxed);
        cb        = cfg_.on_rotate;

        FILE* f = std::fopen(new_path.c_str(), "ab");
        if (!f) {
            errors_.fetch_add(1, std::memory_order_relaxed);
            is_open_.store(false, std::memory_order_relaxed);
            if (cfg_.on_error) cfg_.on_error("rotate fopen failed: " + new_path);
            return false;
        }
        fp_           = f;
        current_path_ = new_path;
        records_.store(0, std::memory_order_relaxed);
        bytes_.store(0, std::memory_order_relaxed);
        is_open_.store(true, std::memory_order_relaxed);
    }
    if (cb) cb(old_path, new_path, rec_count);
    return true;
}

bool TokenStreamRecorder::record(const std::string& token, double weight) noexcept {
    if (!is_open_.load(std::memory_order_relaxed)) return true;  // silent no-op

    std::lock_guard<std::mutex> lk(mutex_);
    if (!fp_) return false;

    RecordHeader hdr{};
    hdr.sequence     = records_.load(std::memory_order_relaxed);
    hdr.timestamp_ns = now_ns();
    hdr.weight       = weight;
    hdr.token_len    = static_cast<uint32_t>(token.size());
    // CRC over first 28 bytes (header without crc field).
    hdr.crc32 = crc32_compute(&hdr, offsetof(RecordHeader, crc32));

    FILE* f = static_cast<FILE*>(fp_);
    if (std::fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_.on_error) cfg_.on_error("fwrite header failed");
        return false;
    }
    if (!token.empty() && std::fwrite(token.data(), 1, token.size(), f) != token.size()) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        if (cfg_.on_error) cfg_.on_error("fwrite token failed");
        return false;
    }

    if (cfg_.sync_write) std::fflush(f);

    uint64_t rec = records_.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t b   = bytes_.fetch_add(sizeof(hdr) + token.size(), std::memory_order_relaxed)
                 + sizeof(hdr) + token.size();

    // Auto-rotate if file size limit exceeded.
    if (cfg_.max_file_bytes > 0 && b >= cfg_.max_file_bytes && !current_path_.empty()) {
        // Build rotated file name: append ".1".
        std::string rot = current_path_ + ".1";
        std::fflush(f);
        std::fclose(f);
        fp_ = nullptr;
        FILE* nf = std::fopen(rot.c_str(), "ab");
        if (nf) {
            fp_           = nf;
            current_path_ = rot;
            records_.store(0, std::memory_order_relaxed);
            bytes_.store(0, std::memory_order_relaxed);
            if (cfg_.on_rotate) cfg_.on_rotate(current_path_, rot, rec);
        } else {
            is_open_.store(false, std::memory_order_relaxed);
            errors_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return true;
}

void TokenStreamRecorder::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

std::string TokenStreamRecorder::to_stats_json() const {
    bool op  = is_open_.load(std::memory_order_relaxed);
    uint64_t rec = records_.load(std::memory_order_relaxed);
    uint64_t by  = bytes_.load(std::memory_order_relaxed);
    uint64_t err = errors_.load(std::memory_order_relaxed);
    std::string path;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        path = current_path_;
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0);
    ss << "{"
       << "\"is_open\":"       << (op ? "true" : "false") << ","
       << "\"path\":\""        << path << "\","
       << "\"records\":"       << rec  << ","
       << "\"bytes\":"         << by   << ","
       << "\"error_count\":"   << err
       << "}";
    return ss.str();
}

} // namespace llmquant
