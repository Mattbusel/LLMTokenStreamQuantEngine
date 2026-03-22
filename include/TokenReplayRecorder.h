#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Binary token-stream recorder and playback engine.
 *
 * Records live token evaluations to a compact binary file and can replay
 * them deterministically offline — reproducing the exact sentiment trajectory
 * for root-cause analysis, parameter tuning, or regression testing.
 *
 * ## On-disk frame format (24 bytes per token)
 * ```
 * [8 bytes] timestamp_ns   : uint64  — nanoseconds since Unix epoch
 * [8 bytes] sentiment_score: double  — raw semantic weight
 * [4 bytes] token_len      : uint32  — length of the token string in bytes
 * [4 bytes] flags          : uint32  — reserved (0); future use for event type
 * [N bytes] token_bytes    : char[]  — token string (not NUL-terminated)
 * ```
 * Frame header (24 bytes) followed immediately by token_len bytes of UTF-8 text.
 *
 * ## File header (32 bytes)
 * ```
 * [8 bytes] magic    : 0x4C4C4D51554E5452  ("LLMQUNTR")
 * [4 bytes] version  : uint32 = 1
 * [4 bytes] reserved : uint32 = 0
 * [8 bytes] frame_count : uint64 (filled in on close)
 * [8 bytes] start_ns    : uint64 (timestamp of first record call)
 * ```
 *
 * ## Thread safety
 * `record()` may be called concurrently from multiple threads.
 * `start_recording()` / `stop_recording()` / `close()` are NOT thread-safe
 * with respect to each other; call from a single control thread.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_REPLAY_RECORDER` (default ON).
 */
class TokenReplayRecorder {
public:
    /**
     * @brief Magic bytes written at the start of every recording file.
     * Spells "LLMQUNTR" in ASCII.
     */
    static constexpr uint64_t kFileMagic = 0x4C4C4D51554E5452ULL;

    /**
     * @brief Current file format version.
     */
    static constexpr uint32_t kFileVersion = 1;

    /**
     * @brief Per-frame header on disk.
     */
    struct FrameHeader {
        uint64_t timestamp_ns{0};
        double   sentiment_score{0.0};
        uint32_t token_len{0};
        uint32_t flags{0};
    };
    static_assert(sizeof(FrameHeader) == 24, "FrameHeader must be 24 bytes");

    /**
     * @brief File header on disk.
     */
    struct FileHeader {
        uint64_t magic{kFileMagic};
        uint32_t version{kFileVersion};
        uint32_t reserved{0};
        uint64_t frame_count{0};  // filled in on close()
        uint64_t start_ns{0};     // set on first record()
    };
    static_assert(sizeof(FileHeader) == 32, "FileHeader must be 32 bytes");

    /**
     * @brief In-memory representation of one recorded token event.
     */
    struct TokenEvent {
        uint64_t    timestamp_ns{0};
        double      sentiment_score{0.0};
        std::string token;
        uint32_t    flags{0};
    };

    /**
     * @brief Playback callback type.
     *
     * Called once per replayed frame in file order.
     * Parameters: (event, frame_index)
     */
    using PlaybackCallback = std::function<void(const TokenEvent&, uint64_t)>;

    /**
     * @brief Construct a recorder that writes to the given file path.
     *
     * File is not opened until start_recording() is called.
     *
     * @param filepath Path to the output recording file.
     */
    explicit TokenReplayRecorder(std::string filepath);

    /** @brief Destructor calls close() if recording is still active. */
    ~TokenReplayRecorder();

    // Non-copyable.
    TokenReplayRecorder(const TokenReplayRecorder&)            = delete;
    TokenReplayRecorder& operator=(const TokenReplayRecorder&) = delete;

    /**
     * @brief Open the file and write the file header.
     *
     * @return true on success; false if the file cannot be opened.
     */
    [[nodiscard]] bool start_recording();

    /**
     * @brief Record a single token evaluation.
     *
     * Thread-safe.  No-op if start_recording() has not been called or
     * if recording has been stopped.
     *
     * @param token           The raw token string that was evaluated.
     * @param sentiment_score The semantic weight returned by the adapter.
     * @param timestamp_ns    Nanosecond timestamp (default: current time).
     */
    void record(const std::string& token,
                double             sentiment_score,
                uint64_t           timestamp_ns = 0);

    /**
     * @brief Flush the write buffer and finalise the frame count in the header.
     *
     * After stop_recording(), record() calls are silently dropped but the file
     * remains open until close() or destruction.  start_recording() may be
     * called again to start a new recording session in a new file.
     */
    void stop_recording();

    /**
     * @brief Flush and close the file.
     */
    void close();

    /**
     * @brief True while a recording session is active.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] bool is_recording() const noexcept {
        return recording_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Total number of frames written in the current session.
     *
     * Thread-safe (atomic read).
     */
    [[nodiscard]] uint64_t frames_written() const noexcept {
        return frames_written_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Static playback API (no recorder instance needed)
    // -----------------------------------------------------------------------

    /**
     * @brief Load all frames from a recording file into memory.
     *
     * @param filepath Path to the recording file.
     * @param events   Output vector, populated on success.
     * @return true on success; false on file error or corrupt header.
     */
    [[nodiscard]] static bool load_recording(const std::string&    filepath,
                                             std::vector<TokenEvent>& events);

    /**
     * @brief Replay a recording file, invoking the callback for each frame.
     *
     * Optionally scales the inter-frame delays by `speed_multiplier`
     * (1.0 = real-time, 0 = as fast as possible).
     *
     * @param filepath         Path to the recording file.
     * @param cb               Callback invoked per frame.
     * @param speed_multiplier Playback speed; 0 = no inter-frame sleep.
     * @return Number of frames replayed; -1 on file error.
     */
    static int64_t replay(const std::string&   filepath,
                          const PlaybackCallback& cb,
                          double               speed_multiplier = 0.0);

    /**
     * @brief Read the file header from a recording file.
     *
     * @param filepath Path to the recording file.
     * @param hdr      Output header, populated on success.
     * @return true on success; false on error or corrupt magic.
     */
    [[nodiscard]] static bool read_file_header(const std::string& filepath,
                                               FileHeader&        hdr);

private:
    std::string filepath_;
    FILE*       file_{nullptr};
    mutable std::mutex write_mutex_;

    std::atomic<bool>     recording_{false};
    std::atomic<uint64_t> frames_written_{0};
    uint64_t              start_ns_{0};
};

} // namespace llmquant
