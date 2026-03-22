#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

/**
 * @brief Measures Shannon entropy of the token sentiment stream.
 *
 * Maintains a fixed-size sliding window of recent sentiment scores and
 * computes the normalised Shannon entropy over B histogram bins.
 *
 * ## Interpretation
 *   - **Low entropy  (→ 0.0)**: sentiment is tightly clustered — the stream
 *     is in a high-conviction regime (all bullish or all bearish).
 *   - **High entropy (→ 1.0)**: sentiment is uniformly distributed — the
 *     stream is noise-like, suggesting low-confidence conditions.
 *
 * ## Usage
 * ```cpp
 * TokenEntropyMeter meter;
 * meter.add(signal.delta_bias_shift);           // called per token
 * double conf_penalty = meter.get_normalized_entropy();  // 0..1 noise
 * signal.confidence *= (1.0 - 0.5 * conf_penalty);
 * ```
 *
 * ## Thread safety
 * `add()` uses an internal mutex.  `get_*` methods are const and lock-free
 * after the entropy has been computed by `add()`.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_ENTROPY_METER` (default ON).
 */
class TokenEntropyMeter {
public:
    /** @brief Number of histogram bins.  More bins = finer resolution. */
    static constexpr int kBins = 16;

    /**
     * @brief Configuration.
     */
    struct Config {
        /** @brief Sliding-window size (number of tokens).  Default 64. */
        int window_size{64};

        /**
         * @brief Sentiment range [lo, hi] mapped to the histogram.
         * Scores outside this range are clamped to the nearest bin edge.
         * Default: [-1.0, 1.0].
         */
        double range_lo{-1.0};
        double range_hi{ 1.0};
    };

    explicit TokenEntropyMeter(Config cfg = Config{})
        : cfg_(cfg), window_(static_cast<std::size_t>(cfg.window_size)) {}

    // Non-copyable.
    TokenEntropyMeter(const TokenEntropyMeter&)            = delete;
    TokenEntropyMeter& operator=(const TokenEntropyMeter&) = delete;

    /**
     * @brief Record one sentiment sample.
     *
     * Inserts into the circular window and recomputes entropy.
     * Thread-safe.
     *
     * @param sentiment Sentiment score in [range_lo, range_hi].
     */
    void add(double sentiment) {
        std::lock_guard<std::mutex> lk(mtx_);
        window_[head_ % static_cast<std::size_t>(cfg_.window_size)] = sentiment;
        ++head_;
        if (count_ < static_cast<std::size_t>(cfg_.window_size)) ++count_;
        recompute_locked();
    }

    /**
     * @brief Normalised Shannon entropy in [0.0, 1.0].
     *
     * 0.0 = all samples in one bin (maximum conviction).
     * 1.0 = samples uniformly distributed (maximum noise).
     */
    [[nodiscard]] double get_normalized_entropy() const noexcept {
        return normalized_entropy_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Raw Shannon entropy in nats (natural log base).
     */
    [[nodiscard]] double get_raw_entropy() const noexcept {
        return raw_entropy_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Number of samples currently in the window.
     */
    [[nodiscard]] std::size_t window_count() const noexcept {
        std::lock_guard<std::mutex> lk(mtx_);
        return count_;
    }

    /**
     * @brief Total samples ever recorded.
     */
    [[nodiscard]] uint64_t total_samples() const noexcept {
        return total_samples_.load(std::memory_order_relaxed);
    }

    /** @brief Reset the window and entropy state. */
    void reset() {
        std::lock_guard<std::mutex> lk(mtx_);
        head_  = 0;
        count_ = 0;
        std::fill(window_.begin(), window_.end(), 0.0);
        normalized_entropy_.store(0.0, std::memory_order_relaxed);
        raw_entropy_.store(0.0, std::memory_order_relaxed);
    }

    /**
     * @brief Update configuration at runtime.
     *
     * Resizes the window (discards existing data).
     * Thread-safe.
     */
    void update_config(const Config& cfg) {
        std::lock_guard<std::mutex> lk(mtx_);
        cfg_ = cfg;
        window_.assign(static_cast<std::size_t>(cfg.window_size), 0.0);
        head_  = 0;
        count_ = 0;
    }

    /**
     * @brief Serialise current state to a JSON object string.
     */
    [[nodiscard]] std::string to_stats_json() const {
        double ne = get_normalized_entropy();
        double re = get_raw_entropy();
        std::size_t cnt = window_count();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "{\"normalized_entropy\":%.6f,\"raw_entropy\":%.6f,"
            "\"window_count\":%zu,\"total_samples\":%llu,"
            "\"window_size\":%d,\"bins\":%d}",
            ne, re, cnt,
            static_cast<unsigned long long>(total_samples_.load(std::memory_order_relaxed)),
            cfg_.window_size, kBins);
        return buf;
    }

private:
    void recompute_locked() {
        total_samples_.fetch_add(1, std::memory_order_relaxed);
        if (count_ == 0) return;

        // Fill histogram.
        std::array<int, kBins> hist{};
        double width = (cfg_.range_hi - cfg_.range_lo) / kBins;
        for (std::size_t i = 0; i < count_; ++i) {
            double v = window_[i];
            int bin = static_cast<int>((v - cfg_.range_lo) / width);
            bin = std::clamp(bin, 0, kBins - 1);
            ++hist[bin];
        }

        // Shannon entropy.
        double h = 0.0;
        double n = static_cast<double>(count_);
        for (int b = 0; b < kBins; ++b) {
            if (hist[b] == 0) continue;
            double p = static_cast<double>(hist[b]) / n;
            h -= p * std::log(p);
        }
        raw_entropy_.store(h, std::memory_order_relaxed);

        // Normalise by max entropy (uniform distribution = log(kBins)).
        double h_max = std::log(static_cast<double>(kBins));
        normalized_entropy_.store(h_max > 0.0 ? h / h_max : 0.0,
                                  std::memory_order_relaxed);
    }

    Config cfg_;
    mutable std::mutex mtx_;
    std::vector<double> window_;
    std::size_t head_{0};
    std::size_t count_{0};

    std::atomic<double>   normalized_entropy_{0.0};
    std::atomic<double>   raw_entropy_{0.0};
    std::atomic<uint64_t> total_samples_{0};
};

} // namespace llmquant
