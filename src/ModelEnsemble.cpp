#include "ModelEnsemble.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <spdlog/spdlog.h>

namespace llmquant {

// ---------------------------------------------------------------------------
// Bit-cast helpers — reinterpret double bits as uint64_t without UB.
// Using memcpy is the standards-compliant approach (no std::bit_cast in C++17
// but this project targets C++20 where std::bit_cast is available; kept as
// memcpy for broadest compiler compatibility and zero overhead).
// ---------------------------------------------------------------------------

uint64_t ModelEnsemble::double_to_bits(double v) noexcept {
    uint64_t bits = 0;
    static_assert(sizeof(double) == sizeof(uint64_t),
                  "ModelEnsemble: double must be 64-bit on this platform");
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

double ModelEnsemble::bits_to_double(uint64_t b) noexcept {
    double v = 0.0;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

// ---------------------------------------------------------------------------
// Spinlock helpers
// ---------------------------------------------------------------------------

void ModelEnsemble::spinlock_acquire() const noexcept {
    while (lock_.test_and_set(std::memory_order_acquire)) {
        // Busy-wait.  This lock is only held during cold-path add/remove
        // channel calls, so contention is effectively zero in production.
    }
}

void ModelEnsemble::spinlock_release() const noexcept {
    lock_.clear(std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ModelEnsemble::ModelEnsemble(Config cfg) : cfg_(std::move(cfg)) {
    if (cfg_.normalisation_range <= 0.0) {
        spdlog::warn("ModelEnsemble: normalisation_range must be > 0; resetting to 2.0");
        cfg_.normalisation_range = 2.0;
    }
    if (cfg_.min_channels == 0) {
        cfg_.min_channels = 1;
    }
}

// ---------------------------------------------------------------------------
// Channel management
// ---------------------------------------------------------------------------

bool ModelEnsemble::add_channel(const std::string& name) {
    if (name.empty()) return false;

    spinlock_acquire();

    uint32_t n = num_channels_.load(std::memory_order_relaxed);

    // Reject duplicate names.
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strncmp(slots_[i].name, name.c_str(), 31) == 0) {
            spinlock_release();
            spdlog::warn("ModelEnsemble::add_channel: duplicate name '{}'", name);
            return false;
        }
    }

    if (n >= kMaxChannels) {
        spinlock_release();
        spdlog::warn("ModelEnsemble::add_channel: kMaxChannels ({}) reached; cannot add '{}'",
                     kMaxChannels, name);
        return false;
    }

    // Register the slot: copy name (truncated to 31 chars + null terminator).
    std::strncpy(slots_[n].name, name.c_str(), 31);
    slots_[n].name[31] = '\0';
    // Bias starts at 0.0, active stays false until first update.
    slots_[n].bias_bits.store(double_to_bits(0.0), std::memory_order_relaxed);
    slots_[n].active.store(false, std::memory_order_relaxed);

    num_channels_.store(n + 1, std::memory_order_release);
    spinlock_release();
    return true;
}

bool ModelEnsemble::remove_channel(const std::string& name) {
    if (name.empty()) return false;

    spinlock_acquire();

    uint32_t n = num_channels_.load(std::memory_order_relaxed);
    int found = -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strncmp(slots_[i].name, name.c_str(), 31) == 0) {
            found = static_cast<int>(i);
            break;
        }
    }

    if (found < 0) {
        spinlock_release();
        return false;
    }

    // Compact the slots array by moving the last slot into the vacated position.
    uint32_t last = n - 1;
    if (static_cast<uint32_t>(found) != last) {
        // Copy slot data from last → found.
        std::strncpy(slots_[found].name, slots_[last].name, 31);
        slots_[found].name[31] = '\0';
        slots_[found].bias_bits.store(
            slots_[last].bias_bits.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        slots_[found].active.store(
            slots_[last].active.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
    // Clear the now-unused last slot.
    slots_[last].name[0] = '\0';
    slots_[last].bias_bits.store(double_to_bits(0.0), std::memory_order_relaxed);
    slots_[last].active.store(false, std::memory_order_relaxed);

    num_channels_.store(last, std::memory_order_release);
    spinlock_release();
    return true;
}

uint32_t ModelEnsemble::channel_count() const noexcept {
    return num_channels_.load(std::memory_order_relaxed);
}

int ModelEnsemble::find_channel(const std::string& name) const noexcept {
    uint32_t n = num_channels_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strncmp(slots_[i].name, name.c_str(), 31) == 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Hot path — lock-free bias update
// ---------------------------------------------------------------------------

bool ModelEnsemble::update_channel(const std::string& name, double bias) noexcept {
    int slot = find_channel(name);
    if (slot < 0) return false;
    return update_channel_by_slot(slot, bias);
}

bool ModelEnsemble::update_channel_by_slot(int slot, double bias) noexcept {
    if (slot < 0 || static_cast<uint32_t>(slot) >= num_channels_.load(std::memory_order_relaxed)) {
        return false;
    }
    slots_[slot].bias_bits.store(double_to_bits(bias), std::memory_order_relaxed);
    slots_[slot].active.store(true, std::memory_order_release);
    update_count_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

std::optional<ModelEnsemble::AggregateResult> ModelEnsemble::aggregate() const noexcept {
    uint32_t n = num_channels_.load(std::memory_order_acquire);
    if (n == 0) return std::nullopt;

    // Collect active biases.
    double biases[kMaxChannels]{};
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < n; ++i) {
        if (slots_[i].active.load(std::memory_order_acquire)) {
            biases[active_count++] = bits_to_double(
                slots_[i].bias_bits.load(std::memory_order_relaxed));
        }
    }

    if (active_count < cfg_.min_channels) return std::nullopt;

    // --- Step 1: mean bias ---
    double sum = 0.0;
    for (uint32_t i = 0; i < active_count; ++i) sum += biases[i];
    double mean_bias = sum / static_cast<double>(active_count);

    // --- Step 2: cross-model agreement score ---
    // agreement = 1.0 - mean(|bias_i - mean_bias|) / normalisation_range
    // Clamp to [0, 1] to handle edge cases where biases exceed the configured
    // normalisation range.
    double mad_sum = 0.0;
    for (uint32_t i = 0; i < active_count; ++i) {
        mad_sum += std::fabs(biases[i] - mean_bias);
    }
    double mad = mad_sum / static_cast<double>(active_count);

    double agreement = 1.0 - mad / cfg_.normalisation_range;
    if (agreement < 0.0) agreement = 0.0;
    if (agreement > 1.0) agreement = 1.0;

    // --- Step 3: confidence-adjusted final signal ---
    double final_signal = mean_bias * agreement;

    AggregateResult result{
        mean_bias,
        agreement,
        final_signal,
        active_count,
        update_count_.load(std::memory_order_relaxed)
    };

    // Fire the callback (if set) outside the hot-path atomics.  The callback
    // is non-null only when configured at construction or update_config(), so
    // this branch is predicted-not-taken in production.
    if (cfg_.on_aggregate) {
        cfg_.on_aggregate(result);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

void ModelEnsemble::reset() noexcept {
    uint32_t n = num_channels_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i) {
        slots_[i].bias_bits.store(double_to_bits(0.0), std::memory_order_relaxed);
        slots_[i].active.store(false, std::memory_order_release);
    }
}

void ModelEnsemble::update_config(const Config& cfg) {
    spinlock_acquire();
    if (cfg.normalisation_range > 0.0) {
        cfg_.normalisation_range = cfg.normalisation_range;
    } else {
        spdlog::warn("ModelEnsemble::update_config: ignoring normalisation_range <= 0");
    }
    cfg_.min_channels = (cfg.min_channels > 0) ? cfg.min_channels : 1u;
    cfg_.on_aggregate = cfg.on_aggregate;
    spinlock_release();
}

} // namespace llmquant
