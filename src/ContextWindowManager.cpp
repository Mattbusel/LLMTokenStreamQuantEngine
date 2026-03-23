#include "ContextWindowManager.h"

#include <algorithm>
#include <cstdio>
#include <numeric>

namespace llmquant {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ContextWindowManager::ContextWindowManager(std::size_t capacity_tokens)
    : capacity_tokens_(capacity_tokens)
{}

// ---------------------------------------------------------------------------
// Adding segments
// ---------------------------------------------------------------------------

bool ContextWindowManager::add_segment(const ContextSegment& segment) {
    if (segment.token_count > capacity_tokens_) {
        return false;  // segment alone exceeds window — can never fit
    }
    if (total_tokens_ + segment.token_count > capacity_tokens_) {
        return false;  // not enough free space
    }
    segments_.push_back(segment);
    total_tokens_ += segment.token_count;
    return true;
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

std::size_t ContextWindowManager::evict_to_fit(std::size_t needed_tokens) {
    std::size_t evicted = 0;

    // Build an eviction order: highest priority value first (most expendable),
    // then by original insertion order (front = oldest).
    // We iterate until we have enough free space or run out of evictable segs.

    bool progress = true;
    while (progress && free_tokens() < needed_tokens) {
        progress = false;

        // Find the index of the segment to evict next.
        // We pick the one with the highest priority value (most evictable)
        // and, among ties, the earliest insertion (lowest index).
        int    best_priority = -1;
        std::size_t best_idx = segments_.size();

        for (std::size_t i = 0; i < segments_.size(); ++i) {
            const auto& seg = segments_[i];
            if (seg.priority == 0) continue;  // system segment — never evict
            if (seg.priority > best_priority) {
                best_priority = seg.priority;
                best_idx      = i;
            }
        }

        if (best_idx == segments_.size()) {
            break;  // no evictable segments remain
        }

        total_tokens_ -= segments_[best_idx].token_count;
        segments_.erase(segments_.begin() + static_cast<std::ptrdiff_t>(best_idx));
        ++evicted;
        progress = true;
    }

    return evicted;
}

bool ContextWindowManager::remove_segment(uint64_t segment_id) {
    for (auto it = segments_.begin(); it != segments_.end(); ++it) {
        if (it->segment_id == segment_id) {
            total_tokens_ -= it->token_count;
            segments_.erase(it);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// State queries
// ---------------------------------------------------------------------------

std::size_t ContextWindowManager::total_tokens() const noexcept {
    return total_tokens_;
}

std::size_t ContextWindowManager::capacity() const noexcept {
    return capacity_tokens_;
}

std::size_t ContextWindowManager::free_tokens() const noexcept {
    return (total_tokens_ <= capacity_tokens_)
        ? capacity_tokens_ - total_tokens_
        : 0;
}

double ContextWindowManager::utilization() const noexcept {
    if (capacity_tokens_ == 0) return 0.0;
    double u = static_cast<double>(total_tokens_)
             / static_cast<double>(capacity_tokens_);
    return u > 1.0 ? 1.0 : u;
}

bool ContextWindowManager::empty() const noexcept {
    return segments_.empty();
}

bool ContextWindowManager::full() const noexcept {
    return total_tokens_ >= capacity_tokens_;
}

std::size_t ContextWindowManager::segment_count() const noexcept {
    return segments_.size();
}

// ---------------------------------------------------------------------------
// Segment access
// ---------------------------------------------------------------------------

const std::vector<ContextSegment>& ContextWindowManager::segments() const noexcept {
    return segments_;
}

const ContextSegment* ContextWindowManager::find_segment(uint64_t segment_id) const noexcept {
    for (const auto& seg : segments_) {
        if (seg.segment_id == segment_id) return &seg;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Modification
// ---------------------------------------------------------------------------

void ContextWindowManager::clear() noexcept {
    segments_.clear();
    total_tokens_ = 0;
}

bool ContextWindowManager::resize_segment(uint64_t segment_id, std::size_t new_token_count) {
    for (auto& seg : segments_) {
        if (seg.segment_id == segment_id) {
            std::size_t new_total = total_tokens_ - seg.token_count + new_token_count;
            if (new_total > capacity_tokens_) return false;
            total_tokens_ = new_total;
            seg.token_count = new_token_count;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string ContextWindowManager::format_summary() const {
    char buf[256];
    std::snprintf(
        buf, sizeof(buf),
        "segments=%zu tokens=%zu/%zu util=%.2f free=%zu",
        segments_.size(),
        total_tokens_,
        capacity_tokens_,
        utilization(),
        free_tokens());
    return buf;
}

} // namespace llmquant
