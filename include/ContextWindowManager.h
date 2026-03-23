#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// ContextSegment
// ---------------------------------------------------------------------------

/**
 * @brief A single segment of the LLM context window.
 *
 * Segments are ranked for eviction by (priority DESC, age ASC): lower
 * priority segments are evicted first; among equal-priority segments the
 * oldest (inserted earliest) is evicted first.
 *
 * Priority values:
 *  - 0 = system  (never evicted by default)
 *  - 1 = important (evicted last)
 *  - 2 = normal   (evicted first)
 */
struct ContextSegment {
    /// Unique identifier for this segment (caller-assigned).
    uint64_t    segment_id{0};

    /// Number of tokens occupied by this segment.
    std::size_t token_count{0};

    /// Eviction priority.  Lower value = higher priority (kept longer).
    int         priority{2};

    /// CRC-32 or any hash of the segment content for identity checks.
    uint32_t    content_hash{0};
};

// ---------------------------------------------------------------------------
// ContextWindowManager
// ---------------------------------------------------------------------------

/**
 * @brief Manages an LLM context window as a collection of named segments.
 *
 * Provides add / evict / query operations with priority-based eviction:
 * lowest-priority segments are removed first; ties are broken by insertion
 * order (oldest first).
 *
 * Thread safety: NOT thread-safe.  External synchronisation required.
 *
 * Usage:
 * @code
 *   ContextWindowManager mgr(8192);   // 8k token window
 *   ContextSegment system_prompt{1, 512, 0, 0xABCD};
 *   mgr.add_segment(system_prompt);   // returns true — fits in window
 *
 *   ContextSegment user_turn{2, 256, 2, 0x1234};
 *   mgr.add_segment(user_turn);       // true — still fits
 *
 *   mgr.evict_to_fit(4096);           // make room for 4096 new tokens
 * @endcode
 */
class ContextWindowManager {
public:
    /**
     * @brief Construct with a fixed context window capacity.
     * @param capacity_tokens  Maximum number of tokens the window can hold.
     */
    explicit ContextWindowManager(std::size_t capacity_tokens);

    // ------------------------------------------------------------------
    // Adding segments
    // ------------------------------------------------------------------

    /**
     * @brief Attempt to add a segment to the context window.
     *
     * If the segment fits within the remaining capacity it is appended and
     * the function returns true.  If it does not fit (even after any prior
     * evictions the caller may have requested) the function returns false
     * without modifying state.
     *
     * @param segment  Segment to add.
     * @return true if the segment was added; false if insufficient capacity.
     */
    bool add_segment(const ContextSegment& segment);

    // ------------------------------------------------------------------
    // Eviction
    // ------------------------------------------------------------------

    /**
     * @brief Evict segments until at least ``needed_tokens`` free capacity is
     *        available (or all evictable segments are removed).
     *
     * Eviction order:
     *  1. Highest priority value first (priority 2 before priority 1).
     *  2. Among equal priority, the segment inserted earliest is evicted first
     *     (FIFO within a priority band).
     *
     * Segments with priority 0 ("system") are never evicted.
     *
     * @param needed_tokens  Target free-token headroom to achieve.
     * @return Number of segments actually evicted.
     */
    std::size_t evict_to_fit(std::size_t needed_tokens);

    /**
     * @brief Remove a specific segment by its segment_id.
     *
     * @param segment_id  ID of the segment to remove.
     * @return true if the segment was found and removed.
     */
    bool remove_segment(uint64_t segment_id);

    // ------------------------------------------------------------------
    // State queries
    // ------------------------------------------------------------------

    /**
     * @brief Total tokens currently occupied.
     */
    [[nodiscard]] std::size_t total_tokens() const noexcept;

    /**
     * @brief Maximum tokens the window can hold.
     */
    [[nodiscard]] std::size_t capacity() const noexcept;

    /**
     * @brief Free token capacity (capacity − total_tokens).
     */
    [[nodiscard]] std::size_t free_tokens() const noexcept;

    /**
     * @brief Fraction of capacity currently used, in [0.0, 1.0].
     */
    [[nodiscard]] double utilization() const noexcept;

    /**
     * @brief Return true if the window is completely empty.
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Return true if the window is at or above capacity.
     */
    [[nodiscard]] bool full() const noexcept;

    /**
     * @brief Number of segments currently in the window.
     */
    [[nodiscard]] std::size_t segment_count() const noexcept;

    // ------------------------------------------------------------------
    // Segment access
    // ------------------------------------------------------------------

    /**
     * @brief Const reference to the ordered list of segments.
     *
     * Segments appear in insertion order.
     */
    [[nodiscard]] const std::vector<ContextSegment>& segments() const noexcept;

    /**
     * @brief Find a segment by id.  Returns nullptr if not found.
     */
    [[nodiscard]] const ContextSegment* find_segment(uint64_t segment_id) const noexcept;

    // ------------------------------------------------------------------
    // Modification
    // ------------------------------------------------------------------

    /**
     * @brief Remove all segments from the window.
     */
    void clear() noexcept;

    /**
     * @brief Update the token count of an existing segment (e.g. after edit).
     *
     * @param segment_id    The segment to update.
     * @param new_token_count  New token count; must not exceed capacity alone.
     * @return true if the segment was found and updated; false if not found
     *         or if the new size would exceed window capacity.
     */
    bool resize_segment(uint64_t segment_id, std::size_t new_token_count);

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    /**
     * @brief Format a single-line summary.
     *
     * Example: "segments=5 tokens=1024/8192 util=0.12 free=7168"
     */
    [[nodiscard]] std::string format_summary() const;

private:
    std::size_t                  capacity_tokens_;
    std::size_t                  total_tokens_{0};
    std::vector<ContextSegment>  segments_;
};

} // namespace llmquant
