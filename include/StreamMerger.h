#pragma once

// ---------------------------------------------------------------------------
// StreamMerger.h
// Merge multiple token streams with configurable ordering policies.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace llmquant {

/// An entry in a token stream.
struct StreamEntry {
    std::string  stream_id;    ///< Source stream identifier.
    std::string  token;        ///< The token payload.
    uint64_t     timestamp_ms; ///< Wall-clock timestamp in milliseconds.
    int          priority;     ///< Higher value = higher precedence.
};

/// Merges multiple token streams with different ordering strategies.
class StreamMerger {
public:
    StreamMerger() = default;

    /// Add an entry to the internal buffer.
    void add_entry(StreamEntry entry);

    /// Return all buffered entries sorted by priority descending,
    /// then by timestamp ascending for ties.
    std::vector<StreamEntry> merge_by_priority() const;

    /// Return all buffered entries sorted chronologically (timestamp ascending).
    std::vector<StreamEntry> merge_by_timestamp() const;

    /// Interleave streams in round-robin order according to `stream_ids`.
    ///
    /// Each stream ID is visited in sequence; one entry per stream per round
    /// (ordered by insertion order within each stream).  Streams with no
    /// remaining entries are skipped.
    std::vector<StreamEntry>
    merge_round_robin(const std::vector<std::string>& stream_ids) const;

    /// Number of distinct streams currently buffered.
    std::size_t stream_count() const;

    /// Total number of entries across all streams.
    std::size_t total_entries() const;

    /// Clear all buffered entries.
    void clear();

private:
    std::vector<StreamEntry> entries_;
};

} // namespace llmquant
