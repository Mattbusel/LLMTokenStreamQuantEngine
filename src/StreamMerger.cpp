#include "StreamMerger.h"

#include <algorithm>
#include <unordered_set>

namespace llmquant {

// ---------------------------------------------------------------------------
// add_entry
// ---------------------------------------------------------------------------

void StreamMerger::add_entry(StreamEntry entry) {
    entries_.push_back(std::move(entry));
}

// ---------------------------------------------------------------------------
// merge_by_priority
// ---------------------------------------------------------------------------

std::vector<StreamEntry> StreamMerger::merge_by_priority() const {
    auto sorted = entries_;
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const StreamEntry& a, const StreamEntry& b) {
            if (a.priority != b.priority) {
                return a.priority > b.priority; // descending priority
            }
            return a.timestamp_ms < b.timestamp_ms; // ascending timestamp
        });
    return sorted;
}

// ---------------------------------------------------------------------------
// merge_by_timestamp
// ---------------------------------------------------------------------------

std::vector<StreamEntry> StreamMerger::merge_by_timestamp() const {
    auto sorted = entries_;
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const StreamEntry& a, const StreamEntry& b) {
            return a.timestamp_ms < b.timestamp_ms;
        });
    return sorted;
}

// ---------------------------------------------------------------------------
// merge_round_robin
// ---------------------------------------------------------------------------

std::vector<StreamEntry>
StreamMerger::merge_round_robin(const std::vector<std::string>& stream_ids) const {
    // Group entries by stream_id preserving insertion order.
    std::unordered_map<std::string, std::vector<const StreamEntry*>> by_stream;
    for (const auto& e : entries_) {
        by_stream[e.stream_id].push_back(&e);
    }

    // Indices into each stream's sub-vector.
    std::unordered_map<std::string, std::size_t> indices;

    std::vector<StreamEntry> result;
    result.reserve(entries_.size());

    bool any_remaining = true;
    while (any_remaining) {
        any_remaining = false;
        for (const auto& sid : stream_ids) {
            auto it = by_stream.find(sid);
            if (it == by_stream.end()) continue;

            std::size_t& idx = indices[sid];
            if (idx < it->second.size()) {
                result.push_back(*(it->second[idx]));
                ++idx;
                any_remaining = true;
            }
        }
    }

    // Also include entries from streams not listed in stream_ids.
    for (const auto& e : entries_) {
        bool in_ids = false;
        for (const auto& sid : stream_ids) {
            if (e.stream_id == sid) { in_ids = true; break; }
        }
        if (!in_ids) {
            result.push_back(e);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// stream_count
// ---------------------------------------------------------------------------

std::size_t StreamMerger::stream_count() const {
    std::unordered_set<std::string> seen;
    for (const auto& e : entries_) {
        seen.insert(e.stream_id);
    }
    return seen.size();
}

// ---------------------------------------------------------------------------
// total_entries
// ---------------------------------------------------------------------------

std::size_t StreamMerger::total_entries() const {
    return entries_.size();
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

void StreamMerger::clear() {
    entries_.clear();
}

} // namespace llmquant
