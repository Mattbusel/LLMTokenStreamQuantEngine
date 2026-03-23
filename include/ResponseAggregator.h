#pragma once

// ---------------------------------------------------------------------------
// ResponseAggregator.h
// Aggregate multiple parallel response streams into a single output sequence.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <limits>
#include <cstdint>

namespace llmquant {

// ---------------------------------------------------------------------------
// Aggregation strategy
// ---------------------------------------------------------------------------

/// How multiple response streams are merged into a single output.
enum class AggregationMethod {
    Concatenate,     ///< Append all streams in stream_id order.
    Interleave,      ///< Round-robin token interleaving across streams.
    MajorityVote,    ///< At each position, pick the token that appears most.
    WeightedAverage, ///< Pick the token with the highest cumulative score.
    BestOf,          ///< Return the single stream with the highest total score.
};

// ---------------------------------------------------------------------------
// Per-stream state
// ---------------------------------------------------------------------------

/// Holds the accumulated tokens, scores, and status for one response stream.
struct ResponseStream {
    int                stream_id;   ///< Unique identifier for this stream.
    std::vector<int>   tokens;      ///< Token IDs appended so far.
    std::vector<float> scores;      ///< Per-token log-probabilities or scores.
    bool               finished{false}; ///< True once finish_stream() is called.
};

// ---------------------------------------------------------------------------
// ResponseAggregator
// ---------------------------------------------------------------------------

/// Merges N parallel response streams into a single token sequence.
///
/// Usage:
///   ResponseAggregator agg(AggregationMethod::BestOf, 3);
///   agg.add_token(0, tok, score);  // stream 0
///   agg.add_token(1, tok, score);  // stream 1
///   agg.finish_stream(0);
///   auto result = agg.aggregate();
class ResponseAggregator {
public:
    /// Constructs the aggregator.
    ///
    /// @param method    How streams are merged (see AggregationMethod).
    /// @param n_streams Expected number of streams (pre-allocates storage).
    ResponseAggregator(AggregationMethod method, int n_streams);

    // ── Token ingestion ───────────────────────────────────────────────────

    /// Append a token to the specified stream.
    ///
    /// @param stream_id  Which stream receives the token.
    /// @param token_id   Token identifier (e.g. vocabulary index).
    /// @param score      Associated log-probability or confidence score.
    void add_token(int stream_id, int token_id, float score);

    /// Mark a stream as complete (no more tokens will be added).
    ///
    /// @param stream_id  Stream to finalise.
    void finish_stream(int stream_id);

    // ── Aggregation ───────────────────────────────────────────────────────

    /// Produce the final merged token sequence according to the method.
    ///
    /// @returns Aggregated token ID vector.
    std::vector<int> aggregate() const;

    /// Pick the winning token from a set of (token_id, score) candidates.
    ///
    /// For MajorityVote: returns the most frequent token_id.
    /// For WeightedAverage: returns the token_id with the highest total score.
    ///
    /// @param candidates  Vector of (token_id, score) pairs.
    /// @returns           Winning token_id.
    int consensus_token(const std::vector<std::pair<int, float>>& candidates) const;

    // ── Query helpers ─────────────────────────────────────────────────────

    /// Return the stream_id whose cumulative score is highest.
    ///
    /// @returns stream_id of the best stream, or -1 if no streams exist.
    int best_stream() const;

    /// True when every registered stream has been finished.
    bool all_finished() const;

    /// Number of streams currently tracked.
    int stream_count() const;

private:
    AggregationMethod method_;
    int               n_streams_;
    std::unordered_map<int, ResponseStream> streams_;

    // ── Private helpers ───────────────────────────────────────────────────

    std::vector<int> aggregate_concatenate() const;
    std::vector<int> aggregate_interleave()  const;
    std::vector<int> aggregate_majority_vote() const;
    std::vector<int> aggregate_weighted_average() const;
    std::vector<int> aggregate_best_of()     const;
};

} // namespace llmquant
