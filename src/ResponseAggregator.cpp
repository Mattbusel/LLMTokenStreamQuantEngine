#include "ResponseAggregator.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace llmquant {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ResponseAggregator::ResponseAggregator(AggregationMethod method, int n_streams)
    : method_(method), n_streams_(n_streams)
{
    // Pre-create stream entries so they are addressable immediately.
    for (int i = 0; i < n_streams; ++i) {
        streams_[i] = ResponseStream{i, {}, {}, false};
    }
}

// ---------------------------------------------------------------------------
// add_token
// ---------------------------------------------------------------------------

void ResponseAggregator::add_token(int stream_id, int token_id, float score) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
        // Auto-register unknown stream IDs.
        streams_[stream_id] = ResponseStream{stream_id, {}, {}, false};
        it = streams_.find(stream_id);
    }
    if (it->second.finished) {
        return; // Silently discard tokens appended after finish.
    }
    it->second.tokens.push_back(token_id);
    it->second.scores.push_back(score);
}

// ---------------------------------------------------------------------------
// finish_stream
// ---------------------------------------------------------------------------

void ResponseAggregator::finish_stream(int stream_id) {
    auto it = streams_.find(stream_id);
    if (it != streams_.end()) {
        it->second.finished = true;
    }
}

// ---------------------------------------------------------------------------
// aggregate
// ---------------------------------------------------------------------------

std::vector<int> ResponseAggregator::aggregate() const {
    switch (method_) {
        case AggregationMethod::Concatenate:     return aggregate_concatenate();
        case AggregationMethod::Interleave:      return aggregate_interleave();
        case AggregationMethod::MajorityVote:    return aggregate_majority_vote();
        case AggregationMethod::WeightedAverage: return aggregate_weighted_average();
        case AggregationMethod::BestOf:          return aggregate_best_of();
    }
    return {};
}

// ---------------------------------------------------------------------------
// consensus_token
// ---------------------------------------------------------------------------

int ResponseAggregator::consensus_token(
    const std::vector<std::pair<int, float>>& candidates) const
{
    if (candidates.empty()) {
        return -1;
    }

    if (method_ == AggregationMethod::MajorityVote) {
        // Most-frequent token wins; ties broken by first occurrence.
        std::unordered_map<int, int> counts;
        int best_tok = candidates[0].first;
        int best_cnt = 0;
        for (const auto& [tok, _score] : candidates) {
            int cnt = ++counts[tok];
            if (cnt > best_cnt) {
                best_cnt = cnt;
                best_tok = tok;
            }
        }
        return best_tok;
    }

    // WeightedAverage (and default): token with the highest cumulative score.
    std::unordered_map<int, float> score_map;
    for (const auto& [tok, sc] : candidates) {
        score_map[tok] += sc;
    }
    int   best_tok   = candidates[0].first;
    float best_score = -std::numeric_limits<float>::infinity();
    for (const auto& [tok, sc] : score_map) {
        if (sc > best_score) {
            best_score = sc;
            best_tok   = tok;
        }
    }
    return best_tok;
}

// ---------------------------------------------------------------------------
// best_stream
// ---------------------------------------------------------------------------

int ResponseAggregator::best_stream() const {
    int   best_id    = -1;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto& [sid, stream] : streams_) {
        float total = 0.0f;
        for (float s : stream.scores) {
            total += s;
        }
        if (total > best_score) {
            best_score = total;
            best_id    = sid;
        }
    }
    return best_id;
}

// ---------------------------------------------------------------------------
// all_finished
// ---------------------------------------------------------------------------

bool ResponseAggregator::all_finished() const {
    for (const auto& [_sid, stream] : streams_) {
        if (!stream.finished) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// stream_count
// ---------------------------------------------------------------------------

int ResponseAggregator::stream_count() const {
    return static_cast<int>(streams_.size());
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::vector<int> ResponseAggregator::aggregate_concatenate() const {
    // Concatenate in ascending stream_id order.
    std::vector<std::pair<int, const ResponseStream*>> ordered;
    for (const auto& [sid, stream] : streams_) {
        ordered.emplace_back(sid, &stream);
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int> result;
    for (const auto& [_sid, stream_ptr] : ordered) {
        result.insert(result.end(),
                      stream_ptr->tokens.begin(),
                      stream_ptr->tokens.end());
    }
    return result;
}

std::vector<int> ResponseAggregator::aggregate_interleave() const {
    // Round-robin across streams in ascending stream_id order.
    std::vector<std::pair<int, const ResponseStream*>> ordered;
    for (const auto& [sid, stream] : streams_) {
        ordered.emplace_back(sid, &stream);
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    if (ordered.empty()) {
        return {};
    }

    std::size_t max_len = 0;
    for (const auto& [_sid, sp] : ordered) {
        if (sp->tokens.size() > max_len) {
            max_len = sp->tokens.size();
        }
    }

    std::vector<int> result;
    result.reserve(max_len * ordered.size());
    for (std::size_t pos = 0; pos < max_len; ++pos) {
        for (const auto& [_sid, sp] : ordered) {
            if (pos < sp->tokens.size()) {
                result.push_back(sp->tokens[pos]);
            }
        }
    }
    return result;
}

std::vector<int> ResponseAggregator::aggregate_majority_vote() const {
    // At each position, collect candidates from all streams and vote.
    std::size_t max_len = 0;
    for (const auto& [_sid, stream] : streams_) {
        if (stream.tokens.size() > max_len) {
            max_len = stream.tokens.size();
        }
    }

    std::vector<int> result;
    result.reserve(max_len);

    for (std::size_t pos = 0; pos < max_len; ++pos) {
        std::vector<std::pair<int, float>> candidates;
        for (const auto& [_sid, stream] : streams_) {
            if (pos < stream.tokens.size()) {
                float sc = (pos < stream.scores.size()) ? stream.scores[pos] : 0.0f;
                candidates.emplace_back(stream.tokens[pos], sc);
            }
        }
        result.push_back(consensus_token(candidates));
    }
    return result;
}

std::vector<int> ResponseAggregator::aggregate_weighted_average() const {
    // At each position, return the token with the highest cumulative score.
    std::size_t max_len = 0;
    for (const auto& [_sid, stream] : streams_) {
        if (stream.tokens.size() > max_len) {
            max_len = stream.tokens.size();
        }
    }

    std::vector<int> result;
    result.reserve(max_len);

    for (std::size_t pos = 0; pos < max_len; ++pos) {
        std::vector<std::pair<int, float>> candidates;
        for (const auto& [_sid, stream] : streams_) {
            if (pos < stream.tokens.size()) {
                float sc = (pos < stream.scores.size()) ? stream.scores[pos] : 0.0f;
                candidates.emplace_back(stream.tokens[pos], sc);
            }
        }
        // Reuse consensus_token with WeightedAverage semantics
        int   best_tok   = candidates.empty() ? -1 : candidates[0].first;
        float best_score = -std::numeric_limits<float>::infinity();
        std::unordered_map<int, float> score_map;
        for (const auto& [tok, sc] : candidates) {
            score_map[tok] += sc;
        }
        for (const auto& [tok, sc] : score_map) {
            if (sc > best_score) {
                best_score = sc;
                best_tok   = tok;
            }
        }
        result.push_back(best_tok);
    }
    return result;
}

std::vector<int> ResponseAggregator::aggregate_best_of() const {
    int bid = best_stream();
    if (bid < 0) {
        return {};
    }
    auto it = streams_.find(bid);
    if (it == streams_.end()) {
        return {};
    }
    return it->second.tokens;
}

} // namespace llmquant
