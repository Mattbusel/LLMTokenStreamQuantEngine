// TokenProfiler.cpp — Implementation of TokenProfiler.

#include "TokenProfiler.h"

#include <algorithm>
#include <cmath>

namespace llmquant {

void TokenProfiler::feed(const std::string& token) {
    auto& s = stats_[token];
    if (s.frequency == 0) {
        s.first_seen = total_tokens_;
    }
    s.position_sum += static_cast<double>(total_tokens_);
    s.last_seen = total_tokens_;
    ++s.frequency;
    ++total_tokens_;
}

void TokenProfiler::feed_batch(const std::vector<std::string>& tokens) {
    for (const auto& t : tokens) {
        feed(t);
    }
}

StreamProfile TokenProfiler::profile(std::size_t top_k) const {
    StreamProfile sp;
    sp.total_tokens = total_tokens_;
    sp.unique_tokens = stats_.size();
    sp.vocab_size = stats_.size();
    sp.entropy = entropy();

    // Build vector of TokenProfile sorted by frequency descending
    std::vector<TokenProfile> profiles;
    profiles.reserve(stats_.size());
    for (const auto& [tok, st] : stats_) {
        TokenProfile tp;
        tp.token = tok;
        tp.frequency = st.frequency;
        tp.avg_position = (st.frequency > 0)
            ? (st.position_sum / static_cast<double>(st.frequency))
            : 0.0;
        tp.first_seen_idx = st.first_seen;
        tp.last_seen_idx = st.last_seen;
        profiles.push_back(std::move(tp));
    }
    std::sort(profiles.begin(), profiles.end(), [](const TokenProfile& a, const TokenProfile& b) {
        return a.frequency > b.frequency;
    });

    std::size_t k = std::min(top_k, profiles.size());
    sp.top_k_tokens.assign(profiles.begin(), profiles.begin() + static_cast<std::ptrdiff_t>(k));

    return sp;
}

double TokenProfiler::entropy() const {
    if (total_tokens_ == 0) {
        return 0.0;
    }
    double h = 0.0;
    const double n = static_cast<double>(total_tokens_);
    for (const auto& [tok, st] : stats_) {
        if (st.frequency > 0) {
            double p = static_cast<double>(st.frequency) / n;
            h -= p * std::log2(p);
        }
    }
    return h;
}

void TokenProfiler::reset() {
    stats_.clear();
    total_tokens_ = 0;
}

} // namespace llmquant
