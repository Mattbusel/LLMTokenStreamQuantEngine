#include "EnsembleSignalVoter.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace llmquant {

EnsembleSignalVoter::EnsembleSignalVoter(Config cfg)
    : cfg_(std::move(cfg))
{}

void EnsembleSignalVoter::register_voter(const std::string& name, double weight) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (index_.count(name)) return;
    index_[name] = voters_.size();
    voters_.push_back({name, std::max(0.0, weight), 0.0, false});
}

EnsembleSignalVoter::ConsensusResult
EnsembleSignalVoter::compute_consensus_locked() const
{
    int total   = static_cast<int>(voters_.size());
    int received = 0;
    double consensus = 0.0;

    if (total == 0) {
        return ConsensusResult{0.0, 0.0, 0, 0, epoch_};
    }

    // Collect active votes.
    std::vector<std::pair<double, double>> active; // (weight, vote)
    double weight_sum = 0.0;
    for (const auto& v : voters_) {
        if (!v.has_voted) continue;
        active.push_back({v.weight, v.current_vote});
        weight_sum += v.weight;
        ++received;
    }

    if (received == 0) {
        return ConsensusResult{0.0, 0.0, 0, total, epoch_};
    }

    if (weight_sum < 1e-12) weight_sum = 1.0;

    switch (cfg_.mode) {
    case VotingMode::Majority: {
        double signed_weight = 0.0;
        for (auto& [w, v] : active)
            signed_weight += (w / weight_sum) * (v >= 0.0 ? 1.0 : -1.0);
        consensus = (signed_weight >= 0.0) ? 1.0 : -1.0;
        break;
    }
    case VotingMode::ConfidenceWeighted: {
        double sum = 0.0;
        for (auto& [w, v] : active) sum += (w / weight_sum) * v;
        consensus = sum;
        break;
    }
    case VotingMode::RankWeighted: {
        // Sort by |vote| descending; rank 0 (highest) gets most weight.
        std::vector<std::size_t> idx(active.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            return std::fabs(active[a].second) > std::fabs(active[b].second);
        });
        int n = static_cast<int>(active.size());
        double rank_sum = static_cast<double>(n * (n + 1)) / 2.0;
        double sum = 0.0;
        for (int r = 0; r < n; ++r) {
            double rank_w = static_cast<double>(n - r) / rank_sum;
            double w      = active[idx[r]].first / weight_sum;
            sum += rank_w * w * active[idx[r]].second;
        }
        // Re-normalize to [-1, 1].
        double max_rank_w = static_cast<double>(n) / rank_sum;
        consensus = (max_rank_w > 0.0) ? sum / max_rank_w : 0.0;
        break;
    }
    }

    // Agreement = fraction of voters on same side as consensus.
    double consensus_sign = (consensus >= 0.0) ? 1.0 : -1.0;
    double agree_w = 0.0;
    for (auto& [w, v] : active)
        if ((v >= 0.0 ? 1.0 : -1.0) == consensus_sign) agree_w += w;
    double agreement = agree_w / weight_sum;

    return ConsensusResult{consensus, agreement, received, total, epoch_};
}

EnsembleSignalVoter::ConsensusResult
EnsembleSignalVoter::vote(const std::string& name, double vote_value)
{
    ConsensusResult result;
    std::function<void(const ConsensusResult&)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_consensus;

        auto it = index_.find(name);
        if (it != index_.end()) {
            auto& v = voters_[it->second];
            v.current_vote = vote_value;
            v.has_voted    = true;
        }

        // Compute consensus if allowed.
        int total    = static_cast<int>(voters_.size());
        int received = 0;
        for (const auto& v : voters_) if (v.has_voted) ++received;
        bool full = (received == total);

        if (full || cfg_.allow_partial) {
            result = compute_consensus_locked();
            last_result_ = result;
            consensus_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            result = ConsensusResult{0.0, 0.0, received, total, epoch_};
        }
    }

    if (cb) cb(result);
    return result;
}

void EnsembleSignalVoter::next_epoch() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& v : voters_) { v.has_voted = false; v.current_vote = 0.0; }
    ++epoch_;
    last_result_.epoch = epoch_;  // keep last_consensus() epoch consistent
}

EnsembleSignalVoter::ConsensusResult
EnsembleSignalVoter::last_consensus() const
{
    std::lock_guard<std::mutex> lk(mutex_);
    return last_result_;
}

bool EnsembleSignalVoter::is_strong_consensus() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_result_.agreement >= cfg_.strong_threshold;
}

int EnsembleSignalVoter::voter_count() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(voters_.size());
}

void EnsembleSignalVoter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& v : voters_) { v.has_voted = false; v.current_vote = 0.0; }
    epoch_ = 0;
    last_result_ = {};
    consensus_count_.store(0, std::memory_order_relaxed);
}

void EnsembleSignalVoter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

std::string EnsembleSignalVoter::to_stats_json() const {
    ConsensusResult r;
    uint64_t cc;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        r  = last_result_;
        cc = consensus_count_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"consensus\":"        << r.consensus       << ","
       << "\"agreement\":"        << r.agreement       << ","
       << "\"votes_received\":"   << r.votes_received  << ","
       << "\"votes_total\":"      << r.votes_total     << ","
       << "\"epoch\":"            << r.epoch           << ","
       << "\"consensus_count\":"  << cc
       << "}";
    return ss.str();
}

} // namespace llmquant
