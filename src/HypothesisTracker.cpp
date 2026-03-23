#include "HypothesisTracker.h"
#include <stdexcept>

HypothesisTracker::HypothesisTracker(int max_hypotheses, int eos_token_id, float length_penalty_alpha)
    : max_hypotheses_(max_hypotheses)
    , eos_token_id_(eos_token_id)
    , length_penalty_alpha_(static_cast<double>(length_penalty_alpha))
{}

void HypothesisTracker::add(Hypothesis hyp) {
    hyp.length_penalty_alpha = length_penalty_alpha_;
    hypotheses_.push_back(std::move(hyp));
}

Hypothesis HypothesisTracker::extend(int hypothesis_idx, int next_token_id, float log_prob) {
    if (hypothesis_idx < 0 || static_cast<size_t>(hypothesis_idx) >= hypotheses_.size()) {
        throw std::out_of_range("hypothesis_idx out of range");
    }
    Hypothesis ext = hypotheses_[static_cast<size_t>(hypothesis_idx)];
    ext.token_ids.push_back(next_token_id);
    ext.log_prob += static_cast<double>(log_prob);
    ext.length_penalty_alpha = length_penalty_alpha_;
    return ext;
}

void HypothesisTracker::prune_to(int n) {
    if (static_cast<int>(hypotheses_.size()) <= n) return;
    std::sort(hypotheses_.begin(), hypotheses_.end(), score_greater);
    hypotheses_.resize(static_cast<size_t>(n));
}

const Hypothesis& HypothesisTracker::best() const {
    if (hypotheses_.empty()) {
        throw std::runtime_error("No hypotheses in tracker");
    }
    const Hypothesis* best_hyp = &hypotheses_[0];
    for (const auto& h : hypotheses_) {
        if (h.normalized_score() > best_hyp->normalized_score()) {
            best_hyp = &h;
        }
    }
    return *best_hyp;
}

std::vector<Hypothesis> HypothesisTracker::n_best(int n) const {
    std::vector<Hypothesis> sorted = hypotheses_;
    std::sort(sorted.begin(), sorted.end(), score_greater);
    if (static_cast<int>(sorted.size()) > n) {
        sorted.resize(static_cast<size_t>(n));
    }
    return sorted;
}

bool HypothesisTracker::all_finished() const {
    if (hypotheses_.empty()) return true;
    for (const auto& h : hypotheses_) {
        if (h.token_ids.empty() || h.token_ids.back() != eos_token_id_) {
            return false;
        }
    }
    return true;
}

std::vector<Hypothesis> HypothesisTracker::finished_hypotheses() const {
    std::vector<Hypothesis> result;
    for (const auto& h : hypotheses_) {
        if (!h.token_ids.empty() && h.token_ids.back() == eos_token_id_) {
            result.push_back(h);
        }
    }
    return result;
}

std::vector<Hypothesis> HypothesisTracker::active_hypotheses() const {
    std::vector<Hypothesis> result;
    for (const auto& h : hypotheses_) {
        if (h.token_ids.empty() || h.token_ids.back() != eos_token_id_) {
            result.push_back(h);
        }
    }
    return result;
}

float HypothesisTracker::diversity_score() const {
    size_t n = hypotheses_.size();
    if (n < 2) return 0.0f;

    double total_disagree = 0.0;
    size_t n_pairs = 0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const auto& a = hypotheses_[i].token_ids;
            const auto& b = hypotheses_[j].token_ids;
            size_t len = std::min(a.size(), b.size());
            if (len == 0) { ++n_pairs; continue; }
            size_t differ = 0;
            for (size_t k = 0; k < len; ++k) {
                if (a[k] != b[k]) ++differ;
            }
            total_disagree += static_cast<double>(differ) / static_cast<double>(len);
            ++n_pairs;
        }
    }
    if (n_pairs == 0) return 0.0f;
    return static_cast<float>(total_disagree / static_cast<double>(n_pairs));
}

int HypothesisTracker::size() const {
    return static_cast<int>(hypotheses_.size());
}
