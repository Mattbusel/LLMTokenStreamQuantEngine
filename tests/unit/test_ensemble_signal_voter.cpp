#include <gtest/gtest.h>

#ifdef LLMQUANT_ENSEMBLE_VOTER_ENABLED

#include "EnsembleSignalVoter.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

// ============================================================
// Default construction
// ============================================================

TEST(EnsembleSignalVoter, DefaultConstructs) {
    EnsembleSignalVoter voter;
    EXPECT_EQ(voter.voter_count(), 0);
    EXPECT_EQ(voter.consensus_count(), 0u);
    EXPECT_FALSE(voter.is_strong_consensus());
}

// ============================================================
// register_voter adds voters
// ============================================================

TEST(EnsembleSignalVoter, RegisterVoterAddsCount) {
    EnsembleSignalVoter voter;
    voter.register_voter("reddit", 1.0);
    voter.register_voter("twitter", 2.0);
    EXPECT_EQ(voter.voter_count(), 2);
}

// ============================================================
// Duplicate registration is idempotent
// ============================================================

TEST(EnsembleSignalVoter, DuplicateRegistrationIgnored) {
    EnsembleSignalVoter voter;
    voter.register_voter("a");
    voter.register_voter("a");
    EXPECT_EQ(voter.voter_count(), 1);
}

// ============================================================
// ConfidenceWeighted mode — weighted mean
// ============================================================

TEST(EnsembleSignalVoter, ConfidenceWeightedMean) {
    EnsembleSignalVoter::Config cfg;
    cfg.mode = EnsembleSignalVoter::VotingMode::ConfidenceWeighted;
    EnsembleSignalVoter voter(cfg);
    voter.register_voter("a", 1.0);
    voter.register_voter("b", 1.0);

    auto r = voter.vote("a", 0.6);
    r = voter.vote("b", 0.4);
    // Equal weights → mean = 0.5
    EXPECT_NEAR(r.consensus, 0.5, 1e-9);
}

// ============================================================
// Majority mode — sign of weighted vote
// ============================================================

TEST(EnsembleSignalVoter, MajorityMode) {
    EnsembleSignalVoter::Config cfg;
    cfg.mode = EnsembleSignalVoter::VotingMode::Majority;
    EnsembleSignalVoter voter(cfg);
    voter.register_voter("a", 1.0);
    voter.register_voter("b", 1.0);
    voter.register_voter("c", 1.0);

    voter.vote("a",  0.5);  // positive
    voter.vote("b",  0.3);  // positive
    auto r = voter.vote("c", -0.9);  // negative — outvoted
    EXPECT_GT(r.consensus, 0.0);  // majority says bullish
}

// ============================================================
// RankWeighted mode produces a finite value in [-1, 1]
// ============================================================

TEST(EnsembleSignalVoter, RankWeightedInBounds) {
    EnsembleSignalVoter::Config cfg;
    cfg.mode = EnsembleSignalVoter::VotingMode::RankWeighted;
    EnsembleSignalVoter voter(cfg);
    voter.register_voter("a", 1.0);
    voter.register_voter("b", 1.0);
    voter.register_voter("c", 1.0);

    voter.vote("a", 0.9);
    voter.vote("b", 0.1);
    auto r = voter.vote("c", -0.5);
    EXPECT_GE(r.consensus, -1.0);
    EXPECT_LE(r.consensus, 1.0);
    EXPECT_TRUE(std::isfinite(r.consensus));
}

// ============================================================
// agreement ∈ [0, 1]
// ============================================================

TEST(EnsembleSignalVoter, AgreementInBounds) {
    EnsembleSignalVoter voter;
    voter.register_voter("x", 1.0);
    voter.register_voter("y", 1.0);
    voter.vote("x", 0.5);
    auto r = voter.vote("y", -0.5);
    EXPECT_GE(r.agreement, 0.0);
    EXPECT_LE(r.agreement, 1.0);
}

// ============================================================
// on_consensus callback fires on each vote
// ============================================================

TEST(EnsembleSignalVoter, ConsensusCallbackFires) {
    EnsembleSignalVoter::Config cfg;
    std::atomic<int> fires{0};
    cfg.on_consensus = [&](const EnsembleSignalVoter::ConsensusResult&) { ++fires; };
    EnsembleSignalVoter voter(cfg);
    voter.register_voter("a");
    voter.register_voter("b");
    voter.vote("a", 0.5);
    voter.vote("b", 0.3);
    EXPECT_EQ(fires.load(), 2);
}

// ============================================================
// next_epoch() clears votes
// ============================================================

TEST(EnsembleSignalVoter, NextEpochClearsVotes) {
    EnsembleSignalVoter voter;
    voter.register_voter("a");
    voter.vote("a", 0.9);
    voter.next_epoch();
    auto r = voter.last_consensus();
    // After next_epoch(), no new votes submitted — consensus still shows last result.
    EXPECT_EQ(r.epoch, 1u);
}

// ============================================================
// reset() clears all state
// ============================================================

TEST(EnsembleSignalVoter, ResetClearsState) {
    EnsembleSignalVoter voter;
    voter.register_voter("a");
    voter.vote("a", 0.8);
    voter.reset();
    EXPECT_EQ(voter.consensus_count(), 0u);
}

// ============================================================
// to_stats_json contains expected keys
// ============================================================

TEST(EnsembleSignalVoter, ToStatsJsonContainsKeys) {
    EnsembleSignalVoter voter;
    voter.register_voter("x");
    voter.vote("x", 0.5);
    std::string json = voter.to_stats_json();
    EXPECT_NE(json.find("consensus"),       std::string::npos);
    EXPECT_NE(json.find("agreement"),       std::string::npos);
    EXPECT_NE(json.find("votes_received"),  std::string::npos);
    EXPECT_NE(json.find("epoch"),           std::string::npos);
}

// ============================================================
// Concurrent vote() is thread-safe
// ============================================================

TEST(EnsembleSignalVoter, ConcurrentVoteSafe) {
    EnsembleSignalVoter::Config cfg;
    cfg.allow_partial = true;
    EnsembleSignalVoter voter(cfg);
    voter.register_voter("a");
    voter.register_voter("b");
    voter.register_voter("c");
    voter.register_voter("d");

    std::vector<std::thread> threads;
    const char* names[] = {"a", "b", "c", "d"};
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 20; ++i) {
                voter.vote(names[t], static_cast<double>(t % 2 == 0 ? 1 : -1) * 0.5);
                voter.next_epoch();
            }
        });
    }
    for (auto& th : threads) th.join();
    // No crash; consensus is valid.
    auto r = voter.last_consensus();
    EXPECT_GE(r.agreement, 0.0);
    EXPECT_LE(r.agreement, 1.0);
}

} // namespace

#else

TEST(EnsembleSignalVoter, DisabledAtBuildTime) { SUCCEED(); }

#endif  // LLMQUANT_ENSEMBLE_VOTER_ENABLED
