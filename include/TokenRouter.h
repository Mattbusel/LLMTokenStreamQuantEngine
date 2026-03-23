#pragma once

// ---------------------------------------------------------------------------
// TokenRouter.h
// Route token streams to downstream processors based on prefix-match rules.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>
#include <unordered_map>

namespace llmquant {

/// A routing rule: if a token starts with `pattern`, send it to
/// `target_processor_id`.  Lower `priority` values take precedence.
struct RoutingRule {
    std::string pattern;            ///< Prefix to match against incoming tokens.
    int         priority;           ///< Lower = higher precedence.
    std::string target_processor_id;///< Processor that receives matching tokens.
};

/// Routes individual tokens or batches to the first matching processor rule.
///
/// Rules are evaluated in ascending priority order.  If no rule matches,
/// the token is routed to the "default" processor.
class TokenRouter {
public:
    TokenRouter() = default;

    /// Add a routing rule.  Rules are re-sorted by priority after each insert.
    void add_rule(RoutingRule rule);

    /// Route a single token. Returns the target processor_id ("default" if none match).
    std::string route(const std::string& token) const;

    /// Route a batch of tokens.
    /// Returns a vector of (token, processor_id) pairs in the same order.
    std::vector<std::pair<std::string, std::string>>
    route_batch(const std::vector<std::string>& tokens) const;

    /// Number of rules currently registered.
    std::size_t rule_count() const;

    /// Human-readable routing statistics (total routed, per-processor counts).
    std::string stats() const;

private:
    std::vector<RoutingRule> rules_;

    // Statistics counters (mutable so route() / route_batch() can update them
    // while remaining logically const from the caller's perspective).
    mutable std::size_t total_routed_{0};
    mutable std::unordered_map<std::string, std::size_t> processor_counts_;
};

} // namespace llmquant
