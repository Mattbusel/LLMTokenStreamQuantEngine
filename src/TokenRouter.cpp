#include "TokenRouter.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// add_rule
// ---------------------------------------------------------------------------

void TokenRouter::add_rule(RoutingRule rule) {
    rules_.push_back(std::move(rule));
    // Keep rules sorted by ascending priority so route() can short-circuit.
    std::stable_sort(rules_.begin(), rules_.end(),
        [](const RoutingRule& a, const RoutingRule& b) {
            return a.priority < b.priority;
        });
}

// ---------------------------------------------------------------------------
// route
// ---------------------------------------------------------------------------

std::string TokenRouter::route(const std::string& token) const {
    for (const auto& rule : rules_) {
        if (token.size() >= rule.pattern.size() &&
            token.compare(0, rule.pattern.size(), rule.pattern) == 0) {
            ++total_routed_;
            ++processor_counts_[rule.target_processor_id];
            return rule.target_processor_id;
        }
    }
    // No rule matched — route to default processor.
    ++total_routed_;
    ++processor_counts_["default"];
    return "default";
}

// ---------------------------------------------------------------------------
// route_batch
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, std::string>>
TokenRouter::route_batch(const std::vector<std::string>& tokens) const {
    std::vector<std::pair<std::string, std::string>> result;
    result.reserve(tokens.size());
    for (const auto& token : tokens) {
        result.emplace_back(token, route(token));
    }
    return result;
}

// ---------------------------------------------------------------------------
// rule_count
// ---------------------------------------------------------------------------

std::size_t TokenRouter::rule_count() const {
    return rules_.size();
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

std::string TokenRouter::stats() const {
    std::ostringstream oss;
    oss << "TokenRouter stats: total_routed=" << total_routed_
        << " rules=" << rules_.size() << "\n";
    for (const auto& [proc, count] : processor_counts_) {
        oss << "  processor=" << proc << " count=" << count << "\n";
    }
    return oss.str();
}

} // namespace llmquant
