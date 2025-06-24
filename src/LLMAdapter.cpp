#pragma once
#include <unordered_map>
#include <string>
#include <vector>

namespace llmquant {

struct SemanticWeight {
    double sentiment_score;     // -1.0 to 1.0
    double confidence_score;    // 0.0 to 1.0  
    double volatility_score;    // 0.0 to 1.0
    double directional_bias;    // -1.0 to 1.0
};

class LLMAdapter {
public:
    LLMAdapter();
    ~LLMAdapter() = default;

    // Core mapping functions
    SemanticWeight map_token_to_weight(const std::string& token) const;
    SemanticWeight map_sequence_to_weight(const std::vector<std::string>& tokens) const;
    
    // Configuration
    void load_sentiment_dictionary(const std::string& filepath);
    void add_token_mapping(const std::string& token, const SemanticWeight& weight);
    
    // Performance metrics
    struct Stats {
        std::atomic<uint64_t> tokens_processed{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
    };
    
    const Stats& get_stats() const { return stats_; }

private:
    void initialize_default_mappings();
    
    std::unordered_map<std::string, SemanticWeight> token_weights_;
    mutable Stats stats_;
};

} // namespace llmquant
