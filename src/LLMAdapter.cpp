#include "LLMAdapter.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace llmquant {

LLMAdapter::LLMAdapter() {
    initialize_default_mappings();
}

SemanticWeight LLMAdapter::map_token_to_weight(const std::string& token) const {
    stats_.tokens_processed++;
    
    auto it = token_weights_.find(token);
    if (it != token_weights_.end()) {
        stats_.cache_hits++;
        return it->second;
    }
    
    stats_.cache_misses++;
    
    // Default neutral weight for unknown tokens
    return SemanticWeight{0.0, 0.5, 0.1, 0.0};
}

SemanticWeight LLMAdapter::map_sequence_to_weight(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) {
        return SemanticWeight{0.0, 0.0, 0.0, 0.0};
    }
    
    std::vector<SemanticWeight> weights;
    weights.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        weights.push_back(map_token_to_weight(token));
    }
    
    // Aggregate weights (simple average with confidence weighting)
    double total_confidence = 0.0;
    SemanticWeight result{0.0, 0.0, 0.0, 0.0};
    
    for (const auto& w : weights) {
        total_confidence += w.confidence_score;
        result.sentiment_score += w.sentiment_score * w.confidence_score;
        result.volatility_score += w.volatility_score * w.confidence_score;
        result.directional_bias += w.directional_bias * w.confidence_score;
    }
    
    if (total_confidence > 0.0) {
        result.sentiment_score /= total_confidence;
        result.volatility_score /= total_confidence;
        result.directional_bias /= total_confidence;
        result.confidence_score = total_confidence / tokens.size();
    }
    
    return result;
}

void LLMAdapter::load_sentiment_dictionary(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open sentiment dictionary: " + filepath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        double sentiment, confidence, volatility, bias;
        
        if (iss >> token >> sentiment >> confidence >> volatility >> bias) {
            add_token_mapping(token, {sentiment, confidence, volatility, bias});
        }
    }
}

void LLMAdapter::add_token_mapping(const std::string& token, const SemanticWeight& weight) {
    token_weights_[token] = weight;
}

void LLMAdapter::initialize_default_mappings() {
    // Fear/Uncertainty tokens
    add_token_mapping("crash", {-0.9, 0.9, 0.8, -0.7});
    add_token_mapping("panic", {-0.8, 0.8, 0.9, -0.8});
    add_token_mapping("collapse", {-0.9, 0.9, 0.7, -0.9});
    add_token_mapping("plunge", {-0.7, 0.8, 0.8, -0.6});
    
    // Certainty/Confidence tokens
    add_token_mapping("inevitable", {0.1, 0.9, 0.3, 0.0});
    add_token_mapping("guarantee", {0.2, 0.9, 0.2, 0.1});
    add_token_mapping("confident", {0.6, 0.8, 0.2, 0.3});
    
    // Directional sentiment
    add_token_mapping("bullish", {0.7, 0.9, 0.4, 0.8});
    add_token_mapping("bearish", {-0.7, 0.9, 0.4, -0.8});
    add_token_mapping("rally", {0.6, 0.8, 0.6, 0.7});
    
    // Volatility implied
    add_token_mapping("volatile", {0.0, 0.7, 0.9, 0.0});
    add_token_mapping("surge", {0.3, 0.8, 0.8, 0.5});
    add_token_mapping("breakout", {0.4, 0.7, 0.7, 0.6});
    
    // Support/Resistance
    add_token_mapping("support", {0.2, 0.6, 0.3, 0.2});
    add_token_mapping("resistance", {-0.1, 0.6, 0.4, -0.2});
    add_token_mapping("momentum", {0.5, 0.7, 0.6, 0.4});
}

} // namespace llmquant
