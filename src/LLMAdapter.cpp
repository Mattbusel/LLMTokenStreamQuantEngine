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
    add_token_mapping("support",    {0.2,  0.6, 0.3, 0.2});
    add_token_mapping("resistance", {-0.1, 0.6, 0.4, -0.2});
    add_token_mapping("momentum",   {0.5,  0.7, 0.6, 0.4});

    // Fear / Uncertainty — additional entries not already mapped above
    add_token_mapping("dump",      {-0.8, 0.85, 0.75, -0.75});
    add_token_mapping("breakdown", {-0.8, 0.85, 0.80, -0.80});
    add_token_mapping("fear",      {-0.7, 0.80, 0.70, -0.60});
    add_token_mapping("selloff",   {-0.8, 0.85, 0.80, -0.75});
    add_token_mapping("tumble",    {-0.7, 0.80, 0.75, -0.65});
    add_token_mapping("rout",      {-0.9, 0.90, 0.85, -0.85});

    // Certainty / Confidence — additional entries not already mapped above
    add_token_mapping("confirmed", {0.3,  0.90, 0.15, 0.2});
    add_token_mapping("certain",   {0.2,  0.90, 0.10, 0.15});
    add_token_mapping("assured",   {0.4,  0.85, 0.10, 0.25});

    // Directional Bullish — additional entries
    add_token_mapping("soar",      {0.7,  0.85, 0.60, 0.75});
    add_token_mapping("moon",      {0.8,  0.80, 0.70, 0.90});
    add_token_mapping("buy",       {0.6,  0.85, 0.40, 0.80});
    add_token_mapping("long",      {0.5,  0.80, 0.35, 0.70});

    // Directional Bearish — additional entries (plunge/dump/breakdown/collapse already mapped)
    add_token_mapping("short",     {-0.5, 0.85, 0.50, -0.80});
    add_token_mapping("sell",      {-0.5, 0.85, 0.45, -0.75});

    // Volatility — additional entries (volatile/surge already mapped)
    add_token_mapping("spike",     {0.0,  0.75, 0.90, 0.0});
    add_token_mapping("whipsaw",   {0.0,  0.70, 0.95, 0.0});
    add_token_mapping("swing",     {0.0,  0.65, 0.85, 0.0});
    add_token_mapping("choppy",    {0.0,  0.70, 0.88, 0.0});
    add_token_mapping("erratic",   {0.0,  0.65, 0.90, 0.0});

    // Neutral / Filler — zero-weight pass-through tokens
    add_token_mapping("the",       {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("and",       {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("is",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("a",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("an",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("in",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("of",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("to",        {0.0,  0.1,  0.0,  0.0});
}

} // namespace llmquant
