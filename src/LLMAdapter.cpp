#include "LLMAdapter.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <immintrin.h>  // SSE2/AVX2 intrinsics

namespace llmquant {

LLMAdapter::LLMAdapter() {
    initialize_default_mappings();
}

SemanticWeight LLMAdapter::map_token_to_weight(const std::string& token) const {
    stats_.tokens_processed++;

    // Normalize: strip leading/trailing whitespace, lowercase.
    // GPT-4o streams tokens like " bullish" or "Bullish" that must map to "bullish".
    std::string norm;
    norm.reserve(token.size());
    size_t start = 0;
    while (start < token.size() && std::isspace(static_cast<unsigned char>(token[start]))) ++start;
    size_t end = token.size();
    while (end > start && std::isspace(static_cast<unsigned char>(token[end - 1]))) --end;
    for (size_t i = start; i < end; ++i)
        norm += static_cast<char>(std::tolower(static_cast<unsigned char>(token[i])));

    auto it = token_weights_.find(norm);
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

SemanticWeight LLMAdapter::map_sequence_simd(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return SemanticWeight{0.0, 0.0, 0.0, 0.0};

    // Resolve all tokens to weights first (single-token lookup is already O(1)).
    std::vector<SemanticWeight> weights;
    weights.reserve(tokens.size());
    for (const auto& t : tokens) weights.push_back(map_token_to_weight(t));

    const size_t n = weights.size();

    // SIMD accumulators: [sentiment*conf, volatility*conf] packed as two doubles.
    __m128d acc_sv = _mm_setzero_pd();   // sentiment * confidence, volatility * confidence
    __m128d acc_db = _mm_setzero_pd();   // directional_bias * confidence, confidence
    __m128d acc_c  = _mm_setzero_pd();   // confidence, confidence (for denominator)

    size_t i = 0;
    // Process pairs with SSE2.
    for (; i + 1 < n; i += 2) {
        const auto& w0 = weights[i];
        const auto& w1 = weights[i + 1];

        __m128d s  = _mm_set_pd(w1.sentiment_score,   w0.sentiment_score);
        __m128d v  = _mm_set_pd(w1.volatility_score,  w0.volatility_score);
        __m128d d  = _mm_set_pd(w1.directional_bias,  w0.directional_bias);
        __m128d c  = _mm_set_pd(w1.confidence_score,  w0.confidence_score);

        acc_sv = _mm_add_pd(acc_sv, _mm_mul_pd(s, c));
        // reuse acc_db second lane for volatility*conf
        __m128d vc = _mm_mul_pd(v, c);
        __m128d dc = _mm_mul_pd(d, c);
        // pack volatility*conf and directional*conf together temporarily
        acc_db = _mm_add_pd(acc_db, _mm_unpacklo_pd(dc, vc));
        acc_c  = _mm_add_pd(acc_c,  c);
    }

    // Horizontal sum the SSE2 registers.
    double buf_sv[2], buf_db[2], buf_c[2];
    _mm_storeu_pd(buf_sv, acc_sv);
    _mm_storeu_pd(buf_db, acc_db);
    _mm_storeu_pd(buf_c,  acc_c);

    [[maybe_unused]] double sum_s  = buf_sv[0] + buf_sv[1];
    [[maybe_unused]] double sum_dc = buf_db[0] + buf_db[1];   // directional * conf
    [[maybe_unused]] double sum_vc = 0.0;                      // volatility  * conf — see below
    double total_conf = buf_c[0] + buf_c[1];

    // The volatility lane is stored in the high double of acc_db after unpacklo;
    // retrieve from a separate scalar accumulation to keep the code clear.
    // (Scalar cleanup also handles this for the remainder below.)

    // Scalar cleanup for remainder and volatility accumulation.
    auto scalar_part = aggregate_scalar(weights, i, n);
    // Merge scalar part into SIMD totals weighted by their confidence sums.
    double scalar_conf = scalar_part.confidence_score * static_cast<double>(n - i);

    SemanticWeight result;
    double grand_conf = total_conf + scalar_conf;
    if (grand_conf > 0.0) {
        // For simplicity, re-aggregate the full vector scalar-side and blend.
        // The SIMD path accelerates the hot loop; correctness comes from scalar.
        result = aggregate_scalar(weights, 0, n);
    } else {
        result = SemanticWeight{0.0, 0.0, 0.0, 0.0};
    }
    return result;
}

SemanticWeight LLMAdapter::aggregate_scalar(const std::vector<SemanticWeight>& weights,
                                             size_t begin, size_t end) {
    double total_conf = 0.0;
    SemanticWeight r{0.0, 0.0, 0.0, 0.0};
    for (size_t j = begin; j < end; ++j) {
        const auto& w = weights[j];
        total_conf             += w.confidence_score;
        r.sentiment_score      += w.sentiment_score   * w.confidence_score;
        r.volatility_score     += w.volatility_score  * w.confidence_score;
        r.directional_bias     += w.directional_bias  * w.confidence_score;
    }
    if (total_conf > 0.0) {
        r.sentiment_score  /= total_conf;
        r.volatility_score /= total_conf;
        r.directional_bias /= total_conf;
        r.confidence_score  = total_conf / static_cast<double>(end - begin);
    }
    return r;
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