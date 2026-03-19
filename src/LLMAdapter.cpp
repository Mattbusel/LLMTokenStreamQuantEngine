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
    // tokens_processed is incremented AFTER the lookup so the invariant
    // cache_hits + cache_misses == tokens_processed always holds (Improvement 12).

    // Fast-path: if the token is already normalized (lowercase ASCII letters,
    // digits, spaces or underscores, no leading/trailing whitespace) skip the
    // allocation and look up the original string directly.
    auto is_normalized = [](const std::string& s) -> bool {
        if (s.empty()) return true;
        if (std::isspace(static_cast<unsigned char>(s.front()))) return false;
        if (std::isspace(static_cast<unsigned char>(s.back()))) return false;
        for (unsigned char c : s) {
            if (c >= 'A' && c <= 'Z') return false;  // uppercase
        }
        return true;
    };

    if (is_normalized(token)) {
        auto it_fast = token_weights_.find(token);
        if (it_fast != token_weights_.end()) {
            stats_.tokens_processed++;
            stats_.cache_hits++;
            return it_fast->second;
        }
        // Token is already normalized but not found — no need to normalize again.
        stats_.tokens_processed++;
        stats_.cache_misses++;
        return SemanticWeight{0.0, 0.5, 0.1, 0.0};
    }

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
        stats_.tokens_processed++;
        stats_.cache_hits++;
        return it->second;
    }

    stats_.tokens_processed++;
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
    // Normalise key to match map_token_to_weight() lookup behaviour:
    // strip leading/trailing whitespace, then lowercase.
    std::string key;
    key.reserve(token.size());
    size_t start = 0;
    while (start < token.size() && std::isspace(static_cast<unsigned char>(token[start]))) ++start;
    size_t end = token.size();
    while (end > start && std::isspace(static_cast<unsigned char>(token[end - 1]))) --end;
    for (size_t i = start; i < end; ++i)
        key += static_cast<char>(std::tolower(static_cast<unsigned char>(token[i])));
    token_weights_[key] = weight;
}

// SSE2-only horizontal add: returns [a[0]+a[1], b[0]+b[1]]
static inline __m128d sse2_hadd(__m128d a, __m128d b) {
    return _mm_add_pd(_mm_unpacklo_pd(a, b), _mm_unpackhi_pd(a, b));
}

SemanticWeight LLMAdapter::map_sequence_simd(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return SemanticWeight{0.0, 0.0, 0.0, 0.0};

    std::vector<SemanticWeight> weights;
    weights.reserve(tokens.size());
    for (const auto& t : tokens) weights.push_back(map_token_to_weight(t));

    const size_t n = weights.size();

    // Four confidence-weighted sums packed into two SSE2 registers:
    //   acc_sv = [ sum(sentiment*conf) , sum(volatility*conf) ]
    //   acc_dc = [ sum(bias*conf)      , sum(conf)            ]
    __m128d acc_sv = _mm_setzero_pd();
    __m128d acc_dc = _mm_setzero_pd();

    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        const auto& w0 = weights[i];
        const auto& w1 = weights[i + 1];

        __m128d c    = _mm_set_pd(w1.confidence_score,  w0.confidence_score);
        __m128d s    = _mm_set_pd(w1.sentiment_score,   w0.sentiment_score);
        __m128d v    = _mm_set_pd(w1.volatility_score,  w0.volatility_score);
        __m128d d    = _mm_set_pd(w1.directional_bias,  w0.directional_bias);

        __m128d sc   = _mm_mul_pd(s, c);
        __m128d vc   = _mm_mul_pd(v, c);
        __m128d dc_v = _mm_mul_pd(d, c);

        // sse2_hadd sums both lanes: [w0_val+w1_val, ...]
        acc_sv = _mm_add_pd(acc_sv, sse2_hadd(sc, vc));
        acc_dc = _mm_add_pd(acc_dc, sse2_hadd(dc_v, c));
    }

    double buf_sv[2], buf_dc[2];
    _mm_storeu_pd(buf_sv, acc_sv);
    _mm_storeu_pd(buf_dc, acc_dc);

    double sum_s = buf_sv[0];
    double sum_v = buf_sv[1];
    double sum_d = buf_dc[0];
    double sum_c = buf_dc[1];

    // Scalar tail for odd n.
    for (; i < n; ++i) {
        const auto& w = weights[i];
        sum_s += w.sentiment_score  * w.confidence_score;
        sum_v += w.volatility_score * w.confidence_score;
        sum_d += w.directional_bias * w.confidence_score;
        sum_c += w.confidence_score;
    }

    SemanticWeight result{};
    if (sum_c > 0.0) {
        result.sentiment_score  = sum_s / sum_c;
        result.volatility_score = sum_v / sum_c;
        result.directional_bias = sum_d / sum_c;
        result.confidence_score = sum_c / static_cast<double>(n);
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