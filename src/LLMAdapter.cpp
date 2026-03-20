#include "LLMAdapter.h"
#include <cmath>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <immintrin.h>  // SSE2/AVX2 intrinsics

namespace llmquant {

LLMAdapter::LLMAdapter() {
    initialize_default_mappings();
}

std::string LLMAdapter::normalize_token(const std::string& token) {
    std::string norm;
    norm.reserve(token.size());
    size_t start = 0;
    while (start < token.size() && std::isspace(static_cast<unsigned char>(token[start]))) ++start;
    size_t end = token.size();
    while (end > start && std::isspace(static_cast<unsigned char>(token[end - 1]))) --end;
    for (size_t i = start; i < end; ++i)
        norm += static_cast<char>(std::tolower(static_cast<unsigned char>(token[i])));
    return norm;
}

SemanticWeight LLMAdapter::map_token_to_weight(const std::string& token) const {
    // tokens_processed is incremented AFTER the lookup so the invariant
    // cache_hits + cache_misses == tokens_processed always holds.

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

    // Normalize via shared helper: strip leading/trailing whitespace, lowercase.
    // GPT-4o streams tokens like " bullish" or "Bullish" that must map to "bullish".
    std::string norm = normalize_token(token);

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
    // Normalise key via shared helper to match map_token_to_weight() lookup behaviour.
    token_weights_[normalize_token(token)] = weight;
}

// SSE2-only horizontal add: returns [a[0]+a[1], b[0]+b[1]]
static inline __m128d sse2_hadd(__m128d a, __m128d b) {
    return _mm_add_pd(_mm_unpacklo_pd(a, b), _mm_unpackhi_pd(a, b));
}

SemanticWeight LLMAdapter::map_sequence_simd(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return SemanticWeight{0.0, 0.0, 0.0, 0.0};

    constexpr size_t kMaxSequenceTokens = 1'000'000;
    const size_t effective_size = tokens.size() > kMaxSequenceTokens
        ? (spdlog::warn("LLMAdapter: sequence length {} exceeds limit {}, truncating",
                        tokens.size(), kMaxSequenceTokens),
           kMaxSequenceTokens)
        : tokens.size();

    std::vector<SemanticWeight> weights;
    weights.reserve(effective_size);
    for (size_t idx = 0; idx < effective_size; ++idx)
        weights.push_back(map_token_to_weight(tokens[idx]));

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

size_t LLMAdapter::get_dictionary_size() const {
    return token_weights_.size();
}

std::vector<std::string> LLMAdapter::get_all_token_keys() const {
    std::vector<std::string> keys;
    keys.reserve(token_weights_.size());
    for (const auto& [tok, wt] : token_weights_) {
        (void)wt;
        keys.push_back(tok);
    }
    return keys;
}

void LLMAdapter::clear_custom_mappings() {
    // Clears ALL mappings (both built-in and custom) since there is no distinction
    // between them in the map. Call initialize_default_mappings() after if needed.
    token_weights_.clear();
}

bool LLMAdapter::contains_token(const std::string& token) const {
    return token_weights_.count(normalize_token(token)) > 0;
}

bool LLMAdapter::remove_token_mapping(const std::string& token) {
    return token_weights_.erase(normalize_token(token)) > 0;
}

bool LLMAdapter::get_token_mapping(const std::string& token, SemanticWeight& weight) const {
    auto it = token_weights_.find(normalize_token(token));
    if (it == token_weights_.end()) return false;
    weight = it->second;
    return true;
}

bool LLMAdapter::update_token_weight(const std::string& token, const SemanticWeight& weight) {
    auto it = token_weights_.find(normalize_token(token));
    if (it == token_weights_.end()) return false;
    it->second = weight;
    return true;
}

size_t LLMAdapter::batch_add_token_mappings(
    const std::unordered_map<std::string, SemanticWeight>& mappings) {
    size_t inserted = 0;
    for (const auto& [tok, wt] : mappings) {
        auto key = normalize_token(tok);
        if (token_weights_.find(key) == token_weights_.end()) ++inserted;
        token_weights_[key] = wt;
    }
    return inserted;
}

LLMAdapter::SentimentDistribution LLMAdapter::get_sentiment_distribution() const {
    SentimentDistribution dist;
    if (token_weights_.empty()) return dist;

    double sum_sentiment  = 0.0;
    double sum_confidence = 0.0;

    for (const auto& [token, weight] : token_weights_) {
        sum_sentiment  += weight.sentiment_score;
        sum_confidence += weight.confidence_score;
        if (weight.sentiment_score < -0.1)
            ++dist.negative_count;
        else if (weight.sentiment_score > 0.1)
            ++dist.positive_count;
        else
            ++dist.neutral_count;
    }

    double n = static_cast<double>(token_weights_.size());
    dist.mean_sentiment  = sum_sentiment  / n;
    dist.mean_confidence = sum_confidence / n;
    return dist;
}

std::vector<std::pair<std::string, double>>
LLMAdapter::top_tokens_by_sentiment(size_t n) const {
    std::vector<std::pair<std::string, double>> result;
    result.reserve(token_weights_.size());
    for (const auto& [tok, weight] : token_weights_) {
        result.emplace_back(tok, weight.sentiment_score);
    }
    // Partial sort: bring the top n (by |sentiment|) to the front in O(N log n).
    size_t take = std::min(n, result.size());
    std::partial_sort(result.begin(),
                      result.begin() + static_cast<std::ptrdiff_t>(take),
                      result.end(),
                      [](const auto& a, const auto& b) {
                          return std::fabs(a.second) > std::fabs(b.second);
                      });
    result.resize(take);
    return result;
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

    // Options market terms
    add_token_mapping("calls",     { 0.5,  0.80, 0.55,  0.65});
    add_token_mapping("puts",      {-0.5,  0.80, 0.55, -0.65});
    add_token_mapping("straddle",  { 0.0,  0.70, 0.90,  0.0});
    add_token_mapping("strangle",  { 0.0,  0.65, 0.88,  0.0});
    add_token_mapping("gamma",     { 0.0,  0.75, 0.85,  0.0});
    add_token_mapping("delta",     { 0.1,  0.65, 0.50,  0.1});
    add_token_mapping("vega",      { 0.0,  0.70, 0.92,  0.0});
    add_token_mapping("iv",        { 0.0,  0.75, 0.90,  0.0});  // implied volatility
    add_token_mapping("dte",       { 0.0,  0.50, 0.40,  0.0});  // days to expiry

    // Macroeconomic sentiment
    add_token_mapping("inflation",  {-0.4,  0.85, 0.70, -0.35});
    add_token_mapping("deflation",  {-0.3,  0.80, 0.60, -0.25});
    add_token_mapping("recession",  {-0.8,  0.90, 0.75, -0.80});
    add_token_mapping("stagflation",{-0.7,  0.85, 0.80, -0.70});
    add_token_mapping("fed",        { 0.0,  0.75, 0.65,  0.0});
    add_token_mapping("rate",       { 0.0,  0.65, 0.50,  0.0});
    add_token_mapping("hike",       {-0.4,  0.80, 0.60, -0.40});
    add_token_mapping("cut",        { 0.4,  0.80, 0.55,  0.40});
    add_token_mapping("pivot",      { 0.5,  0.85, 0.65,  0.50});
    add_token_mapping("gdp",        { 0.1,  0.65, 0.40,  0.10});

    // Risk-off / safe-haven flows
    add_token_mapping("liquidation",{-0.9,  0.90, 0.90, -0.90});
    add_token_mapping("capitulation",{-0.8, 0.85, 0.85, -0.85});
    add_token_mapping("squeeze",    { 0.6,  0.80, 0.80,  0.70});  // short squeeze
    add_token_mapping("deleveraging",{-0.7, 0.85, 0.80, -0.75});
    add_token_mapping("margin",     {-0.3,  0.70, 0.65, -0.30});  // margin call context

    // Positive recovery / accumulation signals
    add_token_mapping("accumulate", { 0.6,  0.80, 0.35,  0.65});
    add_token_mapping("dip",        { 0.3,  0.75, 0.50,  0.40});  // buy the dip
    add_token_mapping("rebound",    { 0.6,  0.80, 0.60,  0.65});
    add_token_mapping("recovery",   { 0.5,  0.80, 0.45,  0.50});
    add_token_mapping("uptrend",    { 0.6,  0.80, 0.40,  0.70});
    add_token_mapping("downtrend",  {-0.6,  0.80, 0.40, -0.70});
    add_token_mapping("oversold",   { 0.5,  0.75, 0.55,  0.55});
    add_token_mapping("overbought", {-0.5,  0.75, 0.55, -0.55});

    // Neutral / Filler — zero-weight pass-through tokens
    add_token_mapping("the",       {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("and",       {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("is",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("a",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("an",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("in",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("of",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("to",        {0.0,  0.1,  0.0,  0.0});

    // Options / derivatives — additional terms (expiry, strike, hedge are new;
    // calls/puts/squeeze/gamma/delta already defined above with better values)
    add_token_mapping("expiry",     { 0.0, 0.50, 0.70,  0.0});
    add_token_mapping("strike",     { 0.0, 0.50, 0.40,  0.0});
    add_token_mapping("hedge",      {-0.1, 0.70, 0.30, -0.10});

    // Retail / crypto sentiment (increasingly common in Reddit feeds)
    add_token_mapping("pump",       { 0.7, 0.75, 0.80,  0.80});
    add_token_mapping("rug",        {-0.9, 0.85, 0.85, -0.90});
    add_token_mapping("fud",        {-0.6, 0.70, 0.60, -0.55});
    add_token_mapping("hodl",       { 0.3, 0.60, 0.20,  0.40});
    add_token_mapping("rekt",       {-0.8, 0.85, 0.80, -0.80});
    add_token_mapping("ath",        { 0.7, 0.75, 0.65,  0.70});

    // Technical analysis — trend / exhaustion signals (new; overbought/oversold/
    // capitulation/dip/margin already defined above with better values)
    add_token_mapping("accumulation",{ 0.5, 0.75, 0.35,  0.55});
    add_token_mapping("distribution",{-0.4, 0.70, 0.45, -0.45});
    add_token_mapping("reversal",   { 0.0, 0.70, 0.80,  0.0});
    add_token_mapping("consolidate",{ 0.0, 0.55, 0.25,  0.0});
    add_token_mapping("parabolic",  { 0.6, 0.75, 0.90,  0.65});
    add_token_mapping("divergence", { 0.0, 0.65, 0.70,  0.0});
    add_token_mapping("liquidated", {-0.7, 0.85, 0.85, -0.75});
}

} // namespace llmquant