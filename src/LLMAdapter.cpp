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
    // Populate hit-count map in parallel with the token dictionary.
    token_hit_counts_.reserve(token_weights_.size());
    for (const auto& kv : token_weights_) {
        token_hit_counts_.emplace(kv.first, std::make_unique<std::atomic<uint64_t>>(0));
    }
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
            auto hc = token_hit_counts_.find(token);
            if (hc != token_hit_counts_.end())
                hc->second->fetch_add(1, std::memory_order_relaxed);
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
        auto hc = token_hit_counts_.find(norm);
        if (hc != token_hit_counts_.end())
            hc->second->fetch_add(1, std::memory_order_relaxed);
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
    std::string norm = normalize_token(token);
    token_weights_[norm] = weight;
    if (token_hit_counts_.find(norm) == token_hit_counts_.end())
        token_hit_counts_.emplace(norm, std::make_unique<std::atomic<uint64_t>>(0));
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

size_t LLMAdapter::count_bullish_tokens() const {
    size_t count = 0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.sentiment_score > 0.0) ++count; }
    return count;
}

size_t LLMAdapter::count_bearish_tokens() const {
    size_t count = 0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.sentiment_score < 0.0) ++count; }
    return count;
}

size_t LLMAdapter::count_neutral_tokens() const {
    size_t count = 0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.sentiment_score == 0.0) ++count; }
    return count;
}

double LLMAdapter::get_avg_confidence() const {
    if (token_weights_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; sum += wt.confidence_score; }
    return sum / static_cast<double>(token_weights_.size());
}

double LLMAdapter::get_min_confidence() const {
    if (token_weights_.empty()) return 0.0;
    double mn = std::numeric_limits<double>::max();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.confidence_score < mn) mn = wt.confidence_score; }
    return mn;
}

double LLMAdapter::get_max_confidence() const {
    if (token_weights_.empty()) return 0.0;
    double mx = std::numeric_limits<double>::lowest();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.confidence_score > mx) mx = wt.confidence_score; }
    return mx;
}

double LLMAdapter::get_confidence_range() const {
    return get_max_confidence() - get_min_confidence();
}

double LLMAdapter::get_avg_sentiment() const {
    if (token_weights_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; sum += wt.sentiment_score; }
    return sum / static_cast<double>(token_weights_.size());
}

double LLMAdapter::get_avg_volatility() const {
    if (token_weights_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; sum += wt.volatility_score; }
    return sum / static_cast<double>(token_weights_.size());
}

double LLMAdapter::get_avg_directional_bias() const {
    if (token_weights_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& [tok, wt] : token_weights_) { (void)tok; sum += wt.directional_bias; }
    return sum / static_cast<double>(token_weights_.size());
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

void LLMAdapter::decay_all_weights(double factor) {
    // Clamp to [0, 1] — negative factor zeroes confidence, >1 is not valid decay.
    if (factor < 0.0) factor = 0.0;
    if (factor > 1.0) factor = 1.0;
    for (auto& [tok, wt] : token_weights_) {
        (void)tok;
        wt.confidence_score *= factor;
    }
}

void LLMAdapter::clear_custom_mappings() {
    // Clears ALL mappings (both built-in and custom) since there is no distinction
    // between them in the map. Call initialize_default_mappings() after if needed.
    token_weights_.clear();
    token_hit_counts_.clear();
}

bool LLMAdapter::contains_token(const std::string& token) const {
    return token_weights_.count(normalize_token(token)) > 0;
}

bool LLMAdapter::contains_any_of(const std::vector<std::string>& tokens) const {
    for (const auto& tok : tokens) {
        if (token_weights_.count(normalize_token(tok)) > 0) return true;
    }
    return false;
}

bool LLMAdapter::remove_token_mapping(const std::string& token) {
    std::string norm = normalize_token(token);
    token_hit_counts_.erase(norm);
    return token_weights_.erase(norm) > 0;
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

std::string LLMAdapter::export_dictionary() const {
    if (token_weights_.empty()) return {};

    // Sort tokens alphabetically for deterministic output.
    std::vector<std::string> keys;
    keys.reserve(token_weights_.size());
    for (const auto& [tok, _] : token_weights_) keys.push_back(tok);
    std::sort(keys.begin(), keys.end());

    std::ostringstream ss;
    ss.precision(6);
    ss << std::fixed;
    for (const auto& key : keys) {
        const auto& w = token_weights_.at(key);
        ss << key << '\t'
           << w.sentiment_score  << '\t'
           << w.confidence_score << '\t'
           << w.volatility_score << '\t'
           << w.directional_bias << '\n';
    }
    return ss.str();
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

std::vector<std::pair<std::string, double>>
LLMAdapter::filter_tokens_by_volatility(double min_volatility, double max_volatility) const {
    std::vector<std::pair<std::string, double>> result;
    for (const auto& [tok, weight] : token_weights_) {
        if (weight.volatility_score >= min_volatility && weight.volatility_score <= max_volatility) {
            result.emplace_back(tok, weight.volatility_score);
        }
    }
    return result;
}

std::vector<std::pair<std::string, double>>
LLMAdapter::filter_tokens_by_sentiment(double min_sentiment, double max_sentiment) const {
    std::vector<std::pair<std::string, double>> result;
    for (const auto& [tok, weight] : token_weights_) {
        if (weight.sentiment_score >= min_sentiment && weight.sentiment_score <= max_sentiment) {
            result.emplace_back(tok, weight.sentiment_score);
        }
    }
    return result;
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

    // Corporate / earnings events
    add_token_mapping("earnings",   { 0.1, 0.80, 0.75,  0.10});
    add_token_mapping("guidance",   { 0.2, 0.75, 0.55,  0.20});
    add_token_mapping("upgrade",    { 0.7, 0.85, 0.50,  0.70});
    add_token_mapping("downgrade",  {-0.7, 0.85, 0.50, -0.70});
    add_token_mapping("beats",      { 0.7, 0.85, 0.60,  0.65});
    add_token_mapping("misses",     {-0.7, 0.85, 0.60, -0.65});
    add_token_mapping("outlook",    { 0.1, 0.70, 0.45,  0.10});
    add_token_mapping("revenue",    { 0.2, 0.70, 0.40,  0.15});
    add_token_mapping("profit",     { 0.5, 0.80, 0.40,  0.45});
    add_token_mapping("loss",       {-0.5, 0.80, 0.50, -0.50});
    add_token_mapping("dividend",   { 0.4, 0.75, 0.20,  0.35});
    add_token_mapping("buyback",    { 0.5, 0.80, 0.30,  0.50});
    add_token_mapping("merger",     { 0.3, 0.75, 0.65,  0.30});
    add_token_mapping("acquisition",{ 0.3, 0.75, 0.65,  0.35});
    add_token_mapping("ipo",        { 0.4, 0.70, 0.70,  0.40});

    // Market-regime / macro signals
    add_token_mapping("risk-on",    { 0.6, 0.80, 0.55,  0.60});
    add_token_mapping("risk-off",   {-0.6, 0.80, 0.60, -0.60});
    add_token_mapping("systemic",   {-0.5, 0.80, 0.75, -0.50});
    add_token_mapping("contagion",  {-0.7, 0.85, 0.80, -0.70});
    add_token_mapping("stimulus",   { 0.5, 0.80, 0.50,  0.50});
    add_token_mapping("tightening", {-0.3, 0.80, 0.55, -0.35});
    add_token_mapping("easing",     { 0.4, 0.80, 0.50,  0.40});
    add_token_mapping("default",    {-0.8, 0.85, 0.80, -0.80});
    add_token_mapping("sanctions",  {-0.5, 0.80, 0.70, -0.45});
    add_token_mapping("tariff",     {-0.3, 0.75, 0.60, -0.30});
    add_token_mapping("deregulation",{ 0.4, 0.70, 0.45, 0.40});
    add_token_mapping("geopolitical",{-0.3, 0.75, 0.70,-0.25});

    // Analyst sentiment
    add_token_mapping("overweight",  { 0.6, 0.80, 0.35,  0.60});
    add_token_mapping("underweight", {-0.6, 0.80, 0.35, -0.60});
    add_token_mapping("outperform",  { 0.6, 0.80, 0.35,  0.60});
    add_token_mapping("underperform",{-0.6, 0.80, 0.35, -0.60});
    add_token_mapping("neutral",     { 0.0, 0.60, 0.15,  0.00});
    add_token_mapping("hold",        { 0.0, 0.65, 0.20,  0.00});
    add_token_mapping("target",      { 0.1, 0.60, 0.30,  0.10});

    // Common neutral filler (additional)
    add_token_mapping("or",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("not",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("with",       {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("for",        {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("as",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("at",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("on",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("it",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("by",         {0.0,  0.1,  0.0,  0.0});
    add_token_mapping("from",       {0.0,  0.1,  0.0,  0.0});

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

std::vector<SemanticWeight> LLMAdapter::batch_map_tokens_to_weights(
        const std::vector<std::string>& tokens) const {
    std::vector<SemanticWeight> results;
    results.reserve(tokens.size());
    for (const auto& tok : tokens)
        results.push_back(map_token_to_weight(tok));
    return results;
}

size_t LLMAdapter::load_dictionary_from_tsv(const std::string& tsv_data) {
    size_t imported = 0;
    std::istringstream ss(tsv_data);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string token;
        double sentiment{}, confidence{}, volatility{}, bias{};
        if (!std::getline(ls, token, '\t')) continue;
        if (!(ls >> sentiment)) continue;
        char sep{}; ls >> sep;
        if (!(ls >> confidence)) continue;
        ls >> sep;
        if (!(ls >> volatility)) continue;
        ls >> sep;
        if (!(ls >> bias)) continue;
        if (!std::isfinite(sentiment) || !std::isfinite(confidence) ||
            !std::isfinite(volatility) || !std::isfinite(bias)) continue;
        SemanticWeight w{sentiment, confidence, volatility, bias};
        auto key = normalize_token(token);
        token_weights_[key] = w;
        ++imported;
    }
    return imported;
}

double LLMAdapter::get_sentiment_range() const {
    if (token_weights_.empty()) return 0.0;
    return get_max_sentiment() - get_min_sentiment();
}

double LLMAdapter::get_min_sentiment() const {
    if (token_weights_.empty()) return 0.0;
    double mn = std::numeric_limits<double>::max();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.sentiment_score < mn) mn = wt.sentiment_score; }
    return mn;
}

double LLMAdapter::get_max_sentiment() const {
    if (token_weights_.empty()) return 0.0;
    double mx = std::numeric_limits<double>::lowest();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.sentiment_score > mx) mx = wt.sentiment_score; }
    return mx;
}

double LLMAdapter::get_min_volatility() const {
    if (token_weights_.empty()) return 0.0;
    double mn = std::numeric_limits<double>::max();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.volatility_score < mn) mn = wt.volatility_score; }
    return mn;
}

double LLMAdapter::get_max_volatility() const {
    if (token_weights_.empty()) return 0.0;
    double mx = std::numeric_limits<double>::lowest();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.volatility_score > mx) mx = wt.volatility_score; }
    return mx;
}

double LLMAdapter::get_min_directional_bias() const {
    if (token_weights_.empty()) return 0.0;
    double mn = std::numeric_limits<double>::max();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.directional_bias < mn) mn = wt.directional_bias; }
    return mn;
}

double LLMAdapter::get_max_directional_bias() const {
    if (token_weights_.empty()) return 0.0;
    double mx = std::numeric_limits<double>::lowest();
    for (const auto& [tok, wt] : token_weights_) { (void)tok; if (wt.directional_bias > mx) mx = wt.directional_bias; }
    return mx;
}

double LLMAdapter::get_volatility_range() const {
    return get_max_volatility() - get_min_volatility();
}

double LLMAdapter::get_directional_bias_range() const {
    return get_max_directional_bias() - get_min_directional_bias();
}

std::vector<std::pair<std::string, double>>
LLMAdapter::filter_tokens_by_confidence(double min_confidence, double max_confidence) const {
    std::vector<std::pair<std::string, double>> result;
    for (const auto& [tok, weight] : token_weights_) {
        if (weight.confidence_score >= min_confidence && weight.confidence_score <= max_confidence)
            result.emplace_back(tok, weight.confidence_score);
    }
    return result;
}

std::vector<std::pair<std::string, double>>
LLMAdapter::top_tokens_by_volatility(size_t n) const {
    std::vector<std::pair<std::string, double>> result;
    result.reserve(token_weights_.size());
    for (const auto& [tok, weight] : token_weights_)
        result.emplace_back(tok, weight.volatility_score);
    size_t take = std::min(n, result.size());
    std::partial_sort(result.begin(),
                      result.begin() + static_cast<std::ptrdiff_t>(take),
                      result.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    result.resize(take);
    return result;
}

size_t LLMAdapter::count_tokens_above_volatility(double threshold) const {
    size_t count = 0;
    for (const auto& [tok, wt] : token_weights_) {
        (void)tok;
        if (wt.volatility_score > threshold) ++count;
    }
    return count;
}

std::string LLMAdapter::format_stats() const {
    uint64_t processed = stats_.tokens_processed.load(std::memory_order_relaxed);
    if (processed == 0) return "tokens=0 (no data)";
    uint64_t hits   = stats_.cache_hits.load(std::memory_order_relaxed);
    uint64_t misses = stats_.cache_misses.load(std::memory_order_relaxed);
    double hit_rate = static_cast<double>(hits) / static_cast<double>(processed);
    std::ostringstream oss;
    oss << "tokens=" << processed
        << " hits="  << hits
        << " misses=" << misses
        << " hit_rate=" << hit_rate
        << " dict_size=" << token_weights_.size();
    return oss.str();
}

uint64_t LLMAdapter::get_token_hit_count(const std::string& token) const {
    auto norm = normalize_token(token);
    auto it = token_hit_counts_.find(norm);
    return (it != token_hit_counts_.end()) ? it->second->load(std::memory_order_relaxed) : 0;
}

std::vector<std::pair<std::string, uint64_t>> LLMAdapter::top_tokens_by_frequency(size_t n) const {
    std::vector<std::pair<std::string, uint64_t>> freq;
    freq.reserve(token_hit_counts_.size());
    for (const auto& kv : token_hit_counts_) {
        freq.emplace_back(kv.first, kv.second->load(std::memory_order_relaxed));
    }
    size_t result_size = std::min(n, freq.size());
    std::partial_sort(freq.begin(), freq.begin() + static_cast<std::ptrdiff_t>(result_size), freq.end(),
        [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
            return a.second > b.second;
        });
    freq.resize(result_size);
    return freq;
}

void LLMAdapter::reset_frequency_counts() {
    for (auto& kv : token_hit_counts_) {
        kv.second->store(0, std::memory_order_relaxed);
    }
}

std::vector<std::pair<std::string, double>>
LLMAdapter::filter_tokens_by_directional_bias(double min_bias, double max_bias) const {
    std::vector<std::pair<std::string, double>> result;
    for (const auto& [tok, weight] : token_weights_) {
        if (weight.directional_bias >= min_bias && weight.directional_bias <= max_bias) {
            result.emplace_back(tok, weight.directional_bias);
        }
    }
    return result;
}

std::vector<std::pair<std::string, double>>
LLMAdapter::top_tokens_by_directional_bias(size_t n) const {
    std::vector<std::pair<std::string, double>> result;
    result.reserve(token_weights_.size());
    for (const auto& [tok, weight] : token_weights_)
        result.emplace_back(tok, weight.directional_bias);
    size_t take = std::min(n, result.size());
    std::partial_sort(result.begin(),
                      result.begin() + static_cast<std::ptrdiff_t>(take),
                      result.end(),
                      [](const auto& a, const auto& b) {
                          return std::abs(a.second) > std::abs(b.second);
                      });
    result.resize(take);
    return result;
}

} // namespace llmquant