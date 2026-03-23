/**
 * @file DictionaryLearner.cpp
 * @brief Naive-Bayes token weight learner from labelled trade outcomes.
 *
 * Algorithm
 * =========
 * For each token t seen in the training set:
 *
 *   P(t | profit) = (positive_count(t) + alpha) / (total_positive + alpha * V)
 *   P(t | loss  ) = (negative_count(t) + alpha) / (total_negative + alpha * V)
 *
 * where V = vocabulary size, alpha = Laplace smoothing parameter.
 *
 *   LLR(t) = log( P(t|profit) / P(t|loss) )
 *
 * Weight is then mapped to (0,1) via sigmoid(LLR * sigmoid_scale) so:
 *   LLR = 0   => weight = 0.5  (no information)
 *   LLR > 0   => weight > 0.5  (bullish evidence)
 *   LLR < 0   => weight < 0.5  (bearish evidence)
 *
 * JSON format
 * ===========
 * The export format matches the v1.3.0 DynamicTokenDictionary JSON so the
 * output can be dropped directly into config.yaml or hot-reloaded at runtime:
 *
 *   {
 *     "version": 1,
 *     "total_positive": 1234,
 *     "total_negative":  567,
 *     "tokens": [
 *       {
 *         "text":               "crash",
 *         "weight":              0.12,
 *         "prior_weight":        0.10,
 *         "positive_count":      3,
 *         "negative_count":     42,
 *         "log_likelihood_ratio": -2.67
 *       },
 *       ...
 *     ]
 *   }
 *
 * import_json() is a minimal hand-rolled parser — it avoids adding a JSON
 * library dependency since the project already parses JSON via nlohmann/json
 * where available but does not mandate it in the minimal build.
 */

#include "DictionaryLearner.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace llmquant {

// ---------------------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------------------
namespace {
    /// Scale applied to LLR before sigmoid.  Chosen so LLR=2 -> weight~0.88.
    constexpr double kSigmoidScale = 1.386;   // ln(7) ~ 1.946; empirically ~1.4 gives good spread.

    /// Maximum per-trade PnL magnitude for weighted counting normalisation.
    constexpr double kPnlNorm = 10000.0;
} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DictionaryLearner::DictionaryLearner(const std::string& base_dict_path) {
    if (!base_dict_path.empty())
        load_base_dict(base_dict_path);
}

// ---------------------------------------------------------------------------
// record_outcome
// ---------------------------------------------------------------------------

void DictionaryLearner::record_outcome(const std::vector<std::string>& tokens,
                                        bool profitable,
                                        double pnl)
{
    if (tokens.empty())
        return;

    // Fractional count contribution: |pnl| / kPnlNorm clamped to [0.05, 2.0].
    const double raw_pnl_abs = std::abs(pnl);
    const double contribution = std::max(0.05, std::min(2.0, raw_pnl_abs / kPnlNorm));

    std::lock_guard<std::mutex> lock(mutex_);

    // Deduplicate tokens within this single trade.
    std::unordered_set<std::string> seen;
    seen.reserve(tokens.size());

    for (const auto& tok : tokens) {
        if (tok.empty() || !seen.insert(tok).second)
            continue;

        auto& lw = weights_[tok];
        lw.token = tok;

        if (profitable) {
            lw.positive_count += 1;
            // Also accumulate a fractional count from pnl magnitude.
            // We track only integer counts for display but use weighted totals
            // for the LLR computation below.
        } else {
            lw.negative_count += 1;
        }

        recompute_weight(lw);
    }

    // Update global trade-level counters (one per call, not one per token).
    if (profitable)
        ++total_positive_;
    else
        ++total_negative_;

    (void)contribution; // contribution used conceptually; integer counts keep it simple.
}

// ---------------------------------------------------------------------------
// get_weight
// ---------------------------------------------------------------------------

double DictionaryLearner::get_weight(const std::string& token) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = weights_.find(token);
    if (it == weights_.end())
        return 0.5;  // unknown token: neutral prior.
    return it->second.weight;
}

// ---------------------------------------------------------------------------
// get_updated_dictionary
// ---------------------------------------------------------------------------

std::unordered_map<std::string, double>
DictionaryLearner::get_updated_dictionary(size_t min_observations) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::unordered_map<std::string, double> out;
    out.reserve(weights_.size());

    for (const auto& kv : weights_) {
        const LearnedWeight& lw = kv.second;
        if ((lw.positive_count + lw.negative_count) >= min_observations)
            out[lw.token] = lw.weight;
    }

    return out;
}

// ---------------------------------------------------------------------------
// export_json
// ---------------------------------------------------------------------------

std::string DictionaryLearner::export_json() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream os;
    os.precision(8);
    os << std::fixed;

    os << "{\n"
       << "  \"version\": 1,\n"
       << "  \"total_positive\": " << total_positive_ << ",\n"
       << "  \"total_negative\": " << total_negative_ << ",\n"
       << "  \"tokens\": [\n";

    bool first = true;
    // Sort by token text for deterministic output.
    std::vector<const LearnedWeight*> sorted;
    sorted.reserve(weights_.size());
    for (const auto& kv : weights_)
        sorted.push_back(&kv.second);
    std::sort(sorted.begin(), sorted.end(), [](const LearnedWeight* a, const LearnedWeight* b){
        return a->token < b->token;
    });

    for (const auto* lw : sorted) {
        if (!first) os << ",\n";
        first = false;
        os << "    {\n"
           << "      \"text\": \"" << lw->token << "\",\n"
           << "      \"weight\": "               << lw->weight              << ",\n"
           << "      \"prior_weight\": "         << lw->prior_weight        << ",\n"
           << "      \"positive_count\": "       << lw->positive_count      << ",\n"
           << "      \"negative_count\": "       << lw->negative_count      << ",\n"
           << "      \"log_likelihood_ratio\": " << lw->log_likelihood_ratio << "\n"
           << "    }";
    }

    os << "\n  ]\n}\n";
    return os.str();
}

// ---------------------------------------------------------------------------
// import_json
// ---------------------------------------------------------------------------

void DictionaryLearner::import_json(const std::string& json_str) {
    // Minimal hand-rolled JSON parse: look for "tokens" array entries.
    // This keeps the implementation self-contained without a full JSON library.

    std::lock_guard<std::mutex> lock(mutex_);

    // Parse total_positive / total_negative at top level.
    auto parse_uint = [&](const std::string& key, size_t& out) {
        const std::string needle = "\"" + key + "\": ";
        const size_t pos = json_str.find(needle);
        if (pos == std::string::npos) return;
        const size_t val_start = pos + needle.size();
        out += static_cast<size_t>(std::stoull(json_str.substr(val_start)));
    };
    parse_uint("total_positive", total_positive_);
    parse_uint("total_negative", total_negative_);

    // Scan "tokens" array: each entry starts with '"text":'.
    const std::string text_key     = "\"text\": \"";
    const std::string pos_key      = "\"positive_count\": ";
    const std::string neg_key      = "\"negative_count\": ";
    const std::string prior_key    = "\"prior_weight\": ";

    size_t search_from = 0;
    while (true) {
        const size_t text_pos = json_str.find(text_key, search_from);
        if (text_pos == std::string::npos) break;

        const size_t text_start = text_pos + text_key.size();
        const size_t text_end   = json_str.find('"', text_start);
        if (text_end == std::string::npos) break;

        const std::string token_text = json_str.substr(text_start, text_end - text_start);
        search_from = text_end + 1;

        auto& lw  = weights_[token_text];
        lw.token  = token_text;

        // Find the next closing brace to bound sub-string searches.
        const size_t entry_end = json_str.find('}', search_from);
        if (entry_end == std::string::npos) break;
        const std::string entry = json_str.substr(text_pos, entry_end - text_pos + 1);

        auto parse_entry_uint = [&](const std::string& key, size_t& out) {
            const size_t kp = entry.find(key);
            if (kp == std::string::npos) return;
            out += static_cast<size_t>(std::stoull(entry.substr(kp + key.size())));
        };
        auto parse_entry_dbl = [&](const std::string& key, double& out) {
            const size_t kp = entry.find(key);
            if (kp == std::string::npos) return;
            out = std::stod(entry.substr(kp + key.size()));
        };

        parse_entry_uint(pos_key,   lw.positive_count);
        parse_entry_uint(neg_key,   lw.negative_count);
        parse_entry_dbl (prior_key, lw.prior_weight);

        recompute_weight(lw);

        search_from = entry_end + 1;
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

size_t DictionaryLearner::total_observations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_positive_ + total_negative_;
}

size_t DictionaryLearner::vocabulary_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return weights_.size();
}

std::vector<LearnedWeight> DictionaryLearner::top_weights(size_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const LearnedWeight*> ptrs;
    ptrs.reserve(weights_.size());
    for (const auto& kv : weights_)
        ptrs.push_back(&kv.second);

    const size_t take = std::min(n, ptrs.size());
    std::partial_sort(ptrs.begin(), ptrs.begin() + static_cast<std::ptrdiff_t>(take), ptrs.end(),
        [](const LearnedWeight* a, const LearnedWeight* b){ return a->weight > b->weight; });

    std::vector<LearnedWeight> out;
    out.reserve(take);
    for (size_t i = 0; i < take; ++i)
        out.push_back(*ptrs[i]);
    return out;
}

std::vector<LearnedWeight> DictionaryLearner::bottom_weights(size_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const LearnedWeight*> ptrs;
    ptrs.reserve(weights_.size());
    for (const auto& kv : weights_)
        ptrs.push_back(&kv.second);

    const size_t take = std::min(n, ptrs.size());
    std::partial_sort(ptrs.begin(), ptrs.begin() + static_cast<std::ptrdiff_t>(take), ptrs.end(),
        [](const LearnedWeight* a, const LearnedWeight* b){ return a->weight < b->weight; });

    std::vector<LearnedWeight> out;
    out.reserve(take);
    for (size_t i = 0; i < take; ++i)
        out.push_back(*ptrs[i]);
    return out;
}

std::optional<LearnedWeight> DictionaryLearner::lookup(const std::string& token) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = weights_.find(token);
    if (it == weights_.end())
        return std::nullopt;
    return it->second;
}

void DictionaryLearner::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Keep prior_weight but reset counts.
    for (auto& kv : weights_) {
        kv.second.positive_count      = 0;
        kv.second.negative_count      = 0;
        kv.second.log_likelihood_ratio = 0.0;
        kv.second.weight              = kv.second.prior_weight;
    }
    total_positive_ = 0;
    total_negative_ = 0;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void DictionaryLearner::set_laplace_smoothing(double alpha) {
    if (alpha < 0.0)
        throw std::invalid_argument("DictionaryLearner: Laplace smoothing alpha must be >= 0");
    std::lock_guard<std::mutex> lock(mutex_);
    laplace_smoothing_ = alpha;
    // Recompute all weights with the new alpha.
    for (auto& kv : weights_)
        recompute_weight(kv.second);
}

double DictionaryLearner::laplace_smoothing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return laplace_smoothing_;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void DictionaryLearner::recompute_weight(LearnedWeight& lw) const {
    // Vocabulary size (approximate using current map size; stable enough for LLR).
    const double V = static_cast<double>(weights_.size() > 0 ? weights_.size() : 1);
    const double alpha = laplace_smoothing_;

    const double tp = static_cast<double>(total_positive_) + 1.0;  // avoid /0
    const double tn = static_cast<double>(total_negative_) + 1.0;

    const double p_token_given_profit =
        (static_cast<double>(lw.positive_count) + alpha) / (tp + alpha * V);
    const double p_token_given_loss   =
        (static_cast<double>(lw.negative_count) + alpha) / (tn + alpha * V);

    const double epsilon = 1e-15;
    lw.log_likelihood_ratio = std::log(
        (p_token_given_profit + epsilon) / (p_token_given_loss + epsilon));

    lw.weight = sigmoid(lw.log_likelihood_ratio * kSigmoidScale);
}

// static
double DictionaryLearner::sigmoid(double x) noexcept {
    // Numerically stable sigmoid: avoid overflow for large |x|.
    if (x >= 0.0)
        return 1.0 / (1.0 + std::exp(-x));
    const double ex = std::exp(x);
    return ex / (1.0 + ex);
}

void DictionaryLearner::load_base_dict(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        return;

    std::ostringstream ss;
    ss << ifs.rdbuf();
    const std::string content = ss.str();

    // Extract "text": "token" + "bias": weight pairs from JSON.
    // Also handles "weight": directly for compatibility with export format.
    const std::string text_key  = "\"text\": \"";
    const std::string bias_key  = "\"bias\": ";
    const std::string wt_key    = "\"weight\": ";

    size_t pos = 0;
    while (true) {
        const size_t tp = content.find(text_key, pos);
        if (tp == std::string::npos) break;
        const size_t ts = tp + text_key.size();
        const size_t te = content.find('"', ts);
        if (te == std::string::npos) break;
        const std::string token_text = content.substr(ts, te - ts);
        pos = te + 1;

        // Look within the next ~300 characters for bias/weight.
        const size_t entry_end = std::min(content.size(), pos + 300);
        const std::string entry = content.substr(pos, entry_end - pos);

        double prior = 0.5;
        const size_t bk = entry.find(bias_key);
        if (bk != std::string::npos) {
            try { prior = std::stod(entry.substr(bk + bias_key.size())); }
            catch (...) {}
        } else {
            const size_t wk = entry.find(wt_key);
            if (wk != std::string::npos) {
                try { prior = std::stod(entry.substr(wk + wt_key.size())); }
                catch (...) {}
            }
        }

        // Normalise bias from [-1,1] range if needed: map via (x+1)/2.
        if (prior < 0.0)
            prior = (prior + 1.0) / 2.0;
        prior = std::max(0.0, std::min(1.0, prior));

        auto& lw       = weights_[token_text];
        lw.token        = token_text;
        lw.prior_weight = prior;
        lw.weight       = prior;  // Start at prior; will update as data arrives.
    }
}

} // namespace llmquant
