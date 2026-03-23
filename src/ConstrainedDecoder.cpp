// ConstrainedDecoder.cpp — Constrained / guided decoding implementation.

#include "ConstrainedDecoder.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ConstrainedDecoder::ConstrainedDecoder(int vocab_size, int eos_id)
    : vocab_size_(vocab_size), eos_id_(eos_id)
{
    if (vocab_size_ <= 0)
        throw std::invalid_argument("ConstrainedDecoder: vocab_size must be > 0");
    if (eos_id_ < 0 || eos_id_ >= vocab_size_)
        throw std::invalid_argument("ConstrainedDecoder: eos_id out of range");
}

// ---------------------------------------------------------------------------
// Constraint management
// ---------------------------------------------------------------------------

void ConstrainedDecoder::add_constraint(Constraint c)
{
    constraints_.push_back(std::move(c));
}

void ConstrainedDecoder::clear_constraints()
{
    constraints_.clear();
}

// ---------------------------------------------------------------------------
// get_allowed_tokens
// ---------------------------------------------------------------------------

std::vector<int> ConstrainedDecoder::get_allowed_tokens(
    const std::vector<int>& generated_so_far) const
{
    std::vector<bool> allowed(static_cast<std::size_t>(vocab_size_), true);

    for (const auto& c : constraints_) {
        switch (c.type) {

        case ConstraintType::AllowList: {
            std::vector<bool> in_list(static_cast<std::size_t>(vocab_size_), false);
            for (int id : c.data) {
                if (id >= 0 && id < vocab_size_)
                    in_list[static_cast<std::size_t>(id)] = true;
            }
            for (int i = 0; i < vocab_size_; ++i) {
                allowed[static_cast<std::size_t>(i)] =
                    allowed[static_cast<std::size_t>(i)] &&
                    in_list[static_cast<std::size_t>(i)];
            }
            break;
        }

        case ConstraintType::DenyList: {
            for (int id : c.data) {
                if (id >= 0 && id < vocab_size_)
                    allowed[static_cast<std::size_t>(id)] = false;
            }
            break;
        }

        case ConstraintType::MaxLength: {
            if (c.max_len > 0 &&
                static_cast<int>(generated_so_far.size()) >= c.max_len) {
                std::fill(allowed.begin(), allowed.end(), false);
                allowed[static_cast<std::size_t>(eos_id_)] = true;
            }
            break;
        }

        case ConstraintType::ForcePrefix: {
            std::size_t gen_len = generated_so_far.size();
            if (gen_len < c.data.size()) {
                int forced = c.data[gen_len];
                std::fill(allowed.begin(), allowed.end(), false);
                if (forced >= 0 && forced < vocab_size_)
                    allowed[static_cast<std::size_t>(forced)] = true;
                else
                    allowed[static_cast<std::size_t>(eos_id_)] = true;
            }
            break;
        }

        case ConstraintType::Regex:
        case ConstraintType::Grammar:
            // Applied externally when surface text is available.
            break;
        }
    }

    std::vector<int> result;
    result.reserve(128);
    for (int i = 0; i < vocab_size_; ++i) {
        if (allowed[static_cast<std::size_t>(i)])
            result.push_back(i);
    }

    if (result.empty())
        result.push_back(eos_id_);

    return result;
}

// ---------------------------------------------------------------------------
// apply_constraints_to_logits
// ---------------------------------------------------------------------------

void ConstrainedDecoder::apply_constraints_to_logits(
    std::vector<float>& logits,
    const std::vector<int>& generated) const
{
    if (static_cast<int>(logits.size()) != vocab_size_)
        return;

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    const std::vector<int> allowed = get_allowed_tokens(generated);
    std::unordered_set<int> allowed_set(allowed.begin(), allowed.end());

    for (int i = 0; i < vocab_size_; ++i) {
        if (allowed_set.find(i) == allowed_set.end())
            logits[static_cast<std::size_t>(i)] = NEG_INF;
    }
}

// ---------------------------------------------------------------------------
// check_allowlist
// ---------------------------------------------------------------------------

bool ConstrainedDecoder::check_allowlist(const std::vector<int>& generated) const
{
    for (const auto& c : constraints_) {
        if (c.type != ConstraintType::AllowList)
            continue;
        std::unordered_set<int> allowed_set(c.data.begin(), c.data.end());
        for (int id : generated) {
            if (allowed_set.find(id) == allowed_set.end())
                return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// check_max_length
// ---------------------------------------------------------------------------

bool ConstrainedDecoder::check_max_length(const std::vector<int>& generated) const
{
    for (const auto& c : constraints_) {
        if (c.type != ConstraintType::MaxLength)
            continue;
        if (c.max_len > 0 && static_cast<int>(generated.size()) > c.max_len)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// matches_prefix
// ---------------------------------------------------------------------------

bool ConstrainedDecoder::matches_prefix(
    const std::vector<int>& generated,
    const std::vector<int>& prefix) const
{
    if (generated.size() < prefix.size())
        return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (generated[i] != prefix[i])
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// simple_regex_match
// ---------------------------------------------------------------------------

bool ConstrainedDecoder::simple_regex_match(
    const std::string& text,
    const std::string& pattern) const
{
    if (pattern.empty())
        return text.empty();

    bool anchored_start = (!pattern.empty() && pattern[0] == '^');
    bool anchored_end   = (!pattern.empty() && pattern.back() == '$');

    const char* pat = pattern.c_str();
    std::size_t plen = pattern.size();

    if (anchored_start) { ++pat; --plen; }
    if (anchored_end && plen > 0) { --plen; }

    const char* txt = text.c_str();
    std::size_t tlen = text.size();

    if (anchored_start && anchored_end) {
        return regex_match_impl(txt, tlen, pat, plen);
    }
    if (anchored_start) {
        for (std::size_t end = 0; end <= tlen; ++end) {
            if (regex_match_impl(txt, end, pat, plen))
                return true;
        }
        return false;
    }
    if (anchored_end) {
        for (std::size_t start = 0; start <= tlen; ++start) {
            if (regex_match_impl(txt + start, tlen - start, pat, plen))
                return true;
        }
        return false;
    }
    // Substring match
    for (std::size_t start = 0; start <= tlen; ++start) {
        for (std::size_t end = start; end <= tlen; ++end) {
            if (regex_match_impl(txt + start, end - start, pat, plen))
                return true;
        }
    }
    return false;
}

bool ConstrainedDecoder::regex_match_impl(
    const char* text,
    std::size_t tlen,
    const char* pattern,
    std::size_t plen) const
{
    if (plen == 0)
        return tlen == 0;

    bool has_quantifier = (plen >= 2 &&
        (pattern[1] == '*' || pattern[1] == '+' || pattern[1] == '?'));

    if (has_quantifier) {
        char quant = pattern[1];
        char ch    = pattern[0];

        auto matches_char = [&](std::size_t i) -> bool {
            if (i >= tlen) return false;
            return (ch == '.') || (ch == text[i]);
        };

        if (quant == '?') {
            if (matches_char(0)) {
                if (regex_match_impl(text + 1, tlen - 1, pattern + 2, plen - 2))
                    return true;
            }
            return regex_match_impl(text, tlen, pattern + 2, plen - 2);
        }
        if (quant == '*') {
            for (std::size_t i = 0; i <= tlen; ++i) {
                if (regex_match_impl(text + i, tlen - i, pattern + 2, plen - 2))
                    return true;
                if (!matches_char(i))
                    break;
            }
            return false;
        }
        if (quant == '+') {
            if (!matches_char(0))
                return false;
            for (std::size_t i = 1; i <= tlen; ++i) {
                if (regex_match_impl(text + i, tlen - i, pattern + 2, plen - 2))
                    return true;
                if (!matches_char(i))
                    break;
            }
            return false;
        }
    }

    char ch = pattern[0];
    if (tlen == 0)
        return false;
    bool char_match = (ch == '.') || (ch == text[0]);
    if (!char_match)
        return false;
    return regex_match_impl(text + 1, tlen - 1, pattern + 1, plen - 1);
}

// ---------------------------------------------------------------------------
// decode_step
// ---------------------------------------------------------------------------

int ConstrainedDecoder::decode_step(
    std::vector<float>& logits,
    const std::vector<int>& generated) const
{
    apply_constraints_to_logits(logits, generated);

    int best_id = eos_id_;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        if (logits[static_cast<std::size_t>(i)] > best_val) {
            best_val = logits[static_cast<std::size_t>(i)];
            best_id  = i;
        }
    }
    return best_id;
}

} // namespace llmquant
