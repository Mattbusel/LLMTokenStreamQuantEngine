#pragma once

#include <string>
#include <vector>

namespace llmquant {

/// Types of decoding constraints that can be applied.
enum class ConstraintType {
    Regex,       ///< Allow tokens whose decoded text matches a regex pattern.
    Grammar,     ///< Allow tokens consistent with a context-free grammar (simplified).
    AllowList,   ///< Only allow tokens explicitly in the allow list.
    DenyList,    ///< Forbid tokens explicitly in the deny list.
    MaxLength,   ///< Stop generation when the sequence reaches max_len tokens.
    ForcePrefix, ///< Force the first N tokens to match a specified prefix sequence.
};

/// A single decoding constraint.
struct Constraint {
    ConstraintType type;
    /// Pattern string for Regex/Grammar constraints.
    std::string pattern;
    /// Token ID list for AllowList / DenyList / ForcePrefix constraints.
    std::vector<int> data;
    /// Maximum sequence length for MaxLength constraint.
    int max_len{0};
};

/**
 * @brief Constrained / guided decoding for autoregressive token generation.
 *
 * Maintains a list of constraints and applies them to the logit vector at
 * each generation step, masking (setting to -inf) any tokens that violate
 * the active constraints.
 *
 * ## Usage
 * @code
 * ConstrainedDecoder decoder(vocab_size, eos_id);
 * decoder.add_constraint({ConstraintType::MaxLength, "", {}, 50});
 * decoder.add_constraint({ConstraintType::DenyList, "", {bad_token_id}, 0});
 *
 * std::vector<int> generated;
 * while (true) {
 *     auto logits = model.next_logits(generated);
 *     int next = decoder.decode_step(logits, generated);
 *     if (next == eos_id) break;
 *     generated.push_back(next);
 * }
 * @endcode
 */
class ConstrainedDecoder {
public:
    /**
     * @param vocab_size Total vocabulary size (number of possible token IDs).
     * @param eos_id     End-of-sequence token ID.
     */
    ConstrainedDecoder(int vocab_size, int eos_id);

    /// Add a constraint to the active constraint list.
    void add_constraint(Constraint c);

    /// Remove all active constraints.
    void clear_constraints();

    /**
     * @brief Return the list of token IDs that are permitted as the next token.
     *
     * Applies all active constraints and returns the intersection of permitted
     * tokens.  If no tokens are permitted, returns a vector containing only
     * the EOS token ID.
     */
    std::vector<int> get_allowed_tokens(const std::vector<int>& generated_so_far) const;

    /**
     * @brief Mask logits in-place: set logit to -inf for banned tokens.
     *
     * This is the fast path used in the hot decoding loop.
     */
    void apply_constraints_to_logits(
        std::vector<float>& logits,
        const std::vector<int>& generated) const;

    /// Check whether the generated sequence satisfies all AllowList constraints.
    bool check_allowlist(const std::vector<int>& generated) const;

    /// Check whether the generated sequence has not exceeded any MaxLength limit.
    bool check_max_length(const std::vector<int>& generated) const;

    /// Check whether the generated sequence starts with the given prefix.
    bool matches_prefix(
        const std::vector<int>& generated,
        const std::vector<int>& prefix) const;

    /**
     * @brief Simple regex match supporting ., *, +, ?, ^, $.
     *
     * This is a hand-written NFA-free recursive regex matcher supporting only
     * the listed metacharacters.  It is intentionally minimal — for production
     * use a proper regex engine such as RE2.
     */
    bool simple_regex_match(
        const std::string& text,
        const std::string& pattern) const;

    /**
     * @brief Apply all constraints to the logits and return the argmax token.
     *
     * Equivalent to: apply_constraints_to_logits() followed by argmax.
     * Returns the EOS token if no token survives masking.
     */
    int decode_step(
        std::vector<float>& logits,
        const std::vector<int>& generated) const;

private:
    int vocab_size_;
    int eos_id_;
    std::vector<Constraint> constraints_;

    /// Regex helper — recursive implementation.
    bool regex_match_impl(
        const char* text,
        std::size_t tlen,
        const char* pattern,
        std::size_t plen) const;
};

} // namespace llmquant
