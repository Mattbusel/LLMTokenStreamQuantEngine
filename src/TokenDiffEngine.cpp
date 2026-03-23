// TokenDiffEngine.cpp — LCS-based diff between token sequences.
#include "TokenDiffEngine.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace llmquant {

// ---------------------------------------------------------------------------
// edit_distance — Wagner–Fischer dynamic programming
// ---------------------------------------------------------------------------

std::size_t TokenDiffEngine::edit_distance(
    const std::vector<std::string>& seq_a,
    const std::vector<std::string>& seq_b) const
{
    const std::size_t m = seq_a.size();
    const std::size_t n = seq_b.size();

    // dp[i][j] = edit distance between seq_a[0..i) and seq_b[0..j)
    std::vector<std::vector<std::size_t>> dp(m + 1, std::vector<std::size_t>(n + 1, 0));

    for (std::size_t i = 0; i <= m; ++i) dp[i][0] = i;
    for (std::size_t j = 0; j <= n; ++j) dp[0][j] = j;

    for (std::size_t i = 1; i <= m; ++i) {
        for (std::size_t j = 1; j <= n; ++j) {
            if (seq_a[i - 1] == seq_b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j],      // delete from a
                                          dp[i][j - 1],      // insert into a
                                          dp[i - 1][j - 1]});// substitute
            }
        }
    }
    return dp[m][n];
}

// ---------------------------------------------------------------------------
// similarity
// ---------------------------------------------------------------------------

double TokenDiffEngine::similarity(
    const std::vector<std::string>& seq_a,
    const std::vector<std::string>& seq_b) const
{
    const std::size_t max_len = std::max(seq_a.size(), seq_b.size());
    if (max_len == 0) return 1.0;
    const double ed = static_cast<double>(edit_distance(seq_a, seq_b));
    return 1.0 - ed / static_cast<double>(max_len);
}

// ---------------------------------------------------------------------------
// diff — LCS back-tracking to produce TokenDiff entries
// ---------------------------------------------------------------------------

std::vector<TokenDiff> TokenDiffEngine::diff(
    const std::vector<std::string>& seq_a,
    const std::vector<std::string>& seq_b) const
{
    const std::size_t m = seq_a.size();
    const std::size_t n = seq_b.size();

    // Build LCS length table.
    std::vector<std::vector<std::size_t>> lcs(m + 1, std::vector<std::size_t>(n + 1, 0));
    for (std::size_t i = 1; i <= m; ++i) {
        for (std::size_t j = 1; j <= n; ++j) {
            if (seq_a[i - 1] == seq_b[j - 1]) {
                lcs[i][j] = lcs[i - 1][j - 1] + 1;
            } else {
                lcs[i][j] = std::max(lcs[i - 1][j], lcs[i][j - 1]);
            }
        }
    }

    // Back-track to recover the edit script.
    std::vector<TokenDiff> result;
    std::size_t i = m, j = n;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && seq_a[i - 1] == seq_b[j - 1]) {
            result.push_back({DiffOp::KEEP, seq_a[i - 1], i - 1});
            --i; --j;
        } else if (j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j])) {
            result.push_back({DiffOp::ADD, seq_b[j - 1], j - 1});
            --j;
        } else {
            result.push_back({DiffOp::DELETE, seq_a[i - 1], i - 1});
            --i;
        }
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// ---------------------------------------------------------------------------
// apply_diff
// ---------------------------------------------------------------------------

std::vector<std::string> TokenDiffEngine::apply_diff(
    const std::vector<std::string>& base,
    const std::vector<TokenDiff>&   diffs) const
{
    std::vector<std::string> result;
    result.reserve(base.size());

    std::size_t base_idx = 0;
    for (const auto& d : diffs) {
        switch (d.op) {
        case DiffOp::KEEP:
            if (base_idx < base.size()) {
                result.push_back(base[base_idx++]);
            }
            break;
        case DiffOp::ADD:
            result.push_back(d.token);
            break;
        case DiffOp::DELETE:
            if (base_idx < base.size()) {
                ++base_idx;  // skip this token
            }
            break;
        }
    }
    return result;
}

}  // namespace llmquant
