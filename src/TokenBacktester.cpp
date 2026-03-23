#include "TokenBacktester.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TokenBacktester::TokenBacktester(Config cfg)
    : cfg_(std::move(cfg))
{}

// ---------------------------------------------------------------------------
// CSV parsing helpers (local, no external dependencies)
// ---------------------------------------------------------------------------

namespace {

// Trim leading/trailing whitespace and \r from a string.
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Split a line by the first of ',' or '\t'.
static std::vector<std::string> split_line(const std::string& line) {
    char delim = (line.find('\t') != std::string::npos) ? '\t' : ',';
    std::vector<std::string> cols;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, delim)) {
        cols.push_back(trim(tok));
    }
    return cols;
}

// Return true if the first field looks like a header (non-numeric).
static bool is_header(const std::vector<std::string>& cols) {
    if (cols.empty()) return false;
    const std::string& first = cols[0];
    if (first.empty()) return false;
    char* end = nullptr;
    std::strtoll(first.c_str(), &end, 10);
    return (end == first.c_str()) || (*end != '\0' && *end != '.' && *end != 'e');
}

} // anon namespace

// ---------------------------------------------------------------------------
// load_price_history (CSV file)
// ---------------------------------------------------------------------------

bool TokenBacktester::load_price_history(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        spdlog::warn("[TokenBacktester] cannot open price history '{}'", path);
        return false;
    }

    bars_.clear();
    std::string line;
    bool header_seen = false;
    int  line_no     = 0;

    while (std::getline(f, line)) {
        ++line_no;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        auto cols = split_line(line);
        if (cols.size() < 5) continue;

        if (!header_seen && is_header(cols)) {
            header_seen = true;
            continue;
        }
        header_seen = true;

        HistoricalBar bar;
        try {
            bar.timestamp_ms = std::stoll(cols[0]);
            bar.open         = std::stod(cols[1]);
            bar.high         = std::stod(cols[2]);
            bar.low          = std::stod(cols[3]);
            bar.close        = std::stod(cols[4]);
            bar.volume       = (cols.size() >= 6) ? std::stod(cols[5]) : 0.0;
        } catch (...) {
            spdlog::warn("[TokenBacktester] price CSV parse error at line {}", line_no);
            continue;
        }
        bars_.push_back(bar);
    }

    // Sort by timestamp (handle unsorted input gracefully).
    std::sort(bars_.begin(), bars_.end(), [](const HistoricalBar& a, const HistoricalBar& b) {
        return a.timestamp_ms < b.timestamp_ms;
    });

    spdlog::info("[TokenBacktester] loaded {} price bars from '{}'", bars_.size(), path);
    return !bars_.empty();
}

// ---------------------------------------------------------------------------
// load_price_history (in-memory)
// ---------------------------------------------------------------------------

void TokenBacktester::load_price_history(std::vector<HistoricalBar> bars) {
    bars_ = std::move(bars);
    std::sort(bars_.begin(), bars_.end(), [](const HistoricalBar& a, const HistoricalBar& b) {
        return a.timestamp_ms < b.timestamp_ms;
    });
}

// ---------------------------------------------------------------------------
// replay_signals (CSV/TSV file)
// ---------------------------------------------------------------------------

bool TokenBacktester::replay_signals(const std::string& signal_log) {
    std::ifstream f(signal_log);
    if (!f.is_open()) {
        spdlog::warn("[TokenBacktester] cannot open signal log '{}'", signal_log);
        return false;
    }

    signals_.clear();
    std::string line;
    bool header_seen = false;
    int  line_no     = 0;

    while (std::getline(f, line)) {
        ++line_no;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        auto cols = split_line(line);
        if (cols.size() < 2) continue;

        if (!header_seen && is_header(cols)) {
            header_seen = true;
            continue;
        }
        header_seen = true;

        SignalEntry entry;
        try {
            entry.signal_time_ns = std::stoll(cols[0]);
            entry.direction      = std::stoi(cols[1]);
            entry.bias           = (cols.size() >= 3) ? std::stod(cols[2]) : 0.0;
            entry.confidence     = (cols.size() >= 4) ? std::stod(cols[3]) : 1.0;
        } catch (...) {
            spdlog::warn("[TokenBacktester] signal log parse error at line {}", line_no);
            continue;
        }

        // Normalise direction to {-1, 0, +1}.
        if (entry.direction > 0) entry.direction = 1;
        else if (entry.direction < 0) entry.direction = -1;

        signals_.push_back(entry);
    }

    spdlog::info("[TokenBacktester] loaded {} signals from '{}'", signals_.size(), signal_log);
    return !signals_.empty();
}

// ---------------------------------------------------------------------------
// replay_signals (in-memory)
// ---------------------------------------------------------------------------

void TokenBacktester::replay_signals(std::vector<SignalEntry> signals) {
    signals_ = std::move(signals);
}

// ---------------------------------------------------------------------------
// match_bar — binary search for nearest bar within tolerance
// ---------------------------------------------------------------------------

int TokenBacktester::match_bar(int64_t signal_ns) const noexcept {
    if (bars_.empty()) return -1;

    // Convert signal timestamp from ns to ms for comparison.
    int64_t signal_ms = signal_ns / 1'000'000LL;

    // Binary search for the insertion point.
    int lo = 0, hi = static_cast<int>(bars_.size()) - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (bars_[static_cast<size_t>(mid)].timestamp_ms < signal_ms) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Check the candidate and its neighbour for the closest match.
    int64_t tol_ms = cfg_.match_tolerance_ns / 1'000'000LL;
    int best        = -1;
    int64_t best_dist = tol_ms + 1;

    for (int candidate : {lo - 1, lo, lo + 1}) {
        if (candidate < 0 || candidate >= static_cast<int>(bars_.size())) continue;
        int64_t dist = std::abs(bars_[static_cast<size_t>(candidate)].timestamp_ms - signal_ms);
        if (dist <= tol_ms && dist < best_dist) {
            best      = candidate;
            best_dist = dist;
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// compute_pnl — main backtest logic
// ---------------------------------------------------------------------------

BacktestResult TokenBacktester::compute_pnl() const {
    std::vector<BacktestResult::SignalResult> results;
    results.reserve(signals_.size());

    for (const auto& sig : signals_) {
        BacktestResult::SignalResult res;
        res.entry   = sig;
        res.matched = false;

        // Confidence filter.
        if (cfg_.apply_confidence_filter && sig.confidence < cfg_.min_confidence) {
            results.push_back(std::move(res));
            continue;
        }

        int bar_idx = match_bar(sig.signal_time_ns);
        if (bar_idx < 0) {
            results.push_back(std::move(res));
            continue;
        }

        const HistoricalBar& bar = bars_[static_cast<size_t>(bar_idx)];
        res.bar     = bar;
        res.matched = true;

        // Gross return: direction * (close - open) / open
        double gross_ret = 0.0;
        if (bar.open > 0.0) {
            gross_ret = static_cast<double>(sig.direction) *
                        (bar.close - bar.open) / bar.open;
        }

        // Transaction cost: proportional to |bias|.
        double cost = cfg_.spread_cost * std::abs(sig.bias);

        res.simulated_return = gross_ret - cost;
        results.push_back(std::move(res));
    }

    return aggregate(std::move(results),
                     static_cast<int>(bars_.size()),
                     static_cast<int>(signals_.size()));
}

// ---------------------------------------------------------------------------
// aggregate — statistics from per-signal results
// ---------------------------------------------------------------------------

BacktestResult TokenBacktester::aggregate(
    std::vector<BacktestResult::SignalResult> records,
    int total_bars,
    int total_signals)
{
    BacktestResult r;
    r.records        = std::move(records);
    r.total_bars     = total_bars;
    r.total_signals  = total_signals;

    // Count matched/unmatched.
    for (const auto& rec : r.records) {
        if (rec.matched) ++r.matched_signals;
        else             ++r.unmatched_signals;
    }

    if (r.matched_signals == 0) return r;

    // Cumulative PnL, drawdown, win count.
    double cumulative = 0.0;
    double peak       = 0.0;
    int    wins       = 0;
    std::vector<double> rets;
    rets.reserve(static_cast<size_t>(r.matched_signals));

    // Hold time: difference in bar timestamps between consecutive matched signals.
    int64_t prev_bar_ts_ms = -1;
    double  sum_hold_ms    = 0.0;
    int     hold_count     = 0;

    for (auto& rec : r.records) {
        if (!rec.matched) continue;

        cumulative += rec.simulated_return;
        rec.cumulative_return = cumulative;

        if (cumulative > peak) peak = cumulative;
        double drawdown = peak - cumulative;
        if (drawdown > r.max_drawdown) r.max_drawdown = drawdown;

        if (rec.simulated_return > 0.0) ++wins;
        rets.push_back(rec.simulated_return);

        if (prev_bar_ts_ms >= 0) {
            double hold_ms = static_cast<double>(
                std::abs(rec.bar.timestamp_ms - prev_bar_ts_ms));
            sum_hold_ms += hold_ms;
            ++hold_count;
        }
        prev_bar_ts_ms = rec.bar.timestamp_ms;
    }

    r.total_return = cumulative;
    r.win_rate     = static_cast<double>(wins) /
                     static_cast<double>(r.matched_signals);

    if (hold_count > 0) {
        r.avg_hold_time_ms = sum_hold_ms / static_cast<double>(hold_count);
    }

    // Sharpe ratio: mean(returns) / std(returns) * sqrt(252).
    if (static_cast<int>(rets.size()) >= 2) {
        double n    = static_cast<double>(rets.size());
        double mean = std::accumulate(rets.begin(), rets.end(), 0.0) / n;
        double sq   = 0.0;
        for (double x : rets) {
            double d = x - mean;
            sq += d * d;
        }
        double std_dev = std::sqrt(sq / (n - 1.0));
        r.sharpe = (std_dev > 1e-12) ? (mean / std_dev * std::sqrt(252.0)) : 0.0;
    }

    return r;
}

// ---------------------------------------------------------------------------
// run — convenience combined loader + compute
// ---------------------------------------------------------------------------

BacktestResult TokenBacktester::run(const std::string& price_csv,
                                     const std::string& signal_log) {
    BacktestResult err;

    if (!load_price_history(price_csv)) {
        err.total_bars = -1;
        return err;
    }
    if (!replay_signals(signal_log)) {
        err.total_signals = -1;
        return err;
    }
    return compute_pnl();
}

// ---------------------------------------------------------------------------
// BacktestResult::to_summary_string
// ---------------------------------------------------------------------------

std::string BacktestResult::to_summary_string() const {
    std::ostringstream s;
    s << "===== TokenBacktester Result =====\n";
    s << "  Price bars loaded  : " << total_bars     << "\n";
    s << "  Signals in log     : " << total_signals  << "\n";
    s << "  Matched signals    : " << matched_signals << "\n";
    s << "  Unmatched signals  : " << unmatched_signals << "\n";
    if (matched_signals > 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", total_return);
        s << "  Total return       : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.4f", sharpe);
        s << "  Sharpe ratio       : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.6f", max_drawdown);
        s << "  Max drawdown       : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.1f%%", win_rate * 100.0);
        s << "  Win rate           : " << buf << "\n";
        std::snprintf(buf, sizeof(buf), "%.1f ms", avg_hold_time_ms);
        s << "  Avg hold time      : " << buf << "\n";
    }
    s << "==================================\n";
    return s.str();
}

// ---------------------------------------------------------------------------
// BacktestResult::to_json
// ---------------------------------------------------------------------------

std::string BacktestResult::to_json() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(6);
    o << "{"
      << "\"total_bars\":"        << total_bars
      << ",\"total_signals\":"    << total_signals
      << ",\"matched_signals\":"  << matched_signals
      << ",\"unmatched_signals\":" << unmatched_signals
      << ",\"total_return\":"     << total_return
      << ",\"sharpe\":"           << sharpe
      << ",\"max_drawdown\":"     << max_drawdown
      << ",\"win_rate\":"         << win_rate
      << ",\"avg_hold_time_ms\":" << avg_hold_time_ms
      << "}";
    return o.str();
}

} // namespace llmquant
