#include "SignalCausalChainTracer.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llmquant {

SignalCausalChainTracer::SignalCausalChainTracer(Config cfg)
    : cfg_(std::move(cfg))
{
    int w = std::max(1, cfg_.window);
    buf_.resize(static_cast<std::size_t>(w));
}

void SignalCausalChainTracer::record_token(const std::string& token, double contribution) {
    std::function<void(const std::string&, double)> cb;
    bool fire_strong = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        int sz = static_cast<int>(buf_.size());
        buf_[static_cast<std::size_t>(head_)] = {token, contribution};
        head_ = (head_ + 1) % sz;
        if (fill_ < sz) ++fill_;

        // Update max contribution
        double abs_c = std::abs(contribution);
        double cur_max = max_contrib_.load(std::memory_order_relaxed);
        if (abs_c > cur_max)
            max_contrib_.store(abs_c, std::memory_order_relaxed);

        if (abs_c >= cfg_.strong_threshold) {
            strong_events_.fetch_add(1, std::memory_order_relaxed);
            fire_strong = true;
            cb = cfg_.on_strong_token;
        }
    }

    total_.fetch_add(1, std::memory_order_relaxed);
    if (fire_strong && cb) cb(token, contribution);
}

std::vector<SignalCausalChainTracer::TokenContribution>
SignalCausalChainTracer::trace(int n) const {
    std::vector<TokenContribution> result;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        int sz = static_cast<int>(buf_.size());
        result.reserve(static_cast<std::size_t>(fill_));
        for (int i = 0; i < fill_; ++i) {
            int idx = (head_ - fill_ + i + sz * 2) % sz;
            result.push_back(buf_[static_cast<std::size_t>(idx)]);
        }
    }
    // Sort descending by |contribution|.
    std::sort(result.begin(), result.end(),
              [](const TokenContribution& a, const TokenContribution& b) {
                  return std::abs(a.contribution) > std::abs(b.contribution);
              });
    if (n > 0 && static_cast<int>(result.size()) > n)
        result.resize(static_cast<std::size_t>(n));
    return result;
}

std::vector<SignalCausalChainTracer::TokenContribution>
SignalCausalChainTracer::window_snapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    int sz = static_cast<int>(buf_.size());
    std::vector<TokenContribution> result;
    result.reserve(static_cast<std::size_t>(fill_));
    for (int i = 0; i < fill_; ++i) {
        int idx = (head_ - fill_ + i + sz * 2) % sz;
        result.push_back(buf_[static_cast<std::size_t>(idx)]);
    }
    return result;
}

void SignalCausalChainTracer::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    int w = static_cast<int>(buf_.size());
    buf_.assign(static_cast<std::size_t>(w), TokenContribution{});
    head_ = fill_ = 0;
    max_contrib_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    strong_events_.store(0, std::memory_order_relaxed);
}

void SignalCausalChainTracer::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    int w = std::max(1, cfg_.window);
    buf_.assign(static_cast<std::size_t>(w), TokenContribution{});
    head_ = fill_ = 0;
    max_contrib_.store(0.0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
    strong_events_.store(0, std::memory_order_relaxed);
}

std::string SignalCausalChainTracer::to_stats_json() const {
    uint64_t tot    = total_.load(std::memory_order_relaxed);
    uint64_t strong = strong_events_.load(std::memory_order_relaxed);
    double   mx     = max_contrib_.load(std::memory_order_relaxed);

    auto top = trace(3);
    std::ostringstream ss;
    ss << "{"
       << "\"total_tokens\":"       << tot    << ","
       << "\"strong_token_events\":" << strong << ","
       << "\"max_contribution\":"   << mx     << ","
       << "\"top_tokens\":[";
    for (std::size_t i = 0; i < top.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"token\":\"" << top[i].token
           << "\",\"contribution\":" << top[i].contribution << "}";
    }
    ss << "]}";
    return ss.str();
}

} // namespace llmquant
