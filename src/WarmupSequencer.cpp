#include "WarmupSequencer.h"

#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>

namespace llmquant {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

WarmupSequencer::WarmupSequencer(Config config)
    : config_(std::move(config)) {}

// ---------------------------------------------------------------------------
// load_from_file
// ---------------------------------------------------------------------------

bool WarmupSequencer::load_from_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        spdlog::warn("[warmup] cannot open token file '{}'", filepath);
        return false;
    }
    config_.synthetic_tokens.clear();
    std::string line;
    while (std::getline(f, line)) {
        // Strip trailing CR/spaces.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        // Expect "token\tscore" or just "token" (score defaults to 0).
        auto tab = line.find('\t');
        WarmupToken wt;
        if (tab == std::string::npos) {
            wt.token     = line;
            wt.sentiment = 0.0;
        } else {
            wt.token     = line.substr(0, tab);
            try {
                wt.sentiment = std::stod(line.substr(tab + 1));
            } catch (...) {
                wt.sentiment = 0.0;
            }
        }
        config_.synthetic_tokens.push_back(std::move(wt));
    }
    spdlog::info("[warmup] loaded {} tokens from '{}'",
                 config_.synthetic_tokens.size(), filepath);
    return true;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

void WarmupSequencer::run(std::function<void(const std::string&, double)> token_cb) {
    if (config_.synthetic_tokens.empty()) {
        spdlog::info("[warmup] no tokens configured — skipping warmup");
        complete_.store(true, std::memory_order_release);
        if (config_.on_complete) config_.on_complete();
        return;
    }

    int repeat = std::max(1, config_.repeat_count);
    spdlog::info("[warmup] starting: {} tokens x{} repetitions",
                 config_.synthetic_tokens.size(), repeat);

    for (int r = 0; r < repeat; ++r) {
        for (const auto& wt : config_.synthetic_tokens) {
            token_cb(wt.token, wt.sentiment);
            tokens_injected_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    spdlog::info("[warmup] complete: {} total tokens injected", tokens_injected_.load());
    complete_.store(true, std::memory_order_release);
    if (config_.on_complete) config_.on_complete();
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void WarmupSequencer::reset() {
    complete_.store(false, std::memory_order_release);
    tokens_injected_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// to_stats_json
// ---------------------------------------------------------------------------

std::string WarmupSequencer::to_stats_json() const {
    std::ostringstream s;
    s << "{\"is_complete\":"      << (is_complete() ? "true" : "false")
      << ",\"tokens_injected\":"  << tokens_injected()
      << ",\"token_types\":"      << config_.synthetic_tokens.size()
      << ",\"repeat_count\":"     << config_.repeat_count
      << "}";
    return s.str();
}

} // namespace llmquant
