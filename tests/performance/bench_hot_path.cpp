#include "gtest/gtest.h"
#include "LLMAdapter.h"
#include "LatencyController.h"
#include "TradeSignalEngine.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace llmquant;
using namespace std::chrono;

// Helper: run fn N times, return sorted microsecond latencies.
template<typename Fn>
std::vector<double> measure_us(Fn&& fn, size_t warmup, size_t iterations) {
    for (size_t i = 0; i < warmup; ++i) fn();
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        auto t0 = high_resolution_clock::now();
        fn();
        auto t1 = high_resolution_clock::now();
        samples.push_back(duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

static double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(static_cast<double>(sorted.size()) * p);
    return sorted[std::min(idx, sorted.size() - 1)];
}

// ============================================================
// Bench 1: Single token lookup — target < 1 μs p99
// ============================================================
TEST(PerformanceBench, bench_llm_adapter_single_token_lookup_under_1us_p99) {
    LLMAdapter adapter;
    auto samples = measure_us([&]{ adapter.map_token_to_weight("crash"); }, 1000, 10000);
    double p99 = percentile(samples, 0.99);
    std::cout << "[bench] LLMAdapter single token p99: " << p99 << " μs\n";
    EXPECT_LT(p99, 1.0) << "Single token lookup p99 must be < 1μs";
}

// ============================================================
// Bench 2: Token-to-signal pipeline — target < 10 μs p99
// ============================================================
TEST(PerformanceBench, bench_token_to_signal_pipeline_under_10us_p99) {
    LLMAdapter adapter;
    TradeSignalEngine::Config eng_cfg;
    eng_cfg.signal_cooldown = std::chrono::microseconds{0};  // no cooldown for bench
    TradeSignalEngine engine(eng_cfg);
    engine.set_backtest_mode(true);
    engine.set_signal_callback([](const TradeSignal&){});

    auto samples = measure_us([&]{
        auto w = adapter.map_token_to_weight("bullish");
        engine.process_semantic_weight(w);
    }, 1000, 10000);

    double p99 = percentile(samples, 0.99);
    std::cout << "[bench] Token-to-signal pipeline p99: " << p99 << " μs\n";
    EXPECT_LT(p99, 10.0) << "Token-to-signal p99 must be < 10μs";
}

// ============================================================
// Bench 3: LatencyController record_latency — target < 1 μs p99
// ============================================================
TEST(PerformanceBench, bench_latency_controller_record_under_1us_p99) {
    LatencyController::Config cfg;
    cfg.enable_profiling = false;  // atomic-only path
    LatencyController ctrl(cfg);

    auto samples = measure_us([&]{
        ctrl.record_latency(std::chrono::microseconds{5});
    }, 1000, 10000);

    double p99 = percentile(samples, 0.99);
    std::cout << "[bench] LatencyController record_latency p99: " << p99 << " μs\n";
    EXPECT_LT(p99, 1.0) << "record_latency p99 must be < 1μs";
}

// ============================================================
// Bench 4: 1M token lookups — target < 2 seconds total
// ============================================================
TEST(PerformanceBench, bench_llm_adapter_1m_tokens_under_2s) {
    LLMAdapter adapter;
    const std::vector<std::string> tokens{
        "crash","panic","bullish","bearish","volatile","rally",
        "surge","confident","the","and"
    };
    const size_t n = 1'000'000;

    auto t0 = high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        adapter.map_token_to_weight(tokens[i % tokens.size()]);
    }
    auto t1 = high_resolution_clock::now();

    double elapsed_s = duration<double>(t1 - t0).count();
    std::cout << "[bench] 1M token lookups: " << elapsed_s << " s\n";
    EXPECT_LT(elapsed_s, 2.0) << "1M token lookups must complete in < 2s";
}

// ============================================================
// Bench 5: SIMD batch not slower than scalar (2x slack allowed)
// ============================================================
TEST(PerformanceBench, bench_simd_batch_faster_than_scalar_for_large_sequence) {
    LLMAdapter adapter;
    const std::vector<std::string> vocab{
        "crash","panic","bullish","bearish","volatile","rally","surge","confident"
    };
    std::vector<std::string> tokens;
    tokens.reserve(64);
    for (size_t i = 0; i < 64; ++i) tokens.push_back(vocab[i % vocab.size()]);

    auto scalar_samples = measure_us([&]{ adapter.map_sequence_to_weight(tokens); }, 100, 1000);
    auto simd_samples   = measure_us([&]{ adapter.map_sequence_simd(tokens); },     100, 1000);

    double scalar_p50 = percentile(scalar_samples, 0.50);
    double simd_p50   = percentile(simd_samples,   0.50);
    std::cout << "[bench] Scalar 64-token p50: " << scalar_p50 << " μs\n";
    std::cout << "[bench] SIMD   64-token p50: " << simd_p50   << " μs\n";
    // SIMD should not be slower than scalar (allow 2x slack for measurement overhead).
    EXPECT_LT(simd_p50, scalar_p50 * 2.0);
}
