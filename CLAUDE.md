# CLAUDE.md — LLMTokenStreamQuantEngine

## What This System Does

Low-latency C++ engine ingesting LLM token streams, mapping semantic weight to trade signals at <10 μs p99 latency. Serves as execution layer beneath ROT (Reddit Options Trader).

## Relation to ROT

This engine consumes structured sentiment from Reddit/LLM pipelines and converts to real-time trade signal adjustments (bias shift, volatility, spread). ROT drives token input; this engine drives signal output.

## Architectural Philosophy

- **Predictive not reactive**: pressure system pushes upstream before queues fill
- **Token-level not batch**: every token modifies signal state, no buffering delay
- **Zero-copy where possible**: memory-mapped streaming planned for high-frequency paths
- **Atomic accumulation**: `std::atomic<double>` for lock-free bias/vol accumulation

## Build Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Test Commands

```bash
cd build && ctest           # run all tests
./tests                     # run test binary directly
./tests --gtest_filter="*integration*"   # integration only
./tests --gtest_filter="*Performance*"  # benchmarks only
```

## Debug Build (with ASan/UBSan)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
./LLMTokenStreamQuantEngine
```

## Lint

```bash
clang-tidy src/*.cpp -- -std=c++20 -Iinclude
clang-format --dry-run src/*.cpp include/*.h
```

## Coding Conventions

- C++20, no raw new/delete (smart pointers only)
- All public methods doc-commented in headers
- namespace llmquant throughout
- -Wall -Wextra -Werror: zero warnings policy
- Atomic types for all hot-path shared state
- Mutex only for sample windows and buffer management (not hot path)

## CMake Details

- CMake 3.20+ required
- Dependencies: spdlog (logging), yaml-cpp (config), GoogleTest (testing), Threads
- Release: -O3 -march=native (no -ffast-math — removed; permits unsafe FP reordering/NaN mishandling in atomics)
- Debug: -g -O0 -fsanitize=address,undefined (via LLMQUANT_ENABLE_ASAN=ON)
- Tests live in tests/unit/ and tests/integration/; tests/CMakeLists.txt is
  included via add_subdirectory(tests) from the root CMakeLists.txt

## Module Status

| Module | Headers | Implementation | Tests | Done |
|--------|---------|----------------|-------|------|
| Config | ✓ | ✓ hot-reload | ✓ | ✓ |
| LLMAdapter | ✓ | ✓ SIMD batch | ✓ | ✓ |
| LatencyController | ✓ | ✓ pressure system | ✓ | ✓ |
| MetricsLogger | ✓ | ✓ | ✓ | ✓ |
| TokenStreamSimulator | ✓ | ✓ lock-free ring | ✓ | ✓ |
| TradeSignalEngine | ✓ | ✓ full fields | ✓ | ✓ |
| OutputSink | ✓ | ✓ (header-only) | ✓ | ✓ |
| LLMStreamClient | ✓ | ✓ TLS via OpenSSL | ✓ | ✓ |
| Deduplicator | ✓ | ✓ hiredis (conditional) | ✓ | ✓ |
| RiskManager | ✓ | ✓ OMS position hooks | ✓ | ✓ |
| OmsAdapter (abstract) | ✓ | ✓ (interface only) | ✓ | ✓ |
| RestOmsAdapter | ✓ | ✓ HTTP/1.1 polling | ✓ | ✓ |
| FixOmsAdapter | ✓ | ✓ FIX 4.2 + session recovery | ✓ | ✓ |
| MockOmsAdapter | ✓ | ✓ deterministic test double | ✓ | ✓ |
| PrometheusExporter | ✓ | ✓ HTTP /metrics scrape | ✓ | ✓ |

## What Still Needs Building
- Nothing critical — all modules complete.
- `evaluate_with_reason` callback re-entrancy deadlock: fixed in cycle 9 with `ewr_mutex_`.
- FIX surrogate-pair Unicode in SequenceReset: still deferred (low priority).
- Async alert dispatch: no longer needed after the re-entrancy fix.

## Non-Obvious Design Decisions

- CMakeLists.txt was originally in include/ — moved to project root for standard
  CMake layout; the old include/CMakeLists.txt has been deleted.
- TradeSignal includes both `timestamp_ns` (uint64, for serialisation) and
  `timestamp` (chrono, for latency calculation in main).
- LLMAdapter uses an unordered_map as a token cache — hot path since all
  production tokens hit it after the first lookup.
- LatencyController uses a fixed-size ring buffer for the sample window — O(1)
  push with no element shifting. Percentiles use `nth_element` (O(N) average)
  rather than a full sort. Thread-local `latency_measurement_start_` makes
  `start/end_measurement` safe to call from multiple threads.
- OutputSink concrete classes (CsvOutputSink, JsonOutputSink, MemoryOutputSink)
  are implemented inline in OutputSink.h because they are thin wrappers over
  std::ofstream / std::vector with no separate compilation unit needed.
- `--list-tokens` and `--export-dict` both exit before the full engine pipeline
  is constructed. They run after `LLMAdapter` construction so the default
  dictionary is populated, but before `TradeSignalEngine`, `RiskManager`, OMS
  adapters etc. are created — intentional to keep startup-inspection flags cheap.
- `export_dictionary()` outputs TSV with no header row (round-trip compatible with
  `load_dictionary_from_tsv()`). `--list-tokens` adds a header row to stdout
  because it is for human inspection, not round-trip import.
- "Tokens processed" in the stats bar and session summary reads from
  `LLMAdapter::get_stats().tokens_processed`, not `variance_n`. The Welford
  variance accumulator resets every 60 s to prevent catastrophic cancellation;
  using it as a token counter caused undercount after the first reset window.

## Test Coverage Summary
| Test File | Count | Coverage |
|-----------|-------|----------|
| test_llm_adapter.cpp | 155 | Token lookup, sequences, SIMD, dictionary analytics, hit counts, top-N frequency, all analytics APIs |
| test_risk_manager.cpp | 125 | Magnitude, confidence, rate, drawdown, OMS gates, evaluate_with_reason, get_most_blocked_gate, batch evaluate |
| test_trade_signal_engine.cpp | 126 | Signals, fields, backtest, cooldown, reset, stats accessors, signal quality, to_json, flush_sinks, format_stats quality_ema |
| test_latency_controller.cpp | 90 | Stats, percentiles, p50, pressure, backoff, reset, concurrent, get_total_latency_us |
| test_config.cpp | 68 | YAML parse, missing fields, hot-reload, risk_thresholds, pressure validation, SemanticWeightsConfig validation |
| test_edge_cases.cpp | 33 | Empty/null inputs, overflow, NaN, invalid params across all major APIs |
| test_metrics_logger.cpp | 30 | Construction, log events, flush, log_trade_signal, log_risk_rejection, log_pipeline_health |
| test_deduplicator.cpp | 30 | Key determinism, TTL, evict, concurrent, Redis stub, facade |
| test_production_readiness.cpp | 26 | Production path coverage for all modules |
| test_oms_adapter.cpp | 24 | Mock/REST/FIX OMS adapter full coverage |
| test_token_stream_simulator.cpp | 23 | Load, callback, ring buffer, emit rate, drop rate |
| test_network_error_paths.cpp | 23 | Network error paths for LLMStreamClient and OMS adapters |
| test_fix_oms_adapter.cpp | 23 | FIX 4.2 session management, heartbeats, sequence recovery |
| test_prometheus_exporter.cpp | 19 | Start/stop, double-start, scrape serves metrics, no-callback |
| test_output_sink.cpp | 19 | CSV, JSON, memory sink, NaN/Inf guard, capacity cap, signal_quality field |
| test_llm_stream_client.cpp | 16 | Connect/stop lifecycle, done callback, error paths |
| test_full_pipeline.cpp (integration) | 16 | 5-stage pipeline end-to-end, all risk gates |
| test_pipeline_integration.cpp (integration) | 12 | Pipeline with OMS and Prometheus wired together |
| test_invariants.cpp | 10 | Dedup key determinism, sentiment sign, risk counter identity, latency avg bounds, confidence interval |
| test_oms_pipeline.cpp (integration) | 8 | Position overlimit blocks signals; safe position; PnL breach; callback fires on approach |
| test_pipeline.cpp (integration) | 8 | End-to-end, latency, accumulation |
| test_chaos.cpp (integration) | 6 | Fear saturation, runaway bias, dedup flood, restart-under-load, mixed pipeline |
| bench_hot_path.cpp (perf) | 5 | Latency budgets, throughput, SIMD vs scalar |
| **Total** | **895** | |
