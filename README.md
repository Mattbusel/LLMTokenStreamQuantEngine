# LLMTokenStreamQuantEngine

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Version](https://img.shields.io/badge/version-1.3.0-blue.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-2102%20passing-brightgreen.svg)](tests/)
[![Dictionary](https://img.shields.io/badge/token%20dictionary-~130%20entries-blue.svg)](src/LLMAdapter.cpp)
[![Download](https://img.shields.io/badge/download-v1.1.0%20Windows%20x64-brightgreen.svg)](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.1.0)

> **Windows users:** grab the pre-built `.exe` from the [v1.1.0 release](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.1.0) — extract, edit `config.yaml`, and run. No build tools required.

A production-grade C++20 engine that ingests a live LLM token stream, maps each token to a quantitative semantic weight, accumulates directional bias and volatility signals with exponential decay, and fires risk-gated trade signals. The end-to-end token-to-signal P99 latency targets sub-10 microseconds in the hot path with zero managed I/O dependencies. As of v1.3.0 the engine also self-learns optimal token weights from labelled trade outcomes via Naive Bayes and correlates signals across multiple assets simultaneously through a rolling Pearson correlation matrix, enabling conviction amplification when correlated assets align and automatic hedge-ratio computation for pairs trading.

---

## Architecture

```
  LLM API / WebSocket feed
          |
          v
   LLMStreamClient          (TLS WebSocket, reconnect, backpressure)
          |
          v
    LLMAdapter              (token -> SemanticWeight, SIMD map_sequence, ~130-entry static dict)
          |             \
          |              DynamicTokenDictionary  (hot-reload YAML/JSON, EMA self-learning)
          |              DictionaryLearner        (Naive Bayes, labelled outcome training)
          v
  TradeSignalEngine         (bias/vol accumulation, exponential decay, signal emission)
          |
          v
  CrossAssetEngine  <-----> [SPY, QQQ, GLD, BTC, ...]
  (rolling Pearson matrix, conviction multiplier, hedge ratio)
          |
          v
  EnsembleSignalVoter       (weighted vote across sub-models)
          |
          v
  RiskManager               (bias/confidence/rate/drawdown gates, position limits)
          |
      pass | block
          |
          v
  OMS Adapter               (FIX 4.2 / REST / Mock)
          |
   [trade execution]

  Side channels (always running):
    MetricsLogger  -->  Prometheus /metrics
    HealthServer   -->  /health (K8s liveness/readiness)
    SignalAuditLog -->  NDJSON append-only audit trail
    BacktestRunner -->  offline token-sequence replay + PnL stats
```

---

## 5-Minute Quickstart

### Prerequisites

| Tool | Version |
|------|---------|
| CMake | 3.20+ |
| C++ compiler | GCC 12+ / Clang 15+ / MSVC 2022 |
| vcpkg | latest (for spdlog, nlohmann-json) |

### Build

```bash
git clone https://github.com/Mattbusel/LLMTokenStreamQuantEngine.git
cd LLMTokenStreamQuantEngine

# Configure (Release, Ninja generator)
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

### Run with simulated tokens

```bash
# Edit config.yaml to enable simulation mode:
#   llm_adapter:
#     mode: simulate
#     simulate_token_interval_ms: 50

./build/LLMTokenStreamQuantEngine --config config.yaml
```

The engine will emit structured JSON signal lines to stdout and write an NDJSON audit log to `signals.ndjson`.

---

## Signal Pipeline Explained

Each token arriving from the LLM stream passes through this sequence:

1. **LLMAdapter::map_token_to_weight** — looks up the token in the static ~130-entry dictionary and optionally the hot-reloadable `DynamicTokenDictionary`. Returns a `SemanticWeight` with `directional_bias`, `volatility_score`, and `confidence_score`.

2. **TradeSignalEngine::process_semantic_weight** — accumulates bias and volatility with per-tick exponential decay (`signal_decay_rate`) and optional time-based decay (`time_decay_half_life_ms`). When accumulated bias exceeds `bias_threshold` the engine emits a `TradeSignal`.

3. **CrossAssetEngine::update_signal** — records the signal in the cross-asset rolling window. Computes conviction multiplier and optionally adjusts signal weight before it reaches the risk layer.

4. **RiskManager::evaluate** — applies hard and soft gates: bias magnitude, minimum confidence, maximum signals per second, maximum drawdown, and position limits. Only passing signals reach the OMS adapter.

---

## Risk Management Explained

`RiskManager` enforces six independent gates in order:

| Gate | Config key | Description |
|------|-----------|-------------|
| Position pre-check | `disable_position_gate` | Rejects if current net position exceeds hard limit before any other check. |
| Bias magnitude | `max_bias_magnitude` | Rejects signals where `|delta_bias_shift|` exceeds threshold. |
| Minimum confidence | `min_confidence` | Rejects signals below confidence floor. |
| Rate limiter | `max_signals_per_second` | Token-bucket rate limit; excess signals are blocked not queued. |
| Drawdown guard | `max_drawdown` | Halts new signals when cumulative drawdown exceeds limit. |
| Dry-run mode | `dry_run_mode` | All signals pass evaluation but OMS is not called; useful for paper trading. |

All gate violations are recorded in `RiskStats` and exported to Prometheus.

---

## Cross-Asset Correlation Guide

`CrossAssetEngine` maintains a rolling Pearson correlation matrix across all tracked asset symbols. It uses Welford's online algorithm for numerically stable incremental covariance, keeping a fixed-size sliding window per symbol.

### Registering assets

Symbols are registered automatically on the first call to `update_signal`:

```cpp
#include "CrossAssetEngine.h"
using namespace llmquant;

CrossAssetEngine engine(/*window_size=*/100);

// Called once per TradeSignalEngine output per asset:
engine.update_signal({"SPY", /*weight=*/0.72, /*confidence=*/0.88, timestamp_ns});
engine.update_signal({"QQQ", /*weight=*/0.65, /*confidence=*/0.81, timestamp_ns});
engine.update_signal({"GLD", /*weight=*/-0.30, /*confidence=*/0.70, timestamp_ns});
```

### Conviction multiplier

```cpp
// Returns value in [0.5, 2.0]:
//   > 1.0 = correlated assets confirm the SPY signal -> increase position size
//   < 1.0 = correlated assets contradict -> reduce position size
double mult = engine.compute_conviction_multiplier("SPY");
double adjusted_size = base_size * mult;
```

### Pairwise correlation

```cpp
double r = engine.get_pairwise_correlation("SPY", "QQQ");
// r in [-1, 1]; |r| < 0.25 treated as uncorrelated
```

### Hedge ratio

Returns the minimum-variance beta of `long_sym` on `short_sym`:

```cpp
double beta = engine.hedge_ratio("SPY", "GLD");
// Negative beta suggests GLD is a valid hedge for long SPY positions
// Bet ratio: short (beta * notional_SPY) in GLD per unit long in SPY
```

### Full correlation matrix snapshot

```cpp
CorrelationMatrix mat = engine.get_correlations();
for (size_t i = 0; i < mat.symbols.size(); ++i)
    for (size_t j = 0; j < mat.symbols.size(); ++j)
        printf("%s vs %s: %.3f\n",
               mat.symbols[i].c_str(),
               mat.symbols[j].c_str(),
               mat.matrix[i][j]);
```

---

## Dictionary Learning Guide

`DictionaryLearner` learns token semantic weights from labelled trade outcomes using a Laplace-smoothed Bernoulli Naive Bayes model.

### Training

```cpp
#include "DictionaryLearner.h"
using namespace llmquant;

// Load priors from the static dictionary (optional, pass "" for uniform 0.5 priors):
DictionaryLearner learner("config/token_dict.json");

// After each trade closes, record the outcome:
std::vector<std::string> active_tokens = {"crash", "selloff", "liquidity"};
learner.record_outcome(active_tokens, /*profitable=*/false, /*pnl=*/-342.50);

active_tokens = {"rally", "breakout", "momentum"};
learner.record_outcome(active_tokens, /*profitable=*/true, /*pnl=*/+820.00);
```

### Inference / hot-reload

```cpp
// Get the current learned weight for a single token:
double w = learner.get_weight("crash");   // ~0.12 after bearish labelling

// Get the full map for hot-reloading into DynamicTokenDictionary:
auto updated = learner.get_updated_dictionary(/*min_observations=*/10);
dynamic_dict.reload(updated);  // zero-downtime update
```

### Top/bottom tokens

```cpp
for (auto& lw : learner.top_weights(10))
    printf("[+] %s  weight=%.3f  LLR=%.3f  pos=%zu neg=%zu\n",
           lw.token.c_str(), lw.weight,
           lw.log_likelihood_ratio,
           lw.positive_count, lw.negative_count);

for (auto& lw : learner.bottom_weights(10))
    printf("[-] %s  weight=%.3f  LLR=%.3f  pos=%zu neg=%zu\n",
           lw.token.c_str(), lw.weight,
           lw.log_likelihood_ratio,
           lw.positive_count, lw.negative_count);
```

### Persistence

```cpp
// Export learned weights to disk (e.g. at end of session):
std::ofstream f("learned_weights.json");
f << learner.export_json();

// Re-import on next startup (counts are merged/accumulated):
std::ifstream g("learned_weights.json");
std::string json_str((std::istreambuf_iterator<char>(g)), {});
learner.import_json(json_str);
```

---

## Configuration Reference

All configuration lives in `config.yaml`. The most important sections:

```yaml
trade_signal_engine:
  bias_sensitivity: 1.0          # Multiplier on directional_bias contributions
  volatility_sensitivity: 1.0    # Multiplier on volatility_score contributions
  bias_threshold: 0.5            # Accumulated bias magnitude to fire a signal
  signal_decay_rate: 0.95        # Per-token exponential decay factor [0, 1]
  time_decay_half_life_ms: 500   # Time-based decay half-life (0 = disabled)

risk_manager:
  max_bias_magnitude: 1.5        # Hard cap on |delta_bias_shift|
  min_confidence: 0.4            # Minimum signal confidence [0, 1]
  max_signals_per_second: 10     # Token-bucket rate limit
  max_drawdown: 5000.0           # Max cumulative drawdown before halt
  dry_run_mode: false            # true = evaluate but skip OMS calls

cross_asset_engine:
  window_size: 100               # Rolling window depth per symbol
  correlation_threshold: 0.25   # Minimum |r| to classify a pair as correlated

dictionary_learner:
  base_dict_path: "config/token_dict.json"
  laplace_smoothing: 1.0         # Laplace alpha; higher = more conservative updates
  min_observations: 10           # Minimum trades per token before weight is applied
  persist_path: "learned_weights.json"  # Written at shutdown, loaded at startup

llm_adapter:
  mode: live                     # live | simulate
  endpoint: "wss://api.example.com/stream"
  api_key: "${OPENAI_API_KEY}"
  simulate_token_interval_ms: 50

prometheus:
  enabled: true
  port: 9090

health_server:
  enabled: true
  port: 8080
```

---

## Performance Benchmarks

All measurements on an AMD Ryzen 9 7950X (16c/32t), GCC 13 `-O3 -march=native`, Release build.

| Path | P50 | P99 | Notes |
|------|-----|-----|-------|
| Token to signal (hot path) | 1.2 µs | 4.8 µs | No OMS call, bias below threshold |
| Token to signal (signal fires) | 2.1 µs | 8.9 µs | Includes RiskManager evaluation |
| CrossAssetEngine::update_signal (5 assets) | 0.6 µs | 1.4 µs | Includes pair accumulator updates |
| CrossAssetEngine::compute_conviction_multiplier | 0.3 µs | 0.7 µs | 5-asset case |
| DictionaryLearner::record_outcome (10 tokens) | 1.8 µs | 3.2 µs | Includes LLR recomputation |
| DictionaryLearner::get_updated_dictionary (500 tokens) | 48 µs | 90 µs | Full map allocation |
| RiskManager::evaluate | 0.4 µs | 0.9 µs | All gates enabled |

Run the benchmark suite:

```bash
cmake --build build --target LLMTokenStreamQuantEngine_bench
./build/LLMTokenStreamQuantEngine_bench
```

---

## What's New in v1.3.0

### Dynamic Token Dictionary (`include/dynamic_dict.hpp`)

| Class | Responsibility |
|-------|----------------|
| `DynamicTokenDictionary` | Lock-free lookup (atomic shared_ptr snapshot), per-category multipliers |
| `DictionaryLoader` | Background file-watcher, supports YAML/JSON/TSV, fires reload callback |
| `CategoryWeights` | Per-category multipliers: `fear`, `bullish`, `bearish`, `volatility`, `corporate`, `macro` |
| `TokenEntry` | Rich entry: `text`, `bias_weight`, `volatility_weight`, `sentiment_score`, `confidence`, `category`, `source` |
| `TokenLearner` | Online EMA weight adjustment from signal-to-outcome correlation |

### CrossAssetEngine (`include/CrossAssetEngine.h`)

Rolling Pearson correlation matrix across N assets with Welford online algorithm.
Conviction multiplier amplifies signals when correlated peers align.
Hedge-ratio computation for pairs and spread trading.

### DictionaryLearner (`include/DictionaryLearner.h`)

Naive Bayes weight learning from labelled trade outcomes.
Laplace-smoothed log-likelihood ratios mapped to [0,1] weights.
Hot-reload into `DynamicTokenDictionary` without process restart.
JSON import/export for cross-session persistence.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow the existing `namespace llmquant` / `#pragma once` / Doxygen style.
3. Add tests in `tests/` using the existing GoogleTest harness. All 2102+ tests must pass.
4. Run the sanitizers before opening a PR:
   ```bash
   cmake -B build-asan -DLLMQUANT_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
   cmake --build build-asan && ctest --test-dir build-asan
   ```
5. Open a pull request with a clear description of the change and its motivation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.
