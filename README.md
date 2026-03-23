# LLMTokenStreamQuantEngine

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-2102%20passing-brightgreen.svg)](tests/)
[![Dictionary](https://img.shields.io/badge/token%20dictionary-~130%20entries-blue.svg)](src/LLMAdapter.cpp)
[![Download](https://img.shields.io/badge/download-v1.1.0%20Windows%20x64-brightgreen.svg)](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.1.0)

> **Windows users:** grab the pre-built `.exe` from the [v1.1.0 release](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.1.0) — extract, edit `config.yaml`, and run. No build tools required.

A production-grade C++20 engine that ingests a live LLM token stream, maps each token to a quantitative semantic weight, accumulates directional bias and volatility signals with exponential decay, and fires risk-gated trade signals. The end-to-end token-to-signal P99 latency targets sub-10 microseconds in the hot path with zero managed I/O dependencies. As of v1.3.0 the engine also self-learns optimal token weights from labelled trade outcomes via Naive Bayes and correlates signals across multiple assets simultaneously through a rolling Pearson correlation matrix, enabling conviction amplification when correlated assets align and automatic hedge-ratio computation for pairs trading.

---

## Alpha Decay Model

### Header: `include/alpha_decay.hpp`  |  Source: `src/alpha_decay.cpp`

Models how a trading signal's alpha (predictive strength) decays over time, supporting four decay profiles and a multi-symbol portfolio aggregator.

**Decay Models (`DecayModel` variant):**

| Model | Formula | Half-life |
|---|---|---|
| `Exponential { half_life_ms }` | s(t) = exp(−λΔt), λ = ln(2)/half\_life\_ms | Analytical: `half_life_ms` |
| `Linear { duration_ms }` | s(t) = max(0, 1 − Δt/duration\_ms) | Analytical: duration\_ms / 2 |
| `PowerLaw { exponent, scale_ms }` | s(t) = (scale / (scale + Δt))^exp | Analytical: scale × (2^(1/exp) − 1) |
| `StepFunction { levels }` | Piecewise constant by threshold | Newton-Raphson / bisection |

**API:**

| Class | Method | Description |
|---|---|---|
| `AlphaDecay` | `current_strength(signal, now_ms)` | Decayed alpha at `now_ms` |
| `AlphaDecay` | `half_life_ms(signal)` | Time for strength to halve |
| `AlphaPortfolio` | `add_signal(symbol, signal)` | Register a signal |
| `AlphaPortfolio` | `net_alpha(symbol, now_ms)` | Sum of active signal strengths |
| `AlphaPortfolio` | `prune(now_ms, threshold=0.01)` | Remove signals below threshold |

```cpp
#include "alpha_decay.hpp"
using namespace llmquant;

AlphaSignal sig{
    .strength        = 1.0,
    .generated_at_ms = 1000,
    .decay_model     = DecayModel{ Exponential{500.0} },
};

double s  = AlphaDecay::current_strength(sig, 1500);  // 0.5
double hl = AlphaDecay::half_life_ms(sig);             // 500.0

AlphaPortfolio portfolio;
portfolio.add_signal("AAPL", sig);
double net = portfolio.net_alpha("AAPL", 2000);  // ~0.25
portfolio.prune(2000, 0.01);
```

**Tests:** `tests/unit/test_alpha_decay.cpp` (25+ GTest cases covering all four decay models and AlphaPortfolio)

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

  v1.4.0 additions:
    RegimeDetectorHMM  -->  3-state HMM (TokenRegime: BULL/BEAR/UNCERTAINTY/CONSOLIDATION/BREAKOUT)
    RegimeFilter       -->  gates trade signals by regime alignment + confidence
    ExecutionQualityTracker --> lock-free 10K ring: p99 latency, fill rate, realised alpha
    TokenBacktester    -->  OHLCV bar matching, per-signal PnL, Sharpe, max drawdown

  Round 3 additions:
    MarketImpactModel  -->  Almgren-Chriss impact: linear + sqrt components in bps
    ExecutionScheduler -->  TWAP / VWAP child-order slicers with historical volume profile
    OptimalExecution   -->  urgency-blended schedule (urgency=1 TWAP, urgency=0 VWAP)

  Cycle 40 additions:
    TokenWindowSummariser --> rolling decay window: WindowSummary {net_bias, confidence, volatility_est}
    BootstrapSignalCI     --> bootstrap CI: ConfidenceInterval {lower, median, upper}
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

## What's New in v1.5.0

### Bug Fix: `RiskManager::update_config` — Drawdown Window Reset

**Problem:** When `drawdown_window` was changed via `update_config()`, the
`cumulative_bias_` and `drawdown_window_start_` from the old window persisted.
This meant a strategy that reduced the window from 5 min → 1 min would
immediately block signals for up to 4 minutes of accumulated history that was
no longer meaningful.

**Fix:** `update_config()` now detects when `drawdown_window` changes and
automatically resets `cumulative_bias_ = 0.0` and `drawdown_window_start_ = now`,
logging the transition at `INFO` level. This matches the natural reset behaviour
that occurs when the window elapses in `check_drawdown()`. Callers that relied
on the old behaviour can call `reset_drawdown()` explicitly before `update_config()`
to preserve it.

---

## What's New in v1.4.0

### Streaming Regime Detector (`include/RegimeDetector.hpp`)

A fully online 3-state Hidden Markov Model regime detector that classifies the
token-sentiment stream into five discrete `TokenRegime` labels in real time.

| Class | Responsibility |
|-------|----------------|
| `TokenRegime` | `BULL_TRENDING`, `BEAR_TRENDING`, `HIGH_UNCERTAINTY`, `CONSOLIDATION`, `BREAKOUT` |
| `RegimeDetectorHMM` | 3-state HMM forward filter (bearish/neutral/bullish); online emission + transition parameter learning via exponential forgetting |
| `RegimeFilter` | Gates trade signals: only fires when the current regime aligns with signal direction; configurable confidence threshold |

**Usage:**

```cpp
#include "RegimeDetector.hpp"
using namespace llmquant;

RegimeDetectorHMM detector;
detector.set_regime_change_callback([](TokenRegime n, TokenRegime o, int64_t ts) {
    spdlog::info("Regime {} → {}", RegimeDetectorHMM::regime_name(o),
                                   RegimeDetectorHMM::regime_name(n));
});

// Feed each incoming token bias:
detector.update(bias, timestamp_ns);

// Read current regime and confidence:
TokenRegime r = detector.current_regime();   // e.g. BULL_TRENDING
double conf   = detector.regime_confidence(); // 0.0–1.0

// Gate a buy signal (+1) through the regime filter:
RegimeFilter filter(detector);
filter.feed(bias, timestamp_ns);
bool should_trade = filter.evaluate(/*direction=*/+1, bias);
```

### Execution Quality Tracker (`include/ExecutionQuality.hpp`)

A lock-free ring buffer (10 000 slots, cache-line aligned) that tracks
signal-to-fill round trips with nanosecond resolution.

| Method | Description |
|--------|-------------|
| `record(ExecutionRecord)` | Lock-free insert into ring buffer; auto-computes slippage from prices |
| `mean_slippage_bps()` | Mean fill slippage across ring snapshot |
| `p99_latency_ns()` | 99th-percentile signal→fill latency via `nth_element` |
| `fill_rate()` | Fraction of recorded signals that were filled |
| `signal_alpha(N)` | Realised alpha of last N filled trades |

**Usage:**

```cpp
#include "ExecutionQuality.hpp"
using namespace llmquant;

ExecutionQualityTracker tracker;

tracker.record({
    .signal_id        = sig.timestamp_ns,
    .signal_time_ns   = sig.timestamp_ns,
    .execution_time_ns = oms_ack_ns,
    .expected_price   = mid_at_signal,
    .actual_price     = fill_price,
    .direction        = 1,
    .filled           = true
});

double slip = tracker.mean_slippage_bps();
int64_t p99 = tracker.p99_latency_ns();
double  fr  = tracker.fill_rate();
double  alpha = tracker.signal_alpha(/*last_n=*/100);
```

### Token Sentiment Backtester (`include/TokenBacktester.hpp`)

Backtests historical token-sentiment signals against OHLCV price bars with
nearest-bar timestamp matching (±100 ms tolerance by default).

| Method | Description |
|--------|-------------|
| `load_price_history(path)` | Load CSV of OHLCV bars (`timestamp_ms,open,high,low,close,volume`) |
| `replay_signals(path)` | Load CSV signal log (`signal_time_ns,direction,bias,confidence`) |
| `compute_pnl()` | Match signals to bars, compute per-signal returns, aggregate stats |
| `BacktestResult` | `total_return`, `sharpe`, `max_drawdown`, `win_rate`, `avg_hold_time_ms` |

**Usage:**

```cpp
#include "TokenBacktester.hpp"
using namespace llmquant;

TokenBacktester bt;
bt.load_price_history("data/SPY_1min.csv");
bt.replay_signals("logs/signals_2024.csv");
BacktestResult res = bt.compute_pnl();

std::cout << res.to_summary_string();
// Sharpe: 1.42  MaxDD: 0.003412  WinRate: 57.3%  AvgHold: 183ms
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

## Round 3 Features

### Execution Timing Optimizer (`include/execution_timing.hpp` + `src/execution_timing.cpp`)

Three composable components for minimising market impact when executing LLM-signal-driven trade orders.

#### MarketImpactModel

Implements the Almgren-Chriss model:

```
impact = eta * sigma * sqrt(Q / V)   [permanent, sqrt component]
       + alpha * sigma * (Q / V)     [temporary, linear component]
```

where Q = order size, V = average daily volume, sigma = volatility.  Both components are returned in basis points.

```cpp
#include "execution_timing.hpp"
using namespace llmquant;

MarketImpactModel model(/*eta=*/0.1, /*alpha=*/0.5);
auto est = model.estimate(
    /*order_qty=*/10'000.0,
    /*avg_volume=*/1'000'000.0,
    /*volatility=*/0.015);

if (est) {
    std::cout << "sqrt impact:   " << est->sqrt_impact_bps   << " bps\n";
    std::cout << "linear impact: " << est->linear_impact_bps << " bps\n";
    std::cout << "total impact:  " << est->total_impact_bps  << " bps\n";
}
```

#### ExecutionScheduler — TWAP and VWAP

```cpp
ExecutionScheduler sched(/*n_slices=*/12, /*start_ms=*/0, /*duration_ms=*/3'600'000);

// TWAP: 12 equal slices over one hour
auto twap = sched.twap(/*total_qty=*/60'000.0, /*spot_price=*/150.0);

// VWAP: slices proportional to intraday volume profile
std::vector<double> volume_profile = {1,2,3,4,5,5,4,3,2,1,1,1};
auto vwap = sched.vwap(60'000.0, 150.0, volume_profile);
```

If `volume_profile` is empty or all zeros, VWAP falls back to equal-weight TWAP slices.  The scheduler normalises the profile automatically.

#### OptimalExecution — urgency-blended schedule

```cpp
OptimalExecution oe(ExecutionScheduler(12, 0, 3'600'000));

// urgency=1.0 → pure TWAP
auto twap_sched = oe.minimize_impact(60'000.0, 1.0, volume_profile);

// urgency=0.0 → pure VWAP
auto vwap_sched = oe.minimize_impact(60'000.0, 0.0, volume_profile);

// urgency=0.5 → blended (linear interpolation of slice quantities)
auto blend_sched = oe.minimize_impact(60'000.0, 0.5, volume_profile);

if (blend_sched) {
    for (auto& sl : blend_sched->slices) {
        std::cout << "  [" << sl.start_ms << " - " << sl.end_ms << "]"
                  << "  qty=" << sl.target_qty << "\n";
    }
}
```

The blended schedule re-normalises child quantities to exactly `order_qty`, ensuring no leakage.

Enable with `LLMQUANT_ENABLE_EXECUTION_TIMING=ON` (default ON).

---

## Round 2 Features

### Sentiment Trend Tracker (`include/sentiment_trend.hpp` + `src/sentiment_trend.cpp`)

Tracks how LLM-derived `delta_bias_shift` evolves over a trading session by fitting
an online OLS regression on a rolling window of signals.

**`SentimentTrendTracker`**:
- Maintains a deque of the last N bias values
- Fits `bias = slope × t + intercept` (OLS) on every push
- Classifies the trend as `Bullish`, `Bearish`, `Flat`, or `Reversing`
- Returns `TrendResult { slope, intercept, r_squared, trend_direction, n_samples }`

**`RegimeSwitchDetector`**:
- Wraps `SentimentTrendTracker`
- Fires a `RegimeSwitch { from, to, confidence, timestamp_ms }` when sentiment
  crosses from positive to negative territory (or vice versa)
- Confirmation requires `confirm_bars` (default 5) consecutive same-direction samples
- While switching: `is_switching()` returns `true` for `uncertainty_bars` (default 20)
  samples — `RiskManager` should pause emission during this window

```cpp
SentimentTrendTracker tracker;
tracker.push(signal);
if (auto res = tracker.fit()) {
    // res->trend_direction: Bullish / Bearish / Flat / Reversing
    // res->slope:           bias units per sample
    // res->r_squared:       goodness-of-fit [0, 1]
}

RegimeSwitchDetector detector;
detector.set_switch_callback([&risk_mgr](const RegimeSwitch& sw) {
    risk_mgr.disable_all_gates();   // halt trading during switch uncertainty
});
detector.push(signal);
if (detector.is_switching()) return;  // skip signal during uncertainty window
```

---

### Market Calendar Integration (`include/market_calendar.hpp` + `src/market_calendar.cpp`)

Provides trading-hours awareness and economic event suppression for the signal pipeline.

**`MarketCalendar`**:

| Market | Session model |
|--------|--------------|
| `NYSE_NASDAQ` | Mon–Fri 09:30–16:00 ET, pre-market 04:00–09:30, AH 16:00–20:00 |
| `CME` | ~23 h session (Mon–Fri), approximated via NYSE extended hours |
| `Crypto` | Always open (24 × 7) |

Returns `TradingSession { is_open, session_type, next_open_ms, next_close_ms }`.

**`EconomicEventFilter`**:

Pre-loaded with all 2026 FOMC, NFP, and CPI dates.  Suppresses signals for
±30 minutes around each event (configurable per-event).  Events can also be
loaded from a JSON file at runtime:

```cpp
EconomicEventFilter filter;
filter.load_defaults_2026();
filter.load_json("/etc/llmquant/custom_events.json");  // optional

if (filter.should_suppress(now_ms)) {
    // near a major event — skip signal emission
}
```

**RiskManager integration** — use the helper function:

```cpp
if (should_suppress_signal(calendar, filter, Market::NYSE_NASDAQ, now_ms)) {
    return;   // market closed or economic event in window
}
```

Covered 2026 events: 8 FOMC meetings, 12 NFP releases, 12 CPI reports (36 events total).

---

## TokenWindowSummariser

`TokenWindowSummariser` (`include/TokenWindowSummariser.hpp`) accumulates
`SemanticWeight` entries in a fixed-capacity sliding window with per-push
exponential decay.  When the window is full (`is_ready() == true`),
`summarise()` returns a `WindowSummary` containing:

| Field | Description |
|-------|-------------|
| `net_bias` | Recency-weighted mean directional bias (signed; positive = bullish) |
| `confidence` | Simple mean confidence score over the window |
| `volatility_est` | Bessel-corrected sample std-dev of bias values — near-term signal uncertainty |
| `token_count` | Number of entries currently in the window |

**Usage:**

```cpp
#include "TokenWindowSummariser.hpp"
using namespace llmquant;

TokenWindowSummariser win(/*window_size=*/20, /*decay=*/0.95f);

// Called once per token from the LLMAdapter:
win.push(semantic_weight);

if (win.is_ready()) {
    auto s = win.summarise();
    // s.net_bias > 0  → window skewed bullish
    // s.volatility_est high → noisy / uncertain window
    // s.confidence → average token mapping confidence
}
```

The decay factor (default 0.95) attenuates older entries on every push, so
the most recent token carries the most weight.  Set `decay = 1.0` for a
uniform (equally-weighted) window.

---

## BootstrapSignalCI

`BootstrapSignalCI` (`include/BootstrapSignalCI.hpp`) maintains a rolling
reservoir of signal observations and estimates a non-parametric bootstrap
confidence interval for the signal mean.  Unlike the jackknife-based
`SignalConfidenceInterval`, it makes no Gaussian assumption and handles
fat-tailed or skewed distributions correctly at the cost of higher
computational work per `compute()` call.

**Usage:**

```cpp
#include "BootstrapSignalCI.hpp"
using namespace llmquant;

BootstrapSignalCI bci(/*n_bootstrap=*/300, /*confidence_level=*/0.95f);

// After each signal is emitted:
bci.add_observation(signal.delta_bias_shift);

if (bci.is_ready()) {
    auto ci = bci.compute();
    // ci.lower, ci.median, ci.upper
    if (ci.lower > 0.0f) {
        // The 95% CI for the mean is entirely positive — high-confidence bullish
    }
}

// JSON snapshot for dashboards:
std::cout << bci.to_stats_json() << "\n";
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bootstrap` | 200 | Re-samples per `compute()` call |
| `confidence_level` | 0.95 | Nominal CI coverage (→ [2.5%, 97.5%] percentiles) |
| `window_size` | 128 | Maximum observations retained (sliding window) |
| `min_observations` | 8 | Minimum before `is_ready()` returns true |

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
