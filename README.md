# LLMTokenStreamQuantEngine

[![CI](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](CHANGELOG.md)

A production-grade C++20 engine that ingests a live LLM token stream, maps each token to a quantitative semantic weight, accumulates directional bias and volatility signals with exponential decay, and fires risk-gated trade signals. The end-to-end token-to-signal P99 latency targets sub-10 microseconds in the hot path. There are zero managed I/O dependencies in the hot path.

---

## What It Does

1. **Token ingestion** — Connects to the OpenAI `gpt-4o` streaming API (or replays a pre-loaded token sequence in simulator mode) and delivers each token to the pipeline.
2. **Deduplication** — An FNV-1a TTL deduplicator filters repeated tokens within a configurable window, with an optional Redis backend for cross-process dedup.
3. **Semantic weighting** — An exact-match dictionary maps each token to a `SemanticWeight` (sentiment, confidence, volatility, directional bias). An SSE2 SIMD path accelerates batch scoring.
4. **Signal generation** — Accumulates directional bias and volatility with exponential decay and emits a `TradeSignal` when the cooldown elapses (realtime) or on every token (backtest).
5. **Risk gating** — A five-gate cascade (magnitude, confidence, rate-limit, drawdown, position/PnL) evaluates each signal before it leaves the engine.
6. **OMS integration** — Pluggable `OmsAdapter` implementations feed live position state into the risk gate: `RestOmsAdapter` (HTTP polling), `FixOmsAdapter` (FIX 4.2 reader), or `MockOmsAdapter` (deterministic test stub).
7. **Observability** — `MetricsLogger` writes structured CSV or NDJSON logs via spdlog; `PrometheusExporter` exposes a `/metrics` scrape endpoint on port 9100; `LatencyController` tracks P50/P95/P99 percentiles and a composite back-pressure signal.

---

## Architecture

```
[OpenAI gpt-4o SSE stream]
         |
   LLMStreamClient          raw TLS socket, zero-copy SSE parser
         |
   Deduplicator              FNV-1a TTL hash, optional Redis backend
         |
   LLMAdapter                token -> SemanticWeight (SSE2 SIMD batch path)
         |
   TradeSignalEngine         exponential-decay accumulators, cooldown gate
         |
   RiskManager               5-gate cascade: magnitude, confidence, rate,
         |                   drawdown, position/PnL
   OmsAdapter  <------------ RestOmsAdapter / FixOmsAdapter / MockOmsAdapter
         |
   OutputSink chain          CsvOutputSink / JsonOutputSink / MemoryOutputSink
```

### Subsystems

| Subsystem | Source | Responsibility |
|-----------|--------|----------------|
| `LLMStreamClient` | `src/LLMStreamClient.cpp` | Zero-dependency TLS client. Raw TCP socket to `api.openai.com:443`, OpenSSL handshake, SSE `data:` line parser. Reconnects after `[DONE]`. |
| `Deduplicator` | `src/Deduplicator.cpp` | FNV-1a 64-bit hash dedup with TTL eviction. Optional Redis backend (`LLMQUANT_REDIS_ENABLED`). |
| `LLMAdapter` | `src/LLMAdapter.cpp` | Token-to-`SemanticWeight` dictionary (~40 built-in entries). SSE2 SIMD aggregate path (`map_sequence_simd`). |
| `TradeSignalEngine` | `src/TradeSignalEngine.cpp` | Exponential-decay bias/volatility accumulators. Cooldown-gated signal emission. Lock-free `std::atomic<double>` CAS loops. Pluggable `OutputSink` chain. |
| `RiskManager` | `src/RiskManager.cpp` | Five-gate cascade. Alert and OMS callbacks. `MetricsLogger` integration for structured rejection logging. |
| `LatencyController` | `src/LatencyController.cpp` | Lock-free P50/P95/P99 percentile tracking. Welford online variance for semantic pressure. Composite back-pressure signal with exponential backoff. |
| `MetricsLogger` | `src/MetricsLogger.cpp` | spdlog-backed CSV and NDJSON structured logging with configurable flush interval. |
| `Config` | `src/Config.cpp` | YAML file loading/saving with range validation. Background file-watcher thread for hot-reload (zero restart). |
| `PrometheusExporter` | `src/PrometheusExporter.cpp` | Lightweight HTTP server on port 9100. Metrics snapshot decoupled from the hot path (updated once per second in the monitoring loop). |
| OMS adapters | `src/{Rest,Fix,Mock}OmsAdapter.cpp` | Pluggable `OmsAdapter` implementations. Mock cycles through deterministic positions; REST polls `GET /positions`; FIX parses ExecutionReport (35=8) and PositionReport (35=AP). |

### Key Design Decisions

- **No exceptions in the hot path.** All hot-path interfaces return `bool` or a result value.
- **Single background thread per stream.** The reader loop owns its socket and reconnects on EOF.
- **Per-request TLS reconnect.** OpenAI closes the connection after `[DONE]`; `SSL_CTX` is reused across reconnects, only the per-connection `SSL*` is torn down.
- **Lock-free accumulators.** `TradeSignalEngine` uses `std::atomic<double>` with CAS loops; no mutex on the hot path.
- **Welford online variance.** Semantic pressure is tracked without storing the full sample history.
- **Prometheus snapshot decoupling.** The monitoring loop builds the metrics string once per second; the scrape thread never contends with `LatencyController` or `TradeSignalEngine`.
- **Structured error logging.** All library code routes diagnostic output through spdlog; no raw `std::cerr` in the library layer.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for lock ordering, risk gate cascade rationale, and SIMD aggregation path details.

---

## Build Instructions

### Prerequisites

| Tool | Version |
|------|---------|
| CMake | 3.20+ |
| C++ compiler | GCC 12+ / Clang 14+ / MSVC 19.44+ |
| spdlog | any recent version |
| yaml-cpp | any recent version |
| nlohmann/json | 3.x |
| GTest | 1.12+ |
| OpenSSL | 1.1+ (optional — enables TLS) |
| hiredis | any (optional — enables Redis dedup) |

### Linux / macOS (GCC or Clang)

```bash
# Install dependencies (Ubuntu / Debian)
sudo apt-get install -y cmake ninja-build libspdlog-dev libyaml-cpp-dev \
    libgtest-dev nlohmann-json3-dev libssl-dev

# Clone and build
git clone https://github.com/Mattbusel/LLMTokenStreamQuantEngine
cd LLMTokenStreamQuantEngine

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLMQUANT_WARNINGS_AS_ERRORS=ON

cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure --parallel 4

# Generate API documentation (requires Doxygen + Graphviz)
cmake --build build --target docs
# Open docs/api/html/index.html
```

#### Debug build with AddressSanitizer

```bash
cmake -B build_asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLMQUANT_ENABLE_ASAN=ON

cmake --build build_asan --parallel

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  ctest --test-dir build_asan --output-on-failure
```

### Windows (MSVC + vcpkg)

```powershell
# Install vcpkg dependencies
vcpkg install spdlog yaml-cpp gtest nlohmann-json openssl --triplet x64-windows

cmake -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DLLMQUANT_WARNINGS_AS_ERRORS=ON

cmake --build build --config Release --parallel

ctest --test-dir build -C Release --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `LLMQUANT_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer + UBSan (non-MSVC Debug builds) |
| `LLMQUANT_WARNINGS_AS_ERRORS` | `ON` | Treat compiler warnings as errors |
| `LLMQUANT_ENABLE_CLANG_TIDY` | `OFF` | Run clang-tidy on every source file during build |

---

## Usage Examples

### Simulator mode (no API key required)

```bash
./build/LLMTokenStreamQuantEngine --no-color
```

Replays a built-in token loop (`crash`, `panic`, `bullish`, `breakout`, ...) through the full signal pipeline at 10 ms/token. The console shows a rolling stats bar with P99 latency, composite pressure, and per-gate block counts.

### Live stream mode (OpenAI gpt-4o)

```bash
# Pass the API key directly
./build/LLMTokenStreamQuantEngine --stream sk-proj-YOUR_KEY_HERE

# Or set the environment variable
export LLMQUANT_API_KEY=sk-proj-YOUR_KEY_HERE
./build/LLMTokenStreamQuantEngine --stream
```

Connects to `api.openai.com:443` over TLS, streams a financial-sentiment completion every 5 seconds, and fires live signals.

### With REST OMS position feed

```bash
./build/LLMTokenStreamQuantEngine --oms 127.0.0.1:8080
```

### Debug raw socket output

```bash
./build/LLMTokenStreamQuantEngine --stream --debug-raw
```

Dumps every raw byte from the TLS socket to stderr for 3 seconds and exits. Useful for diagnosing chunked-encoding or auth failures.

### Custom configuration

```bash
./build/LLMTokenStreamQuantEngine config.yaml
```

All config values are hot-reloaded without restart. See the [Configuration Reference](#configuration-reference) section below.

---

## API Reference

Full Doxygen-generated HTML documentation is produced by `cmake --build build --target docs` and written to `docs/api/html/index.html`.

### Core types

| Type | Header | Description |
|------|--------|-------------|
| `SemanticWeight` | `LLMAdapter.h` | Normalised token weights: sentiment, confidence, volatility, directional bias. All in [-1, 1] except confidence in [0, 1]. |
| `TradeSignal` | `TradeSignalEngine.h` | Emitted signal: delta_bias_shift, volatility_adjustment, spread_modifier, confidence, latency_us, strategy_toggle. |
| `RiskManager::PositionState` | `RiskManager.h` | OMS snapshot: net_position, position_limit, pnl, pnl_limit. |
| `SystemConfig` | `Config.h` | Aggregated YAML-parsed configuration. |

### Key interfaces

```cpp
// Map a single token to a semantic weight.
SemanticWeight LLMAdapter::map_token_to_weight(const std::string& token) const;

// Batch-score a token sequence (SSE2 path).
SemanticWeight LLMAdapter::map_sequence_simd(const std::vector<std::string>& tokens) const;

// Process a weight through the signal engine.
void TradeSignalEngine::process_semantic_weight(const SemanticWeight& weight);

// Evaluate a signal against the risk cascade. Returns true if the signal passes.
bool RiskManager::evaluate(const TradeSignal& signal);

// Register a callback for emitted signals.
void TradeSignalEngine::set_signal_callback(TradeSignalCallback callback);

// Load configuration from YAML.
bool Config::load_from_file(const std::string& filepath);

// Hot-reload on file change.
void Config::start_watching(const std::string& filepath,
                            std::function<void(const SystemConfig&)> on_reload,
                            int poll_interval_ms = 500);
```

---

## Configuration Reference

`config.yaml` controls all runtime parameters and is hot-reloaded without process restart:

```yaml
token_stream:
  data_file_path: "tokens.txt"      # Token file path (ignored when use_memory_stream: true)
  token_interval_ms: 10             # Emission interval in simulator mode (ms)
  buffer_size: 1024                 # SPSC ring buffer capacity
  use_memory_stream: true           # true = built-in token loop; false = file source

trading:
  bias_sensitivity: 1.0             # Scale factor on the directional_bias accumulator
  volatility_sensitivity: 1.0       # Scale factor on the volatility accumulator
  signal_decay_rate: 0.95           # Per-token exponential decay — must be in (0, 1]
  signal_cooldown_us: 1000          # Minimum microseconds between signal emissions

latency:
  target_latency_us: 10             # P99 budget in microseconds; alert fires if exceeded
  sample_window: 1000               # Samples retained for P50/P95/P99 computation
  enable_profiling: false           # true = emit per-measurement latency log entries

logging:
  log_file_path: "signals.log"      # Output file; empty = no file logging
  format: "JSON"                    # "JSON" or "CSV"
  enable_console: false             # true = also log to stdout
  flush_interval_ms: 100            # File sink flush interval
```

All fields have safe compiled-in defaults; missing fields fall through silently.

---

## Risk Gate Reference

Gates are evaluated in cascade order. A signal is rejected at the first gate it fails.

| Gate | Order | Check | Default threshold |
|------|-------|-------|-------------------|
| Magnitude | 1 | `|delta_bias_shift| <= max_bias_magnitude` AND `|volatility_adjustment| <= max_volatility_magnitude` | 1.0 / 1.0 |
| Confidence | 2 | `signal.confidence >= min_confidence` | 0.1 |
| Rate limit | 3 | `signals_in_1s_window < max_signals_per_second` | 100 |
| Drawdown | 4 | `|cumulative_bias + delta_bias_shift| <= max_drawdown` within `drawdown_window` | 5.0 / 60 s |
| Position | 5 | `|net_position + delta_bias_shift| <= position_limit` AND `pnl >= pnl_limit` | from OMS |

A soft warning fires the OMS callback with event `"position_limit_approaching"` at `position_warn_fraction * position_limit` (default 80%) without blocking the signal.

---

## Token Semantic Dictionary

| Category | Tokens | Effect |
|----------|--------|--------|
| Fear / Panic | `crash`, `panic`, `collapse`, `plunge`, `dump`, `breakdown`, `fear`, `selloff`, `tumble`, `rout` | Strong negative BIAS, high VOL |
| Directional Bullish | `bullish`, `rally`, `surge`, `breakout`, `soar`, `moon`, `buy`, `long` | Positive BIAS |
| Directional Bearish | `bearish`, `short`, `sell` | Negative BIAS |
| Volatility | `volatile`, `spike`, `whipsaw`, `swing`, `choppy`, `erratic` | VOL spike, near-zero BIAS |
| Certainty | `inevitable`, `guarantee`, `confident`, `confirmed`, `certain`, `assured` | Confidence boost |
| Neutral filler | `the`, `and`, `is`, `a`, `an`, `in`, `of`, `to` | Near-zero weight on all dimensions |

All entries are in `src/LLMAdapter.cpp::initialize_default_mappings()` and can be extended at runtime via `add_token_mapping()` or loaded in bulk from a whitespace-delimited dictionary file.

---

## Prometheus Metrics Reference

The Prometheus scrape endpoint listens on port 9100 (configurable).

| Metric | Type | Description |
|--------|------|-------------|
| `llmquant_signals_generated_total` | counter | Total trade signals emitted by `TradeSignalEngine`. |
| `llmquant_signals_blocked_total` | counter | Total signals blocked by any risk gate. |
| `llmquant_latency_p99_us` | gauge | P99 token-to-signal latency in microseconds over the last `sample_window` measurements. |
| `llmquant_latency_avg_us` | gauge | Mean token-to-signal latency in microseconds. |

```bash
curl http://localhost:9100/metrics
```

---

## Performance Notes

- **Hot-path latency:** The `LLMAdapter -> TradeSignalEngine` path is allocation-free after startup. The P99 target is sub-10 μs on a modern desktop CPU.
- **SIMD acceleration:** `LLMAdapter::map_sequence_simd()` uses SSE2 intrinsics to process token pairs simultaneously. The scalar `map_sequence_to_weight()` is equivalent but ~2x slower on batches of two or more tokens.
- **Lock-free accumulators:** `TradeSignalEngine` uses `std::atomic<double>` CAS loops to update bias and volatility without a mutex.
- **Ring buffer:** `TokenStreamSimulator` uses a lock-free SPSC ring buffer with power-of-two capacity and cache-line-separated head/tail atomics to avoid false sharing.
- **Back-pressure:** `LatencyController` tracks a composite pressure signal (ingestion rate, semantic variance, queue depth) and exposes an exponential backoff multiplier (1x to 5x) that the monitoring loop can use to throttle upstream token production.
- **Welford online variance:** Semantic pressure is tracked using Welford's algorithm, avoiding the need to store all samples while remaining numerically stable.

---

## Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure --parallel 4

# Run only unit tests
ctest --test-dir build -L unit --output-on-failure

# Run only integration tests
ctest --test-dir build -L integration --output-on-failure
```

The test suite covers unit, integration, property-invariant, and chaos/fault-injection scenarios including backpressure cascade, circuit-breaker recovery, dedup races, retry backoff timing, and adversarial JSON SSE parsing.

---

## License

MIT — see `LICENSE`.
