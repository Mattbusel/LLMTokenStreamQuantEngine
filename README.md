# LLMTokenStreamQuantEngine

[![CI](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-788%20passing-brightgreen.svg)](tests/)
[![Dictionary](https://img.shields.io/badge/token%20dictionary-~130%20entries-blue.svg)](src/LLMAdapter.cpp)
[![Download](https://img.shields.io/badge/download-v1.1.0%20Windows%20x64-brightgreen.svg)](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.1.0)

> **Windows users:** grab the pre-built `.exe` from the [v1.1.0 release](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.1.0) — extract, edit `config.yaml`, and run. No build tools required.

A production-grade C++20 engine that ingests a live LLM token stream, maps each token to a quantitative semantic weight, accumulates directional bias and volatility signals with exponential decay, and fires risk-gated trade signals. The end-to-end token-to-signal P99 latency targets sub-10 microseconds in the hot path. There are zero managed I/O dependencies in the hot path.

---

## What It Does

1. **Token ingestion** — Connects to the OpenAI `gpt-4o` streaming API (or replays a pre-loaded token sequence in simulator mode) and delivers each token to the pipeline.
2. **Deduplication** — An FNV-1a TTL deduplicator filters repeated tokens within a configurable window (default 5 seconds). On Redis disconnect the system falls back transparently to the in-process backend with no signal loss.
3. **Semantic weighting** — An exact-match dictionary maps each token to a `SemanticWeight` (sentiment, confidence, volatility, directional bias). An SSE2 SIMD path accelerates batch scoring. The dictionary exposes a full analytics API: per-field min/max/range/average, sentiment distribution bucketing, top-N ranking, and confidence/volatility filtering.
4. **Signal generation** — Accumulates directional bias and volatility with exponential decay and emits a `TradeSignal` when the cooldown elapses (realtime) or on every token (backtest). Tracks signal efficiency, velocity, aged-out rate, noise filter rate, and a Welford running mean of signal quality.
5. **Risk gating** — A five-gate cascade (magnitude, confidence, rate-limit, drawdown, position/PnL) evaluates each signal before it leaves the engine. Per-gate block rates, utilization ratios, and a `format_stats()` summary are available at runtime.
6. **OMS integration** — Pluggable `OmsAdapter` implementations feed live position state into the risk gate: `RestOmsAdapter` (HTTP polling, tracks error/success rate), `FixOmsAdapter` (FIX 4.2 reader, tracks reconnect count and sequence number), or `MockOmsAdapter` (deterministic test stub).
7. **Observability** — `MetricsLogger` writes structured CSV or NDJSON logs with pipeline health, dedup events, trade signals, and config reloads. `PrometheusExporter` exposes a `/metrics` scrape endpoint on port 9100. `LatencyController` tracks P50/P95/P99 percentiles, sample variance, standard deviation, and a composite back-pressure signal.

---

## Architecture

```
[OpenAI gpt-4o SSE stream]
         |
   LLMStreamClient          raw TLS socket, zero-copy SSE parser
         |                  tracks: tokens_received, reconnect_count
   Deduplicator              FNV-1a TTL hash, optional Redis backend
         |                  tracks: duplicate_rate, novel_rate, total_checked
   LLMAdapter                token -> SemanticWeight (SSE2 SIMD batch path)
         |                  full dictionary analytics API
   TradeSignalEngine         exponential-decay accumulators, cooldown gate
         |                  tracks: efficiency, velocity, aged_out, quality
   RiskManager               5-gate cascade: magnitude, confidence, rate,
         |                   drawdown, position/PnL — per-gate block rates
   OmsAdapter  <------------ RestOmsAdapter / FixOmsAdapter / MockOmsAdapter
         |
   OutputSink chain          CsvOutputSink / JsonOutputSink / MemoryOutputSink
```

### Subsystems

| Subsystem | Source | Responsibility |
|-----------|--------|----------------|
| `LLMStreamClient` | `src/LLMStreamClient.cpp` | Zero-dependency TLS client. Raw TCP socket to `api.openai.com:443`, OpenSSL handshake, SSE `data:` line parser. Reconnects after `[DONE]`. Exposes `tokens_received` and `reconnect_count`. |
| `Deduplicator` | `src/Deduplicator.cpp` | FNV-1a 64-bit hash dedup with TTL eviction. Optional Redis backend (`LLMQUANT_REDIS_ENABLED`). Tracks `duplicate_rate`, `novel_rate`, and `total_checked`. |
| `LLMAdapter` | `src/LLMAdapter.cpp` | Token-to-`SemanticWeight` dictionary (~130 built-in entries). SSE2 SIMD aggregate path (`map_sequence_simd`). Full analytics API: per-field min/max/average/range, distribution buckets, top-N ranking, confidence/volatility filtering, decay, TSV import/export. |
| `TradeSignalEngine` | `src/TradeSignalEngine.cpp` | Exponential-decay bias/volatility accumulators. Cooldown-gated signal emission. Lock-free `std::atomic<double>` CAS loops. Rich stats: `signal_efficiency`, `signal_velocity`, `aged_out_rate`, `noise_filter_rate`, `avg_signal_quality`, `avg_bias_per_token`, `has_pending_bias`, `is_in_cooldown`. |
| `RiskManager` | `src/RiskManager.cpp` | Five-gate cascade. Alert and OMS callbacks. Per-gate block rates (`magnitude_block_rate`, `confidence_block_rate`), `rejection_rate`, `pass_rate`, `drawdown_utilization`, `position_utilization`, `format_stats()`. |
| `LatencyController` | `src/LatencyController.cpp` | Lock-free P50/P95/P99 percentile tracking. Sample variance and standard deviation. Welford online variance for semantic pressure. `is_under_target()`, `format_stats()`, histogram buckets. |
| `MetricsLogger` | `src/MetricsLogger.cpp` | spdlog-backed CSV and NDJSON structured logging. Methods: `log_trade_signal`, `log_config_reload`, `log_pipeline_health`, `log_dedup_event`. Tracks `uptime_ms` and `log_rate`. |
| `Config` | `src/Config.cpp` | YAML file loading/saving with range validation. Background file-watcher thread for hot-reload (zero restart). |
| `TokenStreamSimulator` | `src/TokenStreamSimulator.cpp` | Lock-free SPSC ring buffer. Tracks `tokens_emitted`, `drop_rate`, `emit_rate`, `format_stats()`. |
| `PrometheusExporter` | `src/PrometheusExporter.cpp` | Lightweight HTTP server on port 9100. Metrics snapshot decoupled from the hot path (updated once per second in the monitoring loop). |
| OMS adapters | `src/{Rest,Fix,Mock}OmsAdapter.cpp` | `RestOmsAdapter` polls `GET /positions` and tracks `error_rate`/`success_rate`. `FixOmsAdapter` parses ExecutionReport (35=8) and PositionReport (35=AP), tracking `reconnect_count` and current sequence number. `MockOmsAdapter` cycles through deterministic positions. |

### Key Design Decisions

- **No exceptions in the hot path.** All hot-path interfaces return `bool` or a result value.
- **Single background thread per stream.** The reader loop owns its socket and reconnects on EOF.
- **Per-request TLS reconnect.** OpenAI closes the connection after `[DONE]`; `SSL_CTX` is reused across reconnects, only the per-connection `SSL*` is torn down.
- **Lock-free accumulators.** `TradeSignalEngine` uses `std::atomic<double>` with CAS loops; no mutex on the hot path.
- **Welford online variance.** Semantic pressure and signal quality are tracked without storing the full sample history.
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

### Dump effective configuration and exit

```bash
./build/LLMTokenStreamQuantEngine --dump-config
./build/LLMTokenStreamQuantEngine --dump-config --config /path/to/config.yaml
```

Prints all effective parameter values (YAML defaults + file overrides) as a structured summary and exits with code 0. Useful for verifying hot-reload-eligible fields before starting the engine.

### Quiet (log-file only) mode

```bash
./build/LLMTokenStreamQuantEngine --quiet
```

Suppresses the per-signal console row and rolling stats bar. All data still flows to `MetricsLogger` and the Prometheus endpoint — suitable for daemon / CI deployments where only structured logs are consumed.

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

### LLMAdapter

```cpp
// Token lookup and batch scoring
SemanticWeight map_token_to_weight(const std::string& token) const;
SemanticWeight map_sequence_to_weight(const std::vector<std::string>& tokens) const;
SemanticWeight map_sequence_simd(const std::vector<std::string>& tokens) const;  // SSE2 path
std::vector<SemanticWeight> batch_map_tokens_to_weights(const std::vector<std::string>& tokens) const;

// Dictionary management
void add_token_mapping(const std::string& token, const SemanticWeight& weight);
size_t batch_add_token_mappings(const std::unordered_map<std::string, SemanticWeight>& mappings);
bool update_token_weight(const std::string& token, const SemanticWeight& weight);
bool remove_token_mapping(const std::string& token);
bool get_token_mapping(const std::string& token, SemanticWeight& weight) const;
bool contains_token(const std::string& token) const;
bool contains_any_of(const std::vector<std::string>& tokens) const;
void clear_custom_mappings();
void load_sentiment_dictionary(const std::string& filepath);
size_t get_dictionary_size() const;
std::vector<std::string> get_all_token_keys() const;

// Dictionary analytics — sentiment
double get_avg_sentiment() const;
double get_min_sentiment() const;
double get_max_sentiment() const;
double get_sentiment_range() const;
size_t count_bullish_tokens() const;    // sentiment > 0
size_t count_bearish_tokens() const;    // sentiment < 0
size_t count_neutral_tokens() const;    // sentiment == 0
SentimentDistribution get_sentiment_distribution() const;
std::vector<std::pair<std::string, double>> filter_tokens_by_sentiment(double min, double max) const;
std::vector<std::pair<std::string, double>> top_tokens_by_sentiment(size_t n = 10) const;

// Dictionary analytics — confidence
double get_avg_confidence() const;
double get_min_confidence() const;
double get_max_confidence() const;
double get_confidence_range() const;
std::vector<std::pair<std::string, double>> filter_tokens_by_confidence(double min, double max) const;

// Dictionary analytics — volatility
double get_avg_volatility() const;
double get_min_volatility() const;
double get_max_volatility() const;
double get_volatility_range() const;
size_t count_tokens_above_volatility(double threshold) const;
std::vector<std::pair<std::string, double>> top_tokens_by_volatility(size_t n = 10) const;
std::vector<std::pair<std::string, double>> filter_tokens_by_volatility(double min, double max) const;

// Dictionary analytics — directional bias
double get_avg_directional_bias() const;
double get_min_directional_bias() const;
double get_max_directional_bias() const;
double get_directional_bias_range() const;

// Weight decay and serialisation
void decay_all_weights(double factor);   // multiply all confidence scores by factor in [0, 1]
std::string export_dictionary() const;   // tab-separated, sorted alphabetically
size_t load_dictionary_from_tsv(const std::string& tsv_data);

// Processing stats
Stats get_stats() const noexcept;        // tokens_processed, cache_hits, cache_misses
void reset_stats() noexcept;
double get_cache_hit_rate() const noexcept;

// Token hit-frequency tracking
uint64_t get_token_hit_count(const std::string& token) const;  // lookups matched since construction / last reset
std::vector<std::pair<std::string, uint64_t>> top_tokens_by_frequency(size_t n = 10) const;  // top-N by hit count
void reset_frequency_counts();  // zero all per-token hit counters
```

### TradeSignalEngine

```cpp
// Core signal path
void process_semantic_weight(const SemanticWeight& weight);
void set_signal_callback(TradeSignalCallback callback);
void set_realtime_mode(bool realtime);

// Configuration
TradeSignalEngine::Config get_config() const;
void update_config(const TradeSignalEngine::Config& config);
void set_signal_cooldown(std::chrono::microseconds cooldown);
void set_min_bias_threshold(double threshold);

// Accumulator state
double get_accumulated_bias() const noexcept;
double get_accumulated_volatility() const noexcept;
bool has_pending_bias() const noexcept;
int get_bias_direction() const noexcept;    // +1, 0, or -1
double get_peak_bias() const noexcept;

// Signal stats
uint64_t get_signals_generated() const noexcept;
uint64_t get_signals_suppressed() const noexcept;
uint64_t get_tokens_processed() const noexcept;
uint64_t get_signals_aged_out() const noexcept;
double get_noise_filter_rate() const noexcept;
double get_aged_out_rate() const noexcept;
double get_accumulator_clamp_rate() const noexcept;
double get_signal_efficiency() const noexcept;  // signals_generated / tokens_processed
double get_signal_velocity() const noexcept;    // signals per second since reset
double get_avg_signal_quality() const noexcept;
double get_avg_signal_strength() const noexcept;
double get_avg_bias_per_token() const noexcept;
double get_last_signal_quality() const noexcept;
std::vector<QualityHistogramBucket> get_quality_histogram() const;  // 5 buckets: [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]

// Timing
double get_tokens_per_second() const noexcept;
double get_session_duration_ms() const noexcept;
double get_time_since_last_signal_us() const noexcept;
bool is_in_cooldown() const noexcept;

// TradeSignal helpers (on the TradeSignal struct itself)
std::string to_string() const;  // "bias=<v> vol=<v> spread=<v> conf=<v> quality=<v> lat=<v>us strategy=±1 weight=<v>"
std::string to_json() const;    // compact single-line JSON object with all numeric fields

// Output and introspection
void add_output_sink(std::shared_ptr<OutputSink> sink);
void flush_sinks();
std::string format_stats() const;
Snapshot snapshot() const noexcept;
void reset();
```

### RiskManager

```cpp
// Core evaluation
bool evaluate(const TradeSignal& signal);
std::vector<bool> evaluate_batch(const std::vector<TradeSignal>& signals);

// Configuration
RiskManager::Config get_config() const;
void update_config(const RiskManager::Config& config);
void set_position_limit(double position_limit, double pnl_limit = -10.0);

// Aggregate rates
double get_rejection_rate() const noexcept;      // blocked / total
double get_pass_rate() const noexcept;            // 1 - rejection_rate
double get_magnitude_block_rate() const noexcept;
double get_confidence_block_rate() const noexcept;
uint64_t get_total_signals_evaluated() const noexcept;
std::string get_most_blocked_gate() const;        // "magnitude"|"confidence"|"rate"|"drawdown"|"position"|"pnl"|"none"

// Position and drawdown
double get_net_exposure() const;
double get_position_utilization() const;
double get_drawdown_utilization() const;

// Rate window
double get_signals_per_second() const;
bool is_rate_limited() const;
double get_window_time_elapsed_ms() const;

// Health and diagnostics
bool is_healthy() const;
GateStatus get_gate_status() const;
BlockedByGate get_blocked_by_gate() const noexcept;
std::string format_stats() const;

// Stats
Stats get_stats() const;
void reset_stats();
```

### LatencyController

```cpp
// Measurement
void record_latency(std::chrono::microseconds latency);
void start_measurement();
void end_measurement();

// Stats
LatencyStats get_stats() const;
std::chrono::microseconds get_percentile(double p) const;
size_t get_sample_count() const;
double get_sample_variance_us() const;
double get_stddev_us() const;
double get_throughput_estimate() const noexcept;  // measurements per second
bool is_under_target() const noexcept;
int64_t get_p99_us() const;  // convenience: get_stats().p99_latency.count()
int64_t get_p95_us() const;  // convenience: get_stats().p95_latency.count()
int64_t get_p50_us() const;  // convenience: get_stats().p50_latency.count()

// Back-pressure
PressureState get_pressure() const;
HealthState get_health_state() const;
double get_backoff_multiplier() const;
double get_latency_budget_remaining_us() const;

// Diagnostics
std::string format_stats() const;
std::vector<HistogramBucket> histogram_buckets() const;
void update_config(const Config& config);
void reset_stats();
```

### MetricsLogger

```cpp
// Structured log events
void log_signal(const TradeSignal& signal, bool passed);
void log_trade_signal(double bias, double volatility, double confidence,
                      double latency_us, double quality);
void log_config_reload(const std::string& source_path, bool success);
void log_pipeline_health(double latency_p99_us, double pressure,
                         uint64_t signals_generated, uint64_t signals_blocked);
void log_dedup_event(const std::string& token, bool is_duplicate, uint64_t hash);
void flush();

// Uptime and throughput
double get_uptime_ms() const noexcept;
double get_log_rate() const noexcept;    // log entries per second
uint64_t get_log_entries() const noexcept;
```

### TokenStreamSimulator

```cpp
void load_tokens(const std::vector<std::string>& tokens);
void set_token_callback(TokenCallback callback);
void set_token_interval(std::chrono::milliseconds interval);
void start();
void stop();

uint64_t get_token_count() const noexcept;
uint64_t get_tokens_emitted() const noexcept;
double get_drop_rate() const noexcept;    // ring buffer overflow rate
double get_emit_rate() const noexcept;    // tokens per second
std::string format_stats() const;
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

Per-gate block rates are available at runtime:

```cpp
rm.get_magnitude_block_rate()   // fraction blocked by gate 1
rm.get_confidence_block_rate()  // fraction blocked by gate 2
rm.get_rejection_rate()         // overall blocked / total
rm.format_stats()               // single-line diagnostic string
```

---

## Token Semantic Dictionary

| Category | Tokens | Effect |
|----------|--------|--------|
| Fear / Panic | `crash`, `panic`, `collapse`, `plunge`, `dump`, `breakdown`, `fear`, `selloff`, `tumble`, `rout`, `liquidation`, `capitulation`, `deleveraging` | Strong negative BIAS, high VOL |
| Directional Bullish | `bullish`, `rally`, `surge`, `breakout`, `soar`, `moon`, `buy`, `long`, `accumulate`, `rebound`, `recovery`, `uptrend`, `oversold` | Positive BIAS |
| Directional Bearish | `bearish`, `short`, `sell`, `downtrend`, `overbought`, `distribution` | Negative BIAS |
| Volatility | `volatile`, `spike`, `whipsaw`, `swing`, `choppy`, `erratic`, `straddle`, `strangle`, `gamma`, `vega`, `iv`, `reversal`, `parabolic`, `divergence` | VOL spike, near-zero BIAS |
| Options / Derivatives | `calls`, `puts`, `delta`, `dte`, `expiry`, `strike`, `hedge`, `squeeze` | Options market signals |
| Certainty | `inevitable`, `guarantee`, `confident`, `confirmed`, `certain`, `assured` | Confidence boost |
| Corporate / Earnings | `earnings`, `guidance`, `upgrade`, `downgrade`, `beats`, `misses`, `outlook`, `revenue`, `profit`, `loss`, `dividend`, `buyback`, `merger`, `acquisition`, `ipo` | Fundamental event signals |
| Market Regime / Macro | `inflation`, `deflation`, `recession`, `stagflation`, `fed`, `hike`, `cut`, `pivot`, `gdp`, `risk-on`, `risk-off`, `stimulus`, `tightening`, `easing`, `default`, `sanctions`, `tariff`, `contagion`, `systemic`, `geopolitical` | Macro sentiment |
| Analyst Sentiment | `upgrade`, `downgrade`, `overweight`, `underweight`, `outperform`, `underperform`, `neutral`, `hold`, `target` | Analyst-driven signals |
| Crypto / Retail | `pump`, `rug`, `fud`, `hodl`, `rekt`, `ath`, `dte` | Social-media/Reddit sentiment |
| Neutral filler | `the`, `and`, `is`, `a`, `an`, `in`, `of`, `to`, `or`, `not`, `with`, `for`, `as`, `at`, `on`, `it`, `by`, `from` | Near-zero weight on all dimensions |

All entries are in `src/LLMAdapter.cpp::initialize_default_mappings()` and can be extended at runtime via `add_token_mapping()`, loaded in bulk from a whitespace-delimited file via `load_sentiment_dictionary()`, or imported from a TSV string via `load_dictionary_from_tsv()`. The full dictionary can be exported to TSV via `export_dictionary()`.

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
- **Welford online variance:** Semantic pressure and signal quality are tracked using Welford's algorithm, avoiding the need to store all samples while remaining numerically stable.

---

## Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure --parallel 4

# Run only unit tests
ctest --test-dir build -L unit --output-on-failure

# Run only integration tests
ctest --test-dir build -L integration --output-on-failure

# Run performance benchmarks
ctest --test-dir build -L performance --output-on-failure
```

The test suite has **788 passing tests** (1 skipped: file-permission test, Windows-only) covering:

| Category | Count | Coverage |
|----------|-------|----------|
| Unit tests | ~680 | All modules, every public API method |
| Integration tests | ~50 | End-to-end pipeline, OMS gates, chaos scenarios |
| Invariant tests | ~6 | Dedup key determinism, sentiment sign, counter identity, confidence bounds |
| Performance benchmarks | 5 | Latency budgets, throughput, SIMD vs scalar |
| Chaos / fault injection | ~6 | Fear saturation, runaway bias, dedup flood, restart under load |
| Network error paths | ~23 | LLMStreamClient and OMS adapter error handling |
| Edge cases | ~33 | Empty inputs, NaN, overflow, invalid params |

---

## License

MIT — see `LICENSE`.
