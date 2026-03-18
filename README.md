# LLMTokenStreamQuantEngine

[![CI](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/LLMTokenStreamQuantEngine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Tests](https://img.shields.io/badge/tests-1491%20passing-brightgreen.svg)](tests/)

Connects directly to the OpenAI `gpt-4o` streaming API over a raw TLS socket, ingests the token stream token-by-token, maps each token through a semantic weight dictionary, accumulates directional bias and volatility signals, and fires risk-gated trade signals. End-to-end token-to-signal latency targets sub-10-microsecond P99 in the hot path. Zero managed I/O dependencies in the hot path.

---

## Pipeline

```
[OpenAI gpt-4o Streaming] -> [LLMStreamClient] -> [Deduplicator]
                                                         |
                                                  [LLMAdapter]
                                                (semantic weights)
                                                         |
                                             [TradeSignalEngine]
                                                         |
                                               [RiskManager] -> [OMS Adapter]
                                                             (Mock / REST / FIX)
```

---

## Quickstart

### Prerequisites

- **Windows**: MSVC 19.44+ (Visual Studio 2022 BuildTools)
- **CMake** 3.20+
- **vcpkg** with packages: `spdlog`, `yaml-cpp`, `gtest`, `openssl`, `nlohmann-json`

### Build (Windows / MSVC)

```powershell
git clone https://github.com/Mattbusel/LLMTokenStreamQuantEngine
cd LLMTokenStreamQuantEngine

cmake -B build -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

cmake --build build --config Release
```

### Run - Simulator Mode (no API key required)

```powershell
cd build\Release
.\LLMTokenStreamQuantEngine.exe --no-color
```

Plays back a built-in token loop (`crash`, `panic`, `bullish`, `breakout`, ...) through the full signal pipeline.

### Run - Live Stream Mode (OpenAI gpt-4o)

```powershell
cd build\Release
.\LLMTokenStreamQuantEngine.exe --stream "sk-proj-YOUR_KEY_HERE" --no-color
```

Connects to `api.openai.com:443` over TLS, authenticates, streams a financial sentiment completion every 5 seconds, and fires live signals.

Alternatively, set the environment variable and omit the key argument:

```powershell
$env:LLMQUANT_API_KEY = "sk-proj-YOUR_KEY_HERE"
.\LLMTokenStreamQuantEngine.exe --stream --no-color
```

### Run - Custom Config

```powershell
.\LLMTokenStreamQuantEngine.exe config.yaml --no-color
```

### Debug Raw Socket Output

```powershell
.\LLMTokenStreamQuantEngine.exe --stream --debug-raw
```

Dumps every raw byte from the TLS socket to stderr for 3 seconds and exits. Useful for diagnosing chunked encoding, SSE framing, or auth failures.

---

## Configuration Reference

`config.yaml` controls all runtime parameters and is hot-reloaded without process restart:

```yaml
token_stream:
  data_file_path: "tokens.txt"      # Path to token file (ignored in memory stream mode)
  token_interval_ms: 10             # Emission interval in simulator mode (ms)
  buffer_size: 1024                 # Ring buffer capacity (tokens)
  use_memory_stream: true           # true = built-in token loop; false = file source

trading:
  bias_sensitivity: 1.0             # Scale factor on the directional_bias accumulator
  volatility_sensitivity: 1.0       # Scale factor on the volatility accumulator
  signal_decay_rate: 0.95           # Per-token exponential decay (must be in (0, 1])
  signal_cooldown_us: 1000          # Minimum microseconds between signal emissions

latency:
  target_latency_us: 10             # P99 budget in microseconds; alert fires if exceeded
  sample_window: 1000               # Number of measurements for P50/P99 computation
  enable_profiling: false           # true = emit per-measurement latency log entries

pressure:
  max_ingestion_rate_tps: 10000     # TPS at which ingestion pressure = 1.0
  high_pressure_threshold: 0.8      # Composite pressure above which backoff activates
  max_backoff_multiplier: 5.0       # Maximum signal cooldown multiplier under pressure

semantic_weights:
  fear_multiplier: 1.2              # Scale factor applied to fear-category tokens
  bullish_multiplier: 1.0           # Scale factor applied to bullish-category tokens
  bearish_multiplier: 1.2           # Scale factor applied to bearish-category tokens
  volatility_multiplier: 1.1        # Scale factor applied to volatility-category tokens

logging:
  log_file_path: "signals.log"      # Output file; empty = no file logging
  format: "JSON"                    # "JSON" or "CSV"
  enable_console: false             # true = also log to stdout
  flush_interval_ms: 100            # Flush interval for the file sink
```

All fields have safe defaults; missing fields fall through to the compiled-in defaults without error.

---

## Risk Gate Documentation

Gates are evaluated in cascade order. A signal is rejected at the first gate it fails.

| Gate | Order | Formula | Default Threshold | Reason for Position |
|------|-------|---------|-------------------|---------------------|
| Magnitude | 1 | `|delta_bias_shift| <= max_bias_magnitude` AND `|volatility_adjustment| <= max_volatility_magnitude` | 2.0 / 2.0 | Fastest check; catches accumulator runaway before consuming any state |
| Confidence | 2 | `signal.confidence >= min_confidence` | 0.1 | Stateless; filters semantically weak signals before they use rate quota |
| Rate limit | 3 | `signals_in_window < max_signals_per_second` per 1-second window | 500 | Only valid signals consume rate budget |
| Drawdown | 4 | `|cumulative_bias + delta_bias_shift| <= max_drawdown` within `drawdown_window` | 10.0 / 60 s | Rolling directional exposure cap; resets after window to allow recovery |
| Position | 5 | `|net_position + delta_bias_shift| <= position_limit` AND `pnl >= pnl_limit` | From OMS state | Requires OMS position state; checked last as it is the most expensive |

A soft warning is fired at `position_warn_fraction * position_limit` (default 80%) before the hard position limit is reached. The soft warning fires the OMS callback with event `"position_limit_approaching"` without blocking the signal.

---

## Token-to-Signal Mapping

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

The Prometheus scrape endpoint listens on port 9100 (configurable). Metrics are updated once per second from the monitoring loop and served without holding any hot-path locks.

| Metric name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `llmquant_signals_generated_total` | counter | none | Total trade signals emitted by `TradeSignalEngine`, including those later blocked by `RiskManager`. |
| `llmquant_signals_blocked_total` | counter | none | Total signals blocked by any risk gate. Sum of `signals_blocked_magnitude`, `signals_blocked_confidence`, `signals_blocked_rate`, `signals_blocked_drawdown`, `signals_blocked_position`, and engine-level `signals_suppressed`. |
| `llmquant_latency_p99_us` | gauge | none | 99th-percentile token-to-signal latency in microseconds over the most recent `sample_window` measurements. |
| `llmquant_latency_avg_us` | gauge | none | Mean token-to-signal latency in microseconds over the most recent `sample_window` measurements. |

Scrape with:

```bash
curl http://localhost:9100/metrics
```

Or configure a Prometheus `scrape_config`:

```yaml
scrape_configs:
  - job_name: llmquant
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 5s
```

---

## Architecture Notes

- No exceptions in hot path. All error surfaces return `bool` or `Result`-style values.
- Single background thread per stream. The reader loop owns its socket and reconnects on EOF.
- Per-request TLS reconnect. OpenAI closes after `[DONE]`; client reopens cleanly.
- `SSL_CTX` reused across reconnects. Only the per-connection `SSL*` is torn down.
- Windows CA store injection. vcpkg OpenSSL has no CA bundle; system ROOT certs loaded via `CertOpenSystemStore` and `d2i_X509` at startup.
- Welford online variance. Semantic pressure tracked without storing sample history.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed description of each subsystem.

---

## Tests

```powershell
cmake --build build --config Release --target tests
cd build\Release
.\tests.exe
```

1,491 tests across unit, integration, property-based (proptest-style), and chaos/fault-injection suites. Full pipeline end-to-end covered including backpressure cascade, circuit breaker recovery, dedup races, and retry backoff timing.

---

## License

MIT -- see `LICENSE`.
