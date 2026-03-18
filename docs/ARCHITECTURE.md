# Architecture

This document describes the internal design of `LLMTokenStreamQuantEngine` at a level sufficient to understand latency trade-offs, extend the semantic dictionary, add new risk gates, or swap out the OMS adapter.

---

## Sub-Microsecond Latency Design Decisions

The system is designed to minimise the token-to-signal latency: the elapsed time from the moment a token arrives at the socket buffer to the moment `TradeSignal` is emitted to the callback.

### No Exceptions in the Hot Path

All components on the hot path (`LLMAdapter::map_token_to_weight`, `TradeSignalEngine::process_semantic_weight`, `RiskManager::evaluate`) avoid heap allocation and do not throw exceptions. The `LLMAdapter` lookup is a single `unordered_map::find` on a pre-built hash map. The `TradeSignalEngine` uses lock-free `compare_exchange_weak` loops on `std::atomic<double>` accumulators rather than a mutex.

### Lock-Free Accumulation

`TradeSignalEngine::process_semantic_weight` updates `accumulated_bias_` and `accumulated_volatility_` with CAS (compare-and-swap) loops:

```cpp
double expected = accumulated_bias_.load(std::memory_order_relaxed);
double desired;
do {
    desired = expected * config_.signal_decay_rate + bias_contribution;
} while (!accumulated_bias_.compare_exchange_weak(
             expected, desired,
             std::memory_order_release,
             std::memory_order_relaxed));
```

This is linearisable without a mutex. The CAS loop retries only on contention, which is rare in the single-reader-single-writer use pattern described in the class-level thread-safety comment.

### SSE2 Aggregation Path

`LLMAdapter::map_sequence_simd` uses SSE2 intrinsics to accumulate four confidence-weighted sums in two 128-bit registers (`__m128d`), processing two tokens per iteration. The horizontal add is implemented with `_mm_unpacklo_pd`/`_mm_unpackhi_pd` rather than `_mm_hadd_pd` (which requires SSE3) for broader compatibility. A scalar tail loop handles odd-length sequences.

This path is invoked for multi-token sequences. For the single-token hot path (`map_token_to_weight`), the overhead of SIMD setup exceeds the benefit, so the plain scalar dictionary lookup is used directly.

### Welford Online Variance

Semantic pressure (used by `LatencyController` for backoff decisions) is computed using Welford's online algorithm, which avoids storing the full sample history. The three Welford variables (`variance_n`, `sentiment_mean_accum`, `sentiment_variance_accum`) are reset after 1,000,000 samples to prevent catastrophic cancellation in the floating-point sum-of-squares. The reset is protected by a `std::mutex` because both the token callback (writer) and the monitoring loop (resetter) access all three variables as a unit.

### Prometheus Snapshot Decoupling

The Prometheus scrape endpoint (`PrometheusExporter`) could in theory acquire the latency tracking locks on every scrape, adding jitter to the hot path. Instead, the monitoring loop builds a metrics string once per second and stores it in `prom_snapshot` under a `prom_snapshot_mutex`. The scrape thread returns this cached string without touching any hot-path state. The scrape thread therefore never contends with `LatencyController` or `TradeSignalEngine`.

---

## Semantic Weight Dictionary Design

The dictionary maps normalised token strings to `SemanticWeight` structs with four fields:

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `sentiment_score` | double | [-1, 1] | Overall polarity. Negative = bearish/fearful, positive = bullish. |
| `confidence_score` | double | [0, 1] | Dictionary confidence in the mapping. Tokens with low confidence contribute less to aggregates. |
| `volatility_score` | double | [0, 1] | Implied market volatility contribution. 0 = calm, 1 = high volatility. |
| `directional_bias` | double | [-1, 1] | Net directional pressure. Negative = sell pressure, positive = buy pressure. |

### Normalisation

Before any lookup, the raw token string is stripped of leading/trailing whitespace and lowercased. This handles the common GPT-4o formatting patterns where tokens arrive as `" Bullish"` (leading space, capitalised) rather than `"bullish"`. The same normalisation is applied in `add_token_mapping` so that inserted keys always match the lookup path.

### Default Mappings

`initialize_default_mappings()` loads approximately 40 tokens grouped into five semantic categories:

- **Fear/Panic**: `crash`, `panic`, `collapse`, `plunge`, `dump`, `breakdown`, `fear`, `selloff`, `tumble`, `rout` - high negative sentiment, high volatility, negative directional bias.
- **Certainty/Confidence**: `inevitable`, `guarantee`, `confident`, `confirmed`, `certain`, `assured` - confidence boosts with low volatility.
- **Directional Bullish**: `bullish`, `rally`, `surge`, `breakout`, `soar`, `moon`, `buy`, `long` - positive sentiment, positive directional bias.
- **Directional Bearish**: `bearish`, `short`, `sell` (plus several fear tokens that are bearish by nature) - negative sentiment, negative directional bias.
- **Volatility**: `volatile`, `spike`, `whipsaw`, `swing`, `choppy`, `erratic` - near-zero sentiment and directional bias, high volatility.
- **Neutral fillers**: `the`, `and`, `is`, `a`, `an`, `in`, `of`, `to` - near-zero weight on all dimensions. Present to prevent the deduplicator from passing them to the signal engine with a default neutral weight that could still accumulate spurious bias.

### Runtime Extension

`add_token_mapping()` can insert entries at runtime. `load_sentiment_dictionary()` reads a whitespace-delimited file with the format `<token> <sentiment> <confidence> <volatility> <bias>`. Both methods normalise the key before insertion, ensuring consistency with the lookup path.

---

## Risk Gate Cascade Ordering and Rationale

`RiskManager::evaluate` checks gates in a fixed cascade order. A signal is rejected at the first gate it fails; later gates are not evaluated. This ordering is deliberate:

### Gate 1: Magnitude

```
|delta_bias_shift| <= max_bias_magnitude
|volatility_adjustment| <= max_volatility_magnitude
|spread_modifier| <= max_spread_magnitude
```

Magnitude is checked first because it is the fastest check (three comparisons, no state) and catches unbounded accumulator runaway immediately. A signal with a bias magnitude above the limit is pathological regardless of all other attributes.

### Gate 2: Confidence

```
signal.confidence >= min_confidence
```

Confidence reflects the model's certainty about the token that triggered the signal. A low-confidence signal (generated from a token with an unusual or ambiguous meaning) is filtered before it consumes rate-limit budget or updates the drawdown accumulator.

### Gate 3: Rate Limit

```
signals_in_window < max_signals_per_second
```

The rate-limit window is a sliding 1-second bucket. It is reset on the first signal after the window expires. Rate limiting is third because: magnitude and confidence checks are stateless and fast; only signals that are semantically valid should consume rate-limit quota.

### Gate 4: Drawdown

```
|cumulative_bias + delta_bias_shift| <= max_drawdown
```

The drawdown gate accumulates the net directional bias over a rolling window (default 60 seconds) and halts signals if the cumulative exposure exceeds the threshold. It is placed after the rate gate because a well-structured signal that happens to exceed the drawdown budget is a policy decision, not a data quality issue. The drawdown window resets periodically to allow normal trading to resume after a volatile period.

### Gate 5: Position

```
|net_position + delta_bias_shift| <= position_limit
pnl >= pnl_limit
```

Position and P&L checks are last because they require OMS state updates (which happen asynchronously via the `update_position` callback). The OMS state is accurate to within the latency of the last position update. A soft warning is fired at `position_warn_fraction * position_limit` before the hard limit is reached.

### Mutex Discipline

All five checks run inside a single `std::lock_guard<std::mutex>` to ensure atomicity: a signal cannot pass magnitude but fail rate-limit on a stale counter. Callbacks (alert, OMS) are captured by value inside the lock and fired outside it to prevent deadlock when a callback re-enters `evaluate`.

---

## OMS Adapter Selection Strategy

The OMS adapter is selected at startup based on the `--oms <host:port>` command-line flag:

- **No flag**: `MockOmsAdapter` is used. It cycles through a pre-loaded sequence of `PositionState` values on a background thread, simulating OMS position updates without requiring an external system. Suitable for development, CI, and integration testing.

- **`--oms host:port`**: `RestOmsAdapter` is used. It polls an HTTP endpoint (by convention `GET /position`) for the current position state and converts the JSON response to `PositionState`. The adapter runs on a background thread; position updates are pushed to `RiskManager::update_position` via a callback.

- **FIX**: `FixOmsAdapter` implements a minimal FIX 4.2 session reader that parses `ExecutionReport` (35=8) and `PositionReport` (35=AP) messages. It is available for environments where a FIX drop-copy feed is present.

All three adapters implement the `OmsAdapter` abstract interface:

```cpp
virtual void start() = 0;
virtual void stop() = 0;
virtual void set_position_callback(PositionCallback) = 0;
```

The `main.cpp` instantiation logic creates the adapter via `std::unique_ptr<OmsAdapter>` so that the risk manager and signal callback do not need to know which concrete adapter is active.

### Hot-Reload Config

`Config::start_watching` launches a background polling thread that checks the file modification time of `config.yaml` every `poll_interval_ms` (default 2000 ms). On change, it calls `load_from_file` and then invokes the `on_reload` callback with a snapshot of the new `SystemConfig`. The monitoring loop in `main.cpp` reads `sys_config.trading.bias_sensitivity` on every tick via `config.get_config()`, which acquires the config mutex briefly. The hot-reload callback prints the new sensitivity value to stdout, enabling live tuning without restarting the process.
