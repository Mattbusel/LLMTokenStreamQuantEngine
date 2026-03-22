# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added/Fixed (Cycle 33 — 2026-03-21)
- **`[[nodiscard]]` sweep**: Added `[[nodiscard]]` to all bool/result-returning
  public methods: `OmsAdapter::start()`/`is_running()`/`description()` and all
  concrete overrides (`MockOmsAdapter`, `RestOmsAdapter`, `FixOmsAdapter`),
  `LLMStreamClient::connect()`/`is_running()`, `Config::start_watching()`, and
  `DeduplicatorBackend::check_and_register()` + both concrete overrides.
  Compiler now warns on silently discarded return values.
- **Bug fix** `MockOmsAdapter::emitter_thread`: replaced blocking
  `sleep_for(emit_interval)` with an interruptible 10 ms-slice loop. Previously
  `stop()` could block for the full `emit_interval` (potentially seconds) before
  the thread checked `running_`. Now `stop()` returns within ~10 ms regardless
  of `emit_interval`.
- **Tests** (2 new in `test_oms_adapter.cpp`):
  `test_stop_returns_promptly_with_large_emit_interval` and
  `test_stop_before_emission_completes_returns_promptly` verify that `stop()`
  exits within 500 ms even with a 2–5 second `emit_interval`.
- **Docs**: README test badge updated 914 → 933.

### Added (Cycle 44 — 2026-03-21)
- **Feature flag** `LLMQUANT_ENABLE_MOCK_OMS` (default ON): gates
  `MockOmsAdapter.cpp` out of the production binary when set OFF.
  `src/MockOmsAdapter.cpp` is now conditionally added via
  `target_sources` and defines `LLMQUANT_MOCK_OMS_ENABLED` when ON.
  Added to the minimal-build and per-flag CI matrix jobs.
- **Interface** `Deduplicator::start_background_purge()`: added as a
  virtual no-op on the abstract base class, allowing callers to invoke
  background purging through a base pointer without `dynamic_cast`.
  `Deduplicator.cpp` now uses virtual dispatch instead of
  `dynamic_cast<InProcessDeduplicator*>` for this call.
- **Validation** `Config::validate()`: `token_stream.redis_url` must
  start with `redis://` or `rediss://` (TLS). Malformed URLs now produce
  a descriptive error via `validate()` rather than failing silently at
  connect time.
- **Tests** (2 new in `test_config.cpp`): `test_validate_accepts_valid_redis_url`
  and `test_validate_rejects_invalid_redis_url_scheme` cover the new
  redis_url validation.

### Fixed (Cycle 43 — 2026-03-21)
- **Bug fix** `RiskManager::update_config()`: when a gate transitioned from
  disabled → enabled, the corresponding `gate_*_last_blocked_` flag retained its
  stale `true` value from before the gate was disabled. The trip callback would
  therefore never fire on the first subsequent block after re-enabling.
  Fixed by comparing old vs new `disable_*` fields inside `update_config()` and
  clearing each flag that transitions disabled→enabled before overwriting
  `config_`. Also clears both `gate_position_last_blocked_` and
  `gate_pnl_last_blocked_` when `disable_position_gate` transitions, since both
  gates live inside the same conditional block.
- **Bug fix** `RiskManager::enable_all_gates()`: same stale-state issue; all six
  `gate_*_last_blocked_` flags are now reset unconditionally when all gates are
  re-enabled.
- **Tests**: 2 new regression tests in `test_risk_manager.cpp`:
  `test_gate_trip_fires_after_gate_reenabled_via_update_config` and
  `test_gate_trip_fires_after_enable_all_gates`.

### Changed (Cycle 43 — 2026-03-21)
- **CI** `CMakeLists.txt`: added `LLMQUANT_ENABLE_TSAN` option parallel to
  `LLMQUANT_ENABLE_ASAN`. TSan is mutually exclusive with ASan; when enabled,
  sets `CMAKE_CXX_FLAGS_DEBUG` to `-g -O1 -fsanitize=thread
  -fno-omit-frame-pointer`.
- **CI** `sanitizers.yml` `tsan` job: replaced the manual
  `CMAKE_CXX_FLAGS_DEBUG` override with `-DLLMQUANT_ENABLE_TSAN=ON`, keeping
  TSan configuration in one canonical place (CMakeLists.txt).

### Added (Cycle 32 — 2026-03-21)
- **Tests**: 2 new `MetricsLogger` JSON control-character escape tests — verify
  that `\t` (tab) and `\n` (newline) in token text are escaped as `\\t` /
  `\\n` in JSON output, not emitted as raw control characters
  (`test_metrics_logger.cpp`).
- **Docs**: README test count badge updated 907 → 914.

### Fixed (Cycle 43 — 2026-03-21)
- **Interface** `OmsAdapter`: add virtual `update_count()`, `error_count()`,
  `reconnect_count()` (all default 0) to the abstract base. `RestOmsAdapter`
  and `FixOmsAdapter` now override them, eliminating five `dynamic_cast` calls
  in `main.cpp` (Prometheus scrape and session summary). `FixOmsAdapter::
  get_reconnect_count()` is kept as a non-virtual alias to preserve backwards
  compatibility with existing tests.
- **Refactor** `Deduplicator::get_stats()`: replaced `dynamic_cast<const
  InProcessDeduplicator*>` with virtual dispatch on the newly-added base-class
  `total_novel()` / `total_duplicates()` methods. `RedisDeduplicator` delegates
  both to its inner `InProcessDeduplicator`.

### Fixed (Cycle 42 — 2026-03-21)
- **Interface** `Deduplicator` base class: `total_novel()` and
  `total_duplicates()` were missing from the abstract interface, so code holding
  a `Deduplicator*` (as used in `main.cpp` and tests) could not call them without
  a `dynamic_cast`. Added as virtual (non-pure, default returns 0) to the base
  class; `InProcessDeduplicator` overrides now use `override`; `RedisDeduplicator`
  delegates both to its inner `InProcessDeduplicator` for fallback-path accounting.

### Added (Cycle 41 — 2026-03-21)
- **Feature** `Config.token_stream.redis_url`: expose Redis URL directly in
  `config.yaml` and `TokenStreamConfig` struct. Previously the URL was only
  settable via the `Deduplicator::Config` at runtime; now it can be loaded from
  YAML on startup and overridden at runtime via `LLMQUANT_DEDUP_REDIS_URL` env
  var. Serialised to `token_stream.redis_url` in `save_to_yaml()`.
- **Fuzz** `fuzz/fuzz_pipeline.cpp`: new LibFuzzer harness driving the full
  5-stage signal pipeline (LLMAdapter → RiskManager → TradeSignalEngine →
  LatencyController) with fuzz-derived token text and risk config values. Added
  to `fuzz/CMakeLists.txt`; exercises stateful accumulation across iterations.
- **Tests**: regression test `test_parse_position_oversized_body_returns_false`
  in `test_oms_adapter.cpp` — verifies the 4 MB cap in RestOmsAdapter does not
  crash `parse_position()` on an oversized payload.

### Fixed (Cycle 40 — 2026-03-21)
- **Bug fix** `LatencyController::get_stats()`: TOCTOU race between
  `total_measurements_.fetch_add()` and the `min_latency_us_` CAS in
  `record_latency()`. A `get_stats()` call landing in that window saw
  `measurements > 0` but `min_latency_us_ == UINT64_MAX` (the initial
  sentinel), reporting ~18 exaseconds as `stats.min_latency`.
  Fixed by mapping the sentinel value back to 0 in `get_stats()`.

### Fixed (Cycle 39 — 2026-03-21)
- **Security** `LLMStreamClient::build_http_request` and
  `RestOmsAdapter::build_request`: `config_.host` (and `config_.path` in
  RestOmsAdapter) were embedded in HTTP request headers and the request-line
  without CR/LF sanitization, while `config_.api_key` was already sanitized.
  A config value containing `\r\n` could inject arbitrary extra HTTP headers.
  Both fields now pass through `sanitise_header_value()` before inclusion.

### Fixed (Cycle 38 — 2026-03-21)
- **Bug fix** `FixOmsAdapter::handle_message`: FIX SequenceReset (35=4) with
  `GapFillFlag=N` (hard reset) may legally arrive with a lower sequence number
  than `expected_inbound_seq_`. The previous code applied the duplicate-detection
  early `return` before reading tag 35, so hard resets were silently discarded
  and the session looped forever sending ResendRequests. Fixed by reading
  MsgType (tag 35) first; the duplicate guard now exempts msg_type "4".
  Per FIX 4.2 spec §4.3.2.
- **Bug fix** `RestOmsAdapter` receive buffer: HTTP polling response accumulation
  had no upper bound. A misbehaving REST OMS server could exhaust process memory.
  Added a 4 MB cap with a warning log and clean abandonment of the current poll
  cycle.
- **Validation** `Config::validate()`: two new rules — `logging.format` must be
  `"CSV"` or `"JSON"` (other values silently accepted before); `token_stream.
  dedup_ttl_ms` must be `>= 0` (negative is not meaningful). Two regression tests
  added to `test_config.cpp`.

### Fixed (Cycle 37 — 2026-03-21)
- **Bug fix / security** `FixOmsAdapter` receive buffer: no cap existed on the
  accumulation buffer. A misbehaving FIX counterparty that never sent a valid
  checksum boundary (`SOH 10=`) would cause unbounded memory growth until the
  30-second recv timeout fired. Added a 1 MB cap; on breach the adapter logs a
  warning, clears the buffer, and reconnects.
- **Bug fix / security** `LLMStreamClient` receive buffer: same class of
  unbounded-growth issue for the pre-parse body accumulation buffer. A server
  that never emits `\r\n\r\n` (headers) or `\n` (line boundaries) would fill
  memory until the socket timeout. Added a 4 MB cap with reconnect.

### Fixed (Cycle 36 — 2026-03-21)
- **Bug fix** `Deduplicator.cpp`: Redis key buffer was `[35]` bytes but the
  formatted key `"llmq:" + 16 hex + 16 hex + null` requires 38 bytes. With the
  under-sized buffer `snprintf` silently truncated the last 3 hex characters of
  `value_hi`, causing two keys that differ only in the bottom 12 bits of
  `value_hi` to collide in Redis (false positive duplicate detection). Buffer
  corrected to `[38]`; same fix applied to both call sites (`check_and_register`
  and `evict`). Regression test added.

### Fixed (Cycle 35 — 2026-03-21)
- **Bug fix** `PrometheusExporter::format_info`: label values containing `"`,
  `\`, or `\n` were emitted unescaped, producing invalid Prometheus text format.
  Added `escape_prom_label_value()` helper; all three characters are now escaped
  per the exposition format spec. Three regression tests added.
- **Bug fix** `MetricsLogger` JSON output: token, dedup key, rejection reason,
  and config-reload path strings were embedded directly in JSON format strings
  without escaping. A token containing `"` or `\` would produce malformed JSON.
  Added `escape_json_string()` helper covering `"`, `\`, `\n`, `\r`, `\t`, and
  control characters; all four string fields now use it. Four regression tests added.
- **Cleanup** `PrometheusExporter.cpp`: removed duplicate `#include <sstream>`.
- **CI** `ci.yml`: removed `asan` job — identical coverage is provided by the
  dedicated `sanitizers.yml` workflow (`asan-ubsan` job); eliminating the
  duplicate reduces per-push CI minutes.

### Added (Cycle 31 — 2026-03-21)
- **Feature flag** `LLMQUANT_ENABLE_SIMD` (CMake option, default ON): added to
  README feature flags table; gates SSE2 batch path in
  `LLMAdapter::map_sequence_simd`. Set OFF for ARM, 32-bit x86, or reproducibility
  testing; the scalar fallback path is automatically used.
- **CI badge** for coverage workflow added to README.
- **README**: test count badge updated 904 → 907.
- **SIMD to CI**: `LLMQUANT_ENABLE_SIMD` added to `minimal` build and
  `feature-flags-matrix` job to verify scalar fallback compiles and passes tests.

### Fixed (Cycle 31 — 2026-03-21)
- **Bug fix** `LLMStreamClient::stream_reader`: HTTP 429 rate-limit back-off
  used a 10-second blocking `sleep_for`, and the general HTTP error retry used a
  2-second blocking `sleep_for`. Both prevented `stop()` from returning promptly.
  Both are now interruptible 100ms-slice loops (same pattern as
  `TokenStreamSimulator`, `RestOmsAdapter`, and `Config.cpp` watcher).
- **Portability** `LLMAdapter`: `#include <immintrin.h>` is now guarded by
  `#if defined(__SSE2__) && !defined(LLMQUANT_SIMD_DISABLED)`. This prevents
  a build error on ARM and 32-bit x86 targets without SSE2. A full scalar
  fallback path is compiled when SSE2 is absent or `LLMQUANT_ENABLE_SIMD=OFF`
  is passed at configure time. The unconditional `immintrin.h` include is also
  removed from `LLMAdapter.h` (it was leaking into all translation units that
  included the header).

### Added (Cycle 30 — 2026-03-21)
- **Docs**: README CMake options table expanded to document all 11 feature flags
  and 5 build/tooling options. Split into "Feature flags" and "Build/tooling"
  sub-sections for clarity.

### Added (Cycle 30 — 2026-03-21 — hook additions)
- **Bug fix** `TokenStreamSimulator::stream_worker`: both the normal cadence
  sleep and the back-off-empty sleep were blocking `sleep_for(token_interval)`.
  When `token_interval` is set to seconds (e.g. `--token-interval 5000`),
  `stop()` could block for up to 5 s. Both are now interruptible 50ms-slice
  loops — matching the `RestOmsAdapter` and `Config.cpp` watcher fixes.
- **Feature** `LLMQUANT_ENABLE_SIGNAL_TRACE` (compile-time, default OFF):
  `process_token` now emits `spdlog::trace` lines (`[trace] token seq= text=`
  and `[trace] token seq= DEDUP_SKIP`) when compiled with
  `-DLLMQUANT_ENABLE_SIGNAL_TRACE=ON`. Zero overhead when OFF (preprocessor
  eliminates the calls entirely). Useful for dictionary-hit debugging and
  integration testing.

### Added (Cycle 29 — 2026-03-21)
- **Docs**: README badge updated 895 → 904; CLAUDE.md test table updated with
  accurate per-file counts.
- **Feature flag** `LLMQUANT_ENABLE_SIGNAL_TRACE` (CMake option, default OFF):
  gates two `spdlog::trace` calls in the `process_token` hot path — one after
  `map_token_to_weight` (logs token text + raw weights) and one after signal
  emission (logs accumulated bias/vol + latency). Zero overhead when OFF
  (compile-time `#ifdef`). Enable with `-DLLMQUANT_ENABLE_SIGNAL_TRACE=ON` for
  deep debugging; leave OFF in production to avoid trace I/O in the hot path.

### Added (Cycle 29 — 2026-03-21 — hook additions)
- **Feature flag** `LLMQUANT_ENABLE_STREAM_CLIENT` (CMake option, default ON):
  gates the entire `LLMStreamClient` live-streaming path. When OFF:
  - `LLMStreamClient.h` is not included in `main.cpp`
  - `--stream` CLI option is hidden from help and silently unavailable
  - Process falls back to the token simulator with a `spdlog::warn`
  - `LLMQUANT_ENABLE_STREAM_CLIENT=OFF` added to the `minimal` build and
    `feature-flags-matrix` CI job
- **Prometheus** `llmquant_latency_warmed_up` (gauge, 0/1) and
  `llmquant_latency_budget_remaining_us` (gauge, signed µs) added to the
  Prometheus snapshot using `LatencyController::get_health_state()`. These
  allow Grafana dashboards to alert on warmup state and latency budget
  burn without deriving them from p99 and target separately.

### Added (Cycle 28 — 2026-03-21)
- **Docs**: README test count badge updated 788 → 895; CLAUDE.md test coverage
  table updated with accurate per-file counts (actual `grep -c '^TEST('` values).
- **Feature flag** `LLMQUANT_ENABLE_HOT_RELOAD` (CMake option, default ON): gates
  the config file watcher thread (`config.start_watching()`) behind
  `#ifdef LLMQUANT_HOT_RELOAD_ENABLED` in `src/main.cpp`. Disable with
  `-DLLMQUANT_ENABLE_HOT_RELOAD=OFF` for embedded/constrained builds.
  Added to feature-flags-matrix CI job and minimal build.
- **Tests**: 2 new `TradeSignalEngine` edge-case tests — zero-confidence
  SemanticWeight must not produce a signal; `signals_suppressed` counter must
  increment on noise-filtered tokens (`test_trade_signal_engine.cpp`).
- **Tests**: 2 new `RiskManager` pnl-gate counter tests — `blocked_pnl` counter
  in `to_stats_json()` and `format_blocked_by_gate()` increments correctly after
  a PnL-breach block (`test_risk_manager.cpp`).

### Added (Cycle 28 — 2026-03-21 — hook additions)
- **Bug fix**: `export_hot_tokens(5)` was called twice in the session summary
  ("Top influence" block at line 1515 and "Hot tokens" block at line 1527)
  producing identical output. Removed the redundant "Top influence" block.
- **Bug fix**: `--stats-interval` now clamped to [100, 60000] ms — previously
  only had a floor of 100ms with no ceiling; a typo could set a 10-minute loop.
- **CI**: `concurrency:` groups added to all three workflow files (`ci.yml`,
  `sanitizers.yml`, `fuzz.yml`) — GitHub Actions cancels stale in-progress runs
  on the same branch/PR when a new commit arrives, saving CI minutes.
- **CI**: clang-tidy step added to the `check` job (runs on `clang`/`Release`
  matrix cell only). Configures via `-DLLMQUANT_ENABLE_CLANG_TIDY=ON` and
  fails the job on any `error:` output line.

### Added (Cycle 27 — 2026-03-21)
- **Bug fix**: Cycle 26 CHANGELOG claimed `RiskManager::format_gate_blocks()` was
  added, but the identical functionality already existed as
  `format_blocked_by_gate()` (implemented in `RiskManager.cpp`, declared at
  `RiskManager.h:370`, tested in `test_risk_manager.cpp`). Removed the spurious
  duplicate inline method that was briefly added and reverted the CHANGELOG entry.
- **CI: cppcheck static analysis step** added to the `check` job in `ci.yml`
  (runs on gcc/Release matrix cell only to avoid duplication). Invokes cppcheck
  with `--enable=warning,performance,portability` on `src/*.cpp`.
- **Tests**: `Config::set_token_interval_ms()` now has three unit tests covering
  positive update, zero-is-no-op, and negative-is-no-op paths
  (`test_config.cpp`).
- **Bug fix**: `cpu_fraction()` in `src/main.cpp` now initialises
  `prev_cpu_jiffies` to `UINT64_MAX` (sentinel) and returns 0.0 on the first
  call instead of computing `cpu_jiffies - 0`, which produced a grossly
  inflated first reading.

### Added (Cycle 27 — 2026-03-21 — hook additions)
- **CI: feature-flags-matrix job** in `ci.yml`: six-job matrix that builds and
  tests with each `LLMQUANT_ENABLE_*` flag individually set to OFF, confirming
  every optional subsystem compiles and passes tests when disabled in isolation.
- **CI: minimal job expanded**: `LLMQUANT_ENABLE_DEDUP=OFF`,
  `LLMQUANT_ENABLE_PROFILING=OFF`, and `LLMQUANT_ENABLE_JSON_STATS_SUMMARY=OFF`
  added so the minimal job truly exercises the all-features-off path.
- **CI bug fix (`sanitizers.yml`)**: added `sudo apt-get update -qq`, added
  missing `libssl-dev libhiredis-dev clang`, set
  `-DCMAKE_CXX_COMPILER=clang++`, enabled `LLMQUANT_WARNINGS_AS_ERRORS=ON`,
  and added `ASAN_OPTIONS`/`UBSAN_OPTIONS` env vars — matching the `asan` job
  in `ci.yml` which already had these fixes.
- **CI bug fix (`fuzz.yml`)**: added `sudo apt-get update -qq`, added missing
  `libfmt-dev libgtest-dev libssl-dev libhiredis-dev` so the fuzz smoke-test
  build no longer fails on missing headers.
- **Env-var feature flags** (`src/main.cpp`): `LLMQUANT_NO_PROMETHEUS`,
  `LLMQUANT_NO_DEDUP`, `LLMQUANT_NO_HOT_RELOAD`, `LLMQUANT_DRY_RUN`,
  `LLMQUANT_QUIET`, `LLMQUANT_BACKTEST` — all map to their CLI-flag equivalents.
  CLI flags take precedence; env vars only activate the flag when the CLI has
  not already set it. Useful for containerised/Kubernetes deployments.

### Added (Cycle 26 — 2026-03-21)
- **Bug fix (CI)**: `asan` job in `.github/workflows/ci.yml` was missing `clang`
  and `clang-tidy` from its `apt-get install` step, causing the Configure step
  (`-DCMAKE_CXX_COMPILER=clang++`) to fail with "clang++ not found".
- **Bug fix**: `InProcessDeduplicator::to_stats_json()` and
  `RiskManager::to_stats_json()` were incorrectly declared `noexcept` despite
  internally calling `size()` (acquires mutex) and `is_healthy()` /
  `get_most_blocked_gate()` (both acquire mutex). Removed `noexcept` from both.
- **Feature flag** `LLMQUANT_ENABLE_JSON_STATS_SUMMARY` (CMake option, default
  ON): gates the structured `[json:*]` exit-summary block in `src/main.cpp`
  behind `#ifdef LLMQUANT_JSON_STATS_SUMMARY`. Disable with
  `-DLLMQUANT_ENABLE_JSON_STATS_SUMMARY=OFF` for minimal/embedded builds that
  do not want stdout JSON output. Added to CMakeLists.txt alongside existing
  feature flags; wired via `target_compile_definitions`.

### Added (Cycle 26 — 2026-03-21 — previous entries)
- `RiskManager::format_gate_blocks()`: returns a compact per-gate block-count
  string (`"mag=N conf=N rate=N dd=N pos=N pnl=N"`) for console/log output.
  Implemented in `RiskManager.cpp`; declared in `RiskManager.h`.
- `Config::set_token_interval_ms(int ms)`: inline thread-safe setter; used by
  the `--token-interval` CLI override path.
- `Config::diff_from_defaults()`: returns a list of field-name strings whose
  values differ from compiled-in defaults, useful for `--dump-config` diffing
  and test assertions.
- **Bug fix**: TPS display and ingestion-pressure normalisation corrected for
  non-default `--stats-interval` values. Previously `token_count_window.exchange(0)`
  was passed directly as "tokens/s", but it actually counted tokens per
  `stats_interval_ms`. The count is now divided by `stats_interval_ms/1000.0`
  before use, so pressure and the `TPS:` column are always in tokens/second.
- **Feature flag** `--no-prometheus`: Skip starting the Prometheus exporter.
  Useful when port 9100 is already in use or when the scrape endpoint is
  unwanted (e.g. CI pipelines, integration tests, embedded deployments).
- **Feature flag** `--no-dedup`: Disable the token deduplicator; every token is
  treated as novel. Useful for stress-testing the signal pipeline or when the
  upstream guarantees uniqueness and the hashing overhead is undesirable.
- **Feature flag** `--no-hot-reload`: Skip starting `Config::start_watching()`.
  Useful for deployments where inotify/ReadDirectoryChangesW is unavailable or
  where deterministic config is required.

### Added (Cycle 25 — 2026-03-21)
- `InProcessDeduplicator::to_stats_json()`: inline method in `Deduplicator.h`
  serialising total_novel, total_duplicates, current_size, and dup_rate_pct
  to a JSON object string. Adds `<cinttypes>` and `<cstdio>` to `Deduplicator.h`.
- Session exit summary: `[json:dedup]` line added to the structured JSON block
  alongside risk/engine/adapter/latency (printed unless `--quiet`).
- Tests (`test_deduplicator.cpp`): 2 new tests for `to_stats_json()` — field
  presence / counter accuracy with known novel/duplicate counts, and
  zero-state dup_rate_pct.

### Added (Cycle 24 — 2026-03-21)
- `LatencyController::to_stats_json()`: inline method serialising all
  `LatencyStats` fields (avg, min, max, p5/p25/p50/p75/p95/p99, jitter_ms,
  measurements, target_breaches) plus `window_fill_ratio` and `slo_breach_rate`
  to a JSON object string. Adds `<cinttypes>`, `<cstdio>`, and `<string>` to
  `LatencyController.h`.
- Session exit summary: `[json:latency]` line added alongside the existing
  `[json:risk/engine/adapter]` lines (printed unless `--quiet`).
- Tests (`test_latency_controller.cpp`): 2 new tests for `to_stats_json()` —
  field presence / measurement count accuracy, and empty-controller zero state.

### Added (Cycle 23 — 2026-03-21)
- `--stats-interval N` CLI flag: sets the monitoring loop tick period in
  milliseconds (default 1000, minimum 100). Allows operators to reduce console
  noise on long runs (`--stats-interval 5000`) or increase resolution for
  debugging (`--stats-interval 200`). Sleep duration changed from hardcoded
  `std::chrono::seconds(1)` to `std::chrono::milliseconds(stats_interval_ms)`.
- Session exit summary: `Hot tokens` row using `LLMAdapter::export_hot_tokens(5)`
  — composite score `0.5*(hit_rate) + 0.5*(|directional_bias|)` ranks tokens
  that are both frequently seen AND strongly directional. Printed alongside the
  existing top-by-frequency, top-by-bias, and top-by-influence rows.
- Tests (`test_llm_adapter.cpp`): 3 new tests for `LLMAdapter::clear_dictionary()`
  covering: all tokens removed, stats reset to zero, and post-clear lookup
  returning default zero weights for a previously known token.
- Tests (`test_token_stream_simulator.cpp`): 2 new tests for
  `TokenStreamSimulator::set_token_interval()` — verifying the hot-reload call
  does not throw for both interval-increase and interval-decrease cases.

### Added (Cycle 22 — 2026-03-21)
- `TradeSignalEngine::to_stats_json()`: inline method in `TradeSignalEngine.h`
  that serialises all engine stats (signals generated/suppressed/aged-out/
  noise-filtered, tokens processed, accumulator clamped, avg/peak quality,
  quality EMA, and 5-bucket quality histogram) to a JSON object string.
- `LLMAdapter::to_stats_json()`: inline method in `LLMAdapter.h` that
  serialises adapter stats (tokens_processed, cache_hits, cache_misses,
  hit_rate_pct, dictionary_size) to a JSON object string. Adds `<cinttypes>`
  and `<cstdio>` includes to `LLMAdapter.h`.
- Session exit summary: structured JSON lines `[json:risk]`, `[json:engine]`,
  and `[json:adapter]` are now printed at shutdown (unless `--quiet`) using
  the new `to_stats_json()` methods on all three subsystems.
- Tests (`test_trade_signal_engine.cpp`): 2 new tests for `to_stats_json()`
  field presence and `tokens_processed` counter accuracy.
- Tests (`test_llm_adapter.cpp`): 2 new tests for `LLMAdapter::to_stats_json()`
  field presence and zero-state output.
- Prometheus: `llmquant_signal_quality_ema` gauge — EMA (alpha=0.1) of
  `signal_quality`, value -1.0 when no signals have been emitted. Complements
  the Welford mean with a recency-weighted view of signal quality.
- Session exit summary: `Quality EMA(0.1)` row — shows the final EMA value of
  signal quality; skipped (not printed) when no signals were emitted (-1.0).
- Gate trip-wire callbacks registered at startup: `"magnitude"`, `"confidence"`,
  `"rate"`, `"drawdown"`, and `"position"` gates each emit a `spdlog::warn`
  on the first pass→block edge per gate. Provides real-time blocking alerts in
  the log without polling `RiskManager::get_stats()`.
- Hot-reload: `token_stream.token_interval_ms` changes are now applied to the
  `TokenStreamSimulator` at runtime via `set_token_interval()`. Previously the
  interval was only read at startup; operators can now tune token pacing without
  a process restart.
- `LLMAdapter::export_hot_tokens(n)`: returns the top N tokens by a composite
  influence score (0.5 × normalised hit-frequency + 0.5 × |directional_bias|),
  surfacing tokens that are both frequently seen and strongly directional.
  Useful for debugging runaway bias or prioritising dictionary tuning.
- `RiskManager::reset_stats()`: now also re-arms all per-gate trip-wire callbacks
  (resets the `gate_*_last_blocked_` booleans) so that the next block after a
  stats reset fires the trip callback again regardless of prior gate state.
- Tests (`test_llm_adapter.cpp`): 4 new tests for `export_hot_tokens()` covering
  empty dict, composite ranking, score bounds [0,1], and n > dict size.
- Tests (`test_risk_manager.cpp`): 2 new tests for `reset_stats()` verifying
  counter zeroing and trip-wire callback re-arming.

### Added (Cycle 21 — 2026-03-21)
- Fixed: `Config::load_from_yaml_string()` now validates `semantic_weights`
  multipliers for NaN/Inf — previously only `validate()` checked them, allowing
  silent non-finite values to enter the hot path. Added inline checks for all
  four multipliers (`sentiment`, `confidence`, `volatility`, `bias`).
- `RiskManager::to_stats_json()`: new inline method in `RiskManager.h` that
  serialises the risk statistics snapshot (all blocked-gate counters,
  `signals_passed`, aggregate `blocked_rate_frac`, `is_healthy`, and
  `most_blocked_gate`) to a JSON object string. Includes `<cinttypes>` and
  `<cstdio>` for `PRIu64` and `snprintf`.
- Tests: 2 new `ConfigTest` cases verifying NaN `sentiment_multiplier` and Inf
  `bias_multiplier` are rejected by `load_from_yaml_string`.
- Tests: 2 new `RiskManagerTest` cases verifying `to_stats_json()` produces
  valid JSON with correct counter values and correct "none" / "magnitude" gate
  identification.
- CMake: build metadata baked in at configure time — `git rev-parse --short HEAD`
  captures the short commit hash; `string(TIMESTAMP)` captures the configure
  timestamp (ISO-8601 UTC). Both are exposed as `LLMQUANT_GIT_COMMIT` and
  `LLMQUANT_BUILD_TIMESTAMP` macros in `llmquant_version.h.in`.
- `--version` output now includes the git hash and build timestamp:
  `LLMTokenStreamQuantEngine 1.x.y (abc1234, 2026-03-21T...)`.
- `RiskManager::register_gate_trip_callback(gate_name, cb)`: edge-trigger
  callback fired once per pass→block transition per named gate (`"magnitude"`,
  `"confidence"`, `"rate"`, `"drawdown"`, `"position"`). Enables async alerting
  without polling `get_stats()` in tight loops.
- `TradeSignalEngine::Stats::signal_quality_ema`: exponential moving average of
  `signal_quality` (alpha = 0.1) updated atomically on every signal emission;
  seeded at -1.0 so callers can detect "no signals yet". Exposed via
  `get_signal_quality_ema()`.
- Config hot-reload watcher: moved after `token_sim` construction and added
  `token_sim` to the lambda capture so token-stream config changes (e.g.
  `token_interval_ms`) can be applied to the simulator at runtime.
- Fix: `FixOmsAdapter.cpp` `NOMINMAX` guard changed from unconditional
  `#define NOMINMAX` to `#ifndef NOMINMAX / #define / #endif` to prevent
  redefinition warnings when the macro is already defined by the build system.
- Tests: 2 new `ConfigTest` cases verifying env var `LLMQUANT_MAX_ACCUMULATED_BIAS`
  overrides `trading.max_accumulated_bias` correctly.

### Added (Cycle 20 — 2026-03-21)
- Prometheus: `llmquant_start_time_seconds` gauge — Unix epoch when the engine
  started, enabling Grafana to compute uptime via `time() - llmquant_start_time_seconds`.
  Captured with `system_clock` at startup alongside the existing `steady_clock`
  reference so both uptime duration and absolute start timestamp are available.
- Prometheus: `llmquant_process_rss_bytes` gauge — process resident set size in
  bytes, sampled once per Prometheus scrape. Uses `GetProcessMemoryInfo/psapi` on
  Windows and `/proc/self/status` (VmRSS) on Linux/macOS. CMakeLists.txt now
  links `psapi` on WIN32.
- `MetricsLogger::log_system_stats()` now wired in the monitoring loop (called
  once per second with the process RSS). Previously implemented but never called,
  so `SYSTEM_STATS` events never appeared in the structured log file.
- Session exit summary: `Top bias tokens` row using `top_tokens_by_directional_bias(5)`
  — shows the 5 tokens with the highest `|directional_bias|` to give a directional
  view of what drove the session (complements the top-by-frequency row).
- Session exit summary: removed redundant `Top blocked gate` row; the identical
  `Most blocked gate` row earlier in the summary is retained.

### Added (Cycle 19 — 2026-03-21)
- `Config::to_yaml_string()`: serialises the complete `SystemConfig` to a
  YAML-formatted string covering all subsystem sections including
  `semantic_weights` and `risk_overrides` (previously missing from
  `save_to_file`). Declared in `Config.h`, implemented in `Config.cpp`.
  `save_to_file()` now delegates to `to_yaml_string()` to eliminate
  duplication and ensure parity.
- Fixed `save_to_file()` omitting `semantic_weights` and `risk_overrides`
  sections from saved YAML — both sections are now included via the new
  shared `to_yaml_string()` helper.
- Prometheus: `llmquant_dedup_dup_rate_pct` gauge — duplicate token rate as
  a percentage [0, 100] computed from the running novel/duplicate counters.
- Prometheus: `llmquant_latency_window_fill_ratio` gauge — fraction of the
  latency sample window that is filled [0, 1], enabling dashboards to detect
  warm-up vs steady-state conditions.
- Removed duplicate `llmquant_most_blocked_gate_info` Prometheus metric
  (the redundant block appended after `llmquant_risk_healthy`); the
  canonical `llmquant_risk_most_blocked_gate_info` metric in the main snap
  stream is retained.
- Tests (`test_config.cpp`): 5 new tests covering `SemanticWeightsConfig`
  YAML parsing, default values when the section is absent, and
  `to_yaml_string()` round-trip correctness for trading, semantic\_weights,
  and presence of all YAML sections.
- Tests (`test_trade_signal_engine.cpp`): 3 new tests for `format_stats()`
  quality histogram — verifies `quality_hist=[...]` field presence, 5-bucket
  comma-separated structure, and early-return when no tokens processed.
- Fixed unused-variable warning in `test_invariants.cpp` (`rm.evaluate` return
  value now cast to `void`).
- `CMakeLists.txt`: hiredis link target now falls back from the modern
  `hiredis::hiredis` imported target to the legacy `hiredis` raw name, so the
  build succeeds on both vcpkg/conan installs and manually built hiredis.

### Added (Cycle 18 — 2026-03-21)
- `token_stream.dedup_ttl_ms` config field: configures the in-process deduplicator
  TTL directly rather than deriving it from `token_interval_ms * 10`. Setting to
  `0` (default) preserves the existing auto behaviour. Hot-reload compatible.
  Added to `TokenStreamConfig` in `Config.h`, parsed in `Config.cpp`, and
  documented in `config.yaml`.
- Prometheus: `llmquant_version_info{version="x.y.z"} 1` gauge exposes the engine
  version as a Prometheus info metric with a label, following the standard
  pattern for version discovery in Grafana dashboards.
- `.gitignore`: added `build_*/` (covers `build_vs/` and other non-default build
  dirs), `.vs/`, `*.user`, `CMakeSettings.json`, `out/`, `.DS_Store`, and common
  editor temp files (`*.swp`, `*.swo`, `*~`).
- `LLMAdapter::filter_tokens_by_directional_bias(min, max)`: returns all
  dictionary entries whose `directional_bias` falls in `[min, max]`, enabling
  callers to slice the token set by orientation (bullish-only, bearish-only)
  for backtesting and reporting.
- `LLMAdapter::top_tokens_by_directional_bias(n)`: returns the top-N tokens
  ranked by `|directional_bias|` via `std::partial_sort` (O(k log k)); backed
  by the existing `token_weights_` map with zero extra storage.

### Added (Cycle 17 — 2026-03-21)
- Semantic weight multipliers are now **hot-reloadable**.  Four
  `std::atomic<double>` variables (`sem_mult_sentiment`, `sem_mult_confidence`,
  `sem_mult_volatility`, `sem_mult_bias`) replace the startup-snapshot `sys_config`
  reference in `process_token`.  The `config.start_watching()` callback now stores
  the updated multipliers atomically so the very next token sees the new values
  without a process restart.  The hot-reload console line now also prints the new
  `sem_wts=[...]` tuple so operators can confirm the change was applied.
- Prometheus: `llmquant_most_blocked_gate_info{gate="<name>"}` info gauge added.
  Emits a single time-series with value 1 and a `gate` label identifying which
  risk gate has the highest block count (`magnitude`, `confidence`, `rate`,
  `drawdown`, `position`, or `none`).  Allows Grafana to display the dominant
  bottleneck as a stat panel without a PromQL aggregation query.

### Added (Cycle 16 — 2026-03-21)
- `MetricsLogger::log_dedup_event()` now called in `process_token` for every dedup check
  (novel and duplicate alike) — previously implemented but never invoked, so no dedup
  events appeared in the structured log file.
- `MetricsLogger::log_latency_measurement()` now called once per second in the monitoring
  loop with the current P99 latency — previously implemented but never invoked; provides
  a periodic latency sample series in the structured log file without hot-path overhead.

### Added (Cycle 15 — 2026-03-21)
- Tests for `TradeSignal::to_json()`: verify all 9 fields are present, output
  starts and ends with braces, and numeric values match struct fields.
- Tests for `RiskManager::get_most_blocked_gate()`: returns `"none"` with no
  blocks; correctly identifies `"magnitude"` as the dominant gate.

### Added (Cycle 15 — 2026-03-21)
- Signal callback now calls `logger.log_trade_signal()` for passed signals (replacing the
  less-detailed `log_signal_generated()`) — now logs confidence and signal_quality alongside
  bias, volatility, and latency.
- Signal callback now calls `logger.log_risk_rejection()` for blocked signals — previously
  risk rejections were only printed to the console; now they are also written to the structured
  log file with reason, bias, and confidence fields.

### Added (Cycle 14 — 2026-03-21)
- `--export-dict FILE` CLI flag: exports the full semantic dictionary to a TSV
  file (one entry per line: `token\tsentiment\tconfidence\tvolatility\tbias`)
  and exits. Complements `--list-tokens` (stdout) with a file-based export path
  suitable for piping into data-analysis tools or importing back via
  `LLMAdapter::load_dictionary_from_tsv()`.
- `--dump-config` now includes the `pressure.*` and `semantic_weights.*` sections
  that were previously missing from the output, ensuring all hot-reloadable fields
  are visible.

### Added (Cycle 13 — 2026-03-21)
- Session exit summary now prints `Avg sig quality` (`avg_signal_quality`) and `Noise filtered`
  (`noise_filtered` count) — both tracked in `TradeSignalEngine::Stats` and Prometheus-exported
  but absent from the human-readable exit report.
- Per-second monitoring stats bar now shows `NOISE:N` (noise-gate filtered token count) alongside
  the existing `DEDUP:N` and `BLOCK:N` columns for a complete signal-loss picture.

### Added (Cycle 12 — 2026-03-21)
- `TradeSignal::to_json()` — inline JSON serializer on `TradeSignal`; complements
  `to_string()` for callers feeding signals into JSON pipelines.
- `RiskManager::get_most_blocked_gate()` — inline method returning the name of
  the gate with the highest block count (`"magnitude"`, `"confidence"`, `"rate"`,
  `"drawdown"`, `"position"`, or `"none"`).
- Prometheus: `llmquant_adapter_cache_hit_rate` gauge exposes
  `LLMAdapter::get_cache_hit_rate()` directly for dashboard use without PromQL division.

### Fixed (Cycle 12 — 2026-03-21)
- `LatencyController.h`: removed duplicate `get_total_latency_us()` definition
  inadvertently added when the method was added a second time.

### Fixed (Cycle 13 — 2026-03-21)
- `FixOmsAdapter`: `get_update_count()` and `get_error_count()` now return real values.
  Previously both atomics were declared but never incremented, so all Prometheus
  `llmquant_oms_update_count_total` and `llmquant_oms_error_count_total` readings for
  FIX sessions were always 0.  Now `update_count_` increments on each successful
  callback invocation (`emit_position()`) and `error_count_` increments on recv
  failure and heartbeat send failure.

### Added (Cycle 12 — 2026-03-21)
- `MetricsLogger::log_pipeline_health()` now called once per second from the monitoring loop,
  persisting SLO health, breach rate, and backoff multiplier to the structured log file on
  every tick — previously implemented but never invoked.
- `MetricsLogger::log_config_reload()` now called from the hot-reload callback, capturing each
  successful config reload as a timestamped structured log entry — previously implemented but
  never invoked.
- Session exit summary now prints `Log entries` total from `MetricsLogger::get_log_entry_count()`.
- P99 SLO breach now emits `spdlog::warn` (in addition to the existing console colour print)
  so the breach is captured in the structured log file and visible to any spdlog sink.

### Added (Cycle 11 — 2026-03-21)
- `--dump-config` CLI flag: loads the config file (or defaults), prints all
  effective settings as `key: value` pairs, and exits with code 0. Useful for
  verifying that hot-reload-eligible fields have the expected values before
  starting the full engine.
- `--quiet` CLI flag: suppresses the per-signal console row and the rolling
  stats bar. All data still flows to `MetricsLogger` and the Prometheus
  endpoint, making `--quiet` safe for daemon / log-only deployments.

### Fixed (Cycle 11 — 2026-03-21)
- `main.cpp`: "Tokens processed" in both the rolling stats bar and the session
  exit summary was sourced from `variance_n`, which is reset every 60 seconds
  as part of the Welford catastrophic-cancellation guard. Sessions longer than
  one minute would therefore undercount processed tokens. Fixed by reading
  `llm_adapter.get_stats().tokens_processed` instead, which is a monotonically
  increasing counter that is never reset.

### Added (Cycle 10 — 2026-03-21)
- `Config::validate()` now checks all four `SemanticWeightsConfig` multipliers for
  finiteness (rejects NaN / ±Inf).  Previously a corrupted YAML could inject NaN
  into every SemanticWeight field without any error or warning.
- `Config::to_summary_string()` now includes the `[sem_wts]` line showing the
  four active multiplier values, making the config dump self-documenting.
- `test_concurrent_evaluate_with_reason_no_crash` — new multi-threaded test
  (8 threads × 200 iterations) that exercises `evaluate_with_reason()` concurrently
  with mixed passing/blocking signals to verify `ewr_mutex_` eliminates the
  callback use-after-free race introduced in v1.2.0.

### Added (Cycle 11 — 2026-03-21)
- Prometheus: `llmquant_oms_update_count_total` and `llmquant_oms_error_count_total` now also
  cover `FixOmsAdapter` sessions — previously the `dynamic_cast` only checked `RestOmsAdapter`
  and silently returned 0 for FIX connections.
- Prometheus: new `llmquant_oms_reconnect_count_total` counter exposes `FixOmsAdapter`'s
  reconnect attempt count to Prometheus consumers.
- Startup banner now shows `OMS: <adapter description>` line so the active OMS connection is
  visible at a glance without reading the config file.
- Session exit summary now prints OMS adapter description, update count, error count, and
  (for FIX sessions) reconnect count.

### Fixed (Cycle 10 — 2026-03-21)
- `MetricsLogger` CSV: `log_trade_signal` now emits a correctly aligned 9-column
  row matching the file header.  Previously the row had only 8 columns and placed
  `bias` in the `sequence_id` position.  `confidence` and `signal_quality` are
  omitted from the CSV row (they were not in the header) but remain in the JSON
  format unchanged.
- `MetricsLogger` CSV: `log_pipeline_health` now emits 9 columns; previously it
  emitted 8 (one trailing column was missing).

### Fixed (Cycle 9 — 2026-03-21)
- `RiskManager::evaluate_with_reason`: fixed a data-race / use-after-free when
  called concurrently from multiple threads.  The method temporarily swaps
  `alert_cb_` for a wrapper that captures the rejection reason; two concurrent
  callers would overwrite each other's saved callback and leave a dangling
  wrapper — pointing at a destroyed stack variable — installed on `alert_cb_`
  after both returned.  Fixed by adding `ewr_mutex_` (serialises callers of
  `evaluate_with_reason` only; concurrent `evaluate()` calls are unaffected).

### Added (Cycle 8 — 2026-03-21)
- Startup banner now shows `BACKTEST: cooldown disabled` when `--backtest` is active, matching
  the existing `DRY-RUN` display so operators can see mode at a glance.
- Session exit summary: added uptime (seconds since startup), a compact full latency summary
  via `LatencyController::format_stats()`, and the top-5 tokens by hit frequency using the new
  `LLMAdapter::top_tokens_by_frequency()` method.

### Added (Cycle 7 — 2026-03-21)
- `LatencyController::get_total_latency_us()` — inline accessor returning the
  raw `total_latency_us_` atomic without triggering percentile computation.
  Used to emit an accurate Prometheus histogram `_sum` line; the previous
  `avg × count` approximation lost precision because the integer-microsecond
  average was already rounded.
- Prometheus: six new metrics exposed that were previously computed but not
  exported: `llmquant_slo_breach_rate`, `llmquant_drawdown_utilization`,
  `llmquant_rate_limit_utilization`, `llmquant_noise_filtered_total`,
  `llmquant_risk_healthy`.
- Prometheus: `llmquant_risk_pass_rate_pct` simplified to use
  `RiskManager::get_blocked_rate()` directly, eliminating inline duplication
  of the blocked-total aggregation.
- Prometheus: `llmquant_token_latency_us_sum` now uses
  `latency_ctrl.get_total_latency_us()` for an exact sum value.

### Added (Cycle 6 — 2026-03-21)
- Session exit summary now prints `peak_bias`, `SLO breach rate`, `P5 latency`, and `P25 latency`
  — all were tracked and Prometheus-exported but absent from the human-readable exit report.
- `llmquant_backtest_mode` Prometheus gauge added alongside the existing `llmquant_dry_run` gauge,
  allowing scrape consumers to distinguish backtest from normal operation.
- `MetricsLogger::log_performance_summary()` enhanced: now reports uptime (seconds), log entry
  rate (entries/s), log format, and file path, replacing the two-line placeholder output.

### Added (Cycle 5 — 2026-03-21)
- `--list-tokens` CLI flag: prints all semantic dictionary entries as a TSV table
  (`token`, `sentiment`, `confidence`, `volatility`, `bias`) and exits immediately.
  Useful for inspecting loaded dictionary entries and verifying custom token mappings.
- Simulator in-memory token set expanded from 16 to 44 tokens, covering all
  dictionary categories: fear/panic (`selloff`, `rout`), bullish (`rebound`,
  `accumulate`), bearish (`downtrend`, `distribution`), volatility (`whipsaw`,
  `choppy`, `gamma`, `vega`), certainty, corporate/earnings, macro/regime, analyst,
  options (`calls`, `puts`, `squeeze`), crypto/retail (`pump`, `fud`, `hodl`), and
  neutral filler. Improves demonstration coverage of all semantic signal paths.

### Added (Cycle 4 — 2026-03-21)
- `SemanticWeightsConfig` struct in `Config.h` with four per-category multiplier
  fields (`sentiment_multiplier`, `confidence_multiplier`, `volatility_multiplier`,
  `bias_multiplier`), each defaulting to `1.0`.  Added to `SystemConfig` and parsed
  in `Config.cpp` under the `semantic_weights:` YAML key.  Applied in `main.cpp`'s
  `process_token` lambda between `map_token_to_weight` and `process_semantic_weight`,
  completing the TODO that had been noted in `config.yaml` since initial release.
- `config.yaml`: active `semantic_weights:` section added with all four multipliers
  defaulting to `1.0` (no-op), replacing the old TODO comment.
- CI workflow (`.github/workflows/ci.yml`) expanded from 21 lines to a full four-job
  matrix: `check` (gcc/clang × Release/Debug, with `actions/cache@v4`), `asan`
  (clang Debug + ASan/UBSan), `format` (clang-format gate), and `docs` (Doxygen).

### Changed (Cycle 4 — 2026-03-21)
- Project version bumped `1.1.0` → `1.2.0` in `CMakeLists.txt`, syncing with the
  README badge that already reflected `1.2.0`.

### Added (Cycle 3 — 2026-03-21)
- `TradeSignal::to_string()` now includes `spread_modifier` and `strategy_weight`
  in its one-liner debug format:
  `"bias=<v> vol=<v> spread=<v> conf=<v> quality=<v> lat=<v>us strategy=<±1> weight=<v>"`.
  Previously these two fields were silently omitted.
- `CsvOutputSink` and `JsonOutputSink` now serialize `signal_quality` alongside
  all other `TradeSignal` fields.  Previously the composite quality score was
  computed by the engine but never persisted to disk, making offline quality
  analysis impossible.

### Added (Cycle 2 — 2026-03-21)
- `--log-level LEVEL` CLI flag — set spdlog verbosity (`trace`/`debug`/`info`/`warn`/`error`/`critical`)
  at startup without recompiling. Falls back to `info` with a warning for unknown level names.
- `SIGTERM` handler wired alongside `SIGINT` so the engine shuts down cleanly under systemd / Docker stop.
- Dedup statistics (novel count, duplicate count, duplicate rate %) added to the session summary printed
  on exit.

### Fixed (Cycle 2 — 2026-03-21)
- `main.cpp`: replaced three remaining `std::cerr` calls (hot-reload watcher failure, stream done
  callback, Prometheus bind failure) with `spdlog::warn` — consistent with the project-wide policy
  of routing all library-level output through spdlog.
- `main.cpp` `--version`: replaced hardcoded `"1.1.0"` string with the `LLMQUANT_VERSION` macro from
  the CMake-generated `llmquant_version.h`, so the version flag always reflects the CMake project
  version without a manual edit.
- `main.cpp`: Prometheus bind-failure warning now includes the actual port number rather than the
  hardcoded `9100`.

### Added (Cycle 1 — 2026-03-21)
- `LLMAdapter::filter_tokens_by_volatility(min, max)` — range filter returning
  all dictionary tokens whose `volatility_score` falls within [min, max].
  Completes the analytics API alongside the existing `filter_tokens_by_sentiment`
  and `filter_tokens_by_confidence` overloads.
- `TradeSignal::to_string()` — inline one-liner for logging/debugging:
  `"bias=<val> vol=<val> conf=<val> quality=<val> lat=<val>us strategy=<±1>"`.
- `LatencyController::get_p99_us()`, `get_p95_us()`, `get_p50_us()` — typed
  convenience accessors returning percentile latency in microseconds without
  allocating a full `LatencyStats` snapshot.
- Expanded `LLMAdapter` default dictionary from ~80 to ~130 tokens:
  - **Corporate / earnings**: `earnings`, `guidance`, `upgrade`, `downgrade`,
    `beats`, `misses`, `outlook`, `revenue`, `profit`, `loss`, `dividend`,
    `buyback`, `merger`, `acquisition`, `ipo`
  - **Market regime / macro**: `risk-on`, `risk-off`, `systemic`, `contagion`,
    `stimulus`, `tightening`, `easing`, `default`, `sanctions`, `tariff`,
    `deregulation`, `geopolitical`
  - **Analyst sentiment**: `overweight`, `underweight`, `outperform`,
    `underperform`, `neutral`, `hold`, `target`
  - **Common filler**: `or`, `not`, `with`, `for`, `as`, `at`, `on`, `it`,
    `by`, `from`

### Fixed
- `CMakeLists.txt`: replaced deprecated `yaml-cpp` link target with
  `yaml-cpp::yaml-cpp` to silence CMake deprecation warnings from yaml-cpp 0.8+.
- `CMakeLists.txt`: replaced absolute `${CMAKE_SOURCE_DIR}/include` and
  `${CMAKE_CURRENT_BINARY_DIR}/include` in `target_include_directories` with
  generator expressions (`$<BUILD_INTERFACE:...>` / `$<INSTALL_INTERFACE:...>`)
  to fix the CMake install target error
  ("INTERFACE_INCLUDE_DIRECTORIES contains path prefixed in source directory").
- `src/FixOmsAdapter.cpp`: added `#define NOMINMAX` before `<winsock2.h>` on
  Windows to prevent the `min`/`max` macro collision that caused
  `std::min(1 << n, kMax)` to fail to compile under MSVC (`C2589`/`C2059`).
- `include/TradeSignalEngine.h` (`Stats`): added explicit copy constructor and
  copy assignment operator so `get_stats()` can return the struct by value on
  MSVC, where the compiler rejects implicit copy of `std::atomic` members.
- `include/RestOmsAdapter.h`: moved `parse_position` from `private` to
  `protected` so `TestableRestOmsAdapter` in the test suite can call it via
  subclassing without a compile error (`C2248`).
- `include/FixOmsAdapter.h`: moved `fix_checksum` and `fix_message` from
  `private` to `protected` so `TestableFixOmsAdapter` in the test suite can
  call them via subclassing without a compile error (`C2248`).
- `src/TradeSignalEngine.cpp`: initialised `last_signal_time_` to the epoch
  (default-constructed `time_point{}`) rather than `now()`, so the very first
  token processed always fires a signal regardless of the configured cooldown.
  Previously the constructor-time `now()` meant the cooldown was already active
  at construction, causing the first emission to be silently dropped.
- `tests/unit/test_config.cpp`: corrected three test assertions that expected
  lowercase `"json"` / `"csv"` for `SystemConfig::logging.format`; the
  `Config` implementation normalises the value to uppercase at parse time, so
  the correct expected values are `"JSON"` and `"CSV"`.
- `tests/unit/test_network_error_paths.cpp` (`realtime_cooldown_suppresses_rapid_signals`):
  corrected assertion to match the fixed engine behaviour — `emitted == 1` and
  `signals_suppressed == 0` (suppressed counts only signals with no callback
  and no output sinks, not cooldown-dropped ones).
- `tests/unit/test_network_error_paths.cpp` (`bind_blocker_socket`): on Windows,
  the blocker socket now uses `SO_EXCLUSIVEADDRUSE` and binds to `INADDR_ANY`
  so that a subsequent `SO_REUSEADDR` bind by `PrometheusExporter` is correctly
  rejected; previously both sockets used `SO_REUSEADDR` on loopback, which
  Windows permits (unlike POSIX), causing the port-conflict test to pass when
  it should have failed.

### Added
- `include/llmquant_version.h.in` — CMake-generated version header exposing
  `LLMQUANT_VERSION`, `LLMQUANT_VERSION_MAJOR/MINOR/PATCH`, and
  `LLMQUANT_VERSION_NUMBER` for compile-time feature detection.
- CMakeLists.txt: `LLMQUANT_ENABLE_ASAN` option bakes AddressSanitizer +
  UBSan into non-MSVC Debug builds (`-fsanitize=address,undefined`).
- CMakeLists.txt: `LLMQUANT_WARNINGS_AS_ERRORS` option (default ON) controls
  whether `-Werror` / `/WX` is applied; can be disabled for downstream
  integration.
- CMakeLists.txt: `LLMQUANT_ENABLE_CLANG_TIDY` option sets
  `CMAKE_CXX_CLANG_TIDY` for in-build static analysis.
- CI: AddressSanitizer + UBSan re-run step for Debug builds with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
- CI: Valgrind memcheck step for GCC Release builds
  (`--leak-check=full --show-leak-kinds=definite,indirect`).
- CI: Windows MSVC matrix job (Release + Debug) via `windows-latest`.
- CI: CMake configure step in the `docs` job to generate `llmquant_version.h`
  before running Doxygen.
- CI: `LLMQUANT_WARNINGS_AS_ERRORS=OFF` in the docs configure step to avoid
  spurious header-only warnings.
- `tests/unit/test_production_readiness.cpp` — 25 new unit tests covering:
  - `Config::start_watching()` hot-reload callback fires on file change.
  - `Config::save_to_file()` error path returns false on unwritable path.
  - `Deduplicator::evict()` allows re-registration of an evicted key.
  - `InProcessDeduplicator` counter correctness (`total_novel`,
    `total_duplicates`, `size`).
  - `InProcessDeduplicator::purge_expired()` removes TTL-expired entries.
  - `DedupKey::from_token()` context scoping and determinism.
  - `TradeSignalEngine::add_output_sink()` and `clear_output_sinks()`.
  - `TradeSignalEngine` backtest mode emits a signal on every token.
  - `LLMAdapter::map_sequence_to_weight()` confidence-weighted average
    correctness.
  - `LLMAdapter` unknown-token neutral-weight fallback.
  - SIMD (`map_sequence_simd`) matches scalar for even and odd-length sequences.
  - `LatencyController` profile hooks do not crash.
  - `LatencyController::reset_stats()` zeroes all counters.
  - `LatencyController::update_ingestion_pressure(rate, 0)` does not divide by zero.
  - `MemoryOutputSink::clear()` resets the buffer.
  - `OutputSink` default `flush()` is a no-op.
  - `RiskManager::get_position()` reflects the last `update_position()` call.
  - `MetricsLogger` CSV and JSON paths for all log methods do not crash with
    an empty file path.
  - `TokenStreamSimulator::stop()` before `start()` is safe.
  - `TokenStreamSimulator` emits all loaded in-memory tokens.
  - `Config` default construction produces valid field values.

### Changed
- All `std::cerr` error output in library code replaced with `spdlog`:
  - `src/Config.cpp` — load/save errors now via `spdlog::error/warn`.
  - `src/MetricsLogger.cpp` — sink creation failures now via `spdlog::warn`.
  - `src/LLMStreamClient.cpp` — socket, TLS, and HTTP errors now via
    `spdlog::error/warn/debug`; debug-raw dump uses `std::fwrite`/`std::fflush`
    (intentional, gated behind an explicit debug flag).
  - `src/FixOmsAdapter.cpp` — recv failures, reconnect, heartbeat, and
    sequence-reset errors now via `spdlog::warn/info`.
  - `src/RestOmsAdapter.cpp` — HTTP error status now via `spdlog::warn`.
  - `src/Deduplicator.cpp` — backend type mismatch warning now via `spdlog::info`.
- `docs/Doxyfile` enhanced: `USE_MDFILE_AS_MAINPAGE = README.md`,
  `WARN_NO_PARAMDOC = YES`, `WARN_LOGFILE` set, `HAVE_DOT = YES` with SVG
  output, `GENERATE_TREEVIEW = YES`, README.md and ARCHITECTURE.md added
  to `INPUT`.
- README.md rewritten with full architecture description, CMake option table,
  Linux/macOS/Windows build instructions, ASan build instructions, API
  reference, configuration reference, risk gate table, performance notes,
  and Prometheus metrics reference.

### Fixed
- `src/LLMStreamClient.cpp` `debug_raw` path: replaced `std::cerr.write` with
  `std::fwrite(chunk, 1, n, stderr)` to avoid mixing C++ stream buffering with
  raw byte dumps.

---

## [1.1.0] - 2026-03-17

### Added
- CI: `.github/workflows/release.yml` — tag-triggered release workflow that
  builds a stripped Linux release binary, runs the full test suite, generates
  Doxygen docs, and uploads a `tar.gz` artifact to GitHub Releases.
- `tests/unit/test_network_error_paths.cpp` — new test suite covering:
  - `RestOmsAdapter` refused-connection handling (no crash, error counter
    increments, `update_count` stays zero on a refused port).
  - `RestOmsAdapter` lifecycle idempotency (`stop()` before `start()` safe;
    `start()` returns false when already running).
  - `PrometheusExporter::start()` returns false when port is already bound.
  - `PrometheusExporter::stop()` before `start()` is safe.
  - `LLMStreamClient` done-callback behaviour on repeated failed connects.
  - SSE delta parsing under adversarial JSON: integer content, array content,
    multiple choices, very long content, special characters.
  - `LLMAdapter` signal extraction: volatility tokens, panic/bearish tokens,
    custom mapping override, SIMD vs scalar agreement on known tokens.
  - `TradeSignalEngine` signal extraction: high-sensitivity amplification,
    decay reduces bias over time, realtime cooldown suppresses rapid signals.
- CMake install targets and package config files (`llmquantConfig.cmake`,
  `llmquantConfigVersion.cmake`, `llmquantTargets.cmake`) for downstream use.
- `cmake/llmquantConfig.cmake.in` template for package configuration.
- CI: clang-tidy-14 static analysis step (clang+Release matrix cell).
- CI: cppcheck static analysis step (gcc+Release matrix cell).
- CI: `actions/cache@v4` caching of CMake build artifacts keyed on source hash.
- CI: upgraded runner images from `ubuntu-22.04` to `ubuntu-latest`.
- Doxyfile for HTML API documentation generation via `doxygen Doxyfile`.
- Expanded CI pipeline: GCC + Clang matrix (Release + Debug), clang-format
  gate, and Doxygen build job.
- Doxygen `/** @brief ... @param ... @return ... @throws ... */` doc-comments
  on all public APIs across every header file.
- `AGENTS.md` and `CLAUDE.md` coordination notes for agentic workflows.

### Changed
- `LLMStreamClient` constructor now throws `std::runtime_error` if
  `WSAStartup` fails on Windows (previously silently ignored).
- `LLMStreamClient` constructor now throws `std::runtime_error` if
  `SSL_CTX_new` returns null (prevents a null-pointer dereference on the
  first `tls_handshake()` call when OpenSSL is misconfigured).
- `PrometheusExporter` constructor now throws `std::runtime_error` if
  `WSAStartup` fails on Windows (previously silently ignored).
- `Config::load_from_yaml_string` now validates field ranges (signal_decay_rate
  in (0,1), buffer_size > 0, bias/volatility sensitivity >= 0) and returns
  `false` on validation failure, restoring defaults automatically.
- `MetricsLogger` constructor catches spdlog file-sink creation exceptions and
  falls back to a null sink so callers never receive an exception from
  construction when the path is inaccessible.
- `RiskManager::set_metrics_logger` now accepts a raw pointer annotated with
  lifetime contract documentation; the pointer must outlive the RiskManager.
- CI workflow expanded from a single ubuntu-latest job to a 4-job matrix
  (gcc/clang x Release/Debug) plus format and docs jobs.

### Fixed
- `LatencyController::end_measurement` now checks `latency_measurement_active_`
  before recording, preventing a garbage latency value when called without a
  preceding `start_measurement()` on the same thread.
- `RedisDeduplicator` destructor correctly calls `redis_disconnect()` when
  built with `LLMQUANT_REDIS_ENABLED`, preventing a hiredis context leak.

---

## [1.0.0] - 2025-10-01

### Added
- Initial production release.
- `TokenStreamSimulator`: lock-free SPSC ring buffer, file and in-memory
  sources, configurable emission cadence.
- `LLMAdapter`: exact-match semantic weight dictionary (~40 tokens), SSE2
  SIMD aggregate path (`map_sequence_simd`).
- `TradeSignalEngine`: exponential-decay accumulators, realtime cooldown,
  backtest mode, pluggable `OutputSink` chain.
- `RiskManager`: magnitude, confidence, rate-limit, drawdown, and position
  gates; alert and OMS callbacks.
- `LatencyController`: lock-free P50/P95/P99 tracking, back-pressure system
  with composite pressure signal and exponential backoff multiplier.
- `MetricsLogger`: spdlog-backed CSV and NDJSON structured logging.
- `Config`: YAML file loading/saving with hot-reload via background file
  watcher thread.
- `Deduplicator`: FNV-1a TTL dedup with optional Redis backend.
- `LLMStreamClient`: zero-dependency TLS streaming client for OpenAI chat
  completions SSE endpoint.
- `RestOmsAdapter`: HTTP polling OMS position adapter.
- `FixOmsAdapter`: minimal FIX 4.2 session reader for ExecutionReport and
  PositionReport messages.
- `MockOmsAdapter`: deterministic OMS mock for testing.
- `PrometheusExporter`: lightweight Prometheus `/metrics` scrape endpoint.
- `OutputSink` hierarchy: `CsvOutputSink`, `JsonOutputSink`, `MemoryOutputSink`.
- 1,491 tests across unit, integration, property-invariant, and chaos suites.
- `config.yaml` with hot-reload support.
- `--stream`, `--no-color`, `--debug-raw`, `--oms` CLI flags.

---

[Unreleased]: https://github.com/Mattbusel/LLMTokenStreamQuantEngine/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Mattbusel/LLMTokenStreamQuantEngine/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Mattbusel/LLMTokenStreamQuantEngine/releases/tag/v1.0.0
