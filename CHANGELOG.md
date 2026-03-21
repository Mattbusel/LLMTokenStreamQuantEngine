# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
