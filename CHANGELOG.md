# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Nothing yet.

### Changed
- Nothing yet.

### Fixed
- Nothing yet.

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
