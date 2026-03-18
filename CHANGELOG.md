# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
