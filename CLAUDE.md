# CLAUDE.md — LLMTokenStreamQuantEngine

## What This System Does

Low-latency C++ engine ingesting LLM token streams, mapping semantic weight to trade signals at <10 μs p99 latency. Serves as execution layer beneath ROT (Reddit Options Trader).

## Relation to ROT

This engine consumes structured sentiment from Reddit/LLM pipelines and converts to real-time trade signal adjustments (bias shift, volatility, spread). ROT drives token input; this engine drives signal output.

## Architectural Philosophy

- **Predictive not reactive**: pressure system pushes upstream before queues fill
- **Token-level not batch**: every token modifies signal state, no buffering delay
- **Zero-copy where possible**: memory-mapped streaming planned for high-frequency paths
- **Atomic accumulation**: `std::atomic<double>` for lock-free bias/vol accumulation

## Build Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Test Commands

```bash
cd build && ctest           # run all tests
./tests                     # run test binary directly
./tests --gtest_filter="*integration*"   # integration only
./tests --gtest_filter="*Performance*"  # benchmarks only
```

## Debug Build (with ASan/UBSan)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
./LLMTokenStreamQuantEngine
```

## Lint

```bash
clang-tidy src/*.cpp -- -std=c++20 -Iinclude
clang-format --dry-run src/*.cpp include/*.h
```

## Coding Conventions

- C++20, no raw new/delete (smart pointers only)
- All public methods doc-commented in headers
- namespace llmquant throughout
- -Wall -Wextra -Werror: zero warnings policy
- Atomic types for all hot-path shared state
- Mutex only for sample windows and buffer management (not hot path)

## CMake Details

- CMake 3.20+ required
- Dependencies: spdlog (logging), yaml-cpp (config), GoogleTest (testing), Threads
- Release: -O3 -march=native -ffast-math
- Debug: -g -O0 -fsanitize=address,undefined
- Tests live in tests/unit/ and tests/integration/; tests/CMakeLists.txt is
  included via add_subdirectory(tests) from the root CMakeLists.txt

## Module Status

| Module | Headers | Implementation | Tests | Done |
|--------|---------|----------------|-------|------|
| Config | ✓ | ✓ hot-reload | ✓ | ✓ |
| LLMAdapter | ✓ | ✓ SIMD batch | ✓ | ✓ |
| LatencyController | ✓ | ✓ pressure system | ✓ | ✓ |
| MetricsLogger | ✓ | ✓ | ✓ | ✓ |
| TokenStreamSimulator | ✓ | ✓ lock-free ring | ✓ | ✓ |
| TradeSignalEngine | ✓ | ✓ full fields | ✓ | ✓ |
| OutputSink | ✓ | ✓ (header-only) | ✓ | ✓ |

## What Still Needs Building
- Real-time LLM API streaming integration (direct API streaming, planned)
- Distributed deduplication layer (cross-node, planned)
- Production risk management hooks (position limits, drawdown guards)

## Non-Obvious Design Decisions

- CMakeLists.txt was originally in include/ — moved to project root for standard
  CMake layout; the old include/CMakeLists.txt remains for reference only.
- TradeSignal includes both `timestamp_ns` (uint64, for serialisation) and
  `timestamp` (chrono, for latency calculation in main).
- LLMAdapter uses an unordered_map as a token cache — hot path since all
  production tokens hit it after the first lookup.
- LatencyController keeps a rolling window of samples for percentile
  calculation; erasing from vector front is O(n) but the window is small
  (default 1000 samples) so the amortised cost is acceptable.
- OutputSink concrete classes (CsvOutputSink, JsonOutputSink, MemoryOutputSink)
  are implemented inline in OutputSink.h because they are thin wrappers over
  std::ofstream / std::vector with no separate compilation unit needed.
