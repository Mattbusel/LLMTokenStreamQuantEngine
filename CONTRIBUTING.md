# Contributing to LLMTokenStreamQuantEngine

Thank you for your interest in contributing. This document covers everything you need to build, test, and extend the engine.

---

## Build Requirements

| Requirement | Minimum Version | Notes |
|---|---|---|
| **CMake** | 3.20 | `cmake --version` |
| **Ninja** | 1.11 | Recommended generator (faster incremental builds) |
| **MSVC** | 19.44 (VS 2022) | Windows primary target; GCC/Clang also supported |
| **vcpkg** | latest | Package manager; set `VCPKG_ROOT` environment variable |
| **OpenSSL** | 3.x | TLS support for live OpenAI streaming (`vcpkg install openssl`) |
| **spdlog** | 1.12+ | Structured logging (`vcpkg install spdlog`) |
| **yaml-cpp** | 0.8+ | Config file parsing (`vcpkg install yaml-cpp`) |
| **GTest** | 1.14+ | Unit and integration tests (`vcpkg install gtest`) |
| **nlohmann-json** | 3.11+ | SSE JSON parsing (`vcpkg install nlohmann-json`) |
| **hiredis** | optional | Redis deduplication backend (`vcpkg install hiredis`) |

---

## Build Instructions

### Configure

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

On Windows (PowerShell):

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

To enable TLS (required for live OpenAI streaming), OpenSSL must be found by CMake. The build system detects it automatically via `find_package(OpenSSL QUIET)` and defines `LLMQUANT_TLS_ENABLED` when found.

### Build

```bash
cmake --build build --config Release --parallel
```

### Debug Build (with ASan + UBSan on GCC/Clang)

```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-debug --parallel
```

---

## Running Tests

```bash
ctest --test-dir build -V
```

Or to run only specific test suites:

```bash
ctest --test-dir build -V -R "RiskManager"
```

The test executable is also runnable directly:

```bash
# Windows
./build/Release/tests.exe

# Linux/macOS
./build/tests
```

All 1,491 tests must pass before any pull request is merged. Zero failures, zero new warnings.

---

## Code Style

- **Formatter**: `clang-format` with the `.clang-format` file at the repo root (if absent, use Google style as a baseline).
- **No exceptions in hot path**: All functions on the token-processing critical path (`process_semantic_weight`, `map_token_to_weight`, `evaluate`, the read loop in `LLMStreamClient`) must not throw. Use return values (`bool`, `std::optional`, or result structs) for error signalling.
- **Thread safety**: Document thread-safety contracts in Doxygen class comments. Every class comment must state what methods are safe to call concurrently.
- **No raw owning pointers**: Use `std::unique_ptr` or `std::shared_ptr` for heap ownership. Raw pointers are acceptable only for non-owning observer references.
- **Atomics for stats**: All observable counters must be `std::atomic` so that monitoring-thread reads are safe.
- **Header guards**: Use `#pragma once`. No `#ifndef`/`#define` guards.
- **Namespace**: All production code lives in `namespace llmquant`. Tests may use anonymous or test-scoped namespaces.

---

## Adding a New Token Signal

Follow these steps to add a new semantic token category (e.g. a new directional word "capitulate"):

### Step 1 — Define the SemanticWeight

Open `src/LLMAdapter.cpp` and find `initialize_default_mappings()`. Add a new entry:

```cpp
add_token_mapping("capitulate", SemanticWeight{
    .sentiment_score  = -0.7,   // bearish
    .confidence_score =  0.8,
    .volatility_score =  0.5,
    .directional_bias = -0.6,
});
```

Choose values in [-1.0, 1.0] for sentiment, volatility, and directional_bias; [0.0, 1.0] for confidence.

### Step 2 — Add a Unit Test

Open `tests/test_llm_adapter.cpp` (or the appropriate test file) and add a test case:

```cpp
TEST(LLMAdapterTest, CapitulateMapsToNegativeBias) {
    LLMAdapter adapter;
    auto w = adapter.map_token_to_weight("capitulate");
    EXPECT_LT(w.directional_bias, 0.0);
    EXPECT_GT(w.confidence_score, 0.5);
}
```

### Step 3 — Verify in Simulator Mode

Run the engine without an API key:

```bash
./build/Release/LLMTokenStreamQuantEngine --no-color
```

Modify `main.cpp` temporarily to inject "capitulate" into the in-memory token list and confirm the signal pipeline produces the expected BIAS/VOL output.

### Step 4 — Update the Token-to-Signal Mapping Table in README.md

Add the new token to the appropriate row in the "Token-to-Signal Mapping" table in `README.md`.

### Step 5 — Run All Tests

```bash
cmake --build build --config Release && ctest --test-dir build -V
```

Zero failures required.

---

## Pull Request Checklist

- [ ] `ctest --test-dir build -V` passes with zero failures
- [ ] No new compiler warnings under `-Wall -Wextra -Werror` (or `/W4 /WX` on MSVC)
- [ ] New public functions have Doxygen `/** @brief ... */` comments
- [ ] Thread-safety contract documented for any new class
- [ ] `README.md` updated if a user-visible feature was added
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
