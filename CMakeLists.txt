cmake_minimum_required(VERSION 3.20)
project(LLMTokenStreamQuantEngine VERSION 1.2.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LLMQUANT_ENABLE_ASAN     "Build with AddressSanitizer + UBSan (Debug only)"  OFF)
option(LLMQUANT_ENABLE_TSAN     "Build with ThreadSanitizer (Debug only; mutually exclusive with ASAN)" OFF)
option(LLMQUANT_WARNINGS_AS_ERRORS "Treat all compiler warnings as errors"        ON)
option(LLMQUANT_ENABLE_CLANG_TIDY  "Run clang-tidy on library sources"            OFF)
option(LLMQUANT_ENABLE_FUZZING  "Build libFuzzer targets (requires clang + -fsanitize=fuzzer)" OFF)
option(LLMQUANT_ENABLE_COVERAGE "Build with gcov/lcov coverage instrumentation"   OFF)

# Feature flags — enable/disable optional subsystems for minimal or embedded builds.
option(LLMQUANT_ENABLE_PROMETHEUS  "Build the Prometheus /metrics HTTP scrape endpoint" ON)
option(LLMQUANT_ENABLE_FIX_OMS    "Build the FIX 4.2 OMS adapter (FixOmsAdapter)"      ON)
option(LLMQUANT_ENABLE_REST_OMS   "Build the REST OMS polling adapter (RestOmsAdapter)" ON)
option(LLMQUANT_ENABLE_PROFILING  "Compile latency profiling / percentile tracking"     ON)
option(LLMQUANT_ENABLE_DEDUP      "Build the token deduplication subsystem"             ON)
option(LLMQUANT_ENABLE_JSON_STATS_SUMMARY "Emit structured JSON stats lines to stdout on session exit" ON)
option(LLMQUANT_ENABLE_TLS        "Enable TLS support via OpenSSL in LLMStreamClient (auto-disabled if OpenSSL not found)" ON)
option(LLMQUANT_ENABLE_REDIS      "Enable Redis-backed deduplication via hiredis (auto-disabled if hiredis not found)"     ON)
option(LLMQUANT_ENABLE_HOT_RELOAD    "Enable config file hot-reload watcher thread"                                      ON)
option(LLMQUANT_ENABLE_STREAM_CLIENT "Build the LLMStreamClient live-streaming adapter (disabling also removes OpenSSL dep when TLS=OFF)" ON)
option(LLMQUANT_ENABLE_SIGNAL_TRACE  "Emit per-token spdlog::trace lines in the hot path (high verbosity, disable for production)" OFF)
option(LLMQUANT_ENABLE_SIMD          "Use SSE2 intrinsics in LLMAdapter::map_sequence_simd (disable for ARM/non-SSE2 targets or reproducibility testing)" ON)
option(LLMQUANT_ENABLE_MOCK_OMS      "Build MockOmsAdapter (deterministic test double); disable for production/minimal builds" ON)
option(LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY "Build SentimentTrajectoryAnalyzer (online linear-regression trend detector for token sentiment streams)" ON)
option(LLMQUANT_ENABLE_AUDIT_LOG         "Build the async NDJSON signal audit log (SignalAuditLog)" ON)
option(LLMQUANT_ENABLE_CIRCUIT_BREAKER   "Build the pipeline circuit-breaker (auto-pause on sustained high block rate)" ON)
option(LLMQUANT_ENABLE_BACKTEST          "Build BacktestRunner: full-pipeline token-sequence replay with PnL statistics" ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    set(_warn_flags /W4)
    if(LLMQUANT_WARNINGS_AS_ERRORS)
        list(APPEND _warn_flags /WX)
    endif()
    add_compile_options(${_warn_flags})
    # Suppress MSVC C4996 "deprecated" warnings for standard CRT functions
    # (fopen, strerror, etc.) that are valid C++ standard library calls.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Do NOT set CMAKE_CXX_FLAGS_RELEASE / CMAKE_CXX_FLAGS_DEBUG here — that
    # would overwrite CMake's default /MD CRT linkage flag, causing a CRT
    # mismatch with pre-built spdlog and yaml-cpp packages.  Per-target
    # optimisation flags are added via target_compile_options below instead.
else()
    set(_warn_flags -Wall -Wextra)
    if(LLMQUANT_WARNINGS_AS_ERRORS)
        list(APPEND _warn_flags -Werror)
    endif()
    add_compile_options(${_warn_flags})
    # -ffast-math intentionally removed: it permits FP reordering and
    # NaN/Inf mis-handling that are unsafe in a quant engine where
    # atomic accumulation results must be numerically reproducible.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    if(LLMQUANT_ENABLE_ASAN)
        set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")
    elseif(LLMQUANT_ENABLE_TSAN)
        # TSan uses -O1 so inlined atomics remain visible to the race detector.
        set(CMAKE_CXX_FLAGS_DEBUG   "-g -O1 -fsanitize=thread -fno-omit-frame-pointer")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")
    endif()
endif()

# ---------------------------------------------------------------------------
# clang-tidy integration (optional)
# ---------------------------------------------------------------------------
if(LLMQUANT_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-14 clang-tidy-15 clang-tidy-16)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY
            "${CLANG_TIDY_EXE}"
            "--extra-arg=-std=c++20"
            "--warnings-as-errors="
        )
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    else()
        message(WARNING "LLMQUANT_ENABLE_CLANG_TIDY requested but clang-tidy not found")
    endif()
endif()

# ---------------------------------------------------------------------------
# Build metadata: git commit hash and configure-time timestamp
# ---------------------------------------------------------------------------
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE LLMQUANT_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result
)
if(NOT _git_result EQUAL 0 OR NOT LLMQUANT_GIT_COMMIT)
    set(LLMQUANT_GIT_COMMIT "unknown")
endif()
string(TIMESTAMP LLMQUANT_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)
message(STATUS "Build metadata: git=${LLMQUANT_GIT_COMMIT}  ts=${LLMQUANT_BUILD_TIMESTAMP}")

# ---------------------------------------------------------------------------
# Generated version header
# ---------------------------------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/llmquant_version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/llmquant_version.h"
    @ONLY
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Dependency versions are pinned in vcpkg.json at the project root.
# To use vcpkg: cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
find_package(spdlog        REQUIRED)
# fmt required by spdlog; explicit linkage prevents version mismatch on some distros.
find_package(fmt CONFIG QUIET)
find_package(yaml-cpp      REQUIRED)
find_package(GTest         REQUIRED)
find_package(Threads       REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(OpenSSL QUIET)
if(OpenSSL_FOUND OR OPENSSL_FOUND)
    message(STATUS "OpenSSL found — TLS enabled in LLMStreamClient")
endif()

find_package(hiredis QUIET)
if(hiredis_FOUND OR HIREDIS_FOUND)
    message(STATUS "hiredis found — Redis deduplication enabled")
endif()

# ---------------------------------------------------------------------------
# Engine library — compiled once, linked by executable and tests.
# ---------------------------------------------------------------------------
add_library(llmquant STATIC
    src/TokenStreamSimulator.cpp
    src/TradeSignalEngine.cpp
    src/LatencyController.cpp
    src/LLMAdapter.cpp
    src/MetricsLogger.cpp
    src/Config.cpp
    src/RiskManager.cpp
    src/Deduplicator.cpp
    src/LLMStreamClient.cpp
    src/RestOmsAdapter.cpp
    src/FixOmsAdapter.cpp
    src/PrometheusExporter.cpp
)

# SentimentTrajectoryAnalyzer: optional online trend-detection module.
if(LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY)
    target_sources(llmquant PRIVATE src/SentimentTrajectoryAnalyzer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
    message(STATUS "Feature: SentimentTrajectoryAnalyzer ENABLED (online linear-regression trend detector)")
else()
    message(STATUS "Feature: SentimentTrajectoryAnalyzer DISABLED (LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY=OFF)")
endif()

# SignalAuditLog: async NDJSON audit log — feature-flagged for minimal builds.
if(LLMQUANT_ENABLE_AUDIT_LOG)
    target_sources(llmquant PRIVATE src/SignalAuditLog.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_AUDIT_LOG_ENABLED)
    message(STATUS "Feature: SignalAuditLog ENABLED (async NDJSON signal audit log)")
else()
    message(STATUS "Feature: SignalAuditLog DISABLED (LLMQUANT_ENABLE_AUDIT_LOG=OFF)")
endif()

# PipelineCircuitBreaker: auto-pause when sustained block rate exceeds threshold.
if(LLMQUANT_ENABLE_CIRCUIT_BREAKER)
    target_sources(llmquant PRIVATE src/PipelineCircuitBreaker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CIRCUIT_BREAKER_ENABLED)
    message(STATUS "Feature: PipelineCircuitBreaker ENABLED (auto-pause on high block rate)")
else()
    message(STATUS "Feature: PipelineCircuitBreaker DISABLED (LLMQUANT_ENABLE_CIRCUIT_BREAKER=OFF)")
endif()

# BacktestRunner: full-pipeline token-sequence replay with PnL statistics.
if(LLMQUANT_ENABLE_BACKTEST)
    target_sources(llmquant PRIVATE src/BacktestRunner.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_BACKTEST_ENABLED)
    message(STATUS "Feature: BacktestRunner ENABLED (full-pipeline replay with PnL stats)")
else()
    message(STATUS "Feature: BacktestRunner DISABLED (LLMQUANT_ENABLE_BACKTEST=OFF)")
endif()

# MockOmsAdapter is only compiled when LLMQUANT_ENABLE_MOCK_OMS=ON (default).
# Disable for production builds to strip the test double from the binary.
if(LLMQUANT_ENABLE_MOCK_OMS)
    target_sources(llmquant PRIVATE src/MockOmsAdapter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_MOCK_OMS_ENABLED)
    message(STATUS "Feature: MockOmsAdapter ENABLED (test double included)")
else()
    message(STATUS "Feature: MockOmsAdapter DISABLED (LLMQUANT_ENABLE_MOCK_OMS=OFF)")
endif()

target_include_directories(llmquant
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# MSVC optimisation flags: appended to the target rather than overriding
# CMAKE_CXX_FLAGS_RELEASE so that CMake's default /MD CRT linkage is preserved.
if(MSVC)
    target_compile_options(llmquant PRIVATE
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
    )
    # Prevent windows.h from defining min/max macros that conflict with std::max/std::min.
    target_compile_definitions(llmquant PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(llmquant
    PUBLIC
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        nlohmann_json::nlohmann_json
        Threads::Threads
        # fmt::fmt explicitly linked to prevent version mismatch with spdlog on some distros.
        $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>
)

if(WIN32)
    target_link_libraries(llmquant PUBLIC ws2_32 psapi)
endif()

if(LLMQUANT_ENABLE_TLS AND (OpenSSL_FOUND OR OPENSSL_FOUND))
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TLS_ENABLED)
    target_link_libraries(llmquant PUBLIC OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "Feature: TLS (OpenSSL) ENABLED")
elseif(LLMQUANT_ENABLE_TLS)
    message(STATUS "Feature: TLS requested but OpenSSL not found — TLS disabled")
else()
    message(STATUS "Feature: TLS disabled (LLMQUANT_ENABLE_TLS=OFF)")
endif()

if(LLMQUANT_ENABLE_REDIS AND (hiredis_FOUND OR HIREDIS_FOUND))
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REDIS_ENABLED)
    if(TARGET hiredis::hiredis)
        target_link_libraries(llmquant PUBLIC hiredis::hiredis)
    else()
        target_link_libraries(llmquant PUBLIC hiredis)
    endif()
    message(STATUS "Feature: Redis (hiredis) ENABLED")
elseif(LLMQUANT_ENABLE_REDIS)
    message(STATUS "Feature: Redis requested but hiredis not found — Redis disabled")
else()
    message(STATUS "Feature: Redis disabled (LLMQUANT_ENABLE_REDIS=OFF)")
endif()

# Wire feature flags as compile-time preprocessor symbols.
if(LLMQUANT_ENABLE_PROMETHEUS)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PROMETHEUS_ENABLED)
    message(STATUS "Feature: Prometheus scrape endpoint ENABLED")
else()
    message(STATUS "Feature: Prometheus scrape endpoint DISABLED (LLMQUANT_ENABLE_PROMETHEUS=OFF)")
endif()
if(LLMQUANT_ENABLE_FIX_OMS)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_FIX_OMS_ENABLED)
    message(STATUS "Feature: FIX 4.2 OMS adapter ENABLED")
else()
    message(STATUS "Feature: FIX 4.2 OMS adapter DISABLED (LLMQUANT_ENABLE_FIX_OMS=OFF)")
endif()
if(LLMQUANT_ENABLE_REST_OMS)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REST_OMS_ENABLED)
    message(STATUS "Feature: REST OMS adapter ENABLED")
else()
    message(STATUS "Feature: REST OMS adapter DISABLED (LLMQUANT_ENABLE_REST_OMS=OFF)")
endif()
if(LLMQUANT_ENABLE_PROFILING)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PROFILING_ENABLED)
    message(STATUS "Feature: Latency profiling ENABLED")
else()
    message(STATUS "Feature: Latency profiling DISABLED (LLMQUANT_ENABLE_PROFILING=OFF)")
endif()
if(LLMQUANT_ENABLE_DEDUP)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_DEDUP_ENABLED)
    message(STATUS "Feature: Token deduplication ENABLED")
else()
    message(STATUS "Feature: Token deduplication DISABLED (LLMQUANT_ENABLE_DEDUP=OFF)")
endif()
if(LLMQUANT_ENABLE_JSON_STATS_SUMMARY)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_JSON_STATS_SUMMARY)
    message(STATUS "Feature: JSON stats summary on exit ENABLED")
else()
    message(STATUS "Feature: JSON stats summary on exit DISABLED (LLMQUANT_ENABLE_JSON_STATS_SUMMARY=OFF)")
endif()
if(LLMQUANT_ENABLE_HOT_RELOAD)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_HOT_RELOAD_ENABLED)
    message(STATUS "Feature: Config hot-reload watcher ENABLED")
else()
    message(STATUS "Feature: Config hot-reload watcher DISABLED (LLMQUANT_ENABLE_HOT_RELOAD=OFF)")
endif()
if(LLMQUANT_ENABLE_STREAM_CLIENT)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_STREAM_CLIENT_ENABLED)
    message(STATUS "Feature: LLMStreamClient live-streaming adapter ENABLED")
else()
    message(STATUS "Feature: LLMStreamClient live-streaming adapter DISABLED (LLMQUANT_ENABLE_STREAM_CLIENT=OFF)")
endif()
if(LLMQUANT_ENABLE_SIGNAL_TRACE)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_TRACE_ENABLED)
    message(STATUS "Feature: Per-token signal trace logging ENABLED (high verbosity)")
else()
    message(STATUS "Feature: Per-token signal trace logging DISABLED (LLMQUANT_ENABLE_SIGNAL_TRACE=OFF, default)")
endif()
if(NOT LLMQUANT_ENABLE_SIMD)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIMD_DISABLED)
    message(STATUS "Feature: SSE2 SIMD batch processing DISABLED (LLMQUANT_ENABLE_SIMD=OFF)")
else()
    message(STATUS "Feature: SSE2 SIMD batch processing ENABLED (hardware-conditional)")
endif()

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(LLMTokenStreamQuantEngine src/main.cpp)
target_link_libraries(LLMTokenStreamQuantEngine PRIVATE llmquant)

# ---------------------------------------------------------------------------
# Coverage instrumentation (GCC/Clang, requires LLMQUANT_ENABLE_COVERAGE=ON)
# ---------------------------------------------------------------------------
if(LLMQUANT_ENABLE_COVERAGE)
    if(MSVC)
        message(WARNING "Coverage instrumentation is not supported with MSVC; ignoring LLMQUANT_ENABLE_COVERAGE")
    else()
        target_compile_options(llmquant PUBLIC --coverage -O0 -g)
        target_link_options(llmquant PUBLIC --coverage)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# LibFuzzer targets (LLMQUANT_ENABLE_FUZZING=ON, clang only)
# fuzz/ subdirectory contains root-level harnesses (fuzz_deduplicator,
# fuzz_token_stream); tests/fuzz/ contains the original pipeline harnesses.
# ---------------------------------------------------------------------------
if(LLMQUANT_ENABLE_FUZZING)
    # fuzz/ builds fuzz_deduplicator and fuzz_token_stream.
    add_subdirectory(fuzz)

    # tests/fuzz/fuzz_pipeline.cpp is the only target not covered by fuzz/.
    # Do NOT add fuzz_deduplicator here — it is already defined above and CMake
    # forbids two add_executable() calls with the same target name.
    add_executable(fuzz_pipeline tests/fuzz/fuzz_pipeline.cpp)
    target_include_directories(fuzz_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(fuzz_pipeline PRIVATE llmquant)
    target_compile_options(fuzz_pipeline PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(   fuzz_pipeline PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

# ---------------------------------------------------------------------------
# Install targets and package config (library consumers)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS llmquant LLMTokenStreamQuantEngine
    EXPORT llmquantTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.h.in" EXCLUDE
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/llmquant_version.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT llmquantTargets
    FILE llmquantTargets.cmake
    NAMESPACE llmquant::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llmquant
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/llmquantConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/llmquantConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llmquant
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/llmquantConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/llmquantConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/llmquantConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llmquant
)

# ---------------------------------------------------------------------------
# Docs (optional — requires Doxygen)
# ---------------------------------------------------------------------------
# To generate HTML API documentation:
#   cmake --build build --target docs
# Output is written to docs/api/html/index.html
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()
