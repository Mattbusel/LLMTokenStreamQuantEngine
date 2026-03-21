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
option(LLMQUANT_WARNINGS_AS_ERRORS "Treat all compiler warnings as errors"        ON)
option(LLMQUANT_ENABLE_CLANG_TIDY  "Run clang-tidy on library sources"            OFF)
option(LLMQUANT_ENABLE_FUZZING  "Build libFuzzer targets (requires clang + -fsanitize=fuzzer)" OFF)
option(LLMQUANT_ENABLE_COVERAGE "Build with gcov/lcov coverage instrumentation"   OFF)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    set(_warn_flags /W4)
    if(LLMQUANT_WARNINGS_AS_ERRORS)
        list(APPEND _warn_flags /WX)
    endif()
    add_compile_options(${_warn_flags})
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
    src/MockOmsAdapter.cpp
    src/PrometheusExporter.cpp
)

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
    target_link_libraries(llmquant PUBLIC ws2_32)
endif()

if(OpenSSL_FOUND OR OPENSSL_FOUND)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TLS_ENABLED)
    target_link_libraries(llmquant PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

if(hiredis_FOUND OR HIREDIS_FOUND)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REDIS_ENABLED)
    if(TARGET hiredis::hiredis)
        target_link_libraries(llmquant PUBLIC hiredis::hiredis)
    else()
        target_link_libraries(llmquant PUBLIC hiredis)
    endif()
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
