cmake_minimum_required(VERSION 3.20)
project(LLMTokenStreamQuantEngine VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Werror)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(spdlog   REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest    REQUIRED)
find_package(Threads  REQUIRED)

find_package(OpenSSL QUIET)
if(OpenSSL_FOUND OR OPENSSL_FOUND)
    message(STATUS "OpenSSL found — TLS enabled in LLMStreamClient")
    add_compile_definitions(LLMQUANT_TLS_ENABLED)
endif()

find_package(hiredis QUIET)
if(hiredis_FOUND OR HIREDIS_FOUND)
    message(STATUS "hiredis found — Redis deduplication enabled")
    add_compile_definitions(LLMQUANT_REDIS_ENABLED)
endif()

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
include_directories(include)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
set(ENGINE_SOURCES
    src/main.cpp
    src/TokenStreamSimulator.cpp
    src/TradeSignalEngine.cpp
    src/LatencyController.cpp
    src/LLMAdapter.cpp
    src/MetricsLogger.cpp
    src/Config.cpp
    src/RiskManager.cpp
    src/Deduplicator.cpp
    src/LLMStreamClient.cpp
)

add_executable(LLMTokenStreamQuantEngine ${ENGINE_SOURCES})

target_link_libraries(LLMTokenStreamQuantEngine
    spdlog::spdlog
    yaml-cpp
    Threads::Threads
)

if(WIN32)
    target_link_libraries(LLMTokenStreamQuantEngine ws2_32)
endif()

if(OpenSSL_FOUND OR OPENSSL_FOUND)
    target_link_libraries(LLMTokenStreamQuantEngine OpenSSL::SSL OpenSSL::Crypto)
endif()

if(hiredis_FOUND OR HIREDIS_FOUND)
    target_link_libraries(LLMTokenStreamQuantEngine hiredis)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
