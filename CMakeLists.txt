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
if(MSVC)
    add_compile_options(/W4 /WX)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG   "/Od /Zi")
else()
    add_compile_options(-Wall -Wextra -Werror)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(spdlog        REQUIRED)
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

target_include_directories(llmquant PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(llmquant
    PUBLIC
        spdlog::spdlog
        yaml-cpp
        nlohmann_json::nlohmann_json
        Threads::Threads
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
    target_link_libraries(llmquant PUBLIC hiredis)
endif()

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(LLMTokenStreamQuantEngine src/main.cpp)
target_link_libraries(LLMTokenStreamQuantEngine PRIVATE llmquant)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
