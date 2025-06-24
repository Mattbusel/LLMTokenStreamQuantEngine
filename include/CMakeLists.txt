
cmake_minimum_required(VERSION 3.20)
project(LLMTokenStreamQuantEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

include_directories(include)

add_executable(LLMTokenStreamQuantEngine
    src/main.cpp
    src/TokenStreamSimulator.cpp
    src/TradeSignalEngine.cpp
    src/LatencyController.cpp
    src/LLMAdapter.cpp
    src/MetricsLogger.cpp
    src/Config.cpp
)

target_link_libraries(LLMTokenStreamQuantEngine
    spdlog::spdlog yaml-cpp Threads::Threads
)

enable_testing()
add_executable(tests
    tests/test_token_stream.cpp
    tests/test_trade_signals.cpp
    tests/test_latency.cpp
    src/TokenStreamSimulator.cpp
    src/TradeSignalEngine.cpp
    src/LatencyController.cpp
    src/LLMAdapter.cpp
    src/MetricsLogger.cpp
    src/Config.cpp
)
target_link_libraries(tests GTest::gtest_main spdlog::spdlog yaml-cpp Threads::Threads)
