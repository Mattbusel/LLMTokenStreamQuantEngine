cmake_minimum_required(VERSION 3.20)
project(LLMTokenStreamQuantEngine VERSION 1.1.0 LANGUAGES CXX)

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
    # Do NOT set CMAKE_CXX_FLAGS_RELEASE / CMAKE_CXX_FLAGS_DEBUG here — that
    # would overwrite CMake's default /MD CRT linkage flag, causing a CRT
    # mismatch with pre-built spdlog and yaml-cpp packages.  Per-target
    # optimisation flags are added via target_compile_options below instead.
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
