cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# LibFuzzer targets — only built when ENABLE_FUZZING (alias for the root
# option LLMQUANT_ENABLE_FUZZING) or LLMQUANT_ENABLE_FUZZING is set to ON.
#
# Requires clang++ and -fsanitize=fuzzer support.
#
# Usage:
#   cmake -S .. -B build-fuzz \
#     -DLLMQUANT_ENABLE_FUZZING=ON \
#     -DCMAKE_CXX_COMPILER=clang++ \
#     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
#   cmake --build build-fuzz --target fuzz_deduplicator fuzz_token_stream
# ---------------------------------------------------------------------------

if(LLMQUANT_ENABLE_FUZZING OR ENABLE_FUZZING)

    foreach(_fuzz_target fuzz_pipeline fuzz_deduplicator fuzz_token_stream fuzz_config fuzz_fix_parser)
        add_executable(${_fuzz_target} ${_fuzz_target}.cpp)

        target_include_directories(${_fuzz_target}
            PRIVATE ${CMAKE_SOURCE_DIR}/include
        )

        target_link_libraries(${_fuzz_target}
            PRIVATE llmquant
        )

        # -fsanitize=fuzzer provides LLVMFuzzerRunDriver (replaces main()) and
        # the coverage-guided mutation engine. Address + UBSan detect memory
        # errors at runtime.
        target_compile_options(${_fuzz_target}
            PRIVATE -fsanitize=fuzzer,address,undefined
        )

        target_link_options(${_fuzz_target}
            PRIVATE -fsanitize=fuzzer,address,undefined
        )
    endforeach()

endif()
