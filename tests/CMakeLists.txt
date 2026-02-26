add_executable(tests
    unit/test_config.cpp
    unit/test_llm_adapter.cpp
    unit/test_latency_controller.cpp
    unit/test_metrics_logger.cpp
    unit/test_token_stream_simulator.cpp
    unit/test_trade_signal_engine.cpp
    unit/test_output_sink.cpp
    unit/test_risk_manager.cpp
    unit/test_deduplicator.cpp
    unit/test_llm_stream_client.cpp
    integration/test_pipeline.cpp
    performance/bench_hot_path.cpp
    ${CMAKE_SOURCE_DIR}/src/TokenStreamSimulator.cpp
    ${CMAKE_SOURCE_DIR}/src/TradeSignalEngine.cpp
    ${CMAKE_SOURCE_DIR}/src/LatencyController.cpp
    ${CMAKE_SOURCE_DIR}/src/LLMAdapter.cpp
    ${CMAKE_SOURCE_DIR}/src/MetricsLogger.cpp
    ${CMAKE_SOURCE_DIR}/src/Config.cpp
    ${CMAKE_SOURCE_DIR}/src/RiskManager.cpp
    ${CMAKE_SOURCE_DIR}/src/Deduplicator.cpp
    ${CMAKE_SOURCE_DIR}/src/LLMStreamClient.cpp
)

target_link_libraries(tests
    GTest::gtest_main
    spdlog::spdlog
    yaml-cpp
    Threads::Threads
)

if(WIN32)
    target_link_libraries(tests ws2_32)
endif()

target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME unit_tests COMMAND tests)
