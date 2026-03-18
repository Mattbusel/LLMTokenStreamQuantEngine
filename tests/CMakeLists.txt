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
    unit/test_invariants.cpp
    unit/test_oms_adapter.cpp
    unit/test_fix_oms_adapter.cpp
    unit/test_prometheus_exporter.cpp
    unit/test_edge_cases.cpp
    unit/test_network_error_paths.cpp
    unit/test_production_readiness.cpp
    integration/test_pipeline.cpp
    integration/test_chaos.cpp
    integration/test_oms_pipeline.cpp
    integration/test_full_pipeline.cpp
    integration/test_pipeline_integration.cpp
    performance/bench_hot_path.cpp
)

target_link_libraries(tests PRIVATE GTest::gtest_main llmquant)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Labelled test targets for selective CI execution.
add_test(NAME unit_tests
    COMMAND tests
    --gtest_filter="*Config*:*LLMAdapter*:*Latency*:*Metrics*:*TokenStream*:*TradeSignal*:*OutputSink*:*Risk*:*Dedup*:*LLMStream*:*Invariants*:*OmsAdapter*:*FixOmsAdapter*:*Prometheus*:*EdgeCase*:*NetworkError*:*ProductionReadiness*"
)
set_tests_properties(unit_tests PROPERTIES LABELS "unit")

add_test(NAME integration_tests
    COMMAND tests --gtest_filter="*pipeline*:*Pipeline*:*chaos*:*Chaos*:*OmsPipeline*:*FullPipeline5Stage*:*PipelineIntegration*"
)
set_tests_properties(integration_tests PROPERTIES LABELS "integration")

add_test(NAME performance_tests
    COMMAND tests --gtest_filter="*Performance*:*Bench*:*bench*"
)
set_tests_properties(performance_tests PROPERTIES LABELS "performance")

add_test(NAME all_tests COMMAND tests)
set_tests_properties(all_tests PROPERTIES LABELS "all")
