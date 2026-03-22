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
    unit/test_sentiment_trajectory.cpp
    unit/test_signal_audit_log.cpp
    unit/test_circuit_breaker.cpp
    unit/test_backtest_runner.cpp
    unit/test_kelly_sizer.cpp
    unit/test_health_server.cpp
    unit/test_stale_token_detector.cpp
    unit/test_adaptive_cooldown.cpp
    unit/test_signal_blend_layer.cpp
    unit/test_regime_detector.cpp
    unit/test_token_replay_recorder.cpp
    unit/test_parameter_sweep.cpp
    unit/test_alert_dispatcher.cpp
    unit/test_position_tracker.cpp
    unit/test_token_entropy_meter.cpp
    unit/test_latency_budget_enforcer.cpp
    unit/test_token_entropy_monitor.cpp
    unit/test_trading_hours_guard.cpp
    unit/test_signal_correlation_tracker.cpp
    unit/test_warmup_sequencer.cpp
    unit/test_sentiment_momentum_filter.cpp
    unit/test_signal_decay_envelope.cpp
    unit/test_token_cost_budget.cpp
    unit/test_lead_lag_detector.cpp
    unit/test_gradient_optimiser.cpp
    unit/test_drawdown_protector.cpp
    unit/test_multi_timeframe_aggregator.cpp
    integration/test_pipeline.cpp
    integration/test_chaos.cpp
    integration/test_oms_pipeline.cpp
    integration/test_full_pipeline.cpp
    integration/test_pipeline_integration.cpp
    performance/bench_hot_path.cpp
)

target_link_libraries(tests PRIVATE GTest::gtest_main llmquant)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# MockOmsAdapter is a test double.  Several unit and integration tests use it
# directly.  When LLMQUANT_ENABLE_MOCK_OMS=OFF, MockOmsAdapter.cpp is not
# compiled into the llmquant library, so we compile it directly into the test
# binary here to avoid link errors.
if(NOT LLMQUANT_ENABLE_MOCK_OMS)
    target_sources(tests PRIVATE ${CMAKE_SOURCE_DIR}/src/MockOmsAdapter.cpp)
    # Ensure the tests binary sees the ENABLED define so the same code path runs.
    target_compile_definitions(tests PRIVATE LLMQUANT_MOCK_OMS_ENABLED)
endif()

# Labelled test targets for selective CI execution.
add_test(NAME unit_tests
    COMMAND tests
    --gtest_filter="*Config*:*LLMAdapter*:*Latency*:*Metrics*:*TokenStream*:*TradeSignal*:*OutputSink*:*Risk*:*Dedup*:*LLMStream*:*Invariants*:*OmsAdapter*:*FixOmsAdapter*:*Prometheus*:*EdgeCase*:*NetworkError*:*ProductionReadiness*:*SentimentTrajectory*:*SignalAuditLog*:*CircuitBreaker*:*BacktestRunner*:*KellySizer*:*HealthServer*:*StaleToken*:*AdaptiveCooldown*:*SignalBlend*:*RegimeDetector*:*TokenReplay*:*ParameterSweep*:*AlertDispatcher*:*PositionTracker*:*TokenEntropyMeter*:*LatencyBudgetEnforcer*:*TokenEntropyMonitor*:*TradingHours*:*SignalCorrelation*:*WarmupSequencer*:*SentimentMomentum*:*SignalDecay*:*TokenCostBudget*:*LeadLag*:*GradientOptimiser*:*DrawdownProtector*:*MultiTimeframe*"
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
