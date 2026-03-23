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
    unit/test_regime_transition_model.cpp
    unit/test_signal_throttle.cpp
    unit/test_confidence_calibrator.cpp
    unit/test_anomaly_detector.cpp
    unit/test_bayesian_signal_filter.cpp
    unit/test_volatility_forecaster.cpp
    unit/test_narrative_change_detector.cpp
    unit/test_pnl_attribution_engine.cpp
    unit/test_portfolio_heat_monitor.cpp
    unit/test_token_burst_detector.cpp
    unit/test_signal_persistence_tracker.cpp
    unit/test_rolling_sharpe_bias_tracker.cpp
    unit/test_context_window_budget.cpp
    unit/test_fractal_dimension_estimator.cpp
    unit/test_market_microstructure_filter.cpp
    unit/test_signal_ensemble_layer.cpp
    unit/test_signal_momentum_oscillator.cpp
    unit/test_order_book_simulator.cpp
    unit/test_token_sentiment_heatmap.cpp
    unit/test_cvar_calculator.cpp
    unit/test_temporal_pattern_library.cpp
    unit/test_feedback_loop_detector.cpp
    unit/test_sentiment_cycle_detector.cpp
    unit/test_adaptive_sampling_controller.cpp
    unit/test_mutual_information_estimator.cpp
    unit/test_signal_blind_spot_detector.cpp
    unit/test_signal_surprise_index.cpp
    unit/test_token_stream_health_monitor.cpp
    unit/test_regime_aware_sizer.cpp
    unit/test_confidence_decay_tracker.cpp
    unit/test_cross_asset_correlation_monitor.cpp
    unit/test_token_velocity_tracker.cpp
    unit/test_narrative_momentum_clock.cpp
    unit/test_adaptive_velocity_breaker.cpp
    unit/test_signal_calibration_engine.cpp
    unit/test_token_bias_heatmap.cpp
    unit/test_walk_forward_validator.cpp
    unit/test_order_flow_imbalance_detector.cpp
    unit/test_cross_session_memory.cpp
    unit/test_market_regime_probability_estimator.cpp
    unit/test_signal_replay_buffer.cpp
    unit/test_token_ngram_profiler.cpp
    unit/test_execution_quality_monitor.cpp
    unit/test_sentiment_dispersion_index.cpp
    unit/test_token_influence_attributor.cpp
    unit/test_sentiment_divergence_detector.cpp
    unit/test_adversarial_input_detector.cpp
    unit/test_signal_confidence_interval.cpp
    unit/test_token_stream_recorder.cpp
    unit/test_ensemble_signal_voter.cpp
    unit/test_sentiment_persistence_matrix.cpp
    unit/test_causal_impact_estimator.cpp
    unit/test_sentiment_phase_portrait.cpp
    unit/test_options_flow_bridge.cpp
    unit/test_token_clock_recalibrator.cpp
    unit/test_narrative_topic_classifier.cpp
    unit/test_signal_shadow_portfolio.cpp
    unit/test_token_information_bottleneck.cpp
    unit/test_llm_confidence_band_tracker.cpp
    unit/test_token_importance_decay_scheduler.cpp
    unit/test_regime_switching_router.cpp
    unit/test_token_stream_differencer.cpp
    unit/test_signal_drift_monitor.cpp
    unit/test_signal_lifecycle_tracker.cpp
    unit/test_token_weight_quantiser.cpp
    unit/test_position_concentration_guard.cpp
    unit/test_sentiment_autocorrelation_meter.cpp
    unit/test_signal_strength_indexer.cpp
    unit/test_token_flow_pressure_gauge.cpp
    unit/test_signal_fatigue_meter.cpp
    unit/test_signal_polarization_monitor.cpp
    unit/test_narrative_temperature_gauge.cpp
    unit/test_signal_echo_suppressor.cpp
    unit/test_signal_hurst_estimator.cpp
    unit/test_bias_change_point_detector.cpp
    unit/test_bias_velocity_breaker.cpp
    unit/test_signal_information_ratio_tracker.cpp
    unit/test_narrative_consistency_meter.cpp
    unit/test_signal_oscillation_detector.cpp
    unit/test_token_weight_histogram.cpp
    unit/test_signal_slope_meter.cpp
    unit/test_latency_jitter_monitor.cpp
    unit/test_bias_run_length_encoder.cpp
    unit/test_signal_coverage_meter.cpp
    unit/test_bias_hysteresis_gate.cpp
    unit/test_realized_volatility_tracker.cpp
    unit/test_signal_causal_chain_tracer.cpp
    unit/test_token_dependency_mapper.cpp
    unit/test_bias_frequency_analyser.cpp
    unit/test_signal_entropy_ratchet.cpp
    unit/test_narrative_coherence_scorer.cpp
    unit/test_cross_token_correlation_matrix.cpp
    unit/test_adaptive_position_sizer.cpp
    unit/test_bias_velocity_breaker.cpp
    unit/test_signal_information_ratio_tracker.cpp
    unit/test_narrative_consistency_meter.cpp
    unit/test_signal_oscillation_detector.cpp
    unit/test_bias_clip_monitor.cpp
    unit/test_narrative_intensity_ramp.cpp
    unit/test_sentiment_zscore_tracker.cpp
    unit/test_signal_confluence_detector.cpp
    unit/test_multi_feed_signal_aggregator.cpp
    unit/test_signal_cusum_controller.cpp
    unit/test_bias_momentum_index.cpp
    unit/test_signal_gain_loss_ratio.cpp
    unit/test_token_sentiment_momentum_index.cpp
    unit/test_bias_regime_transition_matrix.cpp
    unit/test_signal_reversal_detector.cpp
    unit/test_adaptive_threshold_controller.cpp
    unit/test_bias_conditional_distribution.cpp
    unit/test_signal_compressor.cpp
    unit/test_bias_rolling_quantile_tracker.cpp
    unit/test_signal_autoregressor.cpp
    unit/test_token_bias_phase_space.cpp
    unit/test_sentiment_topology_mapper.cpp
    unit/test_bias_information_gain.cpp
    unit/test_narrative_drift_detector.cpp
    unit/test_token_sentiment_graph_builder.cpp
    unit/test_bias_kalman_filter.cpp
    unit/test_signal_spectral_entropy.cpp
    unit/test_bias_bootstrap_ci.cpp
    unit/test_wavelet_signal_decomposer.cpp
    unit/test_reinforcement_signal_weighter.cpp
    unit/test_signal_convexity_meter.cpp
    unit/test_bias_garch_estimator.cpp
    unit/test_signal_regime_hmm.cpp
    unit/test_narrative_polarity_index.cpp
    unit/test_signal_residual_analyser.cpp
    unit/test_token_saliency_ranker.cpp
    unit/test_signal_tail_risk_meter.cpp
    unit/test_bias_level_crossing.cpp
    unit/test_signal_cross_correlator.cpp
    unit/test_narrative_volatility_ratio.cpp
    unit/test_signal_parabolic_sar.cpp
    unit/test_narrative_entropy_clock.cpp
    unit/test_signal_decay_halflife.cpp
    unit/test_bayesian_sentiment_prior.cpp
    unit/test_signal_bollinger_bands.cpp
    unit/test_bias_impulse_detector.cpp
    unit/test_signal_trend_strength_index.cpp
    unit/test_narrative_mass_index.cpp
    unit/test_signal_choppiness_index.cpp
    unit/test_signal_acceleration_meter.cpp
    unit/test_narrative_fatigue_detector.cpp
    unit/test_bias_skewness_tracker.cpp
    unit/test_signal_zero_cross_rate.cpp
    unit/test_token_bias_correlogram.cpp
    unit/test_signal_kurtosis_tracker.cpp
    unit/test_narrative_persistence_index.cpp
    unit/test_bias_entropy_rate.cpp
    unit/test_signal_drawdown_meter.cpp
    unit/test_token_cadence_analyser.cpp
    unit/test_signal_mean_reversion_speed.cpp
    unit/test_narrative_cluster_detector.cpp
    unit/test_bias_volatility_breakout.cpp
    unit/test_signal_stochastic_oscillator.cpp
    unit/test_bias_autocorrelation_function.cpp
    unit/test_online_granger_causality.cpp
    unit/test_signal_macd_histogram.cpp
    unit/test_narrative_regime_markov.cpp
    unit/test_bias_concentration_risk.cpp
    unit/test_signal_williams_r.cpp
    unit/test_token_influence_decay.cpp
    unit/test_narrative_polarity_shift.cpp
    unit/test_signal_chande_oscillator.cpp
    unit/test_signal_donchian_channel.cpp
    unit/test_token_bias_histogram.cpp
    unit/test_bias_exponential_smoothing.cpp
    unit/test_signal_relative_vigor_index.cpp
    unit/test_narrative_sentiment_velocity.cpp
    unit/test_bias_zscore_normaliser.cpp
    unit/test_signal_keltner_channel.cpp
    unit/test_token_burst_intensity.cpp
    unit/test_signal_triple_ema_oscillator.cpp
    unit/test_narrative_coherence_tracker.cpp
    unit/test_bias_local_extrema.cpp
    unit/test_signal_adaptive_threshold_filter.cpp
    unit/test_narrative_pressure_gauge.cpp
    unit/test_model_ensemble.cpp
    unit/test_token_window_summariser.cpp
    unit/test_bootstrap_signal_ci.cpp
    unit/test_execution_timing.cpp
    unit/test_alpha_decay.cpp
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
    --gtest_filter="*Config*:*LLMAdapter*:*Latency*:*Metrics*:*TokenStream*:*TradeSignal*:*OutputSink*:*Risk*:*Dedup*:*LLMStream*:*Invariants*:*OmsAdapter*:*FixOmsAdapter*:*Prometheus*:*EdgeCase*:*NetworkError*:*ProductionReadiness*:*SentimentTrajectory*:*SignalAuditLog*:*CircuitBreaker*:*BacktestRunner*:*KellySizer*:*HealthServer*:*StaleToken*:*AdaptiveCooldown*:*SignalBlend*:*RegimeDetector*:*TokenReplay*:*ParameterSweep*:*AlertDispatcher*:*PositionTracker*:*TokenEntropyMeter*:*LatencyBudgetEnforcer*:*TokenEntropyMonitor*:*TradingHours*:*SignalCorrelation*:*WarmupSequencer*:*SentimentMomentum*:*SignalDecay*:*TokenCostBudget*:*LeadLag*:*GradientOptimiser*:*DrawdownProtector*:*MultiTimeframe*:*RegimeTransition*:*SignalThrottle*:*ConfidenceCalibrator*:*AnomalyDetector*:*BayesianSignalFilter*:*VolatilityForecaster*:*NarrativeChange*:*PnLAttribution*:*PortfolioHeat*:*TokenBurstDetector*:*SignalPersistence*:*RollingSharpeBias*:*ContextWindowBudget*:*FractalDimension*:*MarketMicrostructure*:*SignalEnsemble*:*SignalMomentum*:*OrderBookSimulator*:*TokenSentimentHeatmap*:*CVaRCalculator*:*TemporalPattern*:*FeedbackLoop*:*SentimentCycle*:*AdaptiveSampling*:*MutualInformation*:*SignalBlindSpot*:*SignalSurprise*:*StreamHealth*:*RegimeSizer*:*ConfidenceDecay*:*CrossAsset*:*VelocityTracker*:*NarrativeMomentum*:*AdaptiveVelocity*:*SignalCalibration*:*WalkForward*:*OrderFlowImbalance*:*TokenBiasHeatmap*:*CrossSessionMemory*:*MarketRegimeProb*:*SignalReplay*:*TokenNgram*:*ExecutionQuality*:*SentimentDispersion*:*TokenInfluence*:*SentimentDivergence*:*AdversarialInput*:*SignalConfidence*:*TokenStreamRecorder*:*EnsembleSignal*:*SentimentPersistence*:*CausalImpact*:*PhasePortrait*:*OptionsFlow*:*NarrativeTopic*:*TokenClock*:*ShadowPortfolio*:*TokenIB*:*InformationBottleneck*:*ConfidenceBand*:*DecayScheduler*:*RegimeRouter*:*StreamDifferencer*:*SignalDrift*:*LifecycleTracker*:*TokenQuantiser*:*ConcentrationGuard*:*PositionConcentration*:*AutocorrMeter*:*AutoCorr*:*Autocorrelation*:*SignalSSI*:*SignalStrength*:*FlowPressure*:*TokenFlowPressure*:*SignalFatigue*:*FatigueMeter*:*Polarization*:*SignalPolarization*:*NarrativeTemperature*:*TemperatureGauge*:*SignalEchoSuppressor*:*EchoSuppressor*:*HurstEstimator*:*SignalHurst*:*ChangePoint*:*BiasChangePoint*:*VelocityBreaker*:*BiasVelocity*:*InformationRatio*:*IRTracker*:*ConsistencyMeter*:*NarrativeConsistency*:*OscillationDetector*:*SignalOscillation*:*WeightHistogram*:*TokenWeightHistogram*:*SignalSlope*:*SlopeMeter*:*RunLength*:*BiasRunLength*:*CoverageMeter*:*SignalCoverage*:*BiasHysteresis*:*HysteresisGate*:*RealizedVol*:*VolatilityTracker*:*CausalChain*:*CausalTracer*:*DependencyMapper*:*TokenDependency*:*FreqAnalyser*:*FrequencyAnalyser*:*EntropyRatchet*:*CoherenceScorer*:*NarrativeCoherence*:*CrossTokenCorr*:*CorrelationMatrix*:*AdaptiveSizer*:*PositionSizer*:*VelocityBreaker*:*BiasVelocity*:*InformationRatio*:*IRTracker*:*ConsistencyMeter*:*NarrativeConsistency*:*OscillationDetector*:*SignalOscillation*:*ClipMonitor*:*BiasClip*:*IntensityRamp*:*NarrativeIntensity*:*ZScoreTracker*:*SentimentZScore*:*ConfluenceDetector*:*SignalConfluence*:*MultiFeed*:*FeedAggregator*:*SignalCUSUM*:*CUSUMController*:*MomentumIndex*:*BiasMomentum*:*GainLossRatio*:*SignalGainLoss*:*TSMI*:*SentimentMomentum*:*RegimeTransitionMatrix*:*BiasRegimeTransition*:*ReversalDetector*:*SignalReversal*:*AdaptiveThreshold*:*ThresholdController*:*ConditionalDist*:*BiasConditional*:*SignalCompressor*:*LZComplexity*:*RollingQuantile*:*QuantileTracker*:*Autoregressor*:*SignalAR*:*PhaseSpace*:*BiasPhaseSpace*:*TokenBiasPhase*:*TopologyMapper*:*SentimentTopology*:*InformationGain*:*BiasInfoGain*:*NarrativeDrift*:*DriftDetector*:*SentimentGraph*:*GraphBuilder*:*KalmanFilter*:*BiasKalman*:*SpectralEntropy*:*SignalSpectral*:*BootstrapCI*:*BiasBootstrap*:*GarchEstimator*:*BiasGarch*:*RegimeHMM*:*SignalHMM*:*PolarityIndex*:*NarrativePolarity*:*ResidualAnalyser*:*DurbinWatson*:*SaliencyRanker*:*TokenSaliency*:*TailRiskMeter*:*TailRisk*:*LevelCrossing*:*ZeroCrossing*:*CrossCorrelator*:*SignalCross*:*VolatilityRatio*:*NarrativeVol*:*ParabolicSAR*:*SignalSAR*:*BollingerBands*:*SignalBB*:*ImpulseDetector*:*BiasImpulse*:*TrendStrengthIndex*:*TSISignal*:*MassIndex*:*NarrativeMass*:*ChoppinessIndex*:*SignalChop*:*AccelerationMeter*:*SignalAccel*:*FatigueDetector*:*NarrativeFatigue*:*SkewnessTracker*:*BiasSkew*:*ZeroCrossRate*:*SignalZCR*:*BiasCorrelogram*:*TokenCorrelogram*:*KurtosisTracker*:*SignalKurt*:*PersistenceIndex*:*NarrativePersist*:*EntropyRate*:*BiasEntropy*:*DrawdownMeter*:*SignalDD*:*CadenceAnalyser*:*TokenCadence*:*MeanReversionSpeed*:*ARReversion*:*ClusterDetector*:*NarrativeCluster*:*VolatilityBreakout*:*BiasVolBreak*:*StochasticOscillator*:*SignalStoch*:*MACDHistogram*:*SignalMACD*:*RegimeMarkov*:*NarrativeMarkov*:*ConcentrationRisk*:*BiasConc*:*WilliamsR*:*SignalWR*:*InfluenceDecay*:*TokenInfluenceDecay*:*PolarityShift*:*NarrativePolarity*:*ChandeOscillator*:*SignalChande*:*DonchianChannel*:*SignalDonchian*:*BiasHistogram*:*TokenBiasHisto*:*ExpSmoothing*:*BiasExpSmooth*:*RelativeVigor*:*SignalRVI*:*SentimentVelocity*:*NarrativeVel*:*ZScoreNormaliser*:*BiasZScore*:*KeltnerChannel*:*SignalKeltner*:*BurstIntensity*:*TokenBurst*:*TripleEMA*:*SignalTRIX*:*CoherenceTracker*:*NarrativeCoherence*:*LocalExtrema*:*BiasExtreme*:*AdaptiveFilter*:*ThresholdFilter*:*PressureGauge*:*NarrativePressure*:*ModelEnsemble*:*TokenWindowSummariser*:*BootstrapSignalCI*"
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
