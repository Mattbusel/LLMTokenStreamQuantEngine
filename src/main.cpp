#include "TokenStreamSimulator.h"
#include "TradeSignalEngine.h"
#include "LatencyController.h"
#include "LLMAdapter.h"
#include "MetricsLogger.h"
#include "Config.h"
#include "OutputSinkImpl.h"
#include "SignalReplayRunner.h"
#include "WebSocketServer.h"
#ifdef LLMQUANT_DEDUP_ENABLED
#  include "Deduplicator.h"
#endif
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
#  include "LLMStreamClient.h"
#endif
#include "OmsAdapter.h"
#ifdef LLMQUANT_REST_OMS_ENABLED
#  include "RestOmsAdapter.h"
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
#  include "FixOmsAdapter.h"
#endif
#ifdef LLMQUANT_MOCK_OMS_ENABLED
#  include "MockOmsAdapter.h"
#endif
#ifdef LLMQUANT_PROMETHEUS_ENABLED
#  include "PrometheusExporter.h"
#endif
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
#  include "SignalAuditLog.h"
#endif
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
#  include "PipelineCircuitBreaker.h"
#endif
#ifdef LLMQUANT_KELLY_SIZER_ENABLED
#  include "KellyPositionSizer.h"
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
#  include "HealthServer.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
#  include "AdaptiveCooldownController.h"
#endif
#ifdef LLMQUANT_SIGNAL_BLEND_ENABLED
#  include "SignalBlendLayer.h"
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
#  include "StaleTokenDetector.h"
#endif
#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
#  include "RegimeDetector.h"
#endif
#ifdef LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED
#  include "RegimeTransitionModel.h"
#endif
#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
#  include "TokenEntropyMonitor.h"
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
#  include "NarrativeChangeDetector.h"
#endif
#ifdef LLMQUANT_TRADING_HOURS_ENABLED
#  include "TradingHoursGuard.h"
#endif
#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
#  include "SignalCorrelationTracker.h"
#endif
#ifdef LLMQUANT_WARMUP_SEQUENCER_ENABLED
#  include "WarmupSequencer.h"
#endif
#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
#  include "DrawdownProtector.h"
#endif
#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
#  include "MultiTimeframeAggregator.h"
#endif
#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
#  include "VolatilityForecaster.h"
#endif
#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
#  include "BayesianSignalFilter.h"
#endif
#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
#  include "AnomalyDetector.h"
#endif
#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
#  include "TokenBurstDetector.h"
#endif
#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
#  include "SignalPersistenceTracker.h"
#endif
#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
#  include "RollingSharpeBiasTracker.h"
#endif
#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
#  include "OrderBookSimulator.h"
#endif
#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
#  include "TokenSentimentHeatmap.h"
#endif
#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
#  include "SentimentMomentumFilter.h"
#endif
#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)
#  include "PositionTracker.h"
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
#  include "SignalDecayEnvelope.h"
#endif
#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
#  include "LatencyBudgetEnforcer.h"
#endif
#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED
#  include "PnLAttributionEngine.h"
#endif
#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED
#  include "PortfolioHeatMonitor.h"
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
#  include "ContextWindowBudget.h"
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
#  include "FractalDimensionEstimator.h"
#endif
#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
#  include "MarketMicrostructureFilter.h"
#endif
#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
#  include "SignalEnsembleLayer.h"
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
#  include "SignalMomentumOscillator.h"
#endif
#ifdef LLMQUANT_CVAR_ENABLED
#  include "CVaRCalculator.h"
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
#  include "TemporalPatternLibrary.h"
#endif
#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
#  include "FeedbackLoopDetector.h"
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
#  include "SentimentCycleDetector.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
#  include "AdaptiveSamplingController.h"
#endif
#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
#  include "MutualInformationEstimator.h"
#endif
#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
#  include "SignalBlindSpotDetector.h"
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
#  include "SignalSurpriseIndex.h"
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
#  include "TokenStreamHealthMonitor.h"
#endif
#ifdef LLMQUANT_REGIME_SIZER_ENABLED
#  include "RegimeAwareSizer.h"
#endif
#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
#  include "ConfidenceDecayTracker.h"
#endif
#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
#  include "CrossAssetCorrelationMonitor.h"
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
#  include "TokenVelocityTracker.h"
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
#  include "NarrativeMomentumClock.h"
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
#  include "AdaptiveVelocityBreaker.h"
#endif
#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
#  include "SignalCalibrationEngine.h"
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
#  include "TokenBiasHeatmap.h"
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
#  include "OrderFlowImbalanceDetector.h"
#endif
#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
#  include "CrossSessionMemory.h"
#endif
#ifdef LLMQUANT_REGIME_PROB_ENABLED
#  include "MarketRegimeProbabilityEstimator.h"
#endif
#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
#  include "SignalReplayBuffer.h"
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
#  include "TokenNgramProfiler.h"
#endif
#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
#  include "ExecutionQualityMonitor.h"
#endif
#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
#  include "SentimentDispersionIndex.h"
#endif
#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
#  include "SentimentDivergenceDetector.h"
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
#  include "TokenInfluenceAttributor.h"
#endif
#ifdef LLMQUANT_WALK_FORWARD_ENABLED
#  include "WalkForwardValidator.h"
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
#  include "AdversarialInputDetector.h"
#endif
#ifdef LLMQUANT_SIGNAL_CI_ENABLED
#  include "SignalConfidenceInterval.h"
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
#  include "SentimentPersistenceMatrix.h"
#endif
#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
#  include "CausalImpactEstimator.h"
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
#  include "OptionsFlowSentimentBridge.h"
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
#  include "SentimentPhasePortrait.h"
#endif
#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
#  include "NarrativeTopicClassifier.h"
#endif
#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
#  include "TokenClockRecalibrator.h"
#endif
#ifdef LLMQUANT_SHADOW_PORTFOLIO_ENABLED
#  include "SignalShadowPortfolio.h"
#endif
#ifdef LLMQUANT_TOKEN_IB_ENABLED
#  include "TokenInformationBottleneck.h"
#endif
#ifdef LLMQUANT_CONFIDENCE_BAND_ENABLED
#  include "LLMConfidenceBandTracker.h"
#endif
#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
#  include "TokenImportanceDecayScheduler.h"
#endif
#ifdef LLMQUANT_SIGNAL_DRIFT_ENABLED
#  include "SignalDriftMonitor.h"
#endif
#ifdef LLMQUANT_REGIME_ROUTER_ENABLED
#  include "RegimeSwitchingSignalRouter.h"
#endif
#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
#  include "TokenStreamDifferencer.h"
#endif
#ifdef LLMQUANT_LIFECYCLE_TRACKER_ENABLED
#  include "SignalLifecycleTracker.h"
#endif
#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
#  include "TokenWeightQuantiser.h"
#endif
#ifdef LLMQUANT_POSITION_CONCENTRATION_ENABLED
#  include "PositionConcentrationGuard.h"
#endif
#ifdef LLMQUANT_AUTOCORR_METER_ENABLED
#  include "SentimentAutocorrelationMeter.h"
#endif
#ifdef LLMQUANT_SIGNAL_SSI_ENABLED
#  include "SignalStrengthIndexer.h"
#endif
#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
#  include "TokenFlowPressureGauge.h"
#endif
#ifdef LLMQUANT_SIGNAL_FATIGUE_ENABLED
#  include "SignalFatigueMeter.h"
#endif
#ifdef LLMQUANT_POLARIZATION_MONITOR_ENABLED
#  include "SignalPolarizationMonitor.h"
#endif
#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
#  include "NarrativeTemperatureGauge.h"
#endif
#ifdef LLMQUANT_ECHO_SUPPRESSOR_ENABLED
#  include "SignalEchoSuppressor.h"
#endif
#ifdef LLMQUANT_HURST_ESTIMATOR_ENABLED
#  include "SignalHurstEstimator.h"
#endif
#ifdef LLMQUANT_CHANGE_POINT_ENABLED
#  include "BiasChangePointDetector.h"
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
#  include "BiasVelocityBreaker.h"
#endif
#ifdef LLMQUANT_IR_TRACKER_ENABLED
#  include "SignalInformationRatioTracker.h"
#endif
#ifdef LLMQUANT_CONSISTENCY_METER_ENABLED
#  include "NarrativeConsistencyMeter.h"
#endif
#ifdef LLMQUANT_OSCILLATION_DETECTOR_ENABLED
#  include "SignalOscillationDetector.h"
#endif
#ifdef LLMQUANT_WEIGHT_HISTOGRAM_ENABLED
#  include "TokenWeightHistogram.h"
#endif
#ifdef LLMQUANT_SIGNAL_SLOPE_ENABLED
#  include "SignalSlopeMeter.h"
#endif
#ifdef LLMQUANT_LATENCY_JITTER_ENABLED
#  include "LatencyJitterMonitor.h"
#endif
#ifdef LLMQUANT_RUN_LENGTH_ENABLED
#  include "BiasRunLengthEncoder.h"
#endif
#ifdef LLMQUANT_COVERAGE_METER_ENABLED
#  include "SignalCoverageMeter.h"
#endif
#ifdef LLMQUANT_BIAS_HYSTERESIS_ENABLED
#  include "BiasHysteresisGate.h"
#endif
#ifdef LLMQUANT_REALIZED_VOL_ENABLED
#  include "RealizedVolatilityTracker.h"
#endif
#ifdef LLMQUANT_CAUSAL_TRACER_ENABLED
#  include "SignalCausalChainTracer.h"
#endif
#ifdef LLMQUANT_DEPENDENCY_MAPPER_ENABLED
#  include "TokenDependencyMapper.h"
#endif
#ifdef LLMQUANT_FREQ_ANALYSER_ENABLED
#  include "BiasFrequencyAnalyser.h"
#endif
#ifdef LLMQUANT_ENTROPY_RATCHET_ENABLED
#  include "SignalEntropyRatchet.h"
#endif
#ifdef LLMQUANT_COHERENCE_SCORER_ENABLED
#  include "NarrativeCoherenceScorer.h"
#endif
#ifdef LLMQUANT_CROSS_TOKEN_CORR_ENABLED
#  include "CrossTokenCorrelationMatrix.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_SIZER_ENABLED
#  include "AdaptivePositionSizer.h"
#endif
#ifdef LLMQUANT_CLIP_MONITOR_ENABLED
#  include "BiasClipMonitor.h"
#endif
#ifdef LLMQUANT_INTENSITY_RAMP_ENABLED
#  include "NarrativeIntensityRamp.h"
#endif
#ifdef LLMQUANT_ZSCORE_TRACKER_ENABLED
#  include "SentimentZScoreTracker.h"
#endif
#ifdef LLMQUANT_CONFLUENCE_DETECTOR_ENABLED
#  include "SignalConfluenceDetector.h"
#endif
#ifdef LLMQUANT_MULTI_FEED_AGGREGATOR_ENABLED
#  include "MultiFeedSignalAggregator.h"
#endif
#ifdef LLMQUANT_SIGNAL_CUSUM_ENABLED
#  include "SignalCUSUMController.h"
#endif
#ifdef LLMQUANT_MOMENTUM_INDEX_ENABLED
#  include "BiasMomentumIndex.h"
#endif
#ifdef LLMQUANT_GAIN_LOSS_RATIO_ENABLED
#  include "SignalGainLossRatio.h"
#endif
#ifdef LLMQUANT_REGIME_TRANSITION_MATRIX_ENABLED
#  include "BiasRegimeTransitionMatrix.h"
#endif
#ifdef LLMQUANT_REVERSAL_DETECTOR_ENABLED
#  include "SignalReversalDetector.h"
#endif
#ifdef LLMQUANT_TSMI_ENABLED
#  include "TokenSentimentMomentumIndex.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_THRESHOLD_ENABLED
#  include "AdaptiveThresholdController.h"
#endif
#ifdef LLMQUANT_CONDITIONAL_DIST_ENABLED
#  include "BiasConditionalDistribution.h"
#endif
#ifdef LLMQUANT_SIGNAL_COMPRESSOR_ENABLED
#  include "SignalCompressor.h"
#endif
#ifdef LLMQUANT_ROLLING_QUANTILE_ENABLED
#  include "BiasRollingQuantileTracker.h"
#endif
#ifdef LLMQUANT_AUTOREGRESSOR_ENABLED
#  include "SignalAutoregressor.h"
#endif
#ifdef LLMQUANT_PHASE_SPACE_ENABLED
#  include "TokenBiasPhaseSpace.h"
#endif
#ifdef LLMQUANT_TOPOLOGY_MAPPER_ENABLED
#  include "SentimentTopologyMapper.h"
#endif
#ifdef LLMQUANT_INFORMATION_GAIN_ENABLED
#  include "BiasInformationGain.h"
#endif
#ifdef LLMQUANT_NARRATIVE_DRIFT_ENABLED
#  include "NarrativeDriftDetector.h"
#endif
#ifdef LLMQUANT_SENTIMENT_GRAPH_ENABLED
#  include "TokenSentimentGraphBuilder.h"
#endif
#ifdef LLMQUANT_KALMAN_FILTER_ENABLED
#  include "BiasKalmanFilter.h"
#endif
#ifdef LLMQUANT_SPECTRAL_ENTROPY_ENABLED
#  include "SignalSpectralEntropy.h"
#endif
#ifdef LLMQUANT_BOOTSTRAP_CI_ENABLED
#  include "BiasBootstrapCI.h"
#endif
#ifdef LLMQUANT_WAVELET_DECOMPOSER_ENABLED
#  include "WaveletSignalDecomposer.h"
#endif
#ifdef LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED
#  include "ReinforcementSignalWeighter.h"
#endif
#ifdef LLMQUANT_SIGNAL_CONVEXITY_ENABLED
#  include "SignalConvexityMeter.h"
#endif
#ifdef LLMQUANT_GARCH_ESTIMATOR_ENABLED
#  include "BiasGarchEstimator.h"
#endif
#ifdef LLMQUANT_REGIME_HMM_ENABLED
#  include "SignalRegimeHMM.h"
#endif
#ifdef LLMQUANT_POLARITY_INDEX_ENABLED
#  include "NarrativePolarityIndex.h"
#endif
#ifdef LLMQUANT_RESIDUAL_ANALYSER_ENABLED
#  include "SignalResidualAnalyser.h"
#endif
#ifdef LLMQUANT_SALIENCY_RANKER_ENABLED
#  include "TokenSaliencyRanker.h"
#endif
#ifdef LLMQUANT_TAIL_RISK_METER_ENABLED
#  include "SignalTailRiskMeter.h"
#endif
#ifdef LLMQUANT_LEVEL_CROSSING_ENABLED
#  include "BiasLevelCrossing.h"
#endif
#ifdef LLMQUANT_CROSS_CORRELATOR_ENABLED
#  include "SignalCrossCorrelator.h"
#endif
#ifdef LLMQUANT_VOL_RATIO_ENABLED
#  include "NarrativeVolatilityRatio.h"
#endif
#ifdef LLMQUANT_PARABOLIC_SAR_ENABLED
#  include "SignalParabolicSAR.h"
#endif
#ifdef LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED
#  include "NarrativeEntropyClock.h"
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED
#  include "SignalDecayHalfLife.h"
#endif
#ifdef LLMQUANT_BAYESIAN_SENTIMENT_ENABLED
#  include "BayesianSentimentPrior.h"
#endif
#ifdef LLMQUANT_BOLLINGER_BANDS_ENABLED
#  include "SignalBollingerBands.h"
#endif
#ifdef LLMQUANT_IMPULSE_DETECTOR_ENABLED
#  include "BiasImpulseDetector.h"
#endif
#ifdef LLMQUANT_TREND_STRENGTH_INDEX_ENABLED
#  include "SignalTrendStrengthIndex.h"
#endif
#ifdef LLMQUANT_MASS_INDEX_ENABLED
#  include "NarrativeMassIndex.h"
#endif
#ifdef LLMQUANT_CHOPPINESS_INDEX_ENABLED
#  include "SignalChoppinessIndex.h"
#endif
#ifdef LLMQUANT_ACCELERATION_METER_ENABLED
#  include "SignalAccelerationMeter.h"
#endif
#ifdef LLMQUANT_FATIGUE_DETECTOR_ENABLED
#  include "NarrativeFatigueDetector.h"
#endif
#ifdef LLMQUANT_SKEWNESS_TRACKER_ENABLED
#  include "BiasSkewnessTracker.h"
#endif
#ifdef LLMQUANT_ZERO_CROSS_RATE_ENABLED
#  include "SignalZeroCrossRate.h"
#endif
#ifdef LLMQUANT_BIAS_CORRELOGRAM_ENABLED
#  include "TokenBiasCorrelogram.h"
#endif
#ifdef LLMQUANT_KURTOSIS_TRACKER_ENABLED
#  include "SignalKurtosisTracker.h"
#endif
#ifdef LLMQUANT_PERSISTENCE_INDEX_ENABLED
#  include "NarrativePersistenceIndex.h"
#endif
#ifdef LLMQUANT_BIAS_ENTROPY_RATE_ENABLED
#  include "BiasEntropyRate.h"
#endif
#ifdef LLMQUANT_DRAWDOWN_METER_ENABLED
#  include "SignalDrawdownMeter.h"
#endif
#ifdef LLMQUANT_CADENCE_ANALYSER_ENABLED
#  include "TokenCadenceAnalyser.h"
#endif
#ifdef LLMQUANT_MEAN_REVERSION_SPEED_ENABLED
#  include "SignalMeanReversionSpeed.h"
#endif
#ifdef LLMQUANT_CLUSTER_DETECTOR_ENABLED
#  include "NarrativeClusterDetector.h"
#endif
#ifdef LLMQUANT_VOL_BREAKOUT_ENABLED
#  include "BiasVolatilityBreakout.h"
#endif
#ifdef LLMQUANT_STOCHASTIC_OSC_ENABLED
#  include "SignalStochasticOscillator.h"
#endif
#ifdef LLMQUANT_BIAS_ACF_ENABLED
#  include "BiasAutocorrelationFunction.h"
#endif
#ifdef LLMQUANT_ONLINE_GRANGER_ENABLED
#  include "OnlineGrangerCausality.h"
#endif
#ifdef LLMQUANT_MACD_HISTOGRAM_ENABLED
#  include "SignalMACDHistogram.h"
#endif
#ifdef LLMQUANT_REGIME_MARKOV_ENABLED
#  include "NarrativeRegimeMarkov.h"
#endif
#ifdef LLMQUANT_CONCENTRATION_RISK_ENABLED
#  include "BiasConcentrationRisk.h"
#endif
#ifdef LLMQUANT_WILLIAMS_R_ENABLED
#  include "SignalWilliamsR.h"
#endif
#ifdef LLMQUANT_INFLUENCE_DECAY_ENABLED
#  include "TokenInfluenceDecay.h"
#endif
#ifdef LLMQUANT_POLARITY_SHIFT_ENABLED
#  include "NarrativePolarityShift.h"
#endif
#ifdef LLMQUANT_CHANDE_OSC_ENABLED
#  include "SignalChandeOscillator.h"
#endif
#ifdef LLMQUANT_DONCHIAN_CHANNEL_ENABLED
#  include "SignalDonchianChannel.h"
#endif
#ifdef LLMQUANT_BIAS_HISTOGRAM_ENABLED
#  include "TokenBiasHistogram.h"
#endif
#ifdef LLMQUANT_EXP_SMOOTHING_ENABLED
#  include "BiasExponentialSmoothing.h"
#endif
#ifdef LLMQUANT_RELATIVE_VIGOR_ENABLED
#  include "SignalRelativeVigorIndex.h"
#endif
#ifdef LLMQUANT_SENTIMENT_VELOCITY_ENABLED
#  include "NarrativeSentimentVelocity.h"
#endif
#ifdef LLMQUANT_ZSCORE_NORMALISER_ENABLED
#  include "BiasZScoreNormaliser.h"
#endif
#ifdef LLMQUANT_KELTNER_CHANNEL_ENABLED
#  include "SignalKeltnerChannel.h"
#endif
#ifdef LLMQUANT_BURST_INTENSITY_ENABLED
#  include "TokenBurstIntensity.h"
#endif
#ifdef LLMQUANT_TRIPLE_EMA_ENABLED
#  include "SignalTripleEMAOscillator.h"
#endif
#ifdef LLMQUANT_COHERENCE_TRACKER_ENABLED
#  include "NarrativeCoherenceTracker.h"
#endif
#ifdef LLMQUANT_LOCAL_EXTREMA_ENABLED
#  include "BiasLocalExtrema.h"
#endif
#ifdef LLMQUANT_ADAPTIVE_FILTER_ENABLED
#  include "SignalAdaptiveThresholdFilter.h"
#endif
#ifdef LLMQUANT_PRESSURE_GAUGE_ENABLED
#  include "NarrativePressureGauge.h"
#endif
#include "llmquant_version.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
#else
#  include <unistd.h>   // sysconf(_SC_CLK_TCK) for CPU fraction on Linux
#endif

using namespace llmquant;

/// @brief Returns the process RSS (resident set size) in bytes, or 0 if unavailable.
static uint64_t get_process_rss_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<uint64_t>(pmc.WorkingSetSize);
    return 0;
#else
    // Linux/macOS: parse VmRSS from /proc/self/status (kB units).
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            uint64_t kb = 0;
            std::sscanf(line.c_str(), "VmRSS: %llu kB", &kb);
            return kb * 1024ULL;
        }
    }
    return 0;
#endif
}

/// @brief Returns process CPU usage as a fraction [0.0, N_cores] since last call.
///        Returns 0.0 if unavailable.  Not async-signal-safe; call from monitoring thread only.
static double get_process_cpu_fraction() {
#ifdef _WIN32
    static bool  win_initialized = false;
    static FILETIME prev_kernel{}, prev_user{}, prev_wall{};
    FILETIME creation, exit_ft, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit_ft, &kernel, &user))
        return 0.0;
    FILETIME now_ft;
    GetSystemTimeAsFileTime(&now_ft);
    // On first call: seed previous values and return 0 to avoid cumulative-uptime spike.
    if (!win_initialized) {
        win_initialized = true;
        prev_kernel = kernel; prev_user = user; prev_wall = now_ft;
        return 0.0;
    }
    auto to_u64 = [](FILETIME ft) -> uint64_t {
        return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };
    uint64_t k = to_u64(kernel), u = to_u64(user), w = to_u64(now_ft);
    uint64_t dk = k - to_u64(prev_kernel);
    uint64_t du = u - to_u64(prev_user);
    uint64_t dw = w - to_u64(prev_wall);
    prev_kernel = kernel; prev_user = user; prev_wall = now_ft;
    if (dw == 0) return 0.0;
    return static_cast<double>(dk + du) / static_cast<double>(dw);
#else
    // Linux: read /proc/self/stat fields utime+stime (jiffies), compare with wall clock.
    static uint64_t prev_cpu_jiffies = UINT64_MAX;  // sentinel: UINT64_MAX = uninitialized
    static std::chrono::steady_clock::time_point prev_tp = std::chrono::steady_clock::now();
    std::ifstream f("/proc/self/stat");
    if (!f.is_open()) return 0.0;
    std::string stat_line;
    std::getline(f, stat_line);
    // Fields 14 and 15 (1-indexed) are utime and stime; skip past the comm field.
    auto rp = stat_line.rfind(')');
    if (rp == std::string::npos) return 0.0;
    std::istringstream iss(stat_line.substr(rp + 2));
    uint64_t utime = 0, stime = 0;
    std::string tok;
    try {
        for (int i = 3; i <= 15; ++i) {
            if (!(iss >> tok)) break;
            if (i == 14) utime = std::stoull(tok);
            if (i == 15) stime = std::stoull(tok);
        }
    } catch (...) { return 0.0; }
    uint64_t cpu_jiffies = utime + stime;
    auto now = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(now - prev_tp).count();
    // On first call prev_cpu_jiffies == UINT64_MAX (sentinel): seed and return 0.
    if (prev_cpu_jiffies == UINT64_MAX) {
        prev_cpu_jiffies = cpu_jiffies;
        prev_tp = now;
        return 0.0;
    }
    double delta_jiffies = static_cast<double>(cpu_jiffies - prev_cpu_jiffies);
    prev_cpu_jiffies = cpu_jiffies;
    prev_tp = now;
    if (elapsed_s <= 0.0) return 0.0;
    long hz = sysconf(_SC_CLK_TCK);
    return delta_jiffies / (elapsed_s * static_cast<double>(hz > 0 ? hz : 100));
#endif
}

std::atomic<bool> g_running{true};

void signal_handler(int /*signal*/) {
    // std::cout is not async-signal-safe; only set the atomic flag here.
    // The main loop detects g_running==false and prints the shutdown summary.
    g_running = false;
}

int main(int argc, char* argv[]) {
  try {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Record engine start time for uptime metrics.
    const auto engine_start_time = std::chrono::steady_clock::now();
    // Also capture a system_clock snapshot so Prometheus can expose the absolute
    // start timestamp (system_clock and steady_clock have different epochs).
    const int64_t engine_start_unix_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Parse flags before anything else.
    bool        stream_mode    = false;
    std::string stream_api_key;
    bool        no_color       = false;
    bool        debug_raw      = false;
    bool        dry_run         = false;
    bool        backtest_mode   = false;
    bool        list_tokens     = false;
    bool        dump_config     = false;
    bool        validate_config = false;
    bool        quiet           = false;
    std::string export_dict_path;   // non-empty = write TSV to file and exit
    // ── New modes ───────────────────────────────────────────────────────────
    std::string replay_input_path;   // non-empty = run --test-replay mode
    std::string replay_output_path;  // output file for replay trace (empty = stdout)
    bool        replay_verbose   = false;
    bool        websocket_mode   = false;
    std::string ws_host          = "0.0.0.0";
    uint16_t    ws_port          = 9200;
    std::size_t ws_max_sessions  = 64;
    std::string oms_address;
    std::string fix_address;
    std::string config_file    = "config.yaml"; // may be overridden by --config
    uint16_t    stats_port_override = 0;        // 0 = use config value
    int         token_interval_override = 0;    // 0 = use config value
    std::string log_level_str  = "info";        // spdlog level name
    int         stats_interval_ms  = 1000;      // monitoring loop tick period
    bool        no_prometheus  = false;         // skip Prometheus exporter
#ifdef LLMQUANT_DEDUP_ENABLED
    bool        no_dedup       = false;         // disable deduplication at runtime
#endif
    bool        no_hot_reload  = false;         // skip config file watcher
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
    std::string audit_log_path;                  // non-empty = enable audit log at this path
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    uint16_t    health_port_override = 0;        // 0 = use default (8080)
    bool        no_health_server     = false;    // skip health server
#endif
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: LLMTokenStreamQuantEngine [config.yaml] [options]\n"
                "\n"
                "Options:\n"
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
                "  --stream [key]    Enable live LLM stream mode (optional API key)\n"
#endif
                "  --oms host:port   Connect to REST OMS adapter\n"
                "  --fix host:port   Connect to FIX 4.2 OMS adapter\n"
                "  --config path     Path to config YAML (default: config.yaml)\n"
                "  --stats-port N    Override Prometheus metrics port (default: from config, 9100)\n"
                "  --token-interval N  Override token_stream.token_interval_ms (ms between tokens, min 1)\n"
                "  --log-level LEVEL Set spdlog log level: trace|debug|info|warn|error|critical (default: info)\n"
                "  --dry-run         Process tokens through LLMAdapter only; skip signal emission\n"
                "  --backtest        Enable backtest mode (emit signal on every token, no cooldown)\n"
                "  --test-replay FILE  Replay a JSONL token file through the pipeline\n"
                "  --output FILE     Output file for --test-replay trace (default: stdout)\n"
                "  --replay-verbose  Log skipped/malformed lines in --test-replay mode\n"
                "  --websocket       Start WebSocket live feed server\n"
                "  --ws-host HOST    WebSocket bind host (default: 0.0.0.0)\n"
                "  --ws-port PORT    WebSocket bind port (default: 9200)\n"
                "  --ws-max-sessions N  Max concurrent WebSocket clients (default: 64)\n"
                "  --no-color        Disable ANSI colour output\n"
                "  --debug-raw       Print raw LLM stream bytes\n"
                "  --list-tokens     Print the full semantic dictionary and exit\n"
                "  --export-dict FILE  Export semantic dictionary to a TSV file and exit\n"
                "  --dump-config     Print effective configuration and exit\n"
                "  --validate-config Validate configuration, print any errors, exit 0=OK 1=invalid\n"
                "  --quiet           Suppress console signal/stats output (log-file only)\n"
                "  --stats-interval N  Monitoring loop tick period in ms (default: 1000)\n"
                "  --no-prometheus   Disable the Prometheus /metrics scrape endpoint\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                "  --no-dedup        Disable token deduplication (all tokens treated as novel)\n"
#endif
                "  --no-hot-reload   Disable config file hot-reload watcher\n"
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
                "  --audit-log FILE  Write per-signal NDJSON audit log to FILE\n"
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
                "  --health-port N   Override HTTP /health endpoint port (default: 8080)\n"
                "  --no-health       Disable the HTTP /health endpoint\n"
#endif
                "  --version         Print version and exit\n"
                "  --show-flags      Print compile-time feature flags and exit\n"
                "  --help            Print this help and exit\n"
                "\n"
                "Environment:\n"
                "  LLMQUANT_API_KEY        LLM API key (fallback when --stream has no key)\n"
                "  LLMQUANT_NO_PROMETHEUS  Set to 1/true/yes to disable Prometheus endpoint\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                "  LLMQUANT_NO_DEDUP       Set to 1/true/yes to disable token deduplication\n"
#endif
                "  LLMQUANT_NO_HOT_RELOAD  Set to 1/true/yes to disable config hot-reload\n"
                "  LLMQUANT_DRY_RUN        Set to 1/true/yes for dry-run (signal only, no OMS)\n"
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
                "  LLMQUANT_AUDIT_LOG_PATH Path to NDJSON audit log file\n"
#endif
                "  LLMQUANT_QUIET          Set to 1/true/yes to suppress console output\n"
                "  LLMQUANT_BACKTEST       Set to 1/true/yes to enable backtest mode\n"
                "\n"
                "Config file (YAML) keys: token_stream, trading, latency, logging,\n"
                "  pressure, risk_thresholds, risk (override flags).\n";
            return 0;
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "LLMTokenStreamQuantEngine " << LLMQUANT_VERSION
                      << " (" << LLMQUANT_GIT_COMMIT
                      << ", " << LLMQUANT_BUILD_TIMESTAMP << ")\n";
            return 0;
        } else if (arg == "--show-flags") {
            // Print all compile-time feature flag states and exit.
            // Useful for verifying embedded/minimal builds have the expected features.
            std::cout << "Compile-time feature flags:\n"
#ifdef LLMQUANT_PROMETHEUS_ENABLED
                      << "  LLMQUANT_ENABLE_PROMETHEUS    ON\n"
#else
                      << "  LLMQUANT_ENABLE_PROMETHEUS    OFF\n"
#endif
#ifdef LLMQUANT_FIX_OMS_ENABLED
                      << "  LLMQUANT_ENABLE_FIX_OMS       ON\n"
#else
                      << "  LLMQUANT_ENABLE_FIX_OMS       OFF\n"
#endif
#ifdef LLMQUANT_REST_OMS_ENABLED
                      << "  LLMQUANT_ENABLE_REST_OMS      ON\n"
#else
                      << "  LLMQUANT_ENABLE_REST_OMS      OFF\n"
#endif
#ifdef LLMQUANT_DEDUP_ENABLED
                      << "  LLMQUANT_ENABLE_DEDUP         ON\n"
#else
                      << "  LLMQUANT_ENABLE_DEDUP         OFF\n"
#endif
#ifdef LLMQUANT_PROFILING_ENABLED
                      << "  LLMQUANT_ENABLE_PROFILING     ON\n"
#else
                      << "  LLMQUANT_ENABLE_PROFILING     OFF\n"
#endif
#ifdef LLMQUANT_JSON_STATS_SUMMARY
                      << "  LLMQUANT_ENABLE_JSON_STATS    ON\n"
#else
                      << "  LLMQUANT_ENABLE_JSON_STATS    OFF\n"
#endif
#ifdef LLMQUANT_TLS_ENABLED
                      << "  LLMQUANT_ENABLE_TLS           ON\n"
#else
                      << "  LLMQUANT_ENABLE_TLS           OFF\n"
#endif
#ifdef LLMQUANT_REDIS_ENABLED
                      << "  LLMQUANT_ENABLE_REDIS         ON\n"
#else
                      << "  LLMQUANT_ENABLE_REDIS         OFF\n"
#endif
#ifdef LLMQUANT_HOT_RELOAD_ENABLED
                      << "  LLMQUANT_ENABLE_HOT_RELOAD    ON\n"
#else
                      << "  LLMQUANT_ENABLE_HOT_RELOAD    OFF\n"
#endif
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
                      << "  LLMQUANT_ENABLE_STREAM_CLIENT ON\n"
#else
                      << "  LLMQUANT_ENABLE_STREAM_CLIENT OFF\n"
#endif
#ifdef LLMQUANT_SIMD_DISABLED
                      << "  LLMQUANT_ENABLE_SIMD          OFF\n"
#else
                      << "  LLMQUANT_ENABLE_SIMD          ON (SSE2"
#  ifdef LLMQUANT_HAVE_SSE2
                      << " active)\n"
#  else
                      << " not detected — scalar fallback)\n"
#  endif
#endif
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
                      << "  LLMQUANT_ENABLE_SIGNAL_TRACE  ON\n"
#else
                      << "  LLMQUANT_ENABLE_SIGNAL_TRACE  OFF\n"
#endif
                      ;
            return 0;
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
        } else if (arg == "--stream") {
            stream_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                stream_api_key = argv[++i];  // explicit key provided on CLI
#endif
        } else if (arg == "--no-color") {
            no_color = true;
        } else if (arg == "--debug-raw") {
            debug_raw = true;
        } else if (arg == "--list-tokens") {
            list_tokens = true;
        } else if (arg == "--export-dict" && i + 1 < argc) {
            export_dict_path = argv[++i];
        } else if (arg == "--dump-config") {
            dump_config = true;
        } else if (arg == "--validate-config") {
            validate_config = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--backtest") {
            backtest_mode = true;
        // ── New modes ─────────────────────────────────────────────────────────
        } else if (arg == "--test-replay" && i + 1 < argc) {
            replay_input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            replay_output_path = argv[++i];
        } else if (arg == "--replay-verbose" || (arg == "--verbose" && !replay_input_path.empty())) {
            replay_verbose = true;
        } else if (arg == "--websocket" || arg == "--ws") {
            websocket_mode = true;
        } else if (arg == "--ws-host" && i + 1 < argc) {
            ws_host = argv[++i];
        } else if (arg == "--ws-port" && i + 1 < argc) {
            try {
                int p = std::stoi(argv[++i]);
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                ws_port = static_cast<uint16_t>(p);
            }
            catch (...) { std::cerr << "error: --ws-port requires an integer in 1-65535\n"; return 1; }
        } else if (arg == "--ws-max-sessions" && i + 1 < argc) {
            try { ws_max_sessions = static_cast<std::size_t>(std::stoi(argv[++i])); }
            catch (...) { std::cerr << "error: --ws-max-sessions requires a positive integer\n"; return 1; }
        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--stats-port" && i + 1 < argc) {
            try {
                int p = std::stoi(argv[++i]);
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                stats_port_override = static_cast<uint16_t>(p);
            }
            catch (...) { std::cerr << "error: --stats-port requires an integer in range 1-65535\n"; return 1; }
        } else if (arg == "--token-interval" && i + 1 < argc) {
            try { token_interval_override = std::max(1, std::stoi(argv[++i])); }
            catch (...) { std::cerr << "error: --token-interval requires an integer\n"; return 1; }
        } else if (arg == "--log-level" && i + 1 < argc) {
            log_level_str = argv[++i];
        } else if (arg == "--stats-interval" && i + 1 < argc) {
            try {
                stats_interval_ms = std::clamp(std::stoi(argv[++i]), 100, 60000);
            }
            catch (...) { std::cerr << "error: --stats-interval requires an integer\n"; return 1; }
        } else if (arg == "--no-prometheus") {
            no_prometheus = true;
#ifdef LLMQUANT_DEDUP_ENABLED
        } else if (arg == "--no-dedup") {
            no_dedup = true;
#endif
        } else if (arg == "--no-hot-reload") {
            no_hot_reload = true;
        } else if (arg == "--oms" && i + 1 < argc) {
            oms_address = argv[++i];
        } else if (arg == "--fix" && i + 1 < argc) {
            fix_address = argv[++i];
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
        } else if (arg == "--audit-log" && i + 1 < argc) {
            audit_log_path = argv[++i];
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
        } else if (arg == "--health-port" && i + 1 < argc) {
            try {
                int p = std::stoi(argv[++i]);
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                health_port_override = static_cast<uint16_t>(p);
            }
            catch (...) { std::cerr << "error: --health-port requires an integer in range 1-65535\n"; return 1; }
        } else if (arg == "--no-health") {
            no_health_server = true;
#endif
        }
    }

    // Environment variable overrides for runtime feature flags.
    // Useful in containerised / Kubernetes deployments where editing the command
    // line is inconvenient.  CLI flags take precedence; env vars only set the flag
    // when the CLI has NOT already set it.
    //
    // LLMQUANT_NO_PROMETHEUS=1   equivalent to --no-prometheus
    // LLMQUANT_NO_DEDUP=1        equivalent to --no-dedup
    // LLMQUANT_NO_HOT_RELOAD=1   equivalent to --no-hot-reload
    // LLMQUANT_DRY_RUN=1         equivalent to --dry-run
    // LLMQUANT_QUIET=1           equivalent to --quiet
    // LLMQUANT_BACKTEST=1        equivalent to --backtest
    {
        auto env_flag = [](const char* name) -> bool {
#ifdef _WIN32
            char buf[8] = {};
            size_t sz = 0;
            if (getenv_s(&sz, buf, sizeof(buf), name) != 0 || sz == 0) return false;
            const char* v = buf;
#else
            const char* v = std::getenv(name);
#endif
            return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
        };
        if (!no_prometheus  && env_flag("LLMQUANT_NO_PROMETHEUS"))  no_prometheus  = true;
#ifdef LLMQUANT_DEDUP_ENABLED
        if (!no_dedup       && env_flag("LLMQUANT_NO_DEDUP"))       no_dedup       = true;
#endif
        if (!no_hot_reload  && env_flag("LLMQUANT_NO_HOT_RELOAD"))  no_hot_reload  = true;
        if (!dry_run        && env_flag("LLMQUANT_DRY_RUN"))        dry_run        = true;
        if (!quiet          && env_flag("LLMQUANT_QUIET"))          quiet          = true;
        if (!backtest_mode  && env_flag("LLMQUANT_BACKTEST"))       backtest_mode  = true;
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
        // LLMQUANT_AUDIT_LOG_PATH=<file>  set audit log path when --audit-log was not passed
        if (audit_log_path.empty()) {
#ifdef _WIN32
            char env_buf[512] = {};
            size_t env_sz = 0;
            if (getenv_s(&env_sz, env_buf, sizeof(env_buf), "LLMQUANT_AUDIT_LOG_PATH") == 0
                    && env_sz > 0)
                audit_log_path = env_buf;
#else
            const char* env_p = std::getenv("LLMQUANT_AUDIT_LOG_PATH");
            if (env_p && env_p[0] != '\0') audit_log_path = env_p;
#endif
        }
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
        if (!no_health_server && env_flag("LLMQUANT_NO_HEALTH")) no_health_server = true;
#endif
    }

    // Apply log level before any spdlog calls so early warnings are visible.
    {
        auto level = spdlog::level::from_str(log_level_str);
        // from_str returns off for unknown names — warn and fall back to info.
        if (level == spdlog::level::off && log_level_str != "off") {
            spdlog::warn("Unknown --log-level '{}'; defaulting to info", log_level_str);
            level = spdlog::level::info;
        }
        spdlog::set_level(level);
    }

    // API key security: fall back to LLMQUANT_API_KEY env var if not given on CLI.
    // Never echo the key value into logs or the banner.
    if (stream_api_key.empty()) {
        // getenv is safe here: the environment is not modified concurrently
        // during this single-threaded init phase.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
        const char* env_key_raw = std::getenv("LLMQUANT_API_KEY");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        if (const char* env_key = env_key_raw) {
            stream_api_key = env_key;
            spdlog::warn("API key loaded from environment variable LLMQUANT_API_KEY"
                         " — consider using a key file (mode 0600) for better security");
        }
    } else {
        spdlog::debug("API key loaded from command-line argument");
    }

    // Colour helpers — emit empty string when --no-color is active.
    auto C = [&](const char* code) -> const char* { return no_color ? "" : code; };
    // Line helpers — ASCII dividers when --no-color, Unicode otherwise.
    const char* DIV1 = no_color
        ? "  ---------------------------------------------------------\n"
        : "  \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n";
    const char* DIV2 = no_color
        ? "  -----------------------------------------------------------------\n"
        : "  \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n";
    const char* ARROW = no_color ? "->" : "\xe2\x86\x92";

    // Load configuration.
    // config_file may already have been set by --config; otherwise treat the
    // first positional argument (no leading dash) as the config path so that
    // the legacy invocation `engine config.yaml` still works.
    Config config;
    if (config_file == "config.yaml") {
        for (int i = 1; i < argc; ++i) {
            if (argv[i][0] != '-') { config_file = argv[i]; break; }
        }
    }
    bool config_loaded = config.load_from_file(config_file);
    if (!config_loaded) {
        std::cout << "Using default configuration" << std::endl;
        // No config file: fall back to in-memory token stream so no file I/O required.
        config.set_use_memory_stream(true);
    }
    // Apply CLI overrides before capturing sys_config so all subsystems see the
    // effective values (token_sim is constructed later but reads from sys_config).
    if (token_interval_override > 0) {
        config.set_token_interval_ms(token_interval_override);
        spdlog::info("--token-interval: overriding token_interval_ms to {}ms", token_interval_override);
    }

    const auto& sys_config = config.get_config();

    // --dump-config: print effective configuration and exit.
    if (dump_config) {
        const auto& ts  = sys_config.token_stream;
        const auto& tr  = sys_config.trading;
        const auto& lat = sys_config.latency;
        const auto& log = sys_config.logging;
        const auto& met = sys_config.metrics;
        const auto& rt  = sys_config.risk_thresholds;
        std::cout << "# Effective configuration loaded from: " << config_file << "\n"
                  << "token_stream.use_memory_stream:       " << (ts.use_memory_stream ? "true" : "false") << "\n"
                  << "token_stream.token_interval_ms:       " << ts.token_interval_ms << "\n"
                  << "token_stream.buffer_size:             " << ts.buffer_size << "\n"
                  << "token_stream.data_file_path:          " << ts.data_file_path << "\n"
                  << "token_stream.dedup_ttl_ms:            " << ts.dedup_ttl_ms
                      << (ts.dedup_ttl_ms == 0 ? " (auto: 10x token_interval_ms)" : "") << "\n"
                  << "trading.bias_sensitivity:             " << tr.bias_sensitivity << "\n"
                  << "trading.volatility_sensitivity:       " << tr.volatility_sensitivity << "\n"
                  << "trading.signal_decay_rate:            " << tr.signal_decay_rate << "\n"
                  << "trading.signal_cooldown_us:           " << tr.signal_cooldown_us << "\n"
                  << "trading.max_signal_age_us:            " << tr.max_signal_age_us << "\n"
                  << "trading.min_bias_threshold:           " << tr.min_bias_threshold << "\n"
                  << "trading.max_accumulated_bias:         " << tr.max_accumulated_bias << "\n"
                  << "latency.target_latency_us:            " << lat.target_latency_us << "\n"
                  << "latency.sample_window:                " << lat.sample_window << "\n"
                  << "latency.enable_profiling:             " << (lat.enable_profiling ? "true" : "false") << "\n"
                  << "logging.log_file_path:                " << log.log_file_path << "\n"
                  << "logging.format:                       " << log.format << "\n"
                  << "logging.enable_console:               " << (log.enable_console ? "true" : "false") << "\n"
                  << "logging.flush_interval_ms:            " << log.flush_interval_ms << "\n"
                  << "metrics.stats_port:                   " << met.stats_port << "\n"
                  << "metrics.bind_address:                 " << met.bind_address << "\n"
                  << "risk_thresholds.max_bias_magnitude:   " << rt.max_bias_magnitude << "\n"
                  << "risk_thresholds.max_volatility_magnitude: " << rt.max_volatility_magnitude << "\n"
                  << "risk_thresholds.min_confidence:       " << rt.min_confidence << "\n"
                  << "risk_thresholds.max_signals_per_second: " << rt.max_signals_per_second << "\n"
                  << "risk_thresholds.max_drawdown:         " << rt.max_drawdown << "\n"
                  << "risk_thresholds.drawdown_window_s:    " << rt.drawdown_window_s << "\n";
        const auto& pr = sys_config.pressure;
        const auto& sw = sys_config.semantic_weights;
        std::cout << "pressure.max_ingestion_rate_tps:      " << pr.max_ingestion_rate_tps << "\n"
                  << "pressure.backoff_scale_factor:        " << pr.backoff_scale_factor << "\n"
                  << "semantic_weights.sentiment_multiplier: " << sw.sentiment_multiplier << "\n"
                  << "semantic_weights.confidence_multiplier: " << sw.confidence_multiplier << "\n"
                  << "semantic_weights.volatility_multiplier: " << sw.volatility_multiplier << "\n"
                  << "semantic_weights.bias_multiplier:     " << sw.bias_multiplier << "\n";
        // Print non-default fields as a diff section for quick operator review.
        auto diffs = config.diff_from_defaults();
        if (!diffs.empty()) {
            std::cout << "\n# Non-default fields (changed from compiled defaults):\n";
            for (const auto& d : diffs)
                std::cout << "  " << d << "\n";
        } else {
            std::cout << "\n# All fields are at compiled defaults.\n";
        }
        return 0;
    }

    // --validate-config: run the validation suite and report errors.
    if (validate_config) {
        auto errors = config.validate();
        if (errors.empty()) {
            std::cout << "Config OK: " << config_file << " is valid.\n";
            return 0;
        }
        std::cerr << "Config INVALID: " << errors.size() << " error(s) in " << config_file << ":\n";
        for (const auto& e : errors)
            std::cerr << "  - " << e << "\n";
        return 1;
    }

#ifdef LLMQUANT_DEDUP_ENABLED
    // Deduplication layer: skip repeated tokens within a sliding TTL window.
    // Dedup TTL: use config value when set (> 0), else default to 10× the token interval.
    const int dedup_ttl_ms = (sys_config.token_stream.dedup_ttl_ms > 0)
        ? sys_config.token_stream.dedup_ttl_ms
        : sys_config.token_stream.token_interval_ms * 10;
    // Backend selection: Redis (distributed) when redis_url is configured and
    // LLMQUANT_ENABLE_REDIS=ON; otherwise fall back to in-process.
    std::shared_ptr<llmquant::DeduplicatorBackend> dedup_backend;
#if defined(LLMQUANT_REDIS_ENABLED)
    if (!sys_config.token_stream.redis_url.empty()) {
        auto redis_backend = std::make_shared<llmquant::RedisDeduplicator>(
            sys_config.token_stream.redis_url);
        if (redis_backend->is_connected()) {
            spdlog::info("[dedup] using Redis backend: {}", sys_config.token_stream.redis_url);
            redis_backend->set_disconnect_callback([](const std::string& err) {
                spdlog::warn("[dedup] Redis disconnected: {}; falling back to in-process", err);
            });
            dedup_backend = std::move(redis_backend);
        } else {
            spdlog::warn("[dedup] Redis connection failed ({}); falling back to in-process",
                         sys_config.token_stream.redis_url);
        }
    }
#endif
    if (!dedup_backend) {
        auto ip = std::make_shared<llmquant::InProcessDeduplicator>();
        // Prevent unbounded memory growth: purge expired entries every 60 s.
        ip->start_background_purge(60);
        dedup_backend = std::move(ip);
    }
    llmquant::Deduplicator deduplicator(dedup_backend,
        std::chrono::milliseconds(dedup_ttl_ms));
#endif

    // Initialize subsystem components.
    MetricsLogger logger({
        .log_file_path = sys_config.logging.log_file_path,
        .format = sys_config.logging.format == "CSV" ?
                 MetricsLogger::OutputFormat::CSV : MetricsLogger::OutputFormat::JSON,
        .enable_console_output = sys_config.logging.enable_console,
        .flush_interval = std::chrono::milliseconds(sys_config.logging.flush_interval_ms)
    });

    LatencyController latency_ctrl({
        .target_latency = std::chrono::microseconds(sys_config.latency.target_latency_us),
        .sample_window = sys_config.latency.sample_window,
        // When compiled with LLMQUANT_ENABLE_PROFILING=OFF the sample ring buffer
        // is always disabled regardless of the YAML/env runtime setting.
#ifdef LLMQUANT_PROFILING_ENABLED
        .enable_profiling = sys_config.latency.enable_profiling
#else
        .enable_profiling = false
#endif
    });

    // Arrival rate tracking for pressure system.
    std::atomic<uint64_t> token_count_window{0};
    // Welford variance accumulators: plain (non-atomic) variables protected exclusively
    // by variance_mutex so readers can never see an inconsistent state between reset
    // and update (Improvement 2 — fix Welford variance race).
    double   sentiment_variance_accum{0.0};
    double   sentiment_mean_accum{0.0};
    uint64_t variance_n{0};
    // Wall-clock timestamp of the last Welford accumulator reset.
    // Reset every 60 seconds to prevent catastrophic cancellation.
    auto variance_last_reset = std::chrono::steady_clock::now();
    // Protects the three Welford variables as a unit: both the token-callback
    // update and the monitoring-loop reset must hold this mutex so they are
    // never interleaved, which would silently corrupt the variance estimate.
    std::mutex            variance_mutex;

    LLMAdapter llm_adapter;

    // --list-tokens: dump the full semantic dictionary and exit immediately.
    if (list_tokens) {
        auto keys = llm_adapter.get_all_token_keys();
        std::cout << "token\tsentiment\tconfidence\tvolatility\tbias\n";
        for (const auto& k : keys) {
            SemanticWeight w;
            (void)llm_adapter.get_token_mapping(k, w);
            std::cout << k
                      << "\t" << std::fixed << std::setprecision(3) << w.sentiment_score
                      << "\t" << w.confidence_score
                      << "\t" << w.volatility_score
                      << "\t" << w.directional_bias
                      << "\n";
        }
        std::cout << "-- " << keys.size() << " entries --\n";
        return 0;
    }

    // --test-replay FILE: replay a JSONL token file through the pipeline.
    if (!replay_input_path.empty()) {
        llmquant::SignalReplayConfig replay_cfg;
        replay_cfg.input_path   = replay_input_path;
        replay_cfg.output_path  = replay_output_path;
        replay_cfg.verbose      = replay_verbose;

        llmquant::SignalReplayRunner runner(replay_cfg);
        auto result = runner.run();

        if (!result.success) {
            std::cerr << "[error] Replay failed: " << result.error_message << "\n";
            return 1;
        }
        std::cerr << "[replay] Done: "
                  << result.tokens_processed << " tokens, "
                  << result.signals_emitted  << " signals emitted "
                  << "(efficiency=" << (result.efficiency * 100.0) << "%)\n";
        return 0;
    }

    // --websocket: start the WebSocket live feed server.
    if (websocket_mode) {
        llmquant::WebSocketServerConfig ws_cfg;
        ws_cfg.host         = ws_host;
        ws_cfg.port         = ws_port;
        ws_cfg.max_sessions = ws_max_sessions;

        llmquant::WebSocketServer ws_server(ws_cfg);
        ws_server.set_log_callback([](const std::string& msg) {
            std::cerr << "[ws] " << msg << "\n";
        });
        ws_server.start();

        std::cerr << "[ws] WebSocket server running on "
                  << ws_cfg.host << ":" << ws_cfg.port
                  << " (max_sessions=" << ws_cfg.max_sessions << ")\n"
                  << "[ws] Press Ctrl-C to stop.\n";

        // Block until interrupted.
        while (g_running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        ws_server.stop();
        std::cerr << "[ws] Server stopped.\n";
        return 0;
    }

    // --export-dict FILE: write the semantic dictionary to a TSV file and exit.
    if (!export_dict_path.empty()) {
        std::string tsv = llm_adapter.export_dictionary();
        std::ofstream out(export_dict_path);
        if (!out) {
            spdlog::error("--export-dict: cannot open '{}' for writing", export_dict_path);
            return 1;
        }
        out << tsv;
        std::cout << "Exported " << llm_adapter.get_dictionary_size()
                  << " entries to " << export_dict_path << "\n";
        return 0;
    }

    TradeSignalEngine trade_engine({
        .bias_sensitivity     = sys_config.trading.bias_sensitivity,
        .volatility_sensitivity = sys_config.trading.volatility_sensitivity,
        .signal_decay_rate    = sys_config.trading.signal_decay_rate,
        .signal_cooldown      = std::chrono::microseconds(sys_config.trading.signal_cooldown_us),
        .max_signal_age_us    = sys_config.trading.max_signal_age_us,
        .min_bias_threshold   = sys_config.trading.min_bias_threshold,
        .min_vol_threshold    = sys_config.trading.min_vol_threshold,
        .max_accumulated_bias = sys_config.trading.max_accumulated_bias
    });

    // Backtest mode: emit on every token, ignoring the cooldown timer.
    if (backtest_mode) {
        trade_engine.set_backtest_mode(true);
    }

    // Wire an in-memory sink for telemetry (signals accessible for inspection/export).
    auto memory_sink = std::make_shared<llmquant::MemoryOutputSink>();
    trade_engine.add_output_sink(memory_sink);

    // Semantic weight multipliers as atomics so the hot-reload callback can update
    // them without a mutex on the process_token hot path.
    std::atomic<double> sem_mult_sentiment{sys_config.semantic_weights.sentiment_multiplier};
    std::atomic<double> sem_mult_confidence{sys_config.semantic_weights.confidence_multiplier};
    std::atomic<double> sem_mult_volatility{sys_config.semantic_weights.volatility_multiplier};
    std::atomic<double> sem_mult_bias{sys_config.semantic_weights.bias_multiplier};

    // Risk manager — thresholds driven from config (hot-reloadable via YAML).
    const auto& rt = sys_config.risk_thresholds;
    llmquant::RiskManager::Config risk_cfg;
    risk_cfg.max_bias_magnitude       = rt.max_bias_magnitude;
    risk_cfg.max_volatility_magnitude = rt.max_volatility_magnitude;
    risk_cfg.max_spread_magnitude     = rt.max_spread_magnitude;
    risk_cfg.min_confidence           = rt.min_confidence;
    risk_cfg.max_signals_per_second   = rt.max_signals_per_second;
    risk_cfg.max_drawdown             = rt.max_drawdown;
    risk_cfg.drawdown_window          = std::chrono::seconds(rt.drawdown_window_s);
    risk_cfg.position_warn_fraction   = rt.position_warn_fraction;
    risk_cfg.disable_magnitude_gate   = sys_config.risk_overrides.disable_magnitude_gate;
    risk_cfg.disable_confidence_gate  = sys_config.risk_overrides.disable_confidence_gate;
    risk_cfg.disable_rate_gate        = sys_config.risk_overrides.disable_rate_gate;
    risk_cfg.disable_drawdown_gate    = sys_config.risk_overrides.disable_drawdown_gate;
    risk_cfg.disable_position_gate    = sys_config.risk_overrides.disable_position_gate;
    risk_cfg.dry_run_mode             = sys_config.risk_overrides.dry_run_mode;
    llmquant::RiskManager risk_mgr(risk_cfg);
    risk_mgr.set_metrics_logger(&logger);
    // Register gate trip-wire callbacks for real-time alerting on first block.
    // Each callback fires once per pass→block edge; subsequent consecutive blocks
    // on the same gate are silent until the gate passes and trips again.
    for (const char* gate : {"magnitude", "confidence", "rate", "drawdown", "position"}) {
        risk_mgr.set_gate_trip_callback(gate, [gate](const std::string& /*g*/,
                                                      const llmquant::TradeSignal& sig) {
            spdlog::warn("[risk] Gate '{}' tripped: bias={:+.3f} vol={:.3f} conf={:.3f}",
                         gate, sig.delta_bias_shift, sig.volatility_adjustment, sig.confidence);
        });
    }

    // Hot-reload watcher is started after token_sim is constructed (below) so
    // the callback can also reload the token file when data_file_path changes.

    // OMS adapter: MockOmsAdapter by default; REST via --oms, FIX 4.2 via --fix.
    std::unique_ptr<llmquant::OmsAdapter> oms_adapter;
    if (!fix_address.empty()) {
#ifdef LLMQUANT_FIX_OMS_ENABLED
        llmquant::FixOmsAdapter::Config fix_cfg;
        size_t colon = fix_address.find(':');
        if (colon != std::string::npos) {
            fix_cfg.host = fix_address.substr(0, colon);
            try {
                int p = std::stoi(fix_address.substr(colon + 1));
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                fix_cfg.port = static_cast<uint16_t>(p);
            }
            catch (...) { spdlog::error("--fix: invalid port in '{}' (must be 1-65535)", fix_address); return 1; }
        } else {
            fix_cfg.host = fix_address;
        }
        oms_adapter = std::make_unique<llmquant::FixOmsAdapter>(fix_cfg);
#else
        spdlog::error("--fix requested but FIX OMS support was disabled at build time "
                      "(LLMQUANT_ENABLE_FIX_OMS=OFF).");
        return 1;
#endif
    } else if (!oms_address.empty()) {
#ifdef LLMQUANT_REST_OMS_ENABLED
        std::string endpoint = oms_address;
        llmquant::RestOmsAdapter::Config oms_cfg;
        size_t colon = endpoint.find(':');
        if (colon != std::string::npos) {
            oms_cfg.host = endpoint.substr(0, colon);
            try {
                int p = std::stoi(endpoint.substr(colon + 1));
                if (p <= 0 || p > 65535) throw std::out_of_range("port");
                oms_cfg.port = static_cast<uint16_t>(p);
            }
            catch (...) { spdlog::error("--oms: invalid port in '{}' (must be 1-65535)", endpoint); return 1; }
        } else {
            oms_cfg.host = endpoint;
        }
        oms_adapter = std::make_unique<llmquant::RestOmsAdapter>(oms_cfg);
#else
        spdlog::error("--oms requested but REST OMS support was disabled at build time "
                      "(LLMQUANT_ENABLE_REST_OMS=OFF).");
        return 1;
#endif
    } else {
#ifdef LLMQUANT_MOCK_OMS_ENABLED
        auto mock = std::make_unique<llmquant::MockOmsAdapter>();
        mock->load_states({
            {0.1,  1.0,  0.5, -10.0},
            {0.25, 1.0,  0.3, -10.0},
            {-0.1, 1.0, -0.2, -10.0},
        });
        oms_adapter = std::move(mock);
#else
        spdlog::error("No OMS adapter specified and MockOmsAdapter is disabled "
                      "(LLMQUANT_ENABLE_MOCK_OMS=OFF). Pass --oms or --fix, or enable "
                      "LLMQUANT_ENABLE_MOCK_OMS=ON.");
        return 1;
#endif
    }

    oms_adapter->set_position_callback([&](const llmquant::RiskManager::PositionState& state) {
        risk_mgr.update_position(state);
    });
    // OMS alert callback wired after signal callback is registered (see below).
    if (!oms_adapter->start())
        spdlog::warn("[oms] start() returned false — adapter may not be polling");

    TokenStreamSimulator token_sim({
        .token_interval = std::chrono::microseconds(sys_config.token_stream.token_interval_ms * 1000),
        .buffer_size = sys_config.token_stream.buffer_size,
        .use_memory_stream = sys_config.token_stream.use_memory_stream,
        .data_file_path = sys_config.token_stream.data_file_path
    });

    // Start config hot-reload watcher now that all pipeline objects exist.
    // The callback can update every subsystem live, including reloading the
    // token file when token_stream.data_file_path changes at runtime.
    // Disabled when --no-hot-reload is passed OR when hot-reload was compiled
    // out with -DLLMQUANT_ENABLE_HOT_RELOAD=OFF (useful in CI/embedded contexts).
#ifndef LLMQUANT_HOT_RELOAD_ENABLED
    (void)no_hot_reload;  // suppress unused-variable warning
    spdlog::info("Config hot-reload watcher compiled out (LLMQUANT_ENABLE_HOT_RELOAD=OFF)");
    if (false) {
#else
    // prev_hot_config: snapshot of the last-seen config used to diff-log what changed on reload.
    llmquant::SystemConfig prev_hot_config = sys_config;
    if (no_hot_reload) {
        spdlog::info("--no-hot-reload: config file watcher disabled");
    } else if (!config.start_watching(config_file, [&risk_mgr, &trade_engine, &token_sim,
                                              &logger, &config_file,
                                              &sem_mult_sentiment, &sem_mult_confidence,
                                              &sem_mult_volatility, &sem_mult_bias,
                                              &prev_hot_config](const llmquant::SystemConfig& updated) {
        // Config diff: log only fields that changed so operators can see exactly what hot-reload applied.
        {
            const auto& o = prev_hot_config;
            const auto& n = updated;
            auto log_ch = [](const char* key, auto ov, auto nv) {
                if (ov != nv) spdlog::info("[config_diff] {}: {} → {}", key, ov, nv);
            };
            log_ch("trading.bias_sensitivity",       o.trading.bias_sensitivity,       n.trading.bias_sensitivity);
            log_ch("trading.volatility_sensitivity", o.trading.volatility_sensitivity, n.trading.volatility_sensitivity);
            log_ch("trading.signal_decay_rate",      o.trading.signal_decay_rate,      n.trading.signal_decay_rate);
            log_ch("trading.signal_cooldown_us",     o.trading.signal_cooldown_us,     n.trading.signal_cooldown_us);
            log_ch("trading.min_bias_threshold",     o.trading.min_bias_threshold,     n.trading.min_bias_threshold);
            log_ch("trading.min_vol_threshold",      o.trading.min_vol_threshold,      n.trading.min_vol_threshold);
            log_ch("trading.max_accumulated_bias",   o.trading.max_accumulated_bias,   n.trading.max_accumulated_bias);
            log_ch("risk.max_bias_magnitude",        o.risk_thresholds.max_bias_magnitude,       n.risk_thresholds.max_bias_magnitude);
            log_ch("risk.max_volatility_magnitude",  o.risk_thresholds.max_volatility_magnitude, n.risk_thresholds.max_volatility_magnitude);
            log_ch("risk.min_confidence",            o.risk_thresholds.min_confidence,           n.risk_thresholds.min_confidence);
            log_ch("risk.max_signals_per_second",    o.risk_thresholds.max_signals_per_second,   n.risk_thresholds.max_signals_per_second);
            log_ch("risk.max_drawdown",              o.risk_thresholds.max_drawdown,             n.risk_thresholds.max_drawdown);
            log_ch("semantic.sentiment_multiplier",  o.semantic_weights.sentiment_multiplier,    n.semantic_weights.sentiment_multiplier);
            log_ch("semantic.confidence_multiplier", o.semantic_weights.confidence_multiplier,   n.semantic_weights.confidence_multiplier);
            log_ch("semantic.volatility_multiplier", o.semantic_weights.volatility_multiplier,   n.semantic_weights.volatility_multiplier);
            log_ch("semantic.bias_multiplier",       o.semantic_weights.bias_multiplier,         n.semantic_weights.bias_multiplier);
            log_ch("latency.target_latency_us",      o.latency.target_latency_us,                n.latency.target_latency_us);
            prev_hot_config = updated;
        }
        const auto& u = updated.risk_thresholds;
        llmquant::RiskManager::Config new_risk_cfg;
        new_risk_cfg.max_bias_magnitude       = u.max_bias_magnitude;
        new_risk_cfg.max_volatility_magnitude = u.max_volatility_magnitude;
        new_risk_cfg.max_spread_magnitude     = u.max_spread_magnitude;
        new_risk_cfg.min_confidence           = u.min_confidence;
        new_risk_cfg.max_signals_per_second   = u.max_signals_per_second;
        new_risk_cfg.max_drawdown             = u.max_drawdown;
        new_risk_cfg.drawdown_window          = std::chrono::seconds(u.drawdown_window_s);
        new_risk_cfg.position_warn_fraction   = u.position_warn_fraction;
        new_risk_cfg.disable_magnitude_gate   = updated.risk_overrides.disable_magnitude_gate;
        new_risk_cfg.disable_confidence_gate  = updated.risk_overrides.disable_confidence_gate;
        new_risk_cfg.disable_rate_gate        = updated.risk_overrides.disable_rate_gate;
        new_risk_cfg.disable_drawdown_gate    = updated.risk_overrides.disable_drawdown_gate;
        new_risk_cfg.disable_position_gate    = updated.risk_overrides.disable_position_gate;
        new_risk_cfg.dry_run_mode             = updated.risk_overrides.dry_run_mode;
        risk_mgr.update_config(new_risk_cfg);
        llmquant::TradeSignalEngine::Config new_eng_cfg;
        new_eng_cfg.bias_sensitivity       = updated.trading.bias_sensitivity;
        new_eng_cfg.volatility_sensitivity = updated.trading.volatility_sensitivity;
        new_eng_cfg.signal_decay_rate      = updated.trading.signal_decay_rate;
        new_eng_cfg.signal_cooldown        = std::chrono::microseconds(updated.trading.signal_cooldown_us);
        new_eng_cfg.max_signal_age_us      = updated.trading.max_signal_age_us;
        new_eng_cfg.min_bias_threshold     = updated.trading.min_bias_threshold;
        new_eng_cfg.min_vol_threshold      = updated.trading.min_vol_threshold;
        new_eng_cfg.max_accumulated_bias   = updated.trading.max_accumulated_bias;
        trade_engine.update_config(new_eng_cfg);
        // Update semantic weight multipliers atomically — visible to process_token
        // on the very next token without requiring a process restart.
        const auto& sw = updated.semantic_weights;
        sem_mult_sentiment.store(sw.sentiment_multiplier,  std::memory_order_relaxed);
        sem_mult_confidence.store(sw.confidence_multiplier, std::memory_order_relaxed);
        sem_mult_volatility.store(sw.volatility_multiplier, std::memory_order_relaxed);
        sem_mult_bias.store(sw.bias_multiplier,            std::memory_order_relaxed);
        // Reload token file when data_file_path changes and not in memory-stream mode.
        if (!updated.token_stream.use_memory_stream) {
            token_sim.load_tokens_from_file(updated.token_stream.data_file_path);
            spdlog::info("[config] Token file reloaded: {}", updated.token_stream.data_file_path);
        }
        // Apply token pacing changes immediately so interval tuning takes
        // effect without a restart.
        if (updated.token_stream.token_interval_ms > 0) {
            token_sim.set_token_interval(
                std::chrono::microseconds(updated.token_stream.token_interval_ms * 1000));
        }
        logger.log_config_reload(config_file, true);
        std::cout << "\n[config] Hot-reloaded: bias_sensitivity="
                  << updated.trading.bias_sensitivity
                  << "  max_bias=" << u.max_bias_magnitude
                  << "  max_signals/s=" << u.max_signals_per_second
                  << "  sem_wts=[" << sw.sentiment_multiplier << ","
                  << sw.confidence_multiplier << "," << sw.volatility_multiplier
                  << "," << sw.bias_multiplier << "]" << std::endl;
    })) {
        spdlog::warn("Config hot-reload watcher failed to start");
    }
#endif // LLMQUANT_HOT_RELOAD_ENABLED

#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
    // Rolling Shannon entropy of token type diversity.
    // Declared before process_token lambda so the lambda can capture it by ref.
    llmquant::TokenEntropyMonitor entropy_monitor;
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
    // Narrative change detector: cosine similarity break for topic-switch events.
    llmquant::NarrativeChangeDetector narrative_detector;
    narrative_detector.set_break_callback([](double sim) {
        spdlog::info("[narrative] topic break detected — cosine_sim={:.3f}; "
                     "LLM may have switched themes", sim);
    });
#endif

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
    // Stale-token watchdog: fires if no token arrives for >30 s (configurable).
    // Declared before process_token lambda so the lambda can capture it by ref.
    llmquant::StaleTokenDetector stale_detector;
    stale_detector.reset();
    stale_detector.set_stale_callback([](int64_t gap_ms) {
        spdlog::error("[stale_detector] LLM token stream SILENT for {}ms — "
                      "no tokens received; check upstream API / network",
                      gap_ms);
    });
    stale_detector.set_recovery_callback([]() {
        spdlog::info("[stale_detector] LLM token stream RECOVERED — tokens flowing again");
    });
#endif

    // These must be declared BEFORE the process_token lambda so the lambda can
    // capture them by reference even though they appear under #ifdef guards.
#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
    llmquant::SentimentMomentumFilter sentiment_momentum_filter;
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
    llmquant::SignalDecayEnvelope signal_decay;
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
    llmquant::ContextWindowBudget context_budget;
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
    llmquant::TemporalPatternLibrary tpl;
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
    llmquant::TokenStreamHealthMonitor stream_health;
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
    llmquant::TokenBiasHeatmap token_bias_heatmap;
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
    llmquant::OrderFlowImbalanceDetector order_flow_detector;
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
    llmquant::TokenInfluenceAttributor token_influence;
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
    llmquant::TokenNgramProfiler ngram_profiler;
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
    llmquant::AdversarialInputDetector adversarial_detector;
#endif
#ifdef LLMQUANT_TOKEN_IB_ENABLED
    llmquant::TokenInformationBottleneck token_ib;
#endif
#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
    llmquant::NarrativeTopicClassifier narrative_classifier;
#endif
#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
    llmquant::TokenClockRecalibrator token_clock;
#endif
#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
    llmquant::TokenImportanceDecayScheduler decay_scheduler;
#endif
#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
    llmquant::TokenStreamDifferencer stream_differencer;
#endif
#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
    llmquant::TokenWeightQuantiser token_quantiser;
#endif
#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
    llmquant::TokenFlowPressureGauge flow_pressure;
#endif
#ifdef LLMQUANT_LATENCY_JITTER_ENABLED
    llmquant::LatencyJitterMonitor latency_jitter;
#endif
#ifdef LLMQUANT_SIGNAL_CUSUM_ENABLED
    llmquant::SignalCUSUMController signal_cusum;
#endif

    // Shared token processing lambda used by both the simulator and the
    // LLMStreamClient paths.  Encapsulates dedup, latency, logging, and
    // semantic-weight pipeline so neither call site duplicates logic.
    auto process_token = [&](const std::string& text, uint64_t seq_id) {
#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
        entropy_monitor.record(std::hash<std::string>{}(text));
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
        narrative_detector.record(std::hash<std::string>{}(text));
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
        // Feed raw token text into the phrase matcher; fires on complete patterns.
        tpl.push_token(text);
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
        context_budget.consume(1);
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
        stream_health.ping();
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.record_token();
#endif
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
        spdlog::trace("[trace] token seq={} text={}", seq_id, text);
#endif
#ifdef LLMQUANT_DEDUP_ENABLED
        // Skip duplicate tokens within the dedup window (unless --no-dedup).
        if (!no_dedup) {
            auto dedup_result = deduplicator.check(text);
            logger.log_dedup_event(text, dedup_result == llmquant::DedupResult::Duplicate);
            if (dedup_result == llmquant::DedupResult::Duplicate) {
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
                spdlog::trace("[trace] token seq={} DEDUP_SKIP", seq_id);
#endif
                return;
            }
        }
#endif

        latency_ctrl.start_measurement();

        logger.log_token_received(text, seq_id);

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.record_token();
#endif

        auto weight = llm_adapter.map_token_to_weight(text);

#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
        spdlog::trace("token seq={} text='{}' sent={:.4f} conf={:.4f} vol={:.4f} bias={:.4f}",
                      seq_id, text,
                      weight.sentiment_score, weight.confidence_score,
                      weight.volatility_score, weight.directional_bias);
#endif

        // Apply per-category semantic weight multipliers (hot-reloadable).
        // Read atomically so hot-reload updates are visible without a mutex.
        weight.sentiment_score  *= sem_mult_sentiment.load(std::memory_order_relaxed);
        weight.confidence_score *= sem_mult_confidence.load(std::memory_order_relaxed);
        weight.volatility_score *= sem_mult_volatility.load(std::memory_order_relaxed);
        weight.directional_bias *= sem_mult_bias.load(std::memory_order_relaxed);

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
        // Feed raw sentiment score into the momentum filter's trajectory analyzer.
        sentiment_momentum_filter.record_sample(weight.sentiment_score);
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
        // Reinforce the decay envelope with each token's directional bias.
        signal_decay.reinforce(weight.directional_bias);
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
        // Accumulate per-token signed bias so operators can identify dominant tokens.
        token_bias_heatmap.record(text, weight.directional_bias);
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
        // Also feed raw token text into the keyword dictionary matcher.
        order_flow_detector.record(text);
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
        // Track n-gram frequencies; fires on repeated patterns.
        ngram_profiler.push(text);
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
        // Screen each token for weight anomalies, repetition, and vocab inflation.
        adversarial_detector.inspect(text, weight.directional_bias);
#endif
#ifdef LLMQUANT_TOKEN_IB_ENABLED
        // Accumulate per-token weight sample; bias_shift will be filled in the
        // signal callback — here we store the weight with a placeholder 0.0 bias
        // so per-token relevance can be recalculated once a signal fires.
        token_ib.record(text, weight.directional_bias, 0.0);
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
        // Attribute per-token marginal contribution to the current bias.
        token_influence.record(text, weight.directional_bias);
#endif
#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
        // Classify token into macro narrative topic.
        narrative_classifier.classify(text, weight.directional_bias);
#endif
#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
        // Record token arrival timestamp for rate estimation.
        {
            auto tcr_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            token_clock.record_token(tcr_ns);
        }
#endif
#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
        // Record token weight with current timestamp for time-weighted decay.
        {
            auto tds_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            decay_scheduler.record(text, weight.directional_bias, tds_ns);
        }
#endif
#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
        // Track velocity/acceleration/jerk of the token weight series.
        stream_differencer.record(weight.directional_bias);
#endif
#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
        // Stochastic-round the weight to the fixed grid (side effect: tracks error + clamp rate).
        // The quantised value is not fed back into the weight here; the quantiser is used
        // as a monitoring/diagnostic module for integer-precision feasibility analysis.
        (void)token_quantiser.quantise(weight.directional_bias);
#endif
#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
        // Record token arrival time for inter-token EMA pressure tracking.
        {
            auto fp_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            flow_pressure.record_token(fp_ns);
        }
#endif

        // In dry-run mode, tokens are mapped through LLMAdapter for
        // dictionary coverage analysis but no signals are emitted.
        if (!dry_run) {
            trade_engine.process_semantic_weight(weight);
        }
#ifdef LLMQUANT_SIGNAL_TRACE_ENABLED
        spdlog::trace("signal bias={:.4f} vol={:.4f} latency={}us",
                      trade_engine.get_accumulated_bias(),
                      trade_engine.get_accumulated_volatility(),
                      latency_ctrl.get_stats().avg_latency.count());
#endif

        latency_ctrl.end_measurement();
#ifdef LLMQUANT_LATENCY_JITTER_ENABLED
        // Feed the most recent average latency as a per-token sample.
        // avg_latency is a microsecond-resolution chrono duration.
        latency_jitter.record(
            static_cast<double>(latency_ctrl.get_stats().avg_latency.count()));
#endif

        // Track token arrival for ingestion pressure.
        token_count_window++;

        // Welford online variance for semantic pressure.
        // The mutex ensures this three-variable update is never interleaved
        // with the periodic reset in the monitoring loop.
        double current_variance = 0.0;
        {
            std::lock_guard<std::mutex> lk(variance_mutex);
            double s = weight.sentiment_score;
            ++variance_n;
            uint64_t n = variance_n;
            double delta = s - sentiment_mean_accum;
            sentiment_mean_accum += delta / static_cast<double>(n);
            double delta2 = s - sentiment_mean_accum;
            sentiment_variance_accum += delta * delta2;
            current_variance = (n > 1)
                ? (sentiment_variance_accum / static_cast<double>(n - 1))
                : 0.0;
        }

        // Update pressure (semantic only; ingestion + queue updated in monitoring loop).
        latency_ctrl.update_semantic_pressure(current_variance);
    };

    // Set up simulator callback.
    token_sim.set_token_callback([&](const Token& token) {
        process_token(token.text, token.sequence_id);
    });

    // Shared risk-block reason for display on the same line.
    std::string last_block_reason;
    std::mutex  block_reason_mutex;

#ifdef LLMQUANT_AUDIT_LOG_ENABLED
    std::unique_ptr<llmquant::SignalAuditLog> audit_log;
    if (!audit_log_path.empty()) {
        llmquant::SignalAuditLog::Config audit_cfg;
        audit_cfg.filepath = audit_log_path;
        audit_log = std::make_unique<llmquant::SignalAuditLog>(audit_cfg);
        spdlog::info("[audit_log] started — writing to '{}'", audit_log_path);
    }
#endif

#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
    // Circuit-breaker: auto-pause signal emission when block rate stays
    // above the threshold for the configured sustained window.
    llmquant::PipelineCircuitBreaker circuit_breaker;
    circuit_breaker.set_state_change_callback(
        [](llmquant::PipelineCircuitBreaker::State s, double rate) {
            if (s == llmquant::PipelineCircuitBreaker::State::Open) {
                spdlog::error("[circuit_breaker] pipeline OPEN — {:.0f}% of signals blocked; "
                              "signal emission suppressed until block rate drops",
                              rate * 100.0);
            } else if (s == llmquant::PipelineCircuitBreaker::State::Closed) {
                spdlog::info("[circuit_breaker] pipeline CLOSED — recovered");
            } else {
                spdlog::info("[circuit_breaker] pipeline HALF-OPEN — probing recovery");
            }
        });
#endif

#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
    // Latency budget enforcer: tiered SLA escalation (Normal→Warn→Throttle→Drop→Breaker).
    // When the Breaker tier is reached the pipeline circuit breaker is tripped immediately.
    llmquant::LatencyBudgetEnforcer latency_budget_enforcer;
    latency_budget_enforcer.set_warn_callback([](int64_t p99) {
        spdlog::warn("[lbe] p99={}µs — warn budget exceeded", p99);
    });
    latency_budget_enforcer.set_throttle_callback([](int64_t p99) {
        spdlog::warn("[lbe] p99={}µs — throttle tier: slowing token intake", p99);
    });
    latency_budget_enforcer.set_drop_callback([](int64_t p99) {
        spdlog::error("[lbe] p99={}µs — drop tier: signal emission suspended", p99);
    });
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
    latency_budget_enforcer.set_breaker_callback([&](int64_t p99) {
        spdlog::error("[lbe] p99={}µs — breaker tier: tripping circuit breaker", p99);
        circuit_breaker.force_open();
    });
#else
    latency_budget_enforcer.set_breaker_callback([](int64_t p99) {
        spdlog::error("[lbe] p99={}µs — breaker tier: critical latency", p99);
    });
#endif
    latency_budget_enforcer.set_recovery_callback([](int64_t p99) {
        spdlog::info("[lbe] p99={}µs — recovered to Normal tier", p99);
    });
#endif

#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED
    // P&L attribution: attributes realized trade outcomes to sentiment driver categories.
    llmquant::PnLAttributionEngine pnl_attribution;
#endif

#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED
    // Portfolio heat monitor: aggregates cross-instrument risk heat.
    llmquant::PortfolioHeatMonitor portfolio_heat;
    portfolio_heat.set_warn_callback([](double heat) {
        spdlog::warn("[portfolio_heat] heat={:.2f} — approaching risk budget", heat);
    });
    portfolio_heat.set_critical_callback([](double heat) {
        spdlog::error("[portfolio_heat] heat={:.2f} — critical; shedding risk", heat);
    });
    portfolio_heat.set_recovery_callback([](double heat) {
        spdlog::info("[portfolio_heat] heat={:.2f} — recovered to Cool", heat);
    });
#endif

#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
    // ContextWindowBudget: configure the instance declared before process_token.
    {
        llmquant::ContextWindowBudget::Config cb_cfg;
        cb_cfg.capacity = 128000;  // Claude 3 / GPT-4 128k context
        cb_cfg.on_warn = [](uint64_t used, uint64_t cap) {
            spdlog::warn("[ctx_budget] context warn  used={} / {} ({:.0f}%)",
                         used, cap, 100.0 * static_cast<double>(used) / cap);
        };
        cb_cfg.on_critical = [](uint64_t used, uint64_t cap) {
            spdlog::error("[ctx_budget] context CRITICAL  used={} / {} ({:.0f}%)",
                          used, cap, 100.0 * static_cast<double>(used) / cap);
        };
        cb_cfg.on_overflow = [](uint64_t used, uint64_t cap) {
            spdlog::critical("[ctx_budget] context OVERFLOW used={} cap={} — reset required",
                             used, cap);
        };
        context_budget.update_config(cb_cfg);
    }
#endif

#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
    // FractalDimensionEstimator: tracks Hurst exponent of bias stream (H>0.5=trending).
    llmquant::FractalDimensionEstimator fractal_dim;
    {
        llmquant::FractalDimensionEstimator::Config fd_cfg;
        fd_cfg.on_regime_change = [](double prev_h, double new_h) {
            const char* prev_r = (prev_h > 0.55) ? "trending" : (prev_h < 0.45 ? "mean-rev" : "random");
            const char* new_r  = (new_h  > 0.55) ? "trending" : (new_h  < 0.45 ? "mean-rev" : "random");
            spdlog::info("[fractal] Hurst {:.3f}→{:.3f}  {} → {}", prev_h, new_h, prev_r, new_r);
        };
        fractal_dim.update_config(fd_cfg);
    }
#endif

#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
    // MarketMicrostructureFilter: gates signals whose predicted edge < bid-ask + impact cost.
    llmquant::MarketMicrostructureFilter microstructure_filter;
#endif

#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
    // SignalEnsembleLayer: combines bias/vol/confidence sub-signals with online weight learning.
    llmquant::SignalEnsembleLayer signal_ensemble;
    int ens_bias_id = -1, ens_vol_id = -1, ens_conf_id = -1;
    {
        ens_bias_id = signal_ensemble.register_source("bias");
        ens_vol_id  = signal_ensemble.register_source("vol_adj");
        ens_conf_id = signal_ensemble.register_source("confidence");
        llmquant::SignalEnsembleLayer::Config ec;
        ec.on_weight_update = [](const std::vector<double>& w) {
            if (w.size() >= 3)
                spdlog::debug("[ensemble] weights  bias={:.3f}  vol={:.3f}  conf={:.3f}",
                              w[0], w[1], w[2]);
        };
        signal_ensemble.update_config(ec);
    }
#endif

#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
    // SignalMomentumOscillator: MACD-style oscillator on the bias stream.
    // Fires on_cross when the histogram crosses zero (directional momentum shift).
    llmquant::SignalMomentumOscillator signal_momentum_osc;
    {
        llmquant::SignalMomentumOscillator::Config smo_cfg;
        smo_cfg.divergence_threshold = 0.02;
        smo_cfg.on_cross = [](llmquant::SignalMomentumOscillator::CrossDirection dir, double hist) {
            spdlog::info("[smo] histogram zero-cross  dir={}  hist={:.5f}",
                         (dir == llmquant::SignalMomentumOscillator::CrossDirection::Bullish)
                             ? "BULLISH" : "BEARISH",
                         hist);
        };
        smo_cfg.on_divergence = [](double macd, double sig, double hist) {
            spdlog::debug("[smo] divergence  macd={:.5f}  signal={:.5f}  hist={:.5f}",
                          macd, sig, hist);
        };
        signal_momentum_osc.update_config(smo_cfg);
    }
#endif

#ifdef LLMQUANT_CVAR_ENABLED
    // CVaRCalculator: rolling Expected Shortfall at α=0.95 for tail-risk gating.
    // Records delta_bias_shift as a proxy PnL; fires on_breach when CVaR < -5%.
    llmquant::CVaRCalculator cvar_calc;
    {
        llmquant::CVaRCalculator::Config cv_cfg;
        cv_cfg.breach_threshold = -0.05;
        cv_cfg.on_breach = [](double cvar, double var, double alpha) {
            spdlog::warn("[cvar] tail-risk breach  cvar={:.4f}  var={:.4f}  alpha={:.2f}",
                         cvar, var, alpha);
        };
        cvar_calc.update_config(cv_cfg);
    }
#endif

#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
    // TemporalPatternLibrary: configure the instance declared before process_token.
    {
        tpl.register_pattern("earnings_beat",    {"earnings", "beat"},      0.6);
        tpl.register_pattern("earnings_miss",    {"earnings", "miss"},     -0.6);
        tpl.register_pattern("rate_hike",        {"rate", "hike"},         -0.4);
        tpl.register_pattern("rate_cut",         {"rate", "cut"},           0.4);
        tpl.register_pattern("guidance_raised",  {"guidance", "raised"},    0.5);
        tpl.register_pattern("guidance_lowered", {"guidance", "lowered"},  -0.5);
        tpl.register_pattern("short_squeeze",    {"short", "squeeze"},      0.7);
        tpl.register_pattern("margin_call",      {"margin", "call"},       -0.8);
    }
#endif

#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
    // FeedbackLoopDetector: cross-correlation reflexivity trap detector.
    // Warns when the system's own trades appear to drive the LLM sentiment signal.
    llmquant::FeedbackLoopDetector feedback_detector;
    {
        llmquant::FeedbackLoopDetector::Config fb_cfg;
        fb_cfg.threshold  = 0.65;
        fb_cfg.on_feedback = [](double score, int lag) {
            spdlog::warn("[feedback] reflexivity suspected  score={:.3f}  peak_lag={}", score, lag);
        };
        feedback_detector.update_config(fb_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
    // SentimentCycleDetector: detects periodic news-cycle patterns via ACF analysis.
    llmquant::SentimentCycleDetector sentiment_cycle;
    {
        llmquant::SentimentCycleDetector::Config sc_cfg;
        sc_cfg.window_size       = 256;
        sc_cfg.max_lag           = 64;
        sc_cfg.cyclic_threshold  = 0.35;
        sc_cfg.on_period_change  = [](int new_p, int old_p, double strength) {
            spdlog::info("[cycle] dominant period {}→{}  acf={:.3f}", old_p, new_p, strength);
        };
        sentiment_cycle.update_config(sc_cfg);
    }
#endif

#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
    // AdaptiveSamplingController: shrinks poll interval on high activity,
    // grows it on quiet periods to reduce wasteful LLM API calls.
    llmquant::AdaptiveSamplingController adaptive_sampler;
    {
        llmquant::AdaptiveSamplingController::Config as_cfg;
        as_cfg.min_interval_ms  = 10;
        as_cfg.max_interval_ms  = 2000;
        as_cfg.initial_interval_ms = 100;
        as_cfg.on_interval_change = [](int64_t new_ms) {
            spdlog::debug("[sampler] poll interval → {} ms", new_ms);
        };
        adaptive_sampler.update_config(as_cfg);
    }
#endif

#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
    // MutualInformationEstimator: captures non-linear sentiment→return dependency.
    llmquant::MutualInformationEstimator mi_estimator;
    // record(sentiment, return) is called from the OMS PnL callback below.
#endif

#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
    // SignalBlindSpotDetector: flags calendar slots with poor historical win rate.
    llmquant::SignalBlindSpotDetector blind_spot;
    {
        llmquant::SignalBlindSpotDetector::Config bs_cfg;
        bs_cfg.min_samples         = 10;
        bs_cfg.blind_spot_threshold = 0.4;
        bs_cfg.on_blind_spot_found = [](int slot, double wr) {
            spdlog::warn("[blind_spot] hour {} flagged — win_rate={:.2f}", slot, wr);
        };
        blind_spot.update_config(bs_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
    // SignalSurpriseIndex: flags signals that are statistically anomalous
    // relative to the engine's own learned distribution (self-information).
    llmquant::SignalSurpriseIndex signal_surprise;
    {
        llmquant::SignalSurpriseIndex::Config ss_cfg;
        ss_cfg.min_samples             = 50;
        ss_cfg.high_surprise_threshold = 0.80;
        ss_cfg.on_high_surprise = [](double bias, double score) {
            spdlog::warn("[surprise] HIGH I(x)={:.3f}  bias={:.5f} — anomalous signal",
                         score, bias);
        };
        signal_surprise.update_config(ss_cfg);
    }
#endif

    // TokenStreamHealthMonitor: watchdog for feed stalls and token floods.
    {
        llmquant::TokenStreamHealthMonitor::Config sh_cfg;
        sh_cfg.stall_timeout_ms   = 3000;
        sh_cfg.max_tokens_per_sec = 1000.0;
        sh_cfg.on_stall = [](int64_t elapsed_ms) {
            spdlog::warn("[stream_health] STALL detected — {}ms since last token", elapsed_ms);
        };
        sh_cfg.on_flood = [](double rate) {
            spdlog::warn("[stream_health] FLOOD {:.0f} tok/s — exceeds threshold", rate);
        };
        sh_cfg.on_recovery = [] {
            spdlog::info("[stream_health] stream recovered to Healthy state");
        };
        stream_health.update_config(sh_cfg);
    }

#ifdef LLMQUANT_REGIME_SIZER_ENABLED
    // RegimeAwareSizer: scales notional by Hurst exponent × vol-targeting factor.
    // update_hurst() and update_vol() should be called after each signal.
    llmquant::RegimeAwareSizer regime_sizer;
    {
        llmquant::RegimeAwareSizer::Config rs_cfg;
        rs_cfg.target_vol = 0.20;
        rs_cfg.on_size_change = [](double nw, double old) {
            spdlog::info("[regime_sizer] multiplier {:.3f}→{:.3f}", old, nw);
        };
        regime_sizer.update_config(rs_cfg);
    }
#endif

#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
    // ConfidenceDecayTracker: fits exponential decay to signal.confidence over time.
    // Slow decay → news-driven; fast decay → noise spike.
    llmquant::ConfidenceDecayTracker conf_decay;
    {
        llmquant::ConfidenceDecayTracker::Config cd_cfg;
        cd_cfg.fast_decay_threshold_ms = 500.0;
        cd_cfg.on_decay_change = [](double nw, double old) {
            spdlog::info("[conf_decay] half-life {:.0f}ms→{:.0f}ms", old, nw);
        };
        conf_decay.update_config(std::move(cd_cfg));
    }
#endif

#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
    // CrossAssetCorrelationMonitor: rolling Pearson correlation between bias and vol.
    // Register "bias" and "vol" as two virtual "assets" to track their co-movement.
    llmquant::CrossAssetCorrelationMonitor cross_asset_corr;
    {
        llmquant::CrossAssetCorrelationMonitor::Config ca_cfg;
        ca_cfg.on_high_correlation = [](const std::string& a, const std::string& b, double rho) {
            spdlog::info("[cross_asset] high correlation  {}<>{} rho={:.3f}", a, b, rho);
        };
        ca_cfg.on_low_correlation  = [](const std::string& a, const std::string& b, double rho) {
            spdlog::info("[cross_asset] low correlation   {}<>{} rho={:.3f}", a, b, rho);
        };
        cross_asset_corr.update_config(std::move(ca_cfg));
        cross_asset_corr.register_asset("bias");
        cross_asset_corr.register_asset("vol");
        cross_asset_corr.register_asset("confidence");
    }
#endif

#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
    // TokenVelocityTracker: measures first and second time-derivatives of bias.
    // High velocity = rapid sentiment shift; high acceleration = regime change.
    llmquant::TokenVelocityTracker velocity_tracker;
    {
        llmquant::TokenVelocityTracker::Config vt_cfg;
        vt_cfg.window_size         = 16;
        vt_cfg.fast_move_threshold = 0.5;
        vt_cfg.on_fast_move = [](double vel, double accel) {
            spdlog::warn("[velocity] fast move  vel={:.4f}  accel={:.4f}", vel, accel);
        };
        velocity_tracker.update_config(vt_cfg);
    }
#endif

#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
    // NarrativeMomentumClock: four-quadrant investment-clock on smoothed bias.
    // Rising(Q1)/Fading(Q2)/Falling(Q3)/Recovering(Q4) — fires on rotation.
    llmquant::NarrativeMomentumClock narrative_clock;
    {
        llmquant::NarrativeMomentumClock::Config nc_cfg;
        nc_cfg.bias_alpha     = 0.10;
        nc_cfg.velocity_alpha = 0.20;
        nc_cfg.on_quadrant_change = [](llmquant::NarrativeMomentumClock::Quadrant from,
                                       llmquant::NarrativeMomentumClock::Quadrant to,
                                       double b, double v) {
            static const char* names[] = {"Rising", "Fading", "Falling", "Recovering"};
            spdlog::info("[clock] {} → {}  bias={:.4f}  vel={:.5f}",
                         names[static_cast<int>(from)], names[static_cast<int>(to)], b, v);
        };
        narrative_clock.update_config(nc_cfg);
    }
#endif

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
    // AdaptiveVelocityBreaker: EMA-smoothed circuit-breaker that trips when
    // bias changes too rapidly, guarding against reflexive LLM feedback loops.
    llmquant::AdaptiveVelocityBreaker velocity_breaker;
    {
        llmquant::AdaptiveVelocityBreaker::Config vb_cfg;
        vb_cfg.trip_threshold   = 10.0;
        vb_cfg.recovery_factor  = 0.5;
        vb_cfg.velocity_alpha   = 0.3;
        vb_cfg.on_trip = [](double vel) {
            spdlog::warn("[vel_breaker] TRIPPED  smoothed_vel={:.4f}", vel);
        };
        vb_cfg.on_recovery = [](double vel) {
            spdlog::info("[vel_breaker] RECOVERED  smoothed_vel={:.4f}", vel);
        };
        velocity_breaker.update_config(vb_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
    // SignalCalibrationEngine: online Platt-scaling logistic calibration that
    // maps raw confidence scores to well-calibrated P(win) probabilities.
    llmquant::SignalCalibrationEngine signal_calibration;
#endif

#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
    {
        llmquant::OrderFlowImbalanceDetector::Config of_cfg;
        of_cfg.ema_alpha            = 0.1;
        of_cfg.imbalance_threshold  = 0.6;
        of_cfg.on_imbalance = [](double imb, bool buy) {
            spdlog::info("[order_flow] imbalance={:.3f}  side={}", imb, buy ? "BUY" : "SELL");
        };
        order_flow_detector.update_config(of_cfg);
    }
#endif

#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
    // CrossSessionMemory: persist Kelly / drawdown state across restarts
    // so the engine warms up instantly rather than starting cold.
    llmquant::CrossSessionMemory cross_session_mem;
    {
        llmquant::CrossSessionMemory::Config csm_cfg;
        csm_cfg.ignore_missing = true;
        csm_cfg.on_load = [](const std::string& path, uint64_t sess) {
            spdlog::info("[cross_session] loaded session={} from {}", sess, path);
        };
        csm_cfg.on_save = [](const std::string& path, uint64_t sess) {
            spdlog::info("[cross_session] saved  session={} to {}", sess, path);
        };
        cross_session_mem.update_config(csm_cfg);
        cross_session_mem.load();  // warm start
    }
#endif

#ifdef LLMQUANT_REGIME_PROB_ENABLED
    // MarketRegimeProbabilityEstimator: online 2-state HMM Bayesian filter
    // that produces a soft probability distribution over risk-on/risk-off regimes.
    llmquant::MarketRegimeProbabilityEstimator regime_prob_est;
    {
        llmquant::MarketRegimeProbabilityEstimator::Config rp_cfg;
        rp_cfg.min_observations = 20;
        rp_cfg.transition_threshold = 0.70;
        rp_cfg.on_regime_change = [](double prob_on, bool is_on) {
            spdlog::info("[regime_hmm] {} → p_risk_on={:.4f}",
                         is_on ? "RISK-ON" : "RISK-OFF", prob_on);
        };
        regime_prob_est.update_config(rp_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
    // SignalReplayBuffer: retains the last 1024 signals for replay and
    // post-hoc analysis without needing to re-run the full token stream.
    llmquant::SignalReplayBuffer signal_replay;
#endif

#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
    // TokenNgramProfiler: config — tracks 2-gram and 3-gram frequencies;
    // fires on hot n-grams that may indicate adversarial injection or stuck LLM output.
    {
        llmquant::TokenNgramProfiler::Config ng_cfg;
        ng_cfg.hot_threshold = 10;
        ng_cfg.on_hot_ngram = [](const std::string& ng, uint64_t cnt, int n) {
            spdlog::warn("[ngram] hot {}-gram \"{}\" count={}", n, ng, cnt);
        };
        ngram_profiler.update_config(ng_cfg);
    }
#endif

#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
    // ExecutionQualityMonitor: tracks fill latency and slippage between signal
    // emission and OMS acknowledgment; fires on SLA breaches.
    llmquant::ExecutionQualityMonitor exec_quality;
    {
        llmquant::ExecutionQualityMonitor::Config eq_cfg;
        eq_cfg.latency_sla_us   = 5000.0;  // 5 ms SLA
        eq_cfg.slippage_sla_bps = 5.0;
        eq_cfg.on_sla_breach = [](const llmquant::ExecutionQualityMonitor::FillRecord& f,
                                  bool lat_breach, bool slip_breach) {
            spdlog::warn("[exec_quality] SLA breach sig={} lat={:.0f}us slip={:.2f}bps "
                         "lat_breach={} slip_breach={}",
                         f.signal_id, f.latency_us, f.slippage_bps,
                         lat_breach, slip_breach);
        };
        exec_quality.update_config(eq_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
    // SentimentDispersionIndex: measures incoherence across bias / vol / confidence
    // streams via coefficient of variation; fires on high/low dispersion events.
    llmquant::SentimentDispersionIndex sentiment_dispersion;
    {
        llmquant::SentimentDispersionIndex::Config sd_cfg;
        sd_cfg.high_threshold = 0.8;
        sd_cfg.low_threshold  = 0.2;
        sd_cfg.on_high_dispersion = [](double sdi) {
            spdlog::warn("[dispersion] HIGH sdi={:.4f} — signals incoherent", sdi);
        };
        sd_cfg.on_low_dispersion = [](double sdi) {
            spdlog::info("[dispersion] COHERENT sdi={:.4f}", sdi);
        };
        sentiment_dispersion.update_config(sd_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
    // SentimentDivergenceDetector: pairwise EMA divergence across bias/vol/conf.
    // Fires when any pair's |ema_a - ema_b| exceeds the threshold.
    llmquant::SentimentDivergenceDetector sentiment_divergence;
    {
        llmquant::SentimentDivergenceDetector::Config svd_cfg;
        svd_cfg.divergence_threshold  = 0.4;
        svd_cfg.recovery_hysteresis   = 0.7;
        svd_cfg.on_divergence = [](double d, const std::string& a, const std::string& b) {
            spdlog::warn("[divergence] {} <> {} diverge={:.4f}", a, b, d);
        };
        svd_cfg.on_recovery = [](double d) {
            spdlog::info("[divergence] recovered  diverge={:.4f}", d);
        };
        sentiment_divergence.update_config(svd_cfg);
    }
#endif

#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
    // TokenInfluenceAttributor: Shapley-inspired per-token marginal attribution.
    // Declared in the pre-lambda block; configure here.
    {
        llmquant::TokenInfluenceAttributor::Config ti_cfg;
        ti_cfg.window_size = 64;
        ti_cfg.top_k       = 5;
        token_influence.update_config(ti_cfg);
    }
#endif

#ifdef LLMQUANT_WALK_FORWARD_ENABLED
    // WalkForwardValidator: rolling OOS validation — offline/diagnostic tool.
    // Tokens must be loaded via load_tokens() before run() can be called.
    llmquant::WalkForwardValidator::Config wf_cfg;
    wf_cfg.train_size = 200;
    wf_cfg.test_size  = 50;
    wf_cfg.step_size  = 50;
    wf_cfg.optimize   = false;  // skip parameter sweep in live mode
    llmquant::WalkForwardValidator walk_forward(wf_cfg);
#endif

#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
    // AdversarialInputDetector: monitors token stream for weight anomalies,
    // repetition attacks, and vocabulary inflation in real time.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::AdversarialInputDetector::Config ad_cfg;
        ad_cfg.anomaly_threshold    = 4.0;
        ad_cfg.min_warmup_tokens    = 30;
        ad_cfg.max_repeat_fraction  = 0.6;
        ad_cfg.max_novel_fraction   = 0.8;
        ad_cfg.on_anomaly = [](llmquant::AdversarialInputDetector::AnomalyKind kind,
                               const std::string& token, double score) {
            const char* k =
                kind == llmquant::AdversarialInputDetector::AnomalyKind::WeightAnomaly
                ? "weight"
                : kind == llmquant::AdversarialInputDetector::AnomalyKind::RepetitionAttack
                  ? "repetition" : "vocab_inflation";
            spdlog::warn("[adversarial] {} token=\"{}\" score={:.3f}", k, token, score);
        };
        adversarial_detector.update_config(ad_cfg);
    }
#endif

#ifdef LLMQUANT_TOKEN_IB_ENABLED
    // TokenInformationBottleneck: per-token relevance/complexity IB score.
    // Declared in pre-lambda block; configure here.
    {
        llmquant::TokenInformationBottleneck::Config tib_cfg;
        tib_cfg.window_size      = 64;
        tib_cfg.min_observations = 8;
        tib_cfg.prune_threshold  = 0.05;
        tib_cfg.top_k            = 10;
        tib_cfg.on_low_score = [](const std::string& tok, double sc) {
            spdlog::debug("[token_ib] LOW score token=\"{}\" ib={:.4f}", tok, sc);
        };
        tib_cfg.on_recovered = [](const std::string& tok, double sc) {
            spdlog::debug("[token_ib] RECOVERED token=\"{}\" ib={:.4f}", tok, sc);
        };
        token_ib.update_config(tib_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_CI_ENABLED
    // SignalConfidenceInterval: jackknife CI on rolling signal window; narrow
    // CI = high-confidence environment, wide CI = noisy / uncertain signals.
    llmquant::SignalConfidenceInterval signal_ci;
    {
        llmquant::SignalConfidenceInterval::Config ci_cfg;
        ci_cfg.window_size      = 64;
        ci_cfg.z                = 1.96;
        ci_cfg.narrow_threshold = 0.05;
        ci_cfg.wide_threshold   = 0.30;
        ci_cfg.on_narrow_interval = [](double mean, double hw) {
            spdlog::info("[signal_ci] NARROW mean={:.4f} hw={:.4f}", mean, hw);
        };
        ci_cfg.on_wide_interval = [](double mean, double hw) {
            spdlog::warn("[signal_ci] WIDE   mean={:.4f} hw={:.4f}", mean, hw);
        };
        signal_ci.update_config(ci_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
    // SentimentPersistenceMatrix: Markov chain over N discretized bias states.
    // Tracks N×N transition counts, row-normalised probabilities, stickiness,
    // and a stationary-distribution estimate.  Fires on state transitions.
    llmquant::SentimentPersistenceMatrix sentiment_persistence;
    {
        llmquant::SentimentPersistenceMatrix::Config mp_cfg;
        mp_cfg.n_states      = 5;
        mp_cfg.min_row_count = 4;
        mp_cfg.on_state_change = [](int from, int to, double p) {
            spdlog::info("[markov] state {} → {} p={:.3f}", from, to, p);
        };
        sentiment_persistence.update_config(mp_cfg);
    }
#endif

#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
    // CausalImpactEstimator: CUSUM structural-break detector — attributes
    // return regime shifts to preceding LLM sentiment events.
    // Record signals via record_event() and returns via record_return().
    llmquant::CausalImpactEstimator causal_impact;
    {
        llmquant::CausalImpactEstimator::Config ci_causal_cfg;
        ci_causal_cfg.warmup_window  = 50;
        ci_causal_cfg.sensitivity    = 0.001;
        ci_causal_cfg.threshold      = 0.05;
        ci_causal_cfg.event_lookback = 20;
        ci_causal_cfg.on_break = [](double stat, const std::string& label, double impact) {
            spdlog::warn("[causal] break stat={:.4f} event=\"{}\" impact={:.5f}",
                         stat, label.empty() ? "<none>" : label, impact);
        };
        causal_impact.update_config(ci_causal_cfg);
    }
#endif

#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
    // OptionsFlowSentimentBridge: detects divergence between LLM sentiment
    // velocity and options IV skew.  Smart-money bear: narrative bullish but
    // IV skew widening.  Smart-money bull: narrative bearish but calls bid.
    // record_skew() would be driven by a real options feed; here we seed a
    // neutral skew so the detector is ready to accept live updates.
    llmquant::OptionsFlowSentimentBridge options_flow_bridge;
    {
        llmquant::OptionsFlowSentimentBridge::Config ofb_cfg;
        ofb_cfg.velocity_alpha = 0.15;
        ofb_cfg.skew_alpha     = 0.20;
        ofb_cfg.div_threshold  = 0.03;
        ofb_cfg.hysteresis     = 0.30;
        ofb_cfg.min_warmup     = 15;
        ofb_cfg.on_divergence  = [](llmquant::OptionsFlowSentimentBridge::DivergenceKind kind,
                                    double score, double vel, double skew) {
            const char* label = "NONE";
            if (kind == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBear)
                label = "SMART_MONEY_BEAR";
            else if (kind == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBull)
                label = "SMART_MONEY_BULL";
            spdlog::warn("[options_flow] divergence={} score={:.4f} vel={:.4f} skew={:.4f}",
                         label, score, vel, skew);
        };
        options_flow_bridge.update_config(ofb_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
    // SentimentPhasePortrait: discretises the (bias, velocity) state space into
    // an N×N grid, tracking dwell time per cell, dominant attractors, and
    // period-2 oscillation cycles in sentiment dynamics.
    llmquant::SentimentPhasePortrait sentiment_phase_portrait;
    {
        llmquant::SentimentPhasePortrait::Config spp_cfg;
        spp_cfg.grid_size           = 8;
        spp_cfg.velocity_alpha      = 0.2;
        spp_cfg.attractor_threshold = 0.15;
        spp_cfg.cycle_window        = 20;
        spp_cfg.min_visits          = 10;
        spp_cfg.on_attractor_change = [](int row, int col) {
            spdlog::info("[phase_portrait] attractor shifted → cell ({},{})", row, col);
        };
        spp_cfg.on_cycle_detected = [](int r1, int c1, int r2, int c2) {
            spdlog::warn("[phase_portrait] period-2 cycle detected ({},{})↔({},{})", r1, c1, r2, c2);
        };
        sentiment_phase_portrait.update_config(spp_cfg);
    }
#endif

#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
    // NarrativeTopicClassifier: online bag-of-centroids topic labeller.
    // Registers macro-topic buckets keyed on token weight centroids.
    // The dominant topic's signal_multiplier scales the downstream signal.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::NarrativeTopicClassifier::Config ntc_cfg;
        ntc_cfg.freq_alpha     = 0.10;
        ntc_cfg.dominant_alpha = 0.05;
        ntc_cfg.min_warmup     = 20;
        ntc_cfg.on_topic_change = [](const std::string& old_t, const std::string& new_t, double mult) {
            spdlog::info("[narrative_topic] dominant topic {} → {} (mult={:.2f})", old_t, new_t, mult);
        };
        narrative_classifier.update_config(ntc_cfg);
        // Register standard macro-narrative topics
        narrative_classifier.register_topic({"earnings",       0.70, 1.30});
        narrative_classifier.register_topic({"macro",          0.40, 0.90});
        narrative_classifier.register_topic({"geopolitical",   0.60, 0.70});
        narrative_classifier.register_topic({"technical",      0.30, 1.10});
        narrative_classifier.register_topic({"neutral",        0.05, 0.80});
    }
#endif

#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
    // TokenClockRecalibrator: estimates live LLM token emission rate and
    // computes a budget_scale_factor so latency gates stay calibrated even
    // when model throughput varies 2-5× from the default assumption.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::TokenClockRecalibrator::Config tcr_cfg;
        tcr_cfg.window_size            = 64;
        tcr_cfg.target_rate_hz         = 30.0;
        tcr_cfg.min_scale              = 0.25;
        tcr_cfg.max_scale              = 4.0;
        tcr_cfg.rate_change_threshold  = 0.15;
        tcr_cfg.on_rate_change = [](double rate_hz, double scale) {
            spdlog::info("[token_clock] rate={:.1f} Hz  budget_scale={:.2f}x", rate_hz, scale);
        };
        token_clock.update_config(tcr_cfg);
    }
#endif

#ifdef LLMQUANT_SHADOW_PORTFOLIO_ENABLED
    // SignalShadowPortfolio: paper-trades raw signals in parallel with the live
    // strategy, attributing P&L drag to risk constraints and cooldown windows.
    llmquant::SignalShadowPortfolio shadow_portfolio;
    {
        llmquant::SignalShadowPortfolio::Config sp_cfg;
        sp_cfg.unit_size            = 1000.0;
        sp_cfg.max_position         = 10000.0;
        sp_cfg.drag_alert_threshold = 50.0;
        sp_cfg.on_drag_alert = [](double drag, double shad, double live) {
            spdlog::warn("[shadow_portfolio] constraint drag={:.2f}  shadow={:.2f}  live={:.2f}",
                         drag, shad, live);
        };
        shadow_portfolio.update_config(sp_cfg);
    }
#endif

#ifdef LLMQUANT_CONFIDENCE_BAND_ENABLED
    // LLMConfidenceBandTracker: scalar Kalman filter that maintains Bayesian
    // confidence bands around the sentiment EMA.  Bands narrow when observations
    // are consistent and widen when sentiment is erratic.
    llmquant::LLMConfidenceBandTracker confidence_band;
    {
        llmquant::LLMConfidenceBandTracker::Config cb_cfg;
        cb_cfg.process_noise         = 0.001;
        cb_cfg.measurement_noise     = 0.01;
        cb_cfg.initial_variance      = 1.0;
        cb_cfg.z_score               = 2.0;
        cb_cfg.band_change_threshold = 0.10;
        cb_cfg.on_band_change = [](double lo, double center, double hi) {
            spdlog::debug("[conf_band] [{:.4f}, {:.4f}, {:.4f}]  hw={:.4f}",
                          lo, center, hi, (hi - lo) / 2.0);
        };
        confidence_band.update_config(cb_cfg);
    }
#endif

#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
    // TokenImportanceDecayScheduler: applies true wall-clock exponential decay
    // to per-token weights.  A token emitted 30 s ago carries far less signal
    // weight than the same token just received.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::TokenImportanceDecayScheduler::Config tds_cfg;
        tds_cfg.half_life_s  = 10.0;
        tds_cfg.max_age_s    = 60.0;
        tds_cfg.max_entries  = 1024;
        tds_cfg.on_sentiment_flip = [](double old_s, double new_s) {
            spdlog::warn("[decay_sched] sentiment sign flip {:.4f} → {:.4f}", old_s, new_s);
        };
        decay_scheduler.update_config(tds_cfg);
    }
#endif

#ifdef LLMQUANT_REGIME_ROUTER_ENABLED
    // RegimeSwitchingSignalRouter: 3-state FSM (Trending/Ranging/Crash) that
    // routes signals to per-regime configs.  Trending → amplified signals;
    // Ranging → dampened; Crash → minimal position with tight cooldown.
    llmquant::RegimeSwitchingSignalRouter regime_router;
    {
        llmquant::RegimeSwitchingSignalRouter::Config rr_cfg;
        rr_cfg.trend_threshold     = 0.05;
        rr_cfg.crash_vol_threshold = 0.10;
        rr_cfg.hysteresis          = 0.15;
        rr_cfg.vol_alpha           = 0.20;
        rr_cfg.momentum_alpha      = 0.15;
        rr_cfg.trending_cfg  = {1.30, 1.00, 400};
        rr_cfg.ranging_cfg   = {0.70, 0.60, 800};
        rr_cfg.crash_cfg     = {0.20, 0.15, 2000};
        rr_cfg.on_regime_change = [](llmquant::RegimeSwitchingSignalRouter::Regime old_r,
                                     llmquant::RegimeSwitchingSignalRouter::Regime new_r,
                                     double mult) {
            static const char* names[] = {"Trending", "Ranging", "Crash"};
            spdlog::warn("[regime_router] {} → {}  bias_mult={:.2f}",
                         names[static_cast<int>(old_r)],
                         names[static_cast<int>(new_r)], mult);
        };
        regime_router.update_config(rr_cfg);
    }
#endif

#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
    // TokenStreamDifferencer: tracks velocity, acceleration, and jerk of the
    // token weight series.  Jerk spike = abrupt narrative reversal — fires
    // one derivative earlier than a momentum peak.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::TokenStreamDifferencer::Config sd_cfg;
        sd_cfg.ema_alpha       = 0.20;
        sd_cfg.jerk_threshold  = 0.10;
        sd_cfg.hysteresis      = 0.30;
        sd_cfg.on_jerk_spike = [](double raw_jerk, double ema_jerk) {
            spdlog::warn("[differencer] JERK SPIKE raw={:.4f} ema={:.4f}", raw_jerk, ema_jerk);
        };
        stream_differencer.update_config(sd_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_DRIFT_ENABLED
    // SignalDriftMonitor: Wasserstein-1 drift detector comparing recent vs
    // baseline bias-shift distributions.  W1 > threshold → regime shift.
    llmquant::SignalDriftMonitor signal_drift_monitor;
    {
        llmquant::SignalDriftMonitor::Config sdm_cfg;
        sdm_cfg.baseline_size    = 128;
        sdm_cfg.recent_size      = 32;
        sdm_cfg.drift_threshold  = 0.10;
        sdm_cfg.clear_hysteresis = 0.70;
        sdm_cfg.on_drift_detected = [](double w1) {
            spdlog::warn("[drift_monitor] DISTRIBUTION DRIFT W1={:.4f}", w1);
        };
        sdm_cfg.on_drift_cleared = [](double w1) {
            spdlog::info("[drift_monitor] drift cleared W1={:.4f}", w1);
        };
        signal_drift_monitor.update_config(sdm_cfg);
    }
#endif

#ifdef LLMQUANT_LIFECYCLE_TRACKER_ENABLED
    // SignalLifecycleTracker: tracks each emitted signal's lifecycle
    // (birth → peak → 50%-decay half-life → death), and flags zombie signals
    // that persist beyond max_age_s without decaying.
    llmquant::SignalLifecycleTracker lifecycle_tracker;
    {
        llmquant::SignalLifecycleTracker::Config lc_cfg;
        lc_cfg.decay_fraction    = 0.10;
        lc_cfg.halflife_fraction = 0.50;
        lc_cfg.max_age_s         = 300.0;
        lc_cfg.on_death = [](const std::string& id, double hl, double peak) {
            spdlog::debug("[lifecycle] signal '{}' died  halflife={:.1f}s  peak={:.4f}", id, hl, peak);
        };
        lc_cfg.on_zombie = [](const std::string& id, double age) {
            spdlog::warn("[lifecycle] ZOMBIE signal '{}' age={:.1f}s", id, age);
        };
        lifecycle_tracker.update_config(lc_cfg);
    }
#endif

#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
    // TokenWeightQuantiser: stochastic rounding to 256-level fixed grid.
    // Preserves E[w] exactly across many tokens, enabling integer-precision
    // downstream SIMD accumulation without systematic bias.
    // (Declared in the pre-lambda block above; configure here.)
    {
        llmquant::TokenWeightQuantiser::Config tq_cfg;
        tq_cfg.levels               = 256;
        tq_cfg.range_min            = -1.0;
        tq_cfg.range_max            =  1.0;
        tq_cfg.error_alpha          = 0.05;
        tq_cfg.clamp_alert_fraction = 0.05;
        tq_cfg.on_high_clamp_rate = [](double rate, double w) {
            spdlog::warn("[quantiser] HIGH CLAMP RATE {:.1f}% last_weight={:.4f}",
                         rate * 100.0, w);
        };
        token_quantiser.update_config(tq_cfg);
    }
#endif

#ifdef LLMQUANT_POSITION_CONCENTRATION_ENABLED
    // PositionConcentrationGuard: HHI-based theme concentration monitor.
    // Signals tagged with a theme (via dominant_topic from NarrativeTopicClassifier)
    // are tracked; high HHI means signal pipeline is dominated by a single theme.
    llmquant::PositionConcentrationGuard concentration_guard;
    {
        llmquant::PositionConcentrationGuard::Config cg_cfg;
        cg_cfg.window_size             = 64;
        cg_cfg.concentration_threshold = 0.50;
        cg_cfg.clear_hysteresis        = 0.80;
        cg_cfg.on_concentrated = [](double hhi, const std::string& theme) {
            spdlog::warn("[conc_guard] HIGH CONCENTRATION HHI={:.3f} dominant={}", hhi, theme);
        };
        cg_cfg.on_diversified = [](double hhi) {
            spdlog::info("[conc_guard] diversified HHI={:.3f}", hhi);
        };
        concentration_guard.update_config(cg_cfg);
    }
#endif

#ifdef LLMQUANT_AUTOCORR_METER_ENABLED
    // SentimentAutocorrelationMeter: rolling lag-1..5 AC of bias series.
    // High lag-1 AC → trending sentiment (use momentum).
    // Low/negative lag-1 AC → mean reversion (use contrarian).
    llmquant::SentimentAutocorrelationMeter autocorr_meter;
    {
        llmquant::SentimentAutocorrelationMeter::Config ac_cfg;
        ac_cfg.window             = 50;
        ac_cfg.max_lag            = 5;
        ac_cfg.trending_threshold = 0.20;
        ac_cfg.on_regime_change = [](double lag1_ac, bool trending) {
            spdlog::info("[autocorr] lag1={:.3f}  {}",
                         lag1_ac, trending ? "TRENDING" : "mean-reverting");
        };
        autocorr_meter.update_config(ac_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_SSI_ENABLED
    // SignalStrengthIndexer: RSI applied to sentiment bias.
    // SSI > 70 → overbought (sentiment overextended, reversion likely).
    // SSI < 30 → oversold (narrative exhaustion, possible bounce).
    llmquant::SignalStrengthIndexer signal_ssi;
    {
        llmquant::SignalStrengthIndexer::Config ssi_cfg;
        ssi_cfg.period               = 14;
        ssi_cfg.overbought_threshold = 70.0;
        ssi_cfg.oversold_threshold   = 30.0;
        ssi_cfg.on_overbought = [](double ssi_val) {
            spdlog::warn("[ssi] OVERBOUGHT SSI={:.1f} — sentiment may be exhausted", ssi_val);
        };
        ssi_cfg.on_oversold = [](double ssi_val) {
            spdlog::warn("[ssi] OVERSOLD SSI={:.1f} — narrative recovery possible", ssi_val);
        };
        signal_ssi.update_config(ssi_cfg);
    }
#endif

#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
    // TokenFlowPressureGauge: EMA of inter-token arrival intervals.
    // Pressure > 0.85 → tokens arriving faster than target rate → spike alert.
    // (Declared pre-lambda; configured here.)
    {
        llmquant::TokenFlowPressureGauge::Config fp_cfg;
        fp_cfg.target_interval_ns = 33'000'000ULL; // ~30 tok/s
        fp_cfg.ema_alpha          = 0.10;
        fp_cfg.spike_threshold    = 0.85;
        fp_cfg.on_pressure_spike = [](double p) {
            spdlog::warn("[flow_pressure] SPIKE pressure={:.3f} — token rate exceeds target", p);
        };
        flow_pressure.update_config(fp_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_FATIGUE_ENABLED
    // SignalFatigueMeter: tracks consecutive same-direction signal streaks.
    // streak >= 5 → fatigue callback fired; score saturates at streak=10.
    llmquant::SignalFatigueMeter signal_fatigue;
    {
        llmquant::SignalFatigueMeter::Config sf_cfg;
        sf_cfg.fatigue_threshold  = 5;
        sf_cfg.saturation_streak  = 10;
        sf_cfg.on_fatigue = [](int streak, double score) {
            spdlog::warn("[fatigue] FATIGUED streak={} score={:.3f} — reversal risk elevated", streak, score);
        };
        signal_fatigue.update_config(sf_cfg);
    }
#endif

#ifdef LLMQUANT_POLARIZATION_MONITOR_ENABLED
    // SignalPolarizationMonitor: detects bimodal sentiment distribution using
    // Sarle's bimodality coefficient.  BC > 0.6 → two opposing camps active.
    llmquant::SignalPolarizationMonitor polarization_monitor;
    {
        llmquant::SignalPolarizationMonitor::Config pm_cfg;
        pm_cfg.window                 = 64;
        pm_cfg.min_observations       = 8;
        pm_cfg.polarization_threshold = 0.60;
        pm_cfg.clear_hysteresis       = 0.85;
        pm_cfg.on_polarized = [](double bc) {
            spdlog::warn("[polarization] BIMODAL BC={:.3f} — bulls/bears split, regime unstable", bc);
        };
        pm_cfg.on_unified = [](double bc) {
            spdlog::info("[polarization] unified BC={:.3f} — distribution normalized", bc);
        };
        polarization_monitor.update_config(pm_cfg);
    }
#endif

#ifdef LLMQUANT_LATENCY_JITTER_ENABLED
    // LatencyJitterMonitor: MAD-based token processing latency consistency tracker.
    // High MAD → inconsistent processing times → upstream backpressure warranted.
    {
        llmquant::LatencyJitterMonitor::Config ljm_cfg;
        ljm_cfg.window_size         = 64;
        ljm_cfg.min_samples         = 8;
        ljm_cfg.jitter_threshold_us = 500.0;  // 500 μs MAD threshold
        ljm_cfg.clear_hysteresis    = 0.75;
        ljm_cfg.on_jitter_spike = [](double mad) {
            spdlog::warn("[jitter] SPIKE MAD={:.1f}us — latency inconsistent; consider backpressure", mad);
        };
        ljm_cfg.on_jitter_clear = [](double mad) {
            spdlog::info("[jitter] CLEAR MAD={:.1f}us — latency consistency restored", mad);
        };
        latency_jitter.update_config(ljm_cfg);
    }
#endif

#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
    // NarrativeTemperatureGauge: combined |EMA(bias)| × σ(bias) "temperature".
    // Hot narrative = strong direction AND high volatility = elevated execution risk.
    llmquant::NarrativeTemperatureGauge narrative_temperature;
    {
        llmquant::NarrativeTemperatureGauge::Config nt_cfg;
        nt_cfg.window        = 32;
        nt_cfg.ema_alpha     = 0.15;
        nt_cfg.normalizer    = 0.5;
        nt_cfg.hot_threshold = 0.7;
        nt_cfg.cool_hysteresis = 0.8;
        nt_cfg.on_hot = [](double temp) {
            spdlog::warn("[narrative_temp] HOT temperature={:.3f} — strong+volatile narrative, risk elevated", temp);
        };
        nt_cfg.on_cool = [](double temp) {
            spdlog::info("[narrative_temp] cooled temperature={:.3f}", temp);
        };
        narrative_temperature.update_config(nt_cfg);
    }
#endif

#ifdef LLMQUANT_ECHO_SUPPRESSOR_ENABLED
    // SignalEchoSuppressor: rolling fraction of near-duplicate consecutive signals.
    // High echo rate = model is stuck in a narrative loop; suppress downstream amplification.
    llmquant::SignalEchoSuppressor echo_suppressor;
    {
        llmquant::SignalEchoSuppressor::Config es_cfg;
        es_cfg.echo_threshold     = 1e-4;
        es_cfg.window             = 32;
        es_cfg.echo_rate_threshold = 0.6;
        es_cfg.clear_hysteresis   = 0.75;
        es_cfg.on_echo_detected = [](double rate) {
            spdlog::warn("[echo_suppressor] echo state entered rate={:.3f} — model may be in narrative loop", rate);
        };
        es_cfg.on_echo_cleared = [](double rate) {
            spdlog::info("[echo_suppressor] echo cleared rate={:.3f}", rate);
        };
        echo_suppressor.update_config(es_cfg);
    }
#endif

#ifdef LLMQUANT_HURST_ESTIMATOR_ENABLED
    // SignalHurstEstimator: R/S rescaled-range Hurst exponent.
    // H>0.6 = trending (momentum regime — amplify), H<0.4 = mean-reverting (fade).
    llmquant::SignalHurstEstimator hurst_estimator;
    {
        llmquant::SignalHurstEstimator::Config he_cfg;
        he_cfg.window              = 64;
        he_cfg.min_samples         = 16;
        he_cfg.trending_threshold  = 0.60;
        he_cfg.reverting_threshold = 0.40;
        he_cfg.clear_hysteresis    = 0.85;
        he_cfg.on_trending = [](double h) {
            spdlog::info("[hurst] trending regime H={:.3f} — persistent momentum, consider amplifying", h);
        };
        he_cfg.on_mean_reverting = [](double h) {
            spdlog::info("[hurst] mean-reverting regime H={:.3f} — anti-persistent, consider fading", h);
        };
        hurst_estimator.update_config(he_cfg);
    }
#endif

#ifdef LLMQUANT_CHANGE_POINT_ENABLED
    // BiasChangePointDetector: Page-CUSUM sequential change-point detection.
    // Fires on_upshift/on_downshift when sustained bias mean shift is detected.
    llmquant::BiasChangePointDetector change_point_detector;
    {
        llmquant::BiasChangePointDetector::Config cpd_cfg;
        cpd_cfg.reference_mean = 0.0;
        cpd_cfg.allowance      = 0.05;   // detects shifts > 2×allowance = 0.10
        cpd_cfg.threshold      = 1.0;
        cpd_cfg.on_upshift = [](double c) {
            spdlog::warn("[change_point] upside mean shift detected C+={:.3f} — narrative turning bullish", c);
        };
        cpd_cfg.on_downshift = [](double c) {
            spdlog::warn("[change_point] downside mean shift detected C-={:.3f} — narrative turning bearish", c);
        };
        change_point_detector.update_config(cpd_cfg);
    }
#endif

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
    // BiasVelocityBreaker: trips when EMA of |Δbias| exceeds max_velocity.
    // Prevents trading on whipsaw signals between consecutive tokens.
    llmquant::BiasVelocityBreaker bias_vbreaker;
    {
        llmquant::BiasVelocityBreaker::Config vb_cfg;
        vb_cfg.ema_alpha        = 0.2;
        vb_cfg.max_velocity     = 0.15;
        vb_cfg.clear_hysteresis = 0.75;
        vb_cfg.on_trip = [](double v) {
            spdlog::warn("[velocity_breaker] TRIPPED velocity={:.4f} — whipsaw detected, suppress signals", v);
        };
        vb_cfg.on_clear = [](double v) {
            spdlog::info("[velocity_breaker] cleared velocity={:.4f}", v);
        };
        bias_vbreaker.update_config(vb_cfg);
    }
#endif

#ifdef LLMQUANT_IR_TRACKER_ENABLED
    // SignalInformationRatioTracker: rolling IR = μ(bias)/σ(bias).
    // High |IR| = consistent directional signal above noise floor.
    llmquant::SignalInformationRatioTracker ir_tracker;
    {
        llmquant::SignalInformationRatioTracker::Config ir_cfg;
        ir_cfg.window       = 32;
        ir_cfg.min_samples  = 8;
        ir_cfg.reference    = 0.0;
        ir_cfg.ir_threshold = 1.5;
        ir_cfg.on_high_ir = [](double ir) {
            spdlog::info("[ir_tracker] high IR={:.3f} — signal consistently above noise", ir);
        };
        ir_cfg.on_ir_normalized = [](double ir) {
            spdlog::info("[ir_tracker] IR normalized={:.3f}", ir);
        };
        ir_tracker.update_config(ir_cfg);
    }
#endif

#ifdef LLMQUANT_CONSISTENCY_METER_ENABLED
    // NarrativeConsistencyMeter: AAD-based smoothness score.
    // Low score = chaotic, erratic bias jumps = signal quality unreliable.
    llmquant::NarrativeConsistencyMeter consistency_meter;
    {
        llmquant::NarrativeConsistencyMeter::Config cm_cfg;
        cm_cfg.window                = 32;
        cm_cfg.min_samples           = 4;
        cm_cfg.normalizer            = 0.3;
        cm_cfg.consistency_threshold = 0.4;
        cm_cfg.on_inconsistent = [](double s) {
            spdlog::warn("[consistency] INCONSISTENT score={:.3f} — erratic narrative, reduce confidence", s);
        };
        cm_cfg.on_recovered = [](double s) {
            spdlog::info("[consistency] recovered score={:.3f}", s);
        };
        consistency_meter.update_config(cm_cfg);
    }
#endif

#ifdef LLMQUANT_OSCILLATION_DETECTOR_ENABLED
    // SignalOscillationDetector: zero-crossing rate for alternating +/- patterns.
    // High ZCR = model flip-flopping = suppress signal amplification.
    llmquant::SignalOscillationDetector oscillation_detector;
    {
        llmquant::SignalOscillationDetector::Config od_cfg;
        od_cfg.window                = 16;
        od_cfg.min_samples           = 4;
        od_cfg.oscillation_threshold = 0.65;
        od_cfg.clear_hysteresis      = 0.75;
        od_cfg.on_oscillating = [](double zcr) {
            spdlog::warn("[oscillation] ZCR={:.3f} — model oscillating, suppress directional signals", zcr);
        };
        od_cfg.on_stabilized = [](double zcr) {
            spdlog::info("[oscillation] stabilized ZCR={:.3f}", zcr);
        };
        oscillation_detector.update_config(od_cfg);
    }
#endif

#ifdef LLMQUANT_MOMENTUM_INDEX_ENABLED
    // BiasMomentumIndex: MACD-style dual-EMA crossover for bias momentum.
    // Positive histogram = bullish momentum; negative = bearish.
    llmquant::BiasMomentumIndex momentum_index;
    {
        llmquant::BiasMomentumIndex::Config mi_cfg;
        mi_cfg.fast_alpha  = 0.222;   // ~9-period EMA
        mi_cfg.slow_alpha  = 0.0909;  // ~21-period EMA
        mi_cfg.signal_alpha = 0.2;    // ~9-period signal line
        mi_cfg.on_bullish_crossover = [](double hist) {
            spdlog::info("[momentum] BULLISH crossover histogram={:.4f}", hist);
        };
        mi_cfg.on_bearish_crossover = [](double hist) {
            spdlog::warn("[momentum] BEARISH crossover histogram={:.4f}", hist);
        };
        momentum_index.update_config(mi_cfg);
    }
#endif

#ifdef LLMQUANT_GAIN_LOSS_RATIO_ENABLED
    // SignalGainLossRatio: rolling G/L ratio over a fixed window.
    // Ratio > high_gain_threshold = consistent directional edge.
    llmquant::SignalGainLossRatio gain_loss_ratio;
    {
        llmquant::SignalGainLossRatio::Config gl_cfg;
        gl_cfg.window             = 32;
        gl_cfg.min_samples        = 4;
        gl_cfg.high_gain_threshold = 1.5;
        gl_cfg.high_loss_threshold = 0.67;
        gl_cfg.max_ratio          = 10.0;
        gl_cfg.on_high_gain = [](double r) {
            spdlog::info("[gain_loss] high gain edge ratio={:.3f}", r);
        };
        gl_cfg.on_high_loss = [](double r) {
            spdlog::warn("[gain_loss] high loss regime ratio={:.3f}", r);
        };
        gain_loss_ratio.update_config(gl_cfg);
    }
#endif

#ifdef LLMQUANT_REGIME_TRANSITION_MATRIX_ENABLED
    // BiasRegimeTransitionMatrix: empirical Markov P(next|current) over rolling window.
    // Fires when most-probable next regime changes with high probability.
    llmquant::BiasRegimeTransitionMatrix regime_transition;
    {
        llmquant::BiasRegimeTransitionMatrix::Config rt_cfg;
        rt_cfg.n_regimes   = 5;
        rt_cfg.window      = 100;
        rt_cfg.range_min   = -1.0;
        rt_cfg.range_max   =  1.0;
        rt_cfg.alert_prob  = 0.6;
        rt_cfg.on_likely_transition = [](int from, int to, double prob) {
            spdlog::info("[regime_matrix] transition {} → {} prob={:.3f}", from, to, prob);
        };
        regime_transition.update_config(rt_cfg);
    }
#endif

#ifdef LLMQUANT_REVERSAL_DETECTOR_ENABLED
    // SignalReversalDetector: price-action thrust+counter-move pattern in bias stream.
    // Fires when counter-move exceeds reversal_fraction of prior thrust.
    llmquant::SignalReversalDetector reversal_detector;
    {
        llmquant::SignalReversalDetector::Config rd_cfg;
        rd_cfg.reversal_fraction = 0.5;
        rd_cfg.min_thrust        = 0.05;
        rd_cfg.on_bullish_reversal = [](double bias) {
            spdlog::info("[reversal] BULLISH reversal at bias={:.4f}", bias);
        };
        rd_cfg.on_bearish_reversal = [](double bias) {
            spdlog::warn("[reversal] BEARISH reversal at bias={:.4f}", bias);
        };
        reversal_detector.update_config(rd_cfg);
    }
#endif

#ifdef LLMQUANT_TSMI_ENABLED
    // TokenSentimentMomentumIndex: composite ROC+acceleration+SSI momentum in [-1,1].
    // Fires on_momentum_shift when TSMI crosses zero (directional flip).
    llmquant::TokenSentimentMomentumIndex tsmi;
    {
        llmquant::TokenSentimentMomentumIndex::Config tsmi_cfg;
        tsmi_cfg.w_roc      = 0.4;
        tsmi_cfg.w_acc      = 0.3;
        tsmi_cfg.w_str      = 0.3;
        tsmi_cfg.roc_scale  = 0.1;
        tsmi_cfg.acc_scale  = 0.05;
        tsmi_cfg.on_momentum_shift = [](double old_v, double new_v) {
            spdlog::info("[tsmi] momentum flip {:.3f} → {:.3f}", old_v, new_v);
        };
        tsmi.update_config(tsmi_cfg);
    }
#endif

#ifdef LLMQUANT_ADAPTIVE_THRESHOLD_ENABLED
    // AdaptiveThresholdController: vol-adaptive OB/OS thresholds via rolling |Δbias| EMA.
    llmquant::AdaptiveThresholdController adaptive_threshold;
    {
        llmquant::AdaptiveThresholdController::Config at_cfg;
        at_cfg.period           = 20;
        at_cfg.base_overbought  = 70.0;
        at_cfg.base_oversold    = 30.0;
        at_cfg.k_sigma          = 50.0;
        at_cfg.clamp_lo         = 55.0;
        at_cfg.clamp_hi         = 95.0;
        at_cfg.change_threshold = 1.0;
        at_cfg.on_threshold_change = [](double ob, double os) {
            spdlog::info("[adaptive_thresh] OB={:.1f} OS={:.1f} — vol-adjusted thresholds updated", ob, os);
        };
        adaptive_threshold.update_config(at_cfg);
    }
#endif

#ifdef LLMQUANT_CONDITIONAL_DIST_ENABLED
    // BiasConditionalDistribution: P(bias|prev_dir) asymmetry for momentum/mean-rev detection.
    llmquant::BiasConditionalDistribution conditional_dist;
    {
        llmquant::BiasConditionalDistribution::Config cd_cfg;
        cd_cfg.window               = 50;
        cd_cfg.n_bins               = 8;
        cd_cfg.range_min            = -1.0;
        cd_cfg.range_max            =  1.0;
        cd_cfg.asymmetry_threshold  = 0.3;
        cd_cfg.on_asymmetry_detected = [](double tv) {
            spdlog::warn("[cond_dist] ASYMMETRIC tv={:.3f} — momentum/mean-rev regime active", tv);
        };
        cd_cfg.on_symmetry_restored = [](double tv) {
            spdlog::info("[cond_dist] symmetry restored tv={:.3f}", tv);
        };
        conditional_dist.update_config(cd_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_COMPRESSOR_ENABLED
    // SignalCompressor: LZ76 normalised complexity for trend (low) vs noise (high) regime.
    llmquant::SignalCompressor signal_compressor;
    {
        llmquant::SignalCompressor::Config sc_cfg;
        sc_cfg.window           = 64;
        sc_cfg.n_symbols        = 4;
        sc_cfg.range_min        = -1.0;
        sc_cfg.range_max        =  1.0;
        sc_cfg.change_threshold = 0.1;
        sc_cfg.min_samples      = 16;
        sc_cfg.on_complexity_change = [](double old_lzc, double new_lzc) {
            spdlog::info("[compressor] LZC {:.3f} → {:.3f} — {} regime",
                         old_lzc, new_lzc,
                         new_lzc < 0.7 ? "structured/trend" : "noise");
        };
        signal_compressor.update_config(sc_cfg);
    }
#endif

#ifdef LLMQUANT_ROLLING_QUANTILE_ENABLED
    // BiasRollingQuantileTracker: rolling P10-P90 percentiles with IQR and skew.
    llmquant::BiasRollingQuantileTracker rolling_quantile;
    {
        llmquant::BiasRollingQuantileTracker::Config rq_cfg;
        rq_cfg.window                  = 64;
        rq_cfg.min_samples             = 8;
        rq_cfg.median_change_threshold = 0.05;
        rq_cfg.skew_threshold          = 0.3;
        rq_cfg.on_median_shift = [](double old_p50, double new_p50) {
            spdlog::info("[quantile] median shift {:.3f} → {:.3f}", old_p50, new_p50);
        };
        rq_cfg.on_skew_change = [](double skew) {
            spdlog::info("[quantile] skew changed ratio={:.3f}", skew);
        };
        rolling_quantile.update_config(rq_cfg);
    }
#endif

#ifdef LLMQUANT_AUTOREGRESSOR_ENABLED
    // SignalAutoregressor: RLS AR(p) online model fit with prediction-error spike detection.
    llmquant::SignalAutoregressor signal_ar;
    {
        llmquant::SignalAutoregressor::Config ar_cfg;
        ar_cfg.order           = 4;
        ar_cfg.lambda          = 0.97;
        ar_cfg.error_threshold = 0.2;
        ar_cfg.on_prediction_error_spike = [](double err) {
            spdlog::warn("[ar] SPIKE |err|={:.4f} — regime break or structural shift", err);
        };
        signal_ar.update_config(ar_cfg);
    }
#endif

#ifdef LLMQUANT_PHASE_SPACE_ENABLED
    // TokenBiasPhaseSpace: 2D (bias_t, bias_{t-1}) delay-embedding attractor tracking.
    llmquant::TokenBiasPhaseSpace phase_space;
    {
        llmquant::TokenBiasPhaseSpace::Config ps_cfg;
        ps_cfg.grid_size = 8;
        ps_cfg.range_min = -1.0;
        ps_cfg.range_max =  1.0;
        ps_cfg.on_attractor_shift = [](llmquant::TokenBiasPhaseSpace::CellId old_c,
                                       llmquant::TokenBiasPhaseSpace::CellId new_c) {
            spdlog::info("[phase] attractor shift ({},{})→({},{})",
                         old_c.row, old_c.col, new_c.row, new_c.col);
        };
        phase_space.update_config(ps_cfg);
    }
#endif

#ifdef LLMQUANT_TOPOLOGY_MAPPER_ENABLED
    // SentimentTopologyMapper: persistent-homology Betti-0 count for bias stream.
    // Counts structurally distinct sentiment "islands" at each threshold level.
    llmquant::SentimentTopologyMapper topology_mapper;
    {
        llmquant::SentimentTopologyMapper::Config tm_cfg;
        tm_cfg.window           = 64;
        tm_cfg.min_persistence  = 0.05;
        tm_cfg.on_topology_change = [](int old_c, int new_c) {
            spdlog::info("[topo] topology shift: {} → {} sentiment components",
                         old_c, new_c);
        };
        topology_mapper.update_config(tm_cfg);
    }
#endif

#ifdef LLMQUANT_INFORMATION_GAIN_ENABLED
    // BiasInformationGain: normalised mutual information between consecutive bias bins.
    // High NMI → bias has predictable sequential structure; low → near-random.
    llmquant::BiasInformationGain info_gain;
    {
        llmquant::BiasInformationGain::Config ig_cfg;
        ig_cfg.n_bins   = 6;
        ig_cfg.window   = 100;
        ig_cfg.on_mi_change = [](double old_nmi, double new_nmi) {
            spdlog::info("[info_gain] NMI change {:.3f} → {:.3f}", old_nmi, new_nmi);
        };
        info_gain.update_config(ig_cfg);
    }
#endif

#ifdef LLMQUANT_NARRATIVE_DRIFT_ENABLED
    // NarrativeDriftDetector: Page-Hinkley sequential change-point detector.
    // Fires upward/downward drift alarms when accumulated bias shift exceeds λ.
    llmquant::NarrativeDriftDetector narrative_drift;
    {
        llmquant::NarrativeDriftDetector::Config nd_cfg;
        nd_cfg.delta  = 0.02;
        nd_cfg.lambda = 0.5;
        nd_cfg.on_upward_drift = [](double mag) {
            spdlog::warn("[narrative_drift] UPWARD drift U={:.4f}", mag);
        };
        nd_cfg.on_downward_drift = [](double mag) {
            spdlog::warn("[narrative_drift] DOWNWARD drift D={:.4f}", mag);
        };
        narrative_drift.update_config(nd_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_GRAPH_ENABLED
    // TokenSentimentGraphBuilder: directed token-to-token influence graph.
    // Edge w(a→b) = conditional mean bias contribution from a to b.
    llmquant::TokenSentimentGraphBuilder sentiment_graph;
    {
        llmquant::TokenSentimentGraphBuilder::Config sg_cfg;
        sg_cfg.hub_threshold = 0.5;
        sg_cfg.on_hub_detected = [](int tok_id, double in_w) {
            spdlog::info("[sent_graph] HUB token_id={} in_weight={:.3f}",
                         tok_id, in_w);
        };
        sentiment_graph.update_config(sg_cfg);
    }
#endif

#ifdef LLMQUANT_KALMAN_FILTER_ENABLED
    // BiasKalmanFilter: 1-D random-walk Kalman filter for bias smoothing.
    // NIS > 3.84 fires on_model_mismatch signalling regime change or fat-tail event.
    llmquant::BiasKalmanFilter kalman_filter;
    {
        llmquant::BiasKalmanFilter::Config kf_cfg;
        kf_cfg.Q = 0.001;
        kf_cfg.R = 0.1;
        kf_cfg.estimate_R = true;
        kf_cfg.on_model_mismatch = [](double nis) {
            spdlog::warn("[kalman] model mismatch NIS={:.2f} — regime change or tail event", nis);
        };
        kalman_filter.update_config(kf_cfg);
    }
#endif

#ifdef LLMQUANT_SPECTRAL_ENTROPY_ENABLED
    // SignalSpectralEntropy: DCT-II power spectral density entropy of rolling bias window.
    // Low entropy → periodic/trending regime; high → white-noise/random regime.
    llmquant::SignalSpectralEntropy spectral_entropy;
    {
        llmquant::SignalSpectralEntropy::Config se_cfg;
        se_cfg.window            = 64;
        se_cfg.change_threshold  = 0.2;
        se_cfg.on_entropy_change = [](double old_hs, double new_hs) {
            spdlog::info("[spectral] entropy shift {:.3f} → {:.3f}", old_hs, new_hs);
        };
        spectral_entropy.update_config(se_cfg);
    }
#endif

#ifdef LLMQUANT_BOOTSTRAP_CI_ENABLED
    // BiasBootstrapCI: model-free 95% CI for rolling mean via bootstrap resampling.
    // Wide CI → noisy/unreliable signal; narrow CI → high-confidence regime.
    llmquant::BiasBootstrapCI bootstrap_ci;
    {
        llmquant::BiasBootstrapCI::Config bc_cfg;
        bc_cfg.window              = 64;
        bc_cfg.n_bootstrap         = 200;
        bc_cfg.recompute_interval  = 10;
        bc_cfg.width_threshold     = 0.3;
        bc_cfg.on_ci_wide = [](double lo, double hi) {
            spdlog::warn("[bootstrap_ci] WIDE CI [{:.3f}, {:.3f}] — signal unreliable", lo, hi);
        };
        bc_cfg.on_ci_narrow = [](double lo, double hi) {
            spdlog::info("[bootstrap_ci] narrow CI [{:.3f}, {:.3f}] — signal reliable", lo, hi);
        };
        bootstrap_ci.update_config(bc_cfg);
    }
#endif

#ifdef LLMQUANT_GARCH_ESTIMATOR_ENABLED
    // BiasGarchEstimator: GARCH(1,1) conditional volatility for bias stream.
    // Captures vol clustering; fires on_vol_spike when σ_t jumps by vol_spike_ratio.
    llmquant::BiasGarchEstimator garch_est;
    {
        llmquant::BiasGarchEstimator::Config ge_cfg;
        ge_cfg.alpha          = 0.10;
        ge_cfg.beta           = 0.85;
        ge_cfg.sigma2_init    = 0.01;
        ge_cfg.vol_spike_ratio = 1.5;
        ge_cfg.on_vol_spike = [](double sigma) {
            spdlog::warn("[garch] vol spike σ={:.4f} — volatility clustering event", sigma);
        };
        garch_est.update_config(ge_cfg);
    }
#endif

#ifdef LLMQUANT_REGIME_HMM_ENABLED
    // SignalRegimeHMM: 2-state HMM online forward-algorithm regime classifier.
    // Produces soft P(bullish) probability; fires on_regime_change on hard state flip.
    llmquant::SignalRegimeHMM regime_hmm;
    {
        llmquant::SignalRegimeHMM::Config rh_cfg;
        rh_cfg.state_mean     = {-0.05, 0.05};
        rh_cfg.state_std      = {0.12,  0.12};
        rh_cfg.bullish_threshold = 0.60;
        rh_cfg.on_regime_change = [](int from, int to) {
            spdlog::info("[hmm] regime: {} → {} (0=bear,1=bull)", from, to);
        };
        regime_hmm.update_config(rh_cfg);
    }
#endif

#ifdef LLMQUANT_POLARITY_INDEX_ENABLED
    // NarrativePolarityIndex: EMA bull/bear pressure → NPI ∈ (−1,+1).
    // +1 = purely bullish; −1 = purely bearish; fires on flip and extreme events.
    llmquant::NarrativePolarityIndex polarity_idx;
    {
        llmquant::NarrativePolarityIndex::Config pi_cfg;
        pi_cfg.alpha              = 0.05;
        pi_cfg.extreme_threshold  = 0.7;
        pi_cfg.on_polarity_flip = [](double old_npi, double new_npi) {
            spdlog::info("[polarity] NPI flip {:.3f} → {:.3f}", old_npi, new_npi);
        };
        pi_cfg.on_extreme = [](double npi) {
            spdlog::warn("[polarity] EXTREME NPI={:.3f}", npi);
        };
        polarity_idx.update_config(pi_cfg);
    }
#endif

#ifdef LLMQUANT_RESIDUAL_ANALYSER_ENABLED
    // SignalResidualAnalyser: AR(1) residuals + Durbin-Watson DW statistic.
    // DW ≈ 2 → well-specified; DW near 0 or 4 → residual autocorrelation.
    llmquant::SignalResidualAnalyser residual_analyser;
    {
        llmquant::SignalResidualAnalyser::Config ra_cfg;
        ra_cfg.ar_alpha    = 0.10;
        ra_cfg.window      = 50;
        ra_cfg.dw_lo       = 1.5;
        ra_cfg.dw_hi       = 2.5;
        ra_cfg.on_residual_autocorr = [](double dw) {
            spdlog::warn("[residual] DW={:.3f} outside [1.5,2.5] — AR residual structure", dw);
        };
        residual_analyser.update_config(ra_cfg);
    }
#endif

#ifdef LLMQUANT_SALIENCY_RANKER_ENABLED
    // TokenSaliencyRanker: per-token SNR saliency score — bullish/bearish top-K.
    // Ranks tokens by mean_contribution / std_contribution (signal-to-noise ratio).
    llmquant::TokenSaliencyRanker saliency_ranker;
    {
        llmquant::TokenSaliencyRanker::Config sr_cfg;
        sr_cfg.ema_alpha        = 0.10;
        sr_cfg.min_token_count  = 3;
        sr_cfg.on_top_token_change = [](int tok_id, double sal) {
            spdlog::info("[saliency] new top token id={} saliency={:.3f}", tok_id, sal);
        };
        saliency_ranker.update_config(sr_cfg);
    }
#endif

#ifdef LLMQUANT_TAIL_RISK_METER_ENABLED
    // SignalTailRiskMeter: rolling VaR_95 and Expected Shortfall ES_95 for bias stream.
    // ES_95 = mean of all bias values in the 5% left tail — coherent risk measure.
    llmquant::SignalTailRiskMeter tail_risk;
    {
        llmquant::SignalTailRiskMeter::Config tr_cfg;
        tr_cfg.window       = 100;
        tr_cfg.alpha        = 0.95;
        tr_cfg.es_threshold = -0.3;
        tr_cfg.on_tail_event = [](double es) {
            spdlog::warn("[tail_risk] ES_95={:.4f} — extreme tail risk event", es);
        };
        tail_risk.update_config(tr_cfg);
    }
#endif

#ifdef LLMQUANT_LEVEL_CROSSING_ENABLED
    // BiasLevelCrossing: rolling zero-crossing rate (ZCR) — frequency proxy for bias stream.
    // High ZCR → mean-reverting, noisy; low ZCR → trending, low-frequency.
    llmquant::BiasLevelCrossing level_crossing;
    {
        llmquant::BiasLevelCrossing::Config lc_cfg;
        lc_cfg.window               = 50;
        lc_cfg.zcr_change_threshold = 0.1;
        lc_cfg.on_zcr_change = [](double old_zcr, double new_zcr) {
            spdlog::info("[level_cross] ZCR {:.3f} → {:.3f}", old_zcr, new_zcr);
        };
        level_crossing.update_config(lc_cfg);
    }
#endif

#ifdef LLMQUANT_CROSS_CORRELATOR_ENABLED
    // SignalCrossCorrelator: cross-correlation R(τ) between raw bias and EMA bias at lag τ.
    // Dominant lag τ* identifies lead-lag structure: τ*=0 → contemporaneous; τ*>0 → EMA leads.
    llmquant::SignalCrossCorrelator cross_corr_lag;
    {
        llmquant::SignalCrossCorrelator::Config xcor_cfg;
        xcor_cfg.ema_alpha  = 0.10;
        xcor_cfg.window     = 60;
        xcor_cfg.max_lag    = 5;
        xcor_cfg.on_dominant_lag_change = [](int old_l, int new_l, double r) {
            spdlog::info("[xcorr] dominant lag {} → {} (R={:.3f})", old_l, new_l, r);
        };
        cross_corr_lag.update_config(xcor_cfg);
    }
#endif

#ifdef LLMQUANT_VOL_RATIO_ENABLED
    // NarrativeVolatilityRatio: σ_short/σ_long EMA vol ratio — VIX term-structure proxy.
    // Ratio > 1 → near-term uncertainty elevated (fear/panic); < 1 → contango regime.
    llmquant::NarrativeVolatilityRatio vol_ratio;
    {
        llmquant::NarrativeVolatilityRatio::Config vr_cfg;
        vr_cfg.alpha_short      = 0.20;  // ≈ 5-period
        vr_cfg.alpha_long       = 0.04;  // ≈ 25-period
        vr_cfg.change_threshold = 0.2;
        vr_cfg.on_vol_regime_change = [](double old_r, double new_r) {
            spdlog::info("[vol_ratio] ratio {:.3f} → {:.3f}", old_r, new_r);
        };
        vol_ratio.update_config(vr_cfg);
    }
#endif

#ifdef LLMQUANT_PARABOLIC_SAR_ENABLED
    // SignalParabolicSAR: adaptive trailing stop indicator for bias trend tracking.
    // SAR reversal fires when bias crosses the trailing stop from the trending side.
    llmquant::SignalParabolicSAR parabolic_sar;
    {
        llmquant::SignalParabolicSAR::Config ps_cfg;
        ps_cfg.af_start = 0.02;
        ps_cfg.af_step  = 0.02;
        ps_cfg.af_max   = 0.20;
        ps_cfg.on_reversal = [](int dir, double sar_val) {
            spdlog::info("[psar] REVERSAL → {} (SAR={:.4f})", dir > 0 ? "uptrend" : "downtrend", sar_val);
        };
        parabolic_sar.update_config(ps_cfg);
    }
#endif

#ifdef LLMQUANT_BOLLINGER_BANDS_ENABLED
    // SignalBollingerBands: rolling SMA ± k*σ Bollinger Bands over bias stream.
    // Fires on_upper_touch / on_lower_touch on excursions; on_squeeze when bandwidth narrows.
    llmquant::SignalBollingerBands bollinger_bands;
    {
        llmquant::SignalBollingerBands::Config bb_cfg;
        bb_cfg.window            = 20;
        bb_cfg.k                 = 2.0;
        bb_cfg.min_samples       = 10;
        bb_cfg.squeeze_threshold = 0.005;
        bb_cfg.on_upper_touch = [](double bias) {
            spdlog::info("[bollinger] UPPER touch (bias={:.4f})", bias);
        };
        bb_cfg.on_lower_touch = [](double bias) {
            spdlog::info("[bollinger] LOWER touch (bias={:.4f})", bias);
        };
        bb_cfg.on_squeeze = [](double bw) {
            spdlog::info("[bollinger] SQUEEZE (bandwidth={:.4f})", bw);
        };
        bollinger_bands.update_config(bb_cfg);
    }
#endif

#ifdef LLMQUANT_IMPULSE_DETECTOR_ENABLED
    // BiasImpulseDetector: rolling z-score spike detector — identifies sudden bias surges.
    // Fires on_impulse when |z| exceeds z_threshold; tracks max-z for session peak.
    llmquant::BiasImpulseDetector impulse_det;
    {
        llmquant::BiasImpulseDetector::Config id_cfg;
        id_cfg.window      = 30;
        id_cfg.z_threshold = 3.0;
        id_cfg.min_samples = 10;
        id_cfg.on_impulse = [](double z, double bias) {
            spdlog::warn("[impulse] SPIKE z={:.2f} bias={:.4f}", z, bias);
        };
        impulse_det.update_config(id_cfg);
    }
#endif

#ifdef LLMQUANT_TREND_STRENGTH_INDEX_ENABLED
    // SignalTrendStrengthIndex: double-smoothed TSI ∈ (-100, +100) — momentum purity gauge.
    // Positive → sustained up-bias; negative → sustained down-bias; fires on_zero_cross / on_strong.
    llmquant::SignalTrendStrengthIndex trend_strength;
    {
        llmquant::SignalTrendStrengthIndex::Config ts_cfg;
        ts_cfg.r                  = 13;
        ts_cfg.s                  = 7;
        ts_cfg.strength_threshold = 30.0;
        ts_cfg.min_samples        = 20;
        ts_cfg.on_zero_cross = [](double prev_tsi, double curr_tsi) {
            spdlog::info("[tsi] ZERO-CROSS {:.1f} → {:.1f}", prev_tsi, curr_tsi);
        };
        ts_cfg.on_strong = [](double tsi) {
            spdlog::info("[tsi] STRONG TREND tsi={:.1f}", tsi);
        };
        trend_strength.update_config(ts_cfg);
    }
#endif

#ifdef LLMQUANT_MASS_INDEX_ENABLED
    // NarrativeMassIndex: range-expansion mass index — detects narrative reversal setups.
    // Bulge (MI > reversal_bulge) followed by drop below reversal_trigger → reversal signal.
    llmquant::NarrativeMassIndex mass_idx;
    {
        llmquant::NarrativeMassIndex::Config mi_cfg;
        mi_cfg.fast_period      = 9;
        mi_cfg.slow_period      = 25;
        mi_cfg.period           = 25;
        mi_cfg.reversal_bulge   = 27.0;
        mi_cfg.reversal_trigger = 26.5;
        mi_cfg.min_samples      = 15;
        mi_cfg.on_bulge_start = [](double mi) {
            spdlog::info("[mass_idx] BULGE START mi={:.3f}", mi);
        };
        mi_cfg.on_reversal_signal = [](double mi) {
            spdlog::warn("[mass_idx] REVERSAL SIGNAL mi={:.3f}", mi);
        };
        mass_idx.update_config(mi_cfg);
    }
#endif

#ifdef LLMQUANT_CHOPPINESS_INDEX_ENABLED
    // SignalChoppinessIndex: log-normalised ATR/range ratio ∈ [0, 100].
    // < 38.2 → strongly trending; > 61.8 → choppy/sideways; fires on_trending / on_choppy.
    llmquant::SignalChoppinessIndex choppiness;
    {
        llmquant::SignalChoppinessIndex::Config ci_cfg;
        ci_cfg.window              = 14;
        ci_cfg.min_samples         = 8;
        ci_cfg.trending_threshold  = 38.2;
        ci_cfg.choppy_threshold    = 61.8;
        ci_cfg.on_trending = [](double ci) {
            spdlog::info("[choppiness] TRENDING ci={:.1f}", ci);
        };
        ci_cfg.on_choppy = [](double ci) {
            spdlog::info("[choppiness] CHOPPY ci={:.1f}", ci);
        };
        choppiness.update_config(ci_cfg);
    }
#endif

#ifdef LLMQUANT_ACCELERATION_METER_ENABLED
    // SignalAccelerationMeter: EMA-smoothed velocity and acceleration of bias stream.
    // Fires on_inflection when acceleration changes sign; on_surge on large |accel| spikes.
    llmquant::SignalAccelerationMeter accel_meter;
    {
        llmquant::SignalAccelerationMeter::Config am_cfg;
        am_cfg.alpha_vel       = 0.2;
        am_cfg.alpha_acc       = 0.2;
        am_cfg.surge_threshold = 0.005;
        am_cfg.min_samples     = 5;
        am_cfg.on_inflection = [](double vel, double acc) {
            spdlog::info("[accel] INFLECTION vel={:.4f} acc={:.4f}", vel, acc);
        };
        am_cfg.on_surge = [](double acc) {
            spdlog::warn("[accel] SURGE acc={:.4f}", acc);
        };
        accel_meter.update_config(am_cfg);
    }
#endif

#ifdef LLMQUANT_FATIGUE_DETECTOR_ENABLED
    // NarrativeFatigueDetector: dual-EMA ratio saturation sensor for bias responsiveness.
    // is_fatigued() → fast and slow EMAs have converged; new tokens no longer move signal.
    llmquant::NarrativeFatigueDetector fatigue_det;
    {
        llmquant::NarrativeFatigueDetector::Config fd_cfg;
        fd_cfg.alpha_fast        = 0.3;
        fd_cfg.alpha_slow        = 0.05;
        fd_cfg.fatigue_threshold = 0.05;
        fd_cfg.min_samples       = 10;
        fd_cfg.on_fatigue = [](double ratio) {
            spdlog::info("[fatigue] FATIGUED ratio={:.4f}", ratio);
        };
        fd_cfg.on_recovery = [](double ratio) {
            spdlog::info("[fatigue] RECOVERED ratio={:.4f}", ratio);
        };
        fatigue_det.update_config(fd_cfg);
    }
#endif

#ifdef LLMQUANT_SKEWNESS_TRACKER_ENABLED
    // BiasSkewnessTracker: rolling third-moment skewness of bias distribution.
    // Positive skew → right-tail bullish asymmetry; negative → bearish tail-risk.
    llmquant::BiasSkewnessTracker skewness_tracker;
    {
        llmquant::BiasSkewnessTracker::Config sk_cfg;
        sk_cfg.window      = 30;
        sk_cfg.min_samples = 10;
        sk_cfg.on_positive_skew = [](double skew) {
            spdlog::info("[skew] POSITIVE skew={:.4f}", skew);
        };
        sk_cfg.on_negative_skew = [](double skew) {
            spdlog::info("[skew] NEGATIVE skew={:.4f}", skew);
        };
        skewness_tracker.update_config(sk_cfg);
    }
#endif

#ifdef LLMQUANT_ZERO_CROSS_RATE_ENABLED
    // SignalZeroCrossRate: rolling zero-crossing rate in [0, 1].
    // High ZCR → rapid oscillation; low ZCR → sustained directional trend.
    llmquant::SignalZeroCrossRate zcr_meter;
    {
        llmquant::SignalZeroCrossRate::Config zc_cfg;
        zc_cfg.window         = 20;
        zc_cfg.min_samples    = 5;
        zc_cfg.high_threshold = 0.5;
        zc_cfg.low_threshold  = 0.1;
        zc_cfg.on_high_zcr = [](double zcr) {
            spdlog::info("[zcr] HIGH oscillation zcr={:.3f}", zcr);
        };
        zc_cfg.on_low_zcr = [](double zcr) {
            spdlog::info("[zcr] LOW zcr (trending) zcr={:.3f}", zcr);
        };
        zcr_meter.update_config(zc_cfg);
    }
#endif

#ifdef LLMQUANT_BIAS_CORRELOGRAM_ENABLED
    // TokenBiasCorrelogram: sliding multi-lag ACF to detect cyclical bias patterns.
    // dominant_lag() → periodicity of narrative cycle; fires on_cycle_detected.
    llmquant::TokenBiasCorrelogram bias_correlogram;
    {
        llmquant::TokenBiasCorrelogram::Config bc_cfg;
        bc_cfg.window          = 64;
        bc_cfg.max_lag         = 16;
        bc_cfg.min_samples     = 20;
        bc_cfg.cycle_threshold = 0.4;
        bc_cfg.on_cycle_detected = [](int lag, double acf) {
            spdlog::info("[correlogram] CYCLE lag={} acf={:.3f}", lag, acf);
        };
        bias_correlogram.update_config(bc_cfg);
    }
#endif

#ifdef LLMQUANT_KURTOSIS_TRACKER_ENABLED
    // SignalKurtosisTracker: rolling excess kurtosis — fat-tail risk monitor for bias.
    // High kurtosis → frequent outlier spikes; fires on_fat_tail when kurtosis > threshold.
    llmquant::SignalKurtosisTracker kurtosis_tracker;
    {
        llmquant::SignalKurtosisTracker::Config kt_cfg;
        kt_cfg.window            = 30;
        kt_cfg.min_samples       = 10;
        kt_cfg.fat_tail_threshold = 1.0;
        kt_cfg.on_fat_tail = [](double kurt) {
            spdlog::warn("[kurtosis] FAT TAIL excess_kurt={:.3f}", kurt);
        };
        kt_cfg.on_normal_tail = [](double kurt) {
            spdlog::info("[kurtosis] NORMAL TAIL excess_kurt={:.3f}", kurt);
        };
        kurtosis_tracker.update_config(kt_cfg);
    }
#endif

#ifdef LLMQUANT_PERSISTENCE_INDEX_ENABLED
    // NarrativePersistenceIndex: run-length based persistence index ∈ (0, 1].
    // > 0.5 → auto-correlated trending; < 0.5 → mean-reverting oscillation.
    llmquant::NarrativePersistenceIndex persistence_idx;
    {
        llmquant::NarrativePersistenceIndex::Config pi_cfg;
        pi_cfg.window               = 30;
        pi_cfg.min_samples          = 10;
        pi_cfg.persistent_threshold = 0.6;
        pi_cfg.reverting_threshold  = 0.4;
        pi_cfg.on_persistent = [](double pi) {
            spdlog::info("[persistence] TRENDING pi={:.3f}", pi);
        };
        pi_cfg.on_mean_reverting = [](double pi) {
            spdlog::info("[persistence] REVERTING pi={:.3f}", pi);
        };
        persistence_idx.update_config(pi_cfg);
    }
#endif

#ifdef LLMQUANT_BIAS_ENTROPY_RATE_ENABLED
    // BiasEntropyRate: normalised Shannon entropy of the discretised bias stream.
    // 0 → fully structured; 1 → maximum randomness; fires on_high/low_entropy.
    llmquant::BiasEntropyRate bias_entropy;
    {
        llmquant::BiasEntropyRate::Config er_cfg;
        er_cfg.window         = 32;
        er_cfg.n_buckets      = 8;
        er_cfg.min_samples    = 10;
        er_cfg.high_threshold = 0.75;
        er_cfg.low_threshold  = 0.25;
        er_cfg.on_high_entropy = [](double h) {
            spdlog::info("[entropy] HIGH ENTROPY h={:.3f}", h);
        };
        er_cfg.on_low_entropy = [](double h) {
            spdlog::info("[entropy] LOW ENTROPY (structured) h={:.3f}", h);
        };
        bias_entropy.update_config(er_cfg);
    }
#endif

#ifdef LLMQUANT_DRAWDOWN_METER_ENABLED
    // SignalDrawdownMeter: running max-drawdown of cumulative bias.
    // Tracks peak, drawdown, and recovery events for session risk monitoring.
    llmquant::SignalDrawdownMeter drawdown_meter;
    {
        llmquant::SignalDrawdownMeter::Config dm_cfg;
        dm_cfg.drawdown_threshold = 0.1;
        dm_cfg.on_new_drawdown_high = [](double dd) {
            spdlog::warn("[drawdown] NEW MAX dd={:.4f}", dd);
        };
        dm_cfg.on_recovery = [](double cum) {
            spdlog::info("[drawdown] RECOVERY cum={:.4f}", cum);
        };
        drawdown_meter.update_config(dm_cfg);
    }
#endif

#ifdef LLMQUANT_CADENCE_ANALYSER_ENABLED
    // TokenCadenceAnalyser: inter-token arrival timing (IAI mean, CV, burst/gap detection).
    // Detects rapid token floods (bursts) and silent periods (gaps) in the stream.
    llmquant::TokenCadenceAnalyser cadence_analyser;
    {
        llmquant::TokenCadenceAnalyser::Config ca_cfg;
        ca_cfg.window             = 20;
        ca_cfg.min_samples        = 5;
        ca_cfg.burst_threshold_ms = 1.0;
        ca_cfg.gap_threshold_ms   = 200.0;
        ca_cfg.on_burst = [](double iai) {
            spdlog::warn("[cadence] BURST iai={:.2f}ms", iai);
        };
        ca_cfg.on_gap = [](double iai) {
            spdlog::info("[cadence] GAP iai={:.1f}ms", iai);
        };
        cadence_analyser.update_config(ca_cfg);
    }
#endif

#ifdef LLMQUANT_MEAN_REVERSION_SPEED_ENABLED
    // SignalMeanReversionSpeed: OLS AR(1) estimator for κ (speed of mean reversion).
    // θ < 1 → mean-reverting; θ > 1 → explosive/trending; fires on transition.
    llmquant::SignalMeanReversionSpeed mean_rev_speed;
    {
        llmquant::SignalMeanReversionSpeed::Config mr_cfg;
        mr_cfg.window               = 30;
        mr_cfg.min_samples          = 10;
        mr_cfg.fast_kappa_threshold = 1.0;
        mr_cfg.explosive_theta      = 1.0;
        mr_cfg.on_fast_reversion = [](double kappa) {
            spdlog::info("[mean_rev] FAST REVERSION kappa={:.3f}", kappa);
        };
        mr_cfg.on_explosive = [](double theta) {
            spdlog::warn("[mean_rev] EXPLOSIVE theta={:.3f}", theta);
        };
        mean_rev_speed.update_config(mr_cfg);
    }
#endif

#ifdef LLMQUANT_CLUSTER_DETECTOR_ENABLED
    // NarrativeClusterDetector: online k-means bias regime classifier (k=3 regimes).
    // Fires on_cluster_change when narrative regime assignment switches.
    llmquant::NarrativeClusterDetector cluster_detector;
    {
        llmquant::NarrativeClusterDetector::Config cd_cfg;
        cd_cfg.k           = 3;
        cd_cfg.alpha       = 0.1;
        cd_cfg.min_samples = 10;
        cd_cfg.on_cluster_change = [](int old_c, int new_c, double centroid) {
            spdlog::info("[cluster] REGIME CHANGE {} → {} (centroid={:.4f})", old_c, new_c, centroid);
        };
        cluster_detector.update_config(cd_cfg);
    }
#endif

#ifdef LLMQUANT_VOL_BREAKOUT_ENABLED
    // BiasVolatilityBreakout: short/long rolling std ratio for vol expansion/contraction.
    // ratio > breakout_ratio → expansion; ratio < contraction_ratio → compression.
    llmquant::BiasVolatilityBreakout vol_breakout;
    {
        llmquant::BiasVolatilityBreakout::Config vb_cfg;
        vb_cfg.short_window      = 10;
        vb_cfg.long_window       = 30;
        vb_cfg.min_samples       = 10;
        vb_cfg.breakout_ratio    = 1.5;
        vb_cfg.contraction_ratio = 0.7;
        vb_cfg.on_expansion = [](double ratio) {
            spdlog::warn("[vol_breakout] EXPANSION ratio={:.3f}", ratio);
        };
        vb_cfg.on_contraction = [](double ratio) {
            spdlog::info("[vol_breakout] CONTRACTION ratio={:.3f}", ratio);
        };
        vol_breakout.update_config(vb_cfg);
    }
#endif

#ifdef LLMQUANT_STOCHASTIC_OSC_ENABLED
    // SignalStochasticOscillator: %K/%D stochastic exhaustion detector for bias stream.
    // %K near 100 → overbought narrative; %K near 0 → oversold; K/D cross = mean reversion.
    llmquant::SignalStochasticOscillator stochastic_osc;
    {
        llmquant::SignalStochasticOscillator::Config so_cfg;
        so_cfg.k_period             = 14;
        so_cfg.d_period             = 3;
        so_cfg.overbought_threshold = 80.0;
        so_cfg.oversold_threshold   = 20.0;
        so_cfg.on_overbought = [](double k) {
            spdlog::info("[stoch] OVERBOUGHT %K={:.1f}", k);
        };
        so_cfg.on_oversold = [](double k) {
            spdlog::info("[stoch] OVERSOLD %K={:.1f}", k);
        };
        so_cfg.on_kd_bullish_cross = [](double k, double d) {
            spdlog::info("[stoch] K/D BULLISH CROSS %K={:.1f} %D={:.1f}", k, d);
        };
        so_cfg.on_kd_bearish_cross = [](double k, double d) {
            spdlog::info("[stoch] K/D BEARISH CROSS %K={:.1f} %D={:.1f}", k, d);
        };
        stochastic_osc.update_config(so_cfg);
    }
#endif

#ifdef LLMQUANT_BIAS_ACF_ENABLED
    // BiasAutocorrelationFunction: rolling multi-lag ACF for narrative autocorrelation.
    // Positive r1 → trending momentum; negative r1 → mean-reversion.
    llmquant::BiasAutocorrelationFunction bias_acf;
    {
        llmquant::BiasAutocorrelationFunction::Config acf_cfg;
        acf_cfg.window             = 64;
        acf_cfg.max_lag            = 16;
        acf_cfg.min_samples        = 32;
        acf_cfg.periodic_threshold = 0.4;
        acf_cfg.on_periodic_pattern = [](int lag, double r) {
            spdlog::info("[acf] PERIODIC PATTERN dominant_lag={} r={:.3f}", lag, r);
        };
        acf_cfg.on_lag1_sign_change = [](double r1) {
            spdlog::info("[acf] r1 SIGN FLIP {:.3f} (momentum↔mean-rev regime change)", r1);
        };
        bias_acf.update_config(acf_cfg);
    }
#endif

#ifdef LLMQUANT_ONLINE_GRANGER_ENABLED
    // OnlineGrangerCausality: VAR(1) Granger causality between bias and confidence.
    llmquant::OnlineGrangerCausality granger;
    {
        llmquant::OnlineGrangerCausality::Config gc_cfg;
        gc_cfg.window      = 64;
        gc_cfg.min_samples = 32;
        gc_cfg.f_threshold = 3.84;
        gc_cfg.on_x_causes_y = [](double f) {
            spdlog::info("[granger] BIAS->CONFIDENCE F={:.2f} (bias leads confidence)", f);
        };
        gc_cfg.on_y_causes_x = [](double f) {
            spdlog::info("[granger] CONFIDENCE->BIAS F={:.2f} (confidence leads bias)", f);
        };
        gc_cfg.on_bidirectional = []() {
            spdlog::warn("[granger] BIDIRECTIONAL causality — feedback loop detected");
        };
        granger.update_config(gc_cfg);
    }
#endif

#ifdef LLMQUANT_MACD_HISTOGRAM_ENABLED
    // SignalMACDHistogram: MACD fast/slow/signal EMAs applied to bias stream.
    // Histogram zero-cross = momentum shift; divergence = overextended move.
    llmquant::SignalMACDHistogram macd_hist;
    {
        llmquant::SignalMACDHistogram::Config mh_cfg;
        mh_cfg.alpha_fast          = 2.0 / (12 + 1);
        mh_cfg.alpha_slow          = 2.0 / (26 + 1);
        mh_cfg.alpha_signal        = 2.0 / (9 + 1);
        mh_cfg.min_samples         = 26;
        mh_cfg.divergence_threshold = 0.005;
        mh_cfg.on_zero_cross = [](double hist) {
            spdlog::info("[macd] ZERO CROSS hist={:.6f} (momentum direction flip)", hist);
        };
        mh_cfg.on_divergence = [](double hist) {
            spdlog::warn("[macd] DIVERGENCE hist={:.6f} (overextended momentum)", hist);
        };
        macd_hist.update_config(mh_cfg);
    }
#endif

#ifdef LLMQUANT_REGIME_MARKOV_ENABLED
    // NarrativeRegimeMarkov: Markov transition matrix over bias-binned regimes.
    // Steady-state distribution reveals long-run narrative bias tendencies.
    llmquant::NarrativeRegimeMarkov regime_markov;
    {
        llmquant::NarrativeRegimeMarkov::Config rm_cfg;
        rm_cfg.n_regimes   = 5;
        rm_cfg.bias_range  = 0.05;
        rm_cfg.min_samples = 10;
        rm_cfg.on_regime_change = [](int old_s, int new_s) {
            spdlog::info("[regime_markov] REGIME CHANGE {} → {} (narrative shift)", old_s, new_s);
        };
        regime_markov.update_config(rm_cfg);
    }
#endif

#ifdef LLMQUANT_CONCENTRATION_RISK_ENABLED
    // BiasConcentrationRisk: rolling HHI over |bias| magnitudes.
    // High HHI = one dominant bias period; low HHI = dispersed, diversified signal.
    llmquant::BiasConcentrationRisk conc_risk;
    {
        llmquant::BiasConcentrationRisk::Config cr_cfg;
        cr_cfg.window                  = 20;
        cr_cfg.min_samples             = 5;
        cr_cfg.concentration_threshold = 0.25;
        cr_cfg.on_concentrated = [](double hhi) {
            spdlog::warn("[conc_risk] CONCENTRATED hhi={:.4f} (bias dominated by single period)", hhi);
        };
        cr_cfg.on_dispersed = [](double hhi) {
            spdlog::info("[conc_risk] DISPERSED hhi={:.4f} (bias energy spreading out)", hhi);
        };
        conc_risk.update_config(cr_cfg);
    }
#endif

#ifdef LLMQUANT_WILLIAMS_R_ENABLED
    // SignalWilliamsR: Williams %R oscillator over rolling bias window.
    // %R near 0 → overbought narrative; %R near -100 → oversold.
    llmquant::SignalWilliamsR williams_r;
    {
        llmquant::SignalWilliamsR::Config wr_cfg;
        wr_cfg.period               = 14;
        wr_cfg.min_samples          = 5;
        wr_cfg.overbought_threshold = -20.0;
        wr_cfg.oversold_threshold   = -80.0;
        wr_cfg.on_overbought = [](double wr) {
            spdlog::warn("[williams_r] OVERBOUGHT wr={:.1f} (narrative stretched positive)", wr);
        };
        wr_cfg.on_oversold = [](double wr) {
            spdlog::warn("[williams_r] OVERSOLD wr={:.1f} (narrative stretched negative)", wr);
        };
        williams_r.update_config(wr_cfg);
    }
#endif

#ifdef LLMQUANT_INFLUENCE_DECAY_ENABLED
    // TokenInfluenceDecay: separate positive/negative EMA accumulators.
    // Dominance switch = long-run influence regime change; net = directional pressure.
    llmquant::TokenInfluenceDecay influence_decay;
    {
        llmquant::TokenInfluenceDecay::Config id_cfg;
        id_cfg.alpha_pos           = 0.12;
        id_cfg.alpha_neg           = 0.12;
        id_cfg.dominance_threshold = 0.015;
        id_cfg.min_samples         = 10;
        id_cfg.on_positive_dominant = [](double net) {
            spdlog::info("[influence] POS DOMINANT net={:.4f} (bullish influence prevails)", net);
        };
        id_cfg.on_negative_dominant = [](double net) {
            spdlog::info("[influence] NEG DOMINANT net={:.4f} (bearish influence prevails)", net);
        };
        influence_decay.update_config(id_cfg);
    }
#endif

#ifdef LLMQUANT_POLARITY_SHIFT_ENABLED
    // NarrativePolarityShift: tracks positive-bias fraction over rolling window.
    // Bullish shift = narrative turning positive; bearish shift = narrative turning negative.
    llmquant::NarrativePolarityShift polarity_shift;
    {
        llmquant::NarrativePolarityShift::Config ps_cfg;
        ps_cfg.window          = 20;
        ps_cfg.min_samples     = 5;
        ps_cfg.bull_threshold  = 0.65;
        ps_cfg.bear_threshold  = 0.35;
        ps_cfg.on_bullish = [](double frac) {
            spdlog::info("[polarity] BULLISH SHIFT pos_frac={:.3f} (majority bias now positive)", frac);
        };
        ps_cfg.on_bearish = [](double frac) {
            spdlog::info("[polarity] BEARISH SHIFT pos_frac={:.3f} (majority bias now negative)", frac);
        };
        polarity_shift.update_config(ps_cfg);
    }
#endif

#ifdef LLMQUANT_CHANDE_OSC_ENABLED
    // SignalChandeOscillator: CMO = 100*(ΣUp-ΣDown)/(ΣUp+ΣDown).
    // >+50 overbought, <-50 oversold in narrative momentum.
    llmquant::SignalChandeOscillator chande_osc;
    {
        llmquant::SignalChandeOscillator::Config co_cfg;
        co_cfg.window               = 14;
        co_cfg.min_samples          = 5;
        co_cfg.overbought_threshold = 50.0;
        co_cfg.oversold_threshold   = -50.0;
        co_cfg.on_overbought = [](double cmo) {
            spdlog::warn("[chande] OVERBOUGHT cmo={:.1f} (narrative momentum overextended up)", cmo);
        };
        co_cfg.on_oversold = [](double cmo) {
            spdlog::warn("[chande] OVERSOLD cmo={:.1f} (narrative momentum overextended down)", cmo);
        };
        chande_osc.update_config(co_cfg);
    }
#endif

#ifdef LLMQUANT_DONCHIAN_CHANNEL_ENABLED
    // SignalDonchianChannel: rolling high/low bands over bias window.
    // Breakout above upper band = new bias high; below lower = new bias low.
    llmquant::SignalDonchianChannel donchian_ch;
    {
        llmquant::SignalDonchianChannel::Config dc_cfg;
        dc_cfg.window      = 20;
        dc_cfg.min_samples = 5;
        dc_cfg.on_upper_breakout = [](double new_hi, double prev_hi) {
            spdlog::info("[donchian] UPPER BREAKOUT {:.4f} > prev {:.4f} (bias new high)", new_hi, prev_hi);
        };
        dc_cfg.on_lower_breakout = [](double new_lo, double prev_lo) {
            spdlog::info("[donchian] LOWER BREAKOUT {:.4f} < prev {:.4f} (bias new low)", new_lo, prev_lo);
        };
        donchian_ch.update_config(dc_cfg);
    }
#endif

#ifdef LLMQUANT_BIAS_HISTOGRAM_ENABLED
    // TokenBiasHistogram: fixed-bin histogram of bias distribution.
    // Mode changes = distributional shift in narrative bias.
    llmquant::TokenBiasHistogram bias_histogram;
    {
        llmquant::TokenBiasHistogram::Config bh_cfg;
        bh_cfg.n_bins      = 20;
        bh_cfg.lo          = -1.0;
        bh_cfg.hi          =  1.0;
        bh_cfg.min_samples = 5;
        bh_cfg.on_mode_change = [](int new_bin, int prev_bin) {
            spdlog::info("[histogram] MODE CHANGE bin {} → {} (dominant bias region shifted)", prev_bin, new_bin);
        };
        bias_histogram.update_config(bh_cfg);
    }
#endif

#ifdef LLMQUANT_EXP_SMOOTHING_ENABLED
    // BiasExponentialSmoothing: Holt-Winters level+trend smoothing for bias.
    // Large forecast error = structural break in bias dynamics.
    llmquant::BiasExponentialSmoothing exp_smooth;
    {
        llmquant::BiasExponentialSmoothing::Config es_cfg;
        es_cfg.alpha           = 0.2;
        es_cfg.beta            = 0.1;
        es_cfg.error_threshold = 0.05;
        es_cfg.min_samples     = 5;
        es_cfg.on_large_error = [](double actual, double fc) {
            spdlog::warn("[exp_smooth] LARGE ERROR actual={:.4f} forecast={:.4f} (bias regime break)", actual, fc);
        };
        exp_smooth.update_config(es_cfg);
    }
#endif

#ifdef LLMQUANT_RELATIVE_VIGOR_ENABLED
    // SignalRelativeVigorIndex: RVI = EMA(change) / EMA(|bias|) with signal line.
    // Bullish cross (RVI>signal) = upward momentum strengthening.
    llmquant::SignalRelativeVigorIndex rvi_signal;
    {
        llmquant::SignalRelativeVigorIndex::Config rv_cfg;
        rv_cfg.alpha_rvi    = 0.2;
        rv_cfg.alpha_signal = 0.25;
        rv_cfg.min_samples  = 5;
        rv_cfg.on_bullish_cross = [](double rvi) {
            spdlog::info("[rvi] BULLISH CROSS rvi={:.6f} (upward momentum strengthening)", rvi);
        };
        rv_cfg.on_bearish_cross = [](double rvi) {
            spdlog::info("[rvi] BEARISH CROSS rvi={:.6f} (downward momentum strengthening)", rvi);
        };
        rvi_signal.update_config(rv_cfg);
    }
#endif

#ifdef LLMQUANT_SENTIMENT_VELOCITY_ENABLED
    // NarrativeSentimentVelocity: first/second derivative of bias EMA.
    // Positive surge = accelerating bullish bias; negative = bearish acceleration.
    llmquant::NarrativeSentimentVelocity sent_velocity;
    {
        llmquant::NarrativeSentimentVelocity::Config sv_cfg;
        sv_cfg.alpha           = 0.15;
        sv_cfg.surge_threshold = 0.005;
        sv_cfg.min_samples     = 5;
        sv_cfg.on_positive_surge = [](double vel) {
            spdlog::info("[sent_vel] POS SURGE vel={:.6f} (bullish momentum accelerating)", vel);
        };
        sv_cfg.on_negative_surge = [](double vel) {
            spdlog::info("[sent_vel] NEG SURGE vel={:.6f} (bearish momentum accelerating)", vel);
        };
        sent_velocity.update_config(sv_cfg);
    }
#endif

#ifdef LLMQUANT_ZSCORE_NORMALISER_ENABLED
    // BiasZScoreNormaliser: rolling z-score for bias outlier detection.
    // |z| > threshold = statistically extreme bias event relative to recent history.
    llmquant::BiasZScoreNormaliser zscore_norm;
    {
        llmquant::BiasZScoreNormaliser::Config zn_cfg;
        zn_cfg.window            = 30;
        zn_cfg.min_samples       = 10;
        zn_cfg.extreme_threshold = 2.5;
        zn_cfg.on_positive_extreme = [](double z) {
            spdlog::warn("[zscore] POS EXTREME z={:.2f} (statistically high bias outlier)", z);
        };
        zn_cfg.on_negative_extreme = [](double z) {
            spdlog::warn("[zscore] NEG EXTREME z={:.2f} (statistically low bias outlier)", z);
        };
        zscore_norm.update_config(zn_cfg);
    }
#endif

#ifdef LLMQUANT_KELTNER_CHANNEL_ENABLED
    // SignalKeltnerChannel: EMA ± k×ATR dynamic bands for bias.
    // Breakout above upper = overextended positive narrative.
    llmquant::SignalKeltnerChannel keltner_ch;
    {
        llmquant::SignalKeltnerChannel::Config kc_cfg;
        kc_cfg.alpha_ema  = 0.1;
        kc_cfg.alpha_atr  = 0.1;
        kc_cfg.multiplier = 2.0;
        kc_cfg.min_samples = 5;
        kc_cfg.on_upper_break = [](double bias, double upper) {
            spdlog::warn("[keltner] UPPER BREAK bias={:.4f} > upper={:.4f}", bias, upper);
        };
        kc_cfg.on_lower_break = [](double bias, double lower) {
            spdlog::warn("[keltner] LOWER BREAK bias={:.4f} < lower={:.4f}", bias, lower);
        };
        keltner_ch.update_config(kc_cfg);
    }
#endif

#ifdef LLMQUANT_BURST_INTENSITY_ENABLED
    // TokenBurstIntensity: short/long variance ratio burst detector.
    // High ratio = sudden volatility spike in token bias stream.
    llmquant::TokenBurstIntensity burst_intensity;
    {
        llmquant::TokenBurstIntensity::Config bi_cfg;
        bi_cfg.short_window = 5;
        bi_cfg.long_window  = 30;
        bi_cfg.burst_ratio  = 4.0;
        bi_cfg.min_samples  = 15;
        bi_cfg.on_burst_start = [](double ratio) {
            spdlog::warn("[burst] BURST START ratio={:.2f} (token bias volatility spike)", ratio);
        };
        bi_cfg.on_burst_end = [](double ratio) {
            spdlog::info("[burst] BURST END ratio={:.2f} (token bias volatility normalising)", ratio);
        };
        burst_intensity.update_config(bi_cfg);
    }
#endif

#ifdef LLMQUANT_TRIPLE_EMA_ENABLED
    // SignalTripleEMAOscillator: TRIX = ROC of triple-smoothed EMA.
    // TRIX eliminates noise up to 3× the EMA period; trend flips are high-conviction.
    llmquant::SignalTripleEMAOscillator triple_ema;
    {
        llmquant::SignalTripleEMAOscillator::Config te_cfg;
        te_cfg.alpha        = 2.0 / (14 + 1);
        te_cfg.alpha_signal = 2.0 / (9 + 1);
        te_cfg.min_samples  = 14;
        te_cfg.on_zero_cross = [](double trix) {
            spdlog::info("[trix] ZERO CROSS trix={:.6f} (trend flip confirmation)", trix);
        };
        te_cfg.on_signal_cross = [](double trix, double sig) {
            spdlog::info("[trix] SIGNAL CROSS trix={:.6f} sig={:.6f} (early trend signal)", trix, sig);
        };
        triple_ema.update_config(te_cfg);
    }
#endif

#ifdef LLMQUANT_COHERENCE_TRACKER_ENABLED
    // NarrativeCoherenceTracker: fraction of window agreeing with majority direction.
    // High coherence = unified narrative; low = conflicted/mixed signal.
    llmquant::NarrativeCoherenceTracker coherence_tracker;
    {
        llmquant::NarrativeCoherenceTracker::Config ct_cfg;
        ct_cfg.window          = 20;
        ct_cfg.min_samples     = 5;
        ct_cfg.high_threshold  = 0.75;
        ct_cfg.low_threshold   = 0.40;
        ct_cfg.on_high_coherence = [](double coh) {
            spdlog::info("[coherence] HIGH coh={:.3f} (narrative unified)", coh);
        };
        ct_cfg.on_low_coherence = [](double coh) {
            spdlog::warn("[coherence] LOW coh={:.3f} (narrative conflicted)", coh);
        };
        coherence_tracker.update_config(ct_cfg);
    }
#endif

#ifdef LLMQUANT_LOCAL_EXTREMA_ENABLED
    // BiasLocalExtrema: zigzag local peak/trough detector.
    // Peaks and troughs mark bias exhaustion points — potential reversal signals.
    llmquant::BiasLocalExtrema local_extrema;
    {
        llmquant::BiasLocalExtrema::Config le_cfg;
        le_cfg.reversal_threshold = 0.02;
        le_cfg.min_samples        = 5;
        le_cfg.on_peak = [](double val) {
            spdlog::info("[extrema] PEAK confirmed val={:.4f} (local bias maximum)", val);
        };
        le_cfg.on_trough = [](double val) {
            spdlog::info("[extrema] TROUGH confirmed val={:.4f} (local bias minimum)", val);
        };
        local_extrema.update_config(le_cfg);
    }
#endif

#ifdef LLMQUANT_ADAPTIVE_FILTER_ENABLED
    // SignalAdaptiveThresholdFilter: ATR-based adaptive filter — separates signal from noise.
    // Breakout above multiplier×ATR = statistically significant bias move.
    llmquant::SignalAdaptiveThresholdFilter adaptive_filter;
    {
        llmquant::SignalAdaptiveThresholdFilter::Config af_cfg;
        af_cfg.alpha_atr   = 0.1;
        af_cfg.multiplier  = 2.0;
        af_cfg.min_samples = 5;
        af_cfg.on_above = [](double bias, double thresh) {
            spdlog::warn("[adapt_filter] ABOVE bias={:.4f} thresh={:.4f} (significant positive bias)", bias, thresh);
        };
        af_cfg.on_below = [](double bias, double thresh) {
            spdlog::warn("[adapt_filter] BELOW bias={:.4f} thresh=-{:.4f} (significant negative bias)", bias, thresh);
        };
        adaptive_filter.update_config(af_cfg);
    }
#endif

#ifdef LLMQUANT_PRESSURE_GAUGE_ENABLED
    // NarrativePressureGauge: fast/slow EMA spread as directional pressure.
    // Positive pressure = bullish momentum building; negative = bearish.
    llmquant::NarrativePressureGauge pressure_gauge;
    {
        llmquant::NarrativePressureGauge::Config pg_cfg;
        pg_cfg.alpha_fast          = 0.3;
        pg_cfg.alpha_slow          = 0.05;
        pg_cfg.pressure_threshold  = 0.008;
        pg_cfg.min_samples         = 5;
        pg_cfg.on_positive_pressure = [](double p) {
            spdlog::info("[pressure] POSITIVE p={:.6f} (bullish narrative pressure building)", p);
        };
        pg_cfg.on_negative_pressure = [](double p) {
            spdlog::info("[pressure] NEGATIVE p={:.6f} (bearish narrative pressure building)", p);
        };
        pressure_gauge.update_config(pg_cfg);
    }
#endif

#ifdef LLMQUANT_WAVELET_DECOMPOSER_ENABLED
    // WaveletSignalDecomposer: Haar DWT decomposition of signal stream into J=4 frequency bands.
    // Low-level detail spikes → rapid oscillation; high-level → slow regime drift.
    llmquant::WaveletSignalDecomposer wavelet_decomp;
    {
        llmquant::WaveletSignalDecomposer::Config wd_cfg;
        wd_cfg.levels          = 4;
        wd_cfg.window          = 64;
        wd_cfg.spike_threshold = 0.02;
        wd_cfg.on_high_freq_spike = [](int lvl, double energy) {
            spdlog::warn("[wavelet] detail spike at level {} energy={:.4f}", lvl, energy);
        };
        wd_cfg.on_spike_clear = []() {
            spdlog::info("[wavelet] detail energy back to normal");
        };
        wavelet_decomp.update_config(wd_cfg);
    }
#endif

#ifdef LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED
    // ReinforcementSignalWeighter: UCB1 bandit adaptively weights signal components.
    // Arms: primary LLM output ("llm_primary") and technical indicators ("technical").
    llmquant::ReinforcementSignalWeighter rl_weighter;
    {
        llmquant::ReinforcementSignalWeighter::Config rw_cfg;
        rw_cfg.arms = {{"llm_primary", 1.0}, {"technical", 1.0}};
        rw_cfg.exploration_c = 1.414;
        rw_cfg.on_dominant_arm_change = [](const std::string& arm) {
            spdlog::info("[rl_weighter] dominant arm → '{}'", arm);
        };
        rl_weighter.update_config(rw_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_CONVEXITY_ENABLED
    // SignalConvexityMeter: tracks d2 of bias stream to classify acceleration regime.
    llmquant::SignalConvexityMeter convexity_meter;
    {
        llmquant::SignalConvexityMeter::Config cm_cfg;
        cm_cfg.alpha_d1         = 0.3;
        cm_cfg.alpha_d2         = 0.15;
        cm_cfg.convex_threshold = 0.005;
        cm_cfg.min_samples      = 5;
        cm_cfg.on_accelerating = [](double d2) {
            spdlog::info("[convexity] ACCELERATING (d2={:.4f}) — momentum trade", d2);
        };
        cm_cfg.on_decelerating = [](double d2) {
            spdlog::info("[convexity] DECELERATING (d2={:.4f}) — mean-reversion alert", d2);
        };
        cm_cfg.on_stabilized = []() {
            spdlog::info("[convexity] regime STABLE — reduce directional exposure");
        };
        convexity_meter.update_config(cm_cfg);
    }
#endif

#ifdef LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED
    // NarrativeEntropyClock: accumulates KL surprisal from bias stream.
    // Fires when narrative information budget is exhausted (signal is stale/repetitive).
    llmquant::NarrativeEntropyClock entropy_clock;
    {
        llmquant::NarrativeEntropyClock::Config ec_cfg;
        ec_cfg.n_bins        = 20;
        ec_cfg.budget_nats   = 5.0;
        ec_cfg.min_samples   = 10;
        ec_cfg.on_budget_exhausted = [](double nats) {
            spdlog::warn("[entropy_clock] narrative exhausted after {:.2f} nats — signal stale", nats);
        };
        entropy_clock.update_config(ec_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED
    // SignalDecayHalfLife: OLS half-life estimator for signal momentum.
    llmquant::SignalDecayHalfLife decay_halflife;
    {
        llmquant::SignalDecayHalfLife::Config dh_cfg;
        dh_cfg.window         = 32;
        dh_cfg.min_samples    = 8;
        dh_cfg.fast_threshold = 5.0;
        dh_cfg.slow_threshold = 60.0;
        dh_cfg.on_fast_decay = [](double hl) {
            spdlog::warn("[halflife] FAST decay t½={:.1f} samples — momentum fading", hl);
        };
        dh_cfg.on_slow_decay = [](double hl) {
            spdlog::info("[halflife] SLOW decay t½={:.1f} samples — persistent signal", hl);
        };
        decay_halflife.update_config(dh_cfg);
    }
#endif

#ifdef LLMQUANT_BAYESIAN_SENTIMENT_ENABLED
    // BayesianSentimentPrior: Normal-Gamma conjugate posterior over signal mean.
    llmquant::BayesianSentimentPrior bayes_prior;
    {
        llmquant::BayesianSentimentPrior::Config bp_cfg;
        bp_cfg.prior_mean        = 0.0;
        bp_cfg.prior_kappa       = 2.0;
        bp_cfg.prior_alpha       = 2.0;
        bp_cfg.prior_beta        = 0.5;
        bp_cfg.shift_threshold   = 2.5;
        bp_cfg.min_samples       = 5;
        bp_cfg.on_belief_shift = [](double pm, double ps) {
            spdlog::info("[bayes] BELIEF SHIFT: μ={:.4f} ±{:.4f} — signal decisive", pm, ps);
        };
        bp_cfg.on_belief_restored = []() {
            spdlog::info("[bayes] belief restored to prior — signal weakened");
        };
        bayes_prior.update_config(bp_cfg);
    }
#endif

#ifdef LLMQUANT_WEIGHT_HISTOGRAM_ENABLED
    // TokenWeightHistogram: 20-bucket histogram of bias values [-1, 1].
    // Tracks mode bucket and entropy; fires on_distribution_shift on mode change.
    llmquant::TokenWeightHistogram weight_histogram;
    {
        llmquant::TokenWeightHistogram::Config wh_cfg;
        wh_cfg.range_min = -1.0;
        wh_cfg.range_max =  1.0;
        wh_cfg.on_distribution_shift = [](int new_mode) {
            spdlog::info("[histogram] distribution shift — new mode bucket={}", new_mode);
        };
        weight_histogram.update_config(wh_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_SLOPE_ENABLED
    // SignalSlopeMeter: OLS regression slope over rolling window.
    // Positive slope = accelerating bullish; negative = accelerating bearish.
    llmquant::SignalSlopeMeter signal_slope;
    {
        llmquant::SignalSlopeMeter::Config ss_cfg;
        ss_cfg.window                 = 20;
        ss_cfg.acceleration_threshold = 0.02;
        ss_cfg.saturation_slope       = 0.05;
        ss_cfg.on_trend_acceleration  = [](double s) {
            spdlog::info("[slope] ACCELERATING slope={:.4f}", s);
        };
        signal_slope.update_config(ss_cfg);
    }
#endif

#ifdef LLMQUANT_RUN_LENGTH_ENABLED
    // BiasRunLengthEncoder: run-length stats on bias sign sequence.
    // Long runs → persistent directional regime; short runs → noisy/mean-reverting.
    llmquant::BiasRunLengthEncoder run_length;
    {
        llmquant::BiasRunLengthEncoder::Config rl_cfg;
        rl_cfg.long_run_threshold = 10;
        rl_cfg.avg_alpha          = 0.20;
        rl_cfg.on_long_run = [](int dir, int len) {
            spdlog::info("[run_length] LONG RUN dir={} len={} — persistent {} regime",
                         dir, len, dir > 0 ? "bullish" : "bearish");
        };
        run_length.update_config(rl_cfg);
    }
#endif

#ifdef LLMQUANT_COVERAGE_METER_ENABLED
    // SignalCoverageMeter: rolling bias range coverage [0,1].
    // Low coverage = signals stuck in narrow band; high = full dynamic range used.
    llmquant::SignalCoverageMeter coverage_meter;
    {
        llmquant::SignalCoverageMeter::Config cm_cfg;
        cm_cfg.window               = 50;
        cm_cfg.range_min            = -1.0;
        cm_cfg.range_max            =  1.0;
        cm_cfg.expansion_threshold  = 0.5;
        cm_cfg.on_range_expansion   = [](double cov) {
            spdlog::info("[coverage] range expanded coverage={:.3f} — full dynamic range in use", cov);
        };
        coverage_meter.update_config(cm_cfg);
    }
#endif

#ifdef LLMQUANT_BIAS_HYSTERESIS_ENABLED
    // BiasHysteresisGate: N-tick debounce on bias crossings to prevent whipsaw signals.
    llmquant::BiasHysteresisGate hysteresis_gate;
    {
        llmquant::BiasHysteresisGate::Config hg_cfg;
        hg_cfg.enter_threshold = 0.10;
        hg_cfg.exit_threshold  = 0.05;
        hg_cfg.required_ticks  = 3;
        hg_cfg.on_gate_open  = [](double bias) {
            spdlog::info("[hysteresis] GATE OPEN bias={:.4f} — sustained above threshold", bias);
        };
        hg_cfg.on_gate_close = [](double bias) {
            spdlog::info("[hysteresis] gate closed bias={:.4f}", bias);
        };
        hysteresis_gate.update_config(hg_cfg);
    }
#endif

#ifdef LLMQUANT_REALIZED_VOL_ENABLED
    // RealizedVolatilityTracker: rolling sqrt(mean squared return) for dynamic sizing.
    llmquant::RealizedVolatilityTracker realized_vol;
    {
        llmquant::RealizedVolatilityTracker::Config rv_cfg;
        rv_cfg.window           = 30;
        rv_cfg.alert_threshold  = 0.5;
        rv_cfg.clear_hysteresis = 0.7;
        rv_cfg.on_high_volatility = [](double rv) {
            spdlog::warn("[realized_vol] HIGH VOL rv={:.4f} — consider position de-sizing", rv);
        };
        rv_cfg.on_low_volatility = [](double rv) {
            spdlog::info("[realized_vol] vol cleared rv={:.4f}", rv);
        };
        realized_vol.update_config(rv_cfg);
    }
#endif

#ifdef LLMQUANT_CAUSAL_TRACER_ENABLED
    // SignalCausalChainTracer: rolling ring of top-N tokens by signal contribution.
    llmquant::SignalCausalChainTracer causal_tracer;
    {
        llmquant::SignalCausalChainTracer::Config ct_cfg;
        ct_cfg.window            = 16;
        ct_cfg.strong_threshold  = 0.05;
        ct_cfg.on_strong_token   = [](const std::string& tok, double contrib) {
            spdlog::info("[causal] strong token='{}' contrib={:.4f}", tok, contrib);
        };
        causal_tracer.update_config(ct_cfg);
    }
#endif

#ifdef LLMQUANT_DEPENDENCY_MAPPER_ENABLED
    // TokenDependencyMapper: sliding-window co-occurrence matrix with cluster detection.
    llmquant::TokenDependencyMapper dep_mapper;
    {
        llmquant::TokenDependencyMapper::Config dm_cfg;
        dm_cfg.window             = 32;
        dm_cfg.min_count          = 4;
        dm_cfg.min_cluster_size   = 3;
        dm_cfg.recompute_interval = 16;
        dm_cfg.on_cluster_detected = [](const std::vector<int>& ids) {
            spdlog::info("[dep_mapper] cluster detected size={}", ids.size());
        };
        dep_mapper.update_config(dm_cfg);
    }
#endif

#ifdef LLMQUANT_FREQ_ANALYSER_ENABLED
    // BiasFrequencyAnalyser: DCT-II dominant oscillation frequency extractor.
    llmquant::BiasFrequencyAnalyser freq_analyser;
    {
        llmquant::BiasFrequencyAnalyser::Config fa_cfg;
        fa_cfg.window    = 32;
        fa_cfg.on_dominant_frequency_change = [](int old_k, int k, double power) {
            spdlog::info("[freq] dominant k changed {}→{} power={:.4f}", old_k, k, power);
        };
        freq_analyser.update_config(fa_cfg);
    }
#endif

#ifdef LLMQUANT_ENTROPY_RATCHET_ENABLED
    // SignalEntropyRatchet: one-way ratchet on Shannon entropy of the bias distribution.
    llmquant::SignalEntropyRatchet entropy_ratchet;
    {
        llmquant::SignalEntropyRatchet::Config er_cfg;
        er_cfg.n_buckets       = 10;
        er_cfg.spike_threshold = 2.5;
        er_cfg.on_entropy_spike = [](double h) {
            spdlog::warn("[entropy] SPIKE h={:.4f} — unusually high bias diversity", h);
        };
        er_cfg.on_floor_drop = [](double floor) {
            spdlog::info("[entropy] floor dropped new_floor={:.4f}", floor);
        };
        entropy_ratchet.update_config(er_cfg);
    }
#endif

#ifdef LLMQUANT_COHERENCE_SCORER_ENABLED
    // NarrativeCoherenceScorer: |μ|/σ signal-to-noise coherence on rolling bias window.
    llmquant::NarrativeCoherenceScorer coherence_scorer;
    {
        llmquant::NarrativeCoherenceScorer::Config cs_cfg;
        cs_cfg.window           = 30;
        cs_cfg.low_threshold    = 0.5;
        cs_cfg.recover_threshold= 1.0;
        cs_cfg.on_coherence_drop = [](double score) {
            spdlog::warn("[coherence] DROP score={:.3f} — noisy/incoherent narrative", score);
        };
        cs_cfg.on_coherence_recover = [](double score) {
            spdlog::info("[coherence] recovered score={:.3f}", score);
        };
        coherence_scorer.update_config(cs_cfg);
    }
#endif

#ifdef LLMQUANT_CROSS_TOKEN_CORR_ENABLED
    // CrossTokenCorrelationMatrix: rolling Pearson correlation across N bias channels.
    llmquant::CrossTokenCorrelationMatrix cross_corr;
    {
        llmquant::CrossTokenCorrelationMatrix::Config cc_cfg;
        cc_cfg.n_channels            = 2;
        cc_cfg.window                = 30;
        cc_cfg.divergence_threshold  = 0.3;
        cc_cfg.convergence_threshold = 0.7;
        cc_cfg.on_divergence = [](int i, int j, double r) {
            spdlog::warn("[cross_corr] DIVERGED ch{}↔ch{} r={:.3f}", i, j, r);
        };
        cc_cfg.on_convergence = [](int i, int j, double r) {
            spdlog::info("[cross_corr] converged ch{}↔ch{} r={:.3f}", i, j, r);
        };
        cross_corr.update_config(cc_cfg);
    }
#endif

#ifdef LLMQUANT_ADAPTIVE_SIZER_ENABLED
    // AdaptivePositionSizer: multi-factor [0,1] multiplier for dynamic position sizing.
    llmquant::AdaptivePositionSizer pos_sizer;
    {
        llmquant::AdaptivePositionSizer::Config as_cfg;
        as_cfg.coherence_weight  = 0.25;
        as_cfg.confidence_weight = 0.25;
        as_cfg.vol_weight        = 0.25;
        as_cfg.regime_weight     = 0.25;
        as_cfg.rv_max            = 0.5;
        as_cfg.min_multiplier    = 0.05;
        as_cfg.change_threshold  = 0.05;
        as_cfg.on_size_change = [](double old_m, double new_m) {
            spdlog::info("[pos_sizer] mult {:.3f} → {:.3f}", old_m, new_m);
        };
        pos_sizer.update_config(as_cfg);
    }
#endif

#ifdef LLMQUANT_CLIP_MONITOR_ENABLED
    // BiasClipMonitor: rolling clip-rate tracker; high rate = saturated/adversarial input.
    llmquant::BiasClipMonitor clip_monitor;
    {
        llmquant::BiasClipMonitor::Config bcm_cfg;
        bcm_cfg.clip_threshold  = 0.7;
        bcm_cfg.window          = 50;
        bcm_cfg.spike_threshold = 0.3;
        bcm_cfg.on_clip_spike = [](double rate) {
            spdlog::warn("[clip] SPIKE rate={:.3f} — high-saturation bias stream", rate);
        };
        clip_monitor.update_config(bcm_cfg);
    }
#endif

#ifdef LLMQUANT_INTENSITY_RAMP_ENABLED
    // NarrativeIntensityRamp: EMA(|bias|) first derivative for emerging-narrative detection.
    llmquant::NarrativeIntensityRamp intensity_ramp;
    {
        llmquant::NarrativeIntensityRamp::Config ir_cfg;
        ir_cfg.ema_alpha       = 0.15;
        ir_cfg.surge_threshold = 0.02;
        ir_cfg.saturation_ramp = 0.05;
        ir_cfg.on_surge = [](double ramp) {
            spdlog::info("[intensity] SURGE ramp={:.4f} — narrative intensity building", ramp);
        };
        ir_cfg.on_fade = [](double ramp) {
            spdlog::info("[intensity] FADE ramp={:.4f} — narrative exhaustion", ramp);
        };
        intensity_ramp.update_config(ir_cfg);
    }
#endif

#ifdef LLMQUANT_ZSCORE_TRACKER_ENABLED
    // SentimentZScoreTracker: rolling z-score normalization of bias; session-invariant extremes.
    llmquant::SentimentZScoreTracker zscore_tracker;
    {
        llmquant::SentimentZScoreTracker::Config zt_cfg;
        zt_cfg.window            = 50;
        zt_cfg.extreme_threshold = 2.0;
        zt_cfg.on_extreme = [](double z) {
            spdlog::warn("[zscore] EXTREME z={:.3f} — bias far outside recent norm", z);
        };
        zscore_tracker.update_config(zt_cfg);
    }
#endif

#ifdef LLMQUANT_CONFLUENCE_DETECTOR_ENABLED
    // SignalConfluenceDetector: window-based directional agreement score for trend confirmation.
    llmquant::SignalConfluenceDetector confluence;
    {
        llmquant::SignalConfluenceDetector::Config cd_cfg;
        cd_cfg.window    = 10;
        cd_cfg.threshold = 0.6;
        cd_cfg.on_confluence = [](double score, int dir) {
            spdlog::info("[confluence] score={:.3f} dir={} — signals aligned", score, dir);
        };
        confluence.update_config(cd_cfg);
    }
#endif

#ifdef LLMQUANT_MULTI_FEED_AGGREGATOR_ENABLED
    // MultiFeedSignalAggregator: weighted consensus across N LLM feeds.
    // Default config registers a single "primary" feed matching this pipeline.
    // Additional feeds can be wired in at integration time via update_config().
    llmquant::MultiFeedSignalAggregator multi_feed_agg;
    {
        llmquant::MultiFeedSignalAggregator::Config mf_cfg;
        mf_cfg.divergence_threshold = 0.5;
        mf_cfg.clear_hysteresis     = 0.70;
        mf_cfg.min_feeds_active     = 1;  // single feed — no divergence in default pipeline
        mf_cfg.feeds = {{"primary", 1.0, 0.20}};
        mf_cfg.on_divergence = [](double div, const std::string& dom) {
            spdlog::warn("[multi_feed] DIVERGED div={:.3f} dominant={}", div, dom);
        };
        mf_cfg.on_convergence = [](double div) {
            spdlog::info("[multi_feed] converged div={:.3f}", div);
        };
        multi_feed_agg.update_config(mf_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_CUSUM_ENABLED
    // SignalCUSUMController: CUSUM control chart for step-change detection.
    // Complements z-score anomaly detection by accumulating small shifts.
    {
        llmquant::SignalCUSUMController::Config cusum_cfg;
        cusum_cfg.target_mean    = 0.0;
        cusum_cfg.allowance      = 0.05;  // detect shifts > 0.05 from neutral
        cusum_cfg.threshold      = 3.0;
        cusum_cfg.reset_on_alarm = true;
        cusum_cfg.on_upward_shift = [](double s) {
            spdlog::warn("[cusum] UPWARD SHIFT S+={:.3f} — sustained positive sentiment step change", s);
        };
        cusum_cfg.on_downward_shift = [](double s) {
            spdlog::warn("[cusum] DOWNWARD SHIFT S-={:.3f} — sustained negative sentiment step change", s);
        };
        signal_cusum.update_config(cusum_cfg);
    }
#endif

#ifdef LLMQUANT_KELLY_SIZER_ENABLED
    // Kelly Criterion position sizer: scales delta_bias_shift by the optimal
    // fraction given the observed win/loss history.  Outcomes should be fed
    // back via kelly_sizer.record_outcome() from the OMS P&L callback.
    llmquant::KellyPositionSizer kelly_sizer;
#endif

#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
    // Adaptive cooldown: widens signal cooldown when P99 latency exceeds budget,
    // narrows it again during recovery.  Updated by the stats ticker.
    llmquant::AdaptiveCooldownController adaptive_cooldown;
#endif

#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
    // Regime detector: classifies pipeline as Bull/Bear/Volatile/RiskOff/Neutral.
    // Updated per signal in the hot-path callback; logged on regime transitions.
    llmquant::RegimeDetector regime_detector;
#endif
#if defined(LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED) && defined(LLMQUANT_REGIME_DETECTOR_ENABLED)
    // Markov transition model: learns P(next|current) from regime change history.
    llmquant::RegimeTransitionModel regime_transition_model;
    regime_detector.set_regime_change_callback(
        [&](llmquant::RegimeDetector::Regime next, llmquant::RegimeDetector::Regime prev) {
            regime_transition_model.record_transition(prev, next);
        });
#endif

#ifdef LLMQUANT_TRADING_HOURS_ENABLED
    // Market-hours guard: blocks signals outside NYSE/NASDAQ regular session.
    llmquant::TradingHoursGuard trading_hours_guard;
    trading_hours_guard.update_config([]{
        llmquant::TradingHoursGuard::Config cfg;
        cfg.on_session_change = [](bool open) {
            spdlog::info("[trading_hours] session {} — signal gate {}",
                         open ? "OPEN" : "CLOSED",
                         open ? "enabled" : "disabled");
        };
        return cfg;
    }());
#endif

#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
    // Cross-source correlation tracker: watches for diverging/converging sources.
    llmquant::SignalCorrelationTracker signal_corr;
    signal_corr.set_divergence_callback([](const std::string& a, const std::string& b, double r) {
        spdlog::warn("[signal_corr] DIVERGE {}<>{} r={:.3f} — sources moving oppositely", a, b, r);
    });
    signal_corr.set_convergence_callback([](const std::string& a, const std::string& b, double r) {
        spdlog::info("[signal_corr] CONVERGE {}<>{} r={:.3f} — sources in agreement", a, b, r);
    });
#endif

#ifdef LLMQUANT_WARMUP_SEQUENCER_ENABLED
    // Pre-seed EMA accumulators with a short synthetic token burst.
    {
        llmquant::WarmupSequencer::Config wcfg;
        wcfg.synthetic_tokens = {
            {"bullish", 0.6}, {"rally",   0.5}, {"growth",  0.4},
            {"neutral", 0.0}, {"concern", -0.3}, {"crash",  -0.7},
            {"bearish", -0.5}, {"recover", 0.3}, {"stable",  0.1},
        };
        wcfg.repeat_count = 3;
        wcfg.on_complete  = [] { spdlog::info("[warmup] EMA pre-seeding complete"); };
        llmquant::WarmupSequencer warmup(wcfg);
        warmup.run([&](const std::string& tok, double /*sent*/) {
            (void)llm_adapter.map_token_to_weight(tok);
        });
    }
#endif

#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
    // DrawdownProtector: tightens risk thresholds as cumulative losses deepen.
    llmquant::DrawdownProtector drawdown_protector;
    drawdown_protector.update_config([]{
        llmquant::DrawdownProtector::Config cfg;
        cfg.on_tier_change = [](int t, double scale, double dd_pct) {
            spdlog::warn("[drawdown] tier {} active  scale={:.2f}  drawdown={:.1f}%",
                         t, scale, dd_pct * 100.0);
        };
        return cfg;
    }());
#endif

#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
    // MultiTimeframeAggregator: fuses bias signals across 1s/5s/30s/5m EMAs.
    llmquant::MultiTimeframeAggregator multi_tf;
    multi_tf.update_config([]{
        llmquant::MultiTimeframeAggregator::Config cfg;
        cfg.on_divergence = [](double spread, double, double) {
            spdlog::debug("[multi_tf] timeframe divergence spread={:.3f}", spread);
        };
        return cfg;
    }());
#endif

// stale_detector already declared above (before the process_token lambda).

#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
    // VolatilityForecaster: tracks GARCH(1,1) conditional variance of sentiment stream.
    llmquant::VolatilityForecaster vol_forecaster;
    {
        llmquant::VolatilityForecaster::Config vf_cfg;
        vf_cfg.on_high_vol = [](double vol, double /*var*/) {
            spdlog::warn("[vol_fcst] HIGH conditional vol={:.4f}", vol);
        };
        vol_forecaster.update_config(vf_cfg);
    }
#endif

#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
    // BayesianSignalFilter: Beta-Bernoulli posterior confidence per direction.
    llmquant::BayesianSignalFilter bayes_filter;
    {
        llmquant::BayesianSignalFilter::Config bf_cfg;
        bf_cfg.on_low_confidence = [](bool bullish, double post) {
            spdlog::warn("[bayes] low posterior={:.3f} dir={}", post,
                         bullish ? "BULL" : "BEAR");
        };
        bayes_filter.update_config(bf_cfg);
    }
#endif

#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
    // AnomalyDetector: online Z-score anomaly detection on the bias stream.
    llmquant::AnomalyDetector anomaly_detector;
    {
        llmquant::AnomalyDetector::Config ad_cfg;
        ad_cfg.name = "bias";
        ad_cfg.soft_cb = [](const llmquant::AnomalyDetector::AnomalyEvent& e) {
            spdlog::info("[anomaly] soft z={:.2f} val={:.6f}", e.z_score, e.value);
        };
        ad_cfg.hard_cb = [](const llmquant::AnomalyDetector::AnomalyEvent& e) {
            spdlog::warn("[anomaly] HARD z={:.2f} val={:.6f}", e.z_score, e.value);
        };
        anomaly_detector.update_config(ad_cfg);
    }
#endif

#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
    // TokenBurstDetector: flags high token arrival rates to detect backlog flushes.
    llmquant::TokenBurstDetector burst_detector;
    {
        llmquant::TokenBurstDetector::Config bd_cfg;
        bd_cfg.on_burst_start = [](double rate) {
            spdlog::warn("[burst_det] BURST {:.1f} tok/s — throttling may apply", rate);
        };
        bd_cfg.on_burst_end = [](double rate) {
            spdlog::info("[burst_det] burst end {:.1f} tok/s", rate);
        };
        burst_detector.update_config(bd_cfg);
    }
#endif

#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
    // SignalPersistenceTracker: conviction multiplier from directional streak.
    llmquant::SignalPersistenceTracker persistence_tracker;
    {
        llmquant::SignalPersistenceTracker::Config pt_cfg;
        pt_cfg.on_conviction = [](int streak) {
            spdlog::info("[persistence] conviction streak={}", streak);
        };
        persistence_tracker.update_config(pt_cfg);
    }
#endif

#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
    // RollingSharpeBiasTracker: rolling Sharpe of the bias stream.
    llmquant::RollingSharpeBiasTracker rolling_sharpe;
#endif

#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
    // OrderBookSimulator: sentiment-driven LOB for slippage-aware fill estimation.
    llmquant::OrderBookSimulator order_book_sim;
#endif

#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
    // TokenSentimentHeatmap: per-token sentiment distribution for attribution.
    llmquant::TokenSentimentHeatmap sentiment_heatmap;
#endif

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
    // SentimentMomentumFilter: gates trade signals that contradict the macro
    // sentiment trajectory (Improving/Declining/Stable/Volatile).
    {
        llmquant::SentimentMomentumFilter::Config smf_cfg;
        smf_cfg.mode           = llmquant::SentimentMomentumFilter::Mode::Relaxed;
        smf_cfg.scale_by_slope = true;
        smf_cfg.slope_scale    = 0.05;
        sentiment_momentum_filter.update_config(smf_cfg);
    }
#endif

#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)
    // PositionTracker: records open/close trades and feeds realised P&L back
    // into the Kelly sizer to keep position sizing adaptive.
    llmquant::PositionTracker position_tracker(kelly_sizer);
    position_tracker.set_trade_close_callback([](uint64_t id, double ret, bool win) {
        spdlog::info("[pos_tracker] trade#{} closed  return={:.4f}  {}",
                     id, ret, win ? "WIN" : "LOSS");
    });
#endif

#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
    // SignalDecayEnvelope: attenuates accumulated bias after token-stream silence.
    {
        llmquant::SignalDecayEnvelope::Config sd_cfg;
        sd_cfg.half_life_ms = 15'000.0;  // bias halves after 15 s of silence
        sd_cfg.clamp        = true;
        sd_cfg.min_bias     = -2.0;
        sd_cfg.max_bias     =  2.0;
        signal_decay.update_config(sd_cfg);
        signal_decay.set_zero_cross_callback([](double old_b, double new_b) {
            spdlog::info("[signal_decay] bias zero-cross: {:.4f} → {:.4f}", old_b, new_b);
        });
    }
#endif

    // Sparkline ring buffer: 24 most-recent delta_bias_shift values → unicode blocks.
    constexpr int kSparkSlots = 24;
    std::array<double, kSparkSlots> spark_ring{};
    std::atomic<int> spark_head{0};

    // Signal velocity: rate of change of delta_bias_shift (units/second).
    // Written by signal callback (single writer), read by stats ticker.
    double sig_vel_prev_bias = 0.0;
    std::chrono::steady_clock::time_point sig_vel_prev_time{};
    std::atomic<double> sig_velocity{0.0};  // current velocity estimate

    risk_mgr.set_oms_callback([&](const std::string& event,
                                   const llmquant::RiskManager::PositionState&,
                                   const llmquant::TradeSignal&) {
        std::lock_guard<std::mutex> lk(block_reason_mutex);
        last_block_reason = event;
    });

    trade_engine.set_signal_callback([&](const TradeSignal& signal) {
        auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         signal.timestamp.time_since_epoch()).count();
        auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now() - signal.timestamp
                          ).count();

#if defined(LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED) && defined(LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
        // Block signals that contradict the macro sentiment trajectory.
        const TradeSignal momentum_filtered = sentiment_momentum_filter.filter_signal(signal);
#else
        const TradeSignal& momentum_filtered = signal;
#endif

#ifdef LLMQUANT_SIGNAL_DECAY_ENABLED
        // Scale delta_bias_shift by the decay-attenuated envelope factor.
        // Reuse the filtered signal, cloning to allow modification.
        TradeSignal decay_adjusted = momentum_filtered;
        {
            double envelope = signal_decay.decayed_bias();
            if (std::abs(envelope) > 1e-9) {
                // Attenuate: multiply bias by |envelope| / |raw_bias| ratio clamped to [0,1].
                double raw = signal_decay.raw_bias();
                if (std::abs(raw) > 1e-9)
                    decay_adjusted.delta_bias_shift *= std::min(1.0, std::abs(envelope) / std::abs(raw));
            }
        }
        const TradeSignal& pre_kelly_signal = decay_adjusted;
#else
        const TradeSignal& pre_kelly_signal = momentum_filtered;
#endif

#ifdef LLMQUANT_KELLY_SIZER_ENABLED
        // Scale delta_bias_shift by the current Kelly fraction before risk gating.
        const TradeSignal sized_signal = kelly_sizer.size_signal(pre_kelly_signal);
        bool passed = risk_mgr.evaluate(sized_signal);
#else
        const TradeSignal& sized_signal = pre_kelly_signal;
        bool passed = risk_mgr.evaluate(pre_kelly_signal);
#endif

#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
        circuit_breaker.record_signal(!passed);
        // When the circuit is open, treat the signal as blocked regardless
        // of the risk manager decision (unless suppress_when_open is false).
        if (passed && circuit_breaker.is_open()) {
            passed = false;
        }
#endif

#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
        regime_detector.update(signal.delta_bias_shift,
                               signal.volatility_adjustment,
                               !passed);
#endif

#ifdef LLMQUANT_TRADING_HOURS_ENABLED
        // Block signals outside NYSE market hours.
        if (passed && trading_hours_guard.should_block()) {
            passed = false;
        }
#endif

#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
        // Track bias value under the "main" source for correlation monitoring.
        signal_corr.record("main", signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
        // Feed the raw bias into all timeframe EMAs for multi-horizon consensus.
        multi_tf.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
        // Record simulated P&L outcome (passed signals only) into drawdown protector.
        if (passed) {
            // Use a simple proxy: bias * 0.001 as a per-signal PnL estimate.
            drawdown_protector.record_pnl(signal.delta_bias_shift * 0.001);
        }
#endif

#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
        // Update GARCH(1,1) volatility estimate from the raw bias value.
        vol_forecaster.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
        // Record signal direction; outcomes would be fed by a live P&L feed.
        bayes_filter.record_signal(signal.delta_bias_shift > 0.0);
#endif

#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
        // Check for statistical anomalies in the bias stream.
        anomaly_detector.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
        // Record token arrival for burst rate tracking.
        burst_detector.record();
#endif

#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
        // Track directional streak for conviction scoring.
        persistence_tracker.record_bias(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
        // Update rolling Sharpe of the bias stream.
        rolling_sharpe.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
        // Update Hurst exponent estimate with each new bias observation.
        fractal_dim.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
        // Update ensemble sub-signals and train from outcome (passed = positive reward).
        if (ens_bias_id >= 0)  signal_ensemble.update_source(ens_bias_id,  signal.delta_bias_shift);
        if (ens_vol_id  >= 0)  signal_ensemble.update_source(ens_vol_id,   signal.volatility_adjustment);
        if (ens_conf_id >= 0)  signal_ensemble.update_source(ens_conf_id,  signal.confidence);
        signal_ensemble.record_outcome(passed ? 1.0 : -1.0);
#endif

#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
        // Feed bias into MACD oscillator; callbacks fire on zero-crosses.
        signal_momentum_osc.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
        // Update synthetic LOB with the latest bias signal.
        order_book_sim.update_bias(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
        // Record direction–sentiment pair for attribution.
        {
            const char* dir_key = (signal.strategy_toggle > 0) ? "bull"
                                : (signal.strategy_toggle < 0) ? "bear" : "neutral";
            sentiment_heatmap.record(dir_key, signal.delta_bias_shift);
        }
#endif

#ifdef LLMQUANT_CVAR_ENABLED
        // Treat delta_bias_shift as proxy PnL for tail-risk tracking.
        cvar_calc.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
        // Own-activity = |bias shift| emitted; sentiment = raw bias shift.
        feedback_detector.record_own_activity(std::abs(signal.delta_bias_shift));
        feedback_detector.record_sentiment(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
        // Feed bias into ACF cycle detector.
        sentiment_cycle.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
        // Adapt poll interval based on recent signal magnitude.
        adaptive_sampler.record_activity(std::abs(signal.delta_bias_shift));
#endif

#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
        // Compute self-information of this signal relative to learned distribution.
        signal_surprise.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_REGIME_SIZER_ENABLED
        // Feed Hurst exponent and volatility into the regime-aware position sizer.
#  ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
        regime_sizer.update_hurst(fractal_dim.hurst());
#  endif
#  ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
        regime_sizer.update_vol(vol_forecaster.conditional_vol());
#  endif
#endif

#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
        // Track how quickly signal confidence decays over time.
        conf_decay.record(signal.confidence);
#endif

#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
        // Track rolling correlation between bias, vol, and confidence streams.
        cross_asset_corr.record("bias",       signal.delta_bias_shift);
        cross_asset_corr.record("vol",        signal.volatility_adjustment);
        cross_asset_corr.record("confidence", signal.confidence);
#endif

#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
        velocity_tracker.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
        narrative_clock.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
        // Trip the breaker if bias velocity is excessive; block signal when open.
        (void)velocity_breaker.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
        // Record a synthetic outcome: positive confidence = treat as win.
        // In production, wire record_outcome() from the OMS P&L callback.
        signal_calibration.record_outcome(signal.confidence, signal.confidence > 0.5);
#endif

#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
        // Push the raw bias as a signed pressure proxy.
        order_flow_detector.record_pressure(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_REGIME_PROB_ENABLED
        // Feed sentiment + volatility into the HMM filter for regime probability.
        regime_prob_est.update(signal.delta_bias_shift,
                               signal.volatility_adjustment);
#endif

#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
        // Capture signal in the replay ring for post-hoc analysis.
        signal_replay.push(signal);
#endif

#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
        // Measure incoherence across bias, vol, and confidence dimensions.
        sentiment_dispersion.record(std::abs(signal.delta_bias_shift),
                                    signal.volatility_adjustment,
                                    signal.confidence);
#endif

#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
        // Track pairwise divergence between bias, vol, and confidence streams.
        sentiment_divergence.record("bias",       signal.delta_bias_shift);
        sentiment_divergence.record("vol",        signal.volatility_adjustment);
        sentiment_divergence.record("confidence", signal.confidence);
#endif

#ifdef LLMQUANT_SIGNAL_CI_ENABLED
        // Track jackknife CI on bias stream; narrow=reliable, wide=uncertain.
        signal_ci.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
        // Feed each signal's bias shift into the Markov state chain.
        sentiment_persistence.record(signal.delta_bias_shift);
#endif

#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
        // Record the signal as a sentiment event sentinel for causal attribution.
        // Return observations should be fed from the OMS P&L callback in production.
        causal_impact.record_event("signal:" + std::to_string(signal.delta_bias_shift));
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
        // Feed current aggregated bias and elapsed time into the IV-skew divergence
        // detector.  A live options feed would also call record_skew() separately.
        {
            static auto ofb_last_t = std::chrono::steady_clock::now();
            auto ofb_now = std::chrono::steady_clock::now();
            double ofb_dt = std::chrono::duration<double>(ofb_now - ofb_last_t).count();
            ofb_last_t = ofb_now;
            if (ofb_dt > 0.0)
                options_flow_bridge.record_bias(signal.delta_bias_shift, ofb_dt);
        }
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
        sentiment_phase_portrait.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SHADOW_PORTFOLIO_ENABLED
        // Raw signal → shadow portfolio (unconstrained); actual execution → live portfolio.
        shadow_portfolio.record_signal(signal.delta_bias_shift, signal.confidence);
        shadow_portfolio.record_live_signal(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CONFIDENCE_BAND_ENABLED
        // Feed signal bias into the Kalman filter; bands auto-narrow/widen.
        confidence_band.update(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_REGIME_ROUTER_ENABLED
        // Use delta_bias_shift as a proxy for period return for vol/momentum estimation.
        regime_router.record_return(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_DRIFT_ENABLED
        // Track W1 distribution drift between recent and baseline bias-shift windows.
        signal_drift_monitor.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_LIFECYCLE_TRACKER_ENABLED
        // Track the emitted signal's lifecycle (birth → peak → half-life → death).
        // Use a composite ID: bias sign + quantised magnitude.
        {
            std::string sig_id = (signal.delta_bias_shift >= 0.0 ? "bull_" : "bear_")
                               + std::to_string(static_cast<int>(
                                     std::abs(signal.delta_bias_shift) * 10.0));
            lifecycle_tracker.record_signal(sig_id, signal.delta_bias_shift);
            lifecycle_tracker.tick();
        }
#endif
#if defined(LLMQUANT_POSITION_CONCENTRATION_ENABLED) && defined(LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED)
        // Tag signal with the dominant narrative topic for HHI tracking.
        concentration_guard.record_signal(narrative_classifier.dominant_topic(), signal.delta_bias_shift);
#elif defined(LLMQUANT_POSITION_CONCENTRATION_ENABLED)
        concentration_guard.record_signal("default", signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_AUTOCORR_METER_ENABLED
        autocorr_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_SSI_ENABLED
        signal_ssi.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_FATIGUE_ENABLED
        // Track consecutive same-direction bias streaks for reversal risk.
        signal_fatigue.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_POLARIZATION_MONITOR_ENABLED
        // Track bimodality of signal distribution — polarized market = two-camp regime.
        polarization_monitor.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
        // Update narrative temperature: hot when direction and volatility both elevated.
        narrative_temperature.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ECHO_SUPPRESSOR_ENABLED
        // Detect echo state: near-duplicate consecutive signals = narrative stutter.
        echo_suppressor.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_HURST_ESTIMATOR_ENABLED
        // Update Hurst estimate: H>0.6=trending momentum, H<0.4=mean-reverting.
        hurst_estimator.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CHANGE_POINT_ENABLED
        // CUSUM change-point: detect sustained mean shift in signal bias.
        change_point_detector.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
        bias_vbreaker.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_IR_TRACKER_ENABLED
        ir_tracker.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CONSISTENCY_METER_ENABLED
        consistency_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_OSCILLATION_DETECTOR_ENABLED
        oscillation_detector.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_WEIGHT_HISTOGRAM_ENABLED
        // Update bias histogram — tracks distribution shape over session.
        weight_histogram.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_SLOPE_ENABLED
        // Update OLS slope of recent bias values for trend acceleration detection.
        signal_slope.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_RUN_LENGTH_ENABLED
        // Track run-length of bias sign for persistence/mean-reversion detection.
        run_length.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_COVERAGE_METER_ENABLED
        // Update rolling bias range coverage.
        coverage_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BIAS_HYSTERESIS_ENABLED
        hysteresis_gate.evaluate(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_REALIZED_VOL_ENABLED
        realized_vol.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CAUSAL_TRACER_ENABLED
        causal_tracer.record_token("<signal>", signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_DEPENDENCY_MAPPER_ENABLED
        dep_mapper.record_token(static_cast<int>(signal.timestamp_ns & 0xFFFFFFu));
#endif
#ifdef LLMQUANT_FREQ_ANALYSER_ENABLED
        freq_analyser.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ENTROPY_RATCHET_ENABLED
        entropy_ratchet.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_COHERENCE_SCORER_ENABLED
        coherence_scorer.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CROSS_TOKEN_CORR_ENABLED
        cross_corr.record(0, signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ADAPTIVE_SIZER_ENABLED
        pos_sizer.set_confidence(signal.confidence);
        pos_sizer.set_coherence(std::abs(signal.delta_bias_shift));
#endif
#ifdef LLMQUANT_CLIP_MONITOR_ENABLED
        clip_monitor.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_INTENSITY_RAMP_ENABLED
        intensity_ramp.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ZSCORE_TRACKER_ENABLED
        zscore_tracker.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CONFLUENCE_DETECTOR_ENABLED
        confluence.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_CUSUM_ENABLED
        // Update CUSUM accumulators for step-change detection.
        signal_cusum.update(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_MULTI_FEED_AGGREGATOR_ENABLED
        // Feed primary signal into multi-feed aggregator (single-feed mode by default).
        multi_feed_agg.record("primary", signal.delta_bias_shift, signal.confidence);
#endif
#ifdef LLMQUANT_MOMENTUM_INDEX_ENABLED
        momentum_index.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_GAIN_LOSS_RATIO_ENABLED
        gain_loss_ratio.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_REGIME_TRANSITION_MATRIX_ENABLED
        regime_transition.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_REVERSAL_DETECTOR_ENABLED
        reversal_detector.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_TSMI_ENABLED
        tsmi.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ADAPTIVE_THRESHOLD_ENABLED
        adaptive_threshold.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CONDITIONAL_DIST_ENABLED
        conditional_dist.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_COMPRESSOR_ENABLED
        signal_compressor.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ROLLING_QUANTILE_ENABLED
        rolling_quantile.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_AUTOREGRESSOR_ENABLED
        signal_ar.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_PHASE_SPACE_ENABLED
        phase_space.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_TOPOLOGY_MAPPER_ENABLED
        topology_mapper.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_INFORMATION_GAIN_ENABLED
        info_gain.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_NARRATIVE_DRIFT_ENABLED
        narrative_drift.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SENTIMENT_GRAPH_ENABLED
        {
            int tok_id = static_cast<int>(signal.timestamp_ns & 0xFFu); // modulo kMaxTokens=256
            sentiment_graph.record(tok_id, signal.delta_bias_shift);
        }
#endif
#ifdef LLMQUANT_KALMAN_FILTER_ENABLED
        kalman_filter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SPECTRAL_ENTROPY_ENABLED
        spectral_entropy.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BOOTSTRAP_CI_ENABLED
        bootstrap_ci.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_GARCH_ESTIMATOR_ENABLED
        garch_est.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_REGIME_HMM_ENABLED
        regime_hmm.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_POLARITY_INDEX_ENABLED
        polarity_idx.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_RESIDUAL_ANALYSER_ENABLED
        residual_analyser.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SALIENCY_RANKER_ENABLED
        {
            int tok_id = static_cast<int>(
                signal.timestamp_ns % llmquant::TokenSaliencyRanker::kMaxTokens);
            saliency_ranker.record(tok_id, signal.delta_bias_shift);
        }
#endif
#ifdef LLMQUANT_TAIL_RISK_METER_ENABLED
        tail_risk.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_LEVEL_CROSSING_ENABLED
        level_crossing.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CROSS_CORRELATOR_ENABLED
        cross_corr_lag.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_VOL_RATIO_ENABLED
        vol_ratio.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_PARABOLIC_SAR_ENABLED
        parabolic_sar.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BOLLINGER_BANDS_ENABLED
        bollinger_bands.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_IMPULSE_DETECTOR_ENABLED
        impulse_det.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_TREND_STRENGTH_INDEX_ENABLED
        trend_strength.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_MASS_INDEX_ENABLED
        mass_idx.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CHOPPINESS_INDEX_ENABLED
        choppiness.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ACCELERATION_METER_ENABLED
        accel_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_FATIGUE_DETECTOR_ENABLED
        fatigue_det.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SKEWNESS_TRACKER_ENABLED
        skewness_tracker.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ZERO_CROSS_RATE_ENABLED
        zcr_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BIAS_CORRELOGRAM_ENABLED
        bias_correlogram.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_KURTOSIS_TRACKER_ENABLED
        kurtosis_tracker.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_PERSISTENCE_INDEX_ENABLED
        persistence_idx.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BIAS_ENTROPY_RATE_ENABLED
        bias_entropy.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_DRAWDOWN_METER_ENABLED
        drawdown_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CADENCE_ANALYSER_ENABLED
        cadence_analyser.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_MEAN_REVERSION_SPEED_ENABLED
        mean_rev_speed.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CLUSTER_DETECTOR_ENABLED
        cluster_detector.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_VOL_BREAKOUT_ENABLED
        vol_breakout.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_STOCHASTIC_OSC_ENABLED
        stochastic_osc.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BIAS_ACF_ENABLED
        bias_acf.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ONLINE_GRANGER_ENABLED
        // x = bias_shift, y = confidence: test if bias leads confidence or vice versa
        granger.record(signal.delta_bias_shift, signal.confidence);
#endif
#ifdef LLMQUANT_MACD_HISTOGRAM_ENABLED
        macd_hist.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_REGIME_MARKOV_ENABLED
        regime_markov.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CONCENTRATION_RISK_ENABLED
        conc_risk.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_WILLIAMS_R_ENABLED
        williams_r.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_INFLUENCE_DECAY_ENABLED
        influence_decay.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_POLARITY_SHIFT_ENABLED
        polarity_shift.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_CHANDE_OSC_ENABLED
        chande_osc.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_DONCHIAN_CHANNEL_ENABLED
        donchian_ch.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BIAS_HISTOGRAM_ENABLED
        bias_histogram.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_EXP_SMOOTHING_ENABLED
        exp_smooth.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_RELATIVE_VIGOR_ENABLED
        rvi_signal.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SENTIMENT_VELOCITY_ENABLED
        sent_velocity.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ZSCORE_NORMALISER_ENABLED
        zscore_norm.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_KELTNER_CHANNEL_ENABLED
        keltner_ch.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BURST_INTENSITY_ENABLED
        burst_intensity.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_TRIPLE_EMA_ENABLED
        triple_ema.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_COHERENCE_TRACKER_ENABLED
        coherence_tracker.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_LOCAL_EXTREMA_ENABLED
        local_extrema.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_ADAPTIVE_FILTER_ENABLED
        adaptive_filter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_PRESSURE_GAUGE_ENABLED
        pressure_gauge.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_WAVELET_DECOMPOSER_ENABLED
        wavelet_decomp.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_CONVEXITY_ENABLED
        convexity_meter.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED
        // Reward = |bias_shift| * confidence: larger, higher-confidence signals are better.
        rl_weighter.update("llm_primary",
            std::abs(signal.delta_bias_shift) * signal.confidence);
#endif
#ifdef LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED
        entropy_clock.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED
        decay_halflife.record(signal.delta_bias_shift);
#endif
#ifdef LLMQUANT_BAYESIAN_SENTIMENT_ENABLED
        bayes_prior.update(signal.delta_bias_shift);
#endif

        // Record bias value in sparkline ring (lock-free: only one writer thread).
        {
            int idx = spark_head.fetch_add(1, std::memory_order_relaxed) % kSparkSlots;
            spark_ring[idx] = signal.delta_bias_shift;
        }

        // Signal velocity: delta_bias_shift per second (first-order finite difference).
        {
            auto now = std::chrono::steady_clock::now();
            if (sig_vel_prev_time.time_since_epoch().count() != 0) {
                double dt_s = std::chrono::duration<double>(now - sig_vel_prev_time).count();
                if (dt_s > 1e-6) {
                    double vel = (signal.delta_bias_shift - sig_vel_prev_bias) / dt_s;
                    sig_velocity.store(vel, std::memory_order_relaxed);
                }
            }
            sig_vel_prev_bias = signal.delta_bias_shift;
            sig_vel_prev_time = now;
        }

        // Capture the rejection reason once, under the lock, so it is available
        // for both the console gate_str and the structured log call below.
        // Previously the reason was cleared after gate_str was built, making
        // the second read always fall back to "risk".
        std::string block_reason_copy;
        if (!passed) {
            std::lock_guard<std::mutex> lk(block_reason_mutex);
            block_reason_copy = last_block_reason.empty() ? "risk" : last_block_reason;
            if (block_reason_copy.size() > 16) block_reason_copy = block_reason_copy.substr(0, 16);
            last_block_reason.clear();
        }

        std::string gate_str;
        if (passed) {
            gate_str = std::string(" ") + C("\033[32m") + "PASS" + C("\033[0m");
        } else {
            gate_str = std::string(" ") + C("\033[31m") + "BLOCK" + C("\033[0m") + "(" + block_reason_copy + ")";
        }

        // Aligned columns: TIME(ms)  BIAS     VOL      LATENCY  GATE
        // Suppressed in --quiet mode; all data still flows to MetricsLogger.
        if (!quiet) {
            std::cout << "\n  "
                      << std::setw(12) << ts_ms          << "  "
                      << std::setw(8)  << std::fixed << std::setprecision(4)
                                       << sized_signal.delta_bias_shift  << "  "
                      << std::setw(8)  << sized_signal.volatility_adjustment << "  "
                      << std::setw(6)  << latency_us << "μs"
                      << gate_str
                      << std::flush;
        }

        if (passed) {
            logger.log_trade_signal(
                sized_signal.delta_bias_shift,
                sized_signal.volatility_adjustment,
                sized_signal.confidence,
                static_cast<double>(latency_us),
                sized_signal.signal_quality);

#if defined(LLMQUANT_POSITION_TRACKER_ENABLED) && defined(LLMQUANT_KELLY_SIZER_ENABLED)
            // Open a tracked position for this signal.  In a live system the
            // entry_price would come from the OMS fill report; here we use the
            // sized bias as a normalised proxy (1.0 base).
            (void)position_tracker.open_trade(sized_signal, 1.0);
#endif
        } else {
            logger.log_risk_rejection(block_reason_copy,
                                      signal.delta_bias_shift,
                                      signal.confidence);
        }

#ifdef LLMQUANT_AUDIT_LOG_ENABLED
        if (audit_log) {
            audit_log->log_signal(signal, passed, passed ? "" : block_reason_copy);
        }
#endif
    });

    // Load test tokens for simulator path.
    if (sys_config.token_stream.use_memory_stream) {
        token_sim.load_tokens_from_memory({
            // Fear / panic
            "crash", "panic", "collapse", "plunge", "selloff", "rout",
            // Bullish directional
            "bullish", "rally", "surge", "breakout", "rebound", "accumulate",
            // Bearish directional
            "bearish", "short", "downtrend", "distribution",
            // Volatility
            "volatile", "spike", "whipsaw", "choppy", "gamma", "vega",
            // Certainty / confidence
            "inevitable", "guarantee", "confident", "confirmed",
            // Corporate / earnings
            "earnings", "beats", "misses", "guidance", "dividend", "buyback",
            // Macro / regime
            "inflation", "fed", "pivot", "recession", "risk-on", "risk-off",
            // Analyst
            "upgrade", "downgrade", "overweight", "outperform",
            // Options
            "calls", "puts", "squeeze", "hedge",
            // Crypto / retail
            "pump", "fud", "hodl",
            // Neutral filler (tests zero-weight path)
            "the", "and", "is"
        });
    } else {
        token_sim.load_tokens_from_file(sys_config.token_stream.data_file_path);
    }

    // Print banner.
    std::cout << "\n";
    std::cout << "  LLMTokenStreamQuantEngine\n";
    std::cout << DIV1;
    if (stream_mode) {
        std::cout << "  MODE    : LIVE STREAM  (gpt-4o " << ARROW << " api.openai.com:443)\n";
        std::cout << "  PROMPT  : market sentiment / tickers / directional\n";
        std::cout << "  INTERVAL: 5s per request\n";
    } else {
        std::cout << "  MODE    : SIMULATOR  (in-memory token loop)\n";
        std::cout << "  INTERVAL: " << sys_config.token_stream.token_interval_ms << "ms/token\n";
    }
    std::cout << "  OMS     : " << oms_adapter->description() << "\n";
    std::cout << "  LATENCY : target p99 < " << sys_config.latency.target_latency_us << "us\n";
    if (dry_run)
        std::cout << "  DRY-RUN : signals suppressed — dictionary coverage mode\n";
    if (backtest_mode)
        std::cout << "  BACKTEST: cooldown disabled — signal emitted on every token\n";
    std::cout << DIV1 << "\n";
    std::cout << config.to_summary_string() << "\n";
    std::cout << "  TIME(ms)     BIAS      VOL       LATENCY   GATE\n";
    std::cout << DIV2;

#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
    std::unique_ptr<llmquant::LLMStreamClient> stream_client;
    if (stream_mode) {
        llmquant::LLMStreamClient::Config stream_cfg;
        stream_cfg.host         = "api.openai.com";
        stream_cfg.port         = 443;
        stream_cfg.api_key      = stream_api_key;
        stream_cfg.model        = "gpt-4o";
        stream_cfg.use_tls      = true;
        stream_cfg.max_tokens   = 300;
        stream_cfg.loop_interval = std::chrono::seconds(5);
        stream_cfg.debug_raw    = debug_raw;
        stream_cfg.system_prompt =
            "You are a high-frequency financial markets analyst. Every response "
            "must include specific tickers and explicit directional words: "
            "bullish, bearish, crash, surge, panic, breakout, collapse, volatile, "
            "guarantee, inevitable. Be terse and signal-dense.";
        stream_cfg.user_prompt =
            "Give a terse real-time market signal update. Use tickers. "
            "Use words: bullish, bearish, surge, crash, breakout, collapse, volatile.";

        stream_client = std::make_unique<llmquant::LLMStreamClient>(stream_cfg);
        // Each stream token gets a unique monotonically-increasing sequence ID so
        // MetricsLogger and dedup logs can distinguish individual stream tokens.
        std::atomic<uint64_t> stream_seq_id{0};
        stream_client->set_token_callback([&](const std::string& text) {
            process_token(text, stream_seq_id.fetch_add(1, std::memory_order_relaxed));
        });
        stream_client->set_done_callback([](const std::string& err) {
            if (!err.empty())
                spdlog::warn("stream: {}", err);
        });
        if (!stream_client->connect())
            spdlog::warn("[stream] connect() returned false — stream may not start");
    } else {
        token_sim.start();
    }
#else
    // LLMStreamClient compiled out: always use the token simulator.
    if (stream_mode) {
        spdlog::warn("--stream requested but LLMStreamClient was compiled out "
                     "(-DLLMQUANT_ENABLE_STREAM_CLIENT=OFF); falling back to simulator.");
        stream_mode = false;
    }
    token_sim.start();
#endif

    // Prometheus metrics endpoint on port 9100.
    // The snapshot is built once per second in the monitoring loop so the
    // scrape thread never contends with the hot path for latency stats.
    std::string prom_snapshot;
    std::mutex  prom_snapshot_mutex;

#ifdef LLMQUANT_PROMETHEUS_ENABLED
    uint16_t eff_stats_port = (stats_port_override != 0)
                                  ? stats_port_override
                                  : sys_config.metrics.stats_port;
    llmquant::PrometheusExporter prom_exporter({.port = eff_stats_port,
                                                .bind_address = sys_config.metrics.bind_address});
    prom_exporter.set_metrics_callback([&]() -> std::string {
        std::lock_guard<std::mutex> lk(prom_snapshot_mutex);
        return prom_snapshot;
    });
    if (!no_prometheus) {
        if (!prom_exporter.start()) {
            spdlog::warn("PrometheusExporter failed to bind on port {}", eff_stats_port);
        }
    } else {
        spdlog::info("--no-prometheus: Prometheus scrape endpoint disabled");
    }
#else
    (void)stats_port_override;
    if (!no_prometheus)
        spdlog::info("Prometheus scrape endpoint not available "
                     "(built with LLMQUANT_ENABLE_PROMETHEUS=OFF)");
#endif

#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    uint16_t eff_health_port = (health_port_override != 0) ? health_port_override : uint16_t{8080};
    llmquant::HealthServer::Config health_cfg;
    health_cfg.port         = eff_health_port;
    health_cfg.bind_address = sys_config.metrics.bind_address;
    llmquant::HealthServer health_server(health_cfg);
    health_server.set_health_callback([&]() -> std::pair<bool, std::string> {
        auto lc_stats         = latency_ctrl.get_stats();
        auto te_stats         = trade_engine.get_stats();
        const auto& rm_stats  = risk_mgr.get_stats();
        auto uptime_s  = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - engine_start_time).count();
        bool oms_ok    = oms_adapter && oms_adapter->is_running();
#  ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
        bool cb_open         = circuit_breaker.is_open();
        double blk_rate      = circuit_breaker.block_rate();
        std::string cb_name  = circuit_breaker.state_name();
#  else
        bool cb_open         = false;
        double blk_rate      = 0.0;
        std::string cb_name  = "closed";
#  endif
#  if defined(LLMQUANT_DEDUP_ENABLED) && defined(LLMQUANT_REDIS_ENABLED)
        bool redis_ok = deduplicator.is_connected();
#  else
        bool redis_ok = false;
#  endif
#  ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        bool stream_stale = stale_detector.is_stale();
#  else
        bool stream_stale = false;
#  endif
        bool ok = !cb_open && oms_ok && !stream_stale;
        uint64_t blocked = rm_stats.signals_blocked_magnitude.load()
                         + rm_stats.signals_blocked_confidence.load()
                         + rm_stats.signals_blocked_rate.load()
                         + rm_stats.signals_blocked_drawdown.load();
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
            "{\"ok\":%s"
            ",\"uptime_s\":%lld"
            ",\"circuit_breaker\":\"%s\""
            ",\"block_rate\":%.4f"
            ",\"oms_connected\":%s"
            ",\"stream_stale\":%s"
            ",\"p99_latency_us\":%lld"
            ",\"signals_generated\":%llu"
            ",\"signals_blocked\":%llu"
            ",\"redis_connected\":%s"
            ",\"version\":\"%s\"}",
            ok ? "true" : "false",
            static_cast<long long>(uptime_s),
            cb_name.c_str(),
            blk_rate,
            oms_ok ? "true" : "false",
            stream_stale ? "true" : "false",
            static_cast<long long>(lc_stats.p99_latency.count()),
            static_cast<unsigned long long>(te_stats.signals_generated.load()),
            static_cast<unsigned long long>(blocked),
            redis_ok ? "true" : "false",
            LLMQUANT_VERSION
        );
        return {ok, std::string(buf)};
    });
    if (!no_health_server) {
        if (!health_server.start())
            spdlog::warn("[health_server] failed to bind on port {}", eff_health_port);
    } else {
        spdlog::info("--no-health: HTTP /health endpoint disabled");
    }
#endif  // LLMQUANT_HEALTH_SERVER_ENABLED

    // Main monitoring loop — prints a rolling stats bar every second.
    // Interruptible sleep: wake every 100ms to check g_running so that
    // SIGINT/SIGTERM is handled promptly regardless of --stats-interval.
    uint64_t last_tick = 0;
    std::string last_regime;    // For regime-change transition alerts.
    std::string last_morphology; // Last detected sparkline pattern name.
    while (g_running) {
        {
            auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(stats_interval_ms);
            while (g_running && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!g_running) break;
        }

        auto stats    = latency_ctrl.get_stats();
        auto pressure = latency_ctrl.get_pressure();

        // Update ingestion pressure.
        // Normalise the per-tick token count to tokens/second regardless of
        // the configured stats_interval_ms (fixes pressure and TPS display
        // when --stats-interval differs from the default 1000ms).
        uint64_t raw_count = token_count_window.exchange(0);
        double   tps_d     = (stats_interval_ms > 0)
                                 ? (static_cast<double>(raw_count) * 1000.0
                                    / static_cast<double>(stats_interval_ms))
                                 : static_cast<double>(raw_count);
        uint64_t tps = static_cast<uint64_t>(tps_d + 0.5);  // rounded for display
        double   max_tps = stream_mode
                               ? sys_config.pressure.max_ingestion_rate_tps   // gpt-4o emits ~10-30 tokens/s
                               : static_cast<double>(1000000 / std::max(1, sys_config.token_stream.token_interval_ms));
        latency_ctrl.update_ingestion_pressure(tps_d, max_tps);

        // Queue pressure via suppressed-signal count.
        auto eng_stats = trade_engine.get_stats();  // snapshot, not reference
        latency_ctrl.update_queue_pressure(eng_stats.signals_suppressed.load(), 1024);

        double backoff = latency_ctrl.get_backoff_multiplier();
        double cpu_fraction = get_process_cpu_fraction();  // sampled once per loop tick

#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
        // Feed current P99 to the adaptive cooldown controller.
        // If pressure is detected, update the trade engine's cooldown on the fly.
        {
            double p99_us_f = static_cast<double>(stats.p99_latency.count());
            adaptive_cooldown.update_p99(p99_us_f);
            if (adaptive_cooldown.is_under_pressure()) {
                auto new_cd = adaptive_cooldown.get_cooldown();
                auto te_cfg = trade_engine.get_config();
                if (te_cfg.signal_cooldown != new_cd) {
                    te_cfg.signal_cooldown = new_cd;
                    trade_engine.update_config(te_cfg);
                }
            }
        }
#endif

#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
        // Poll stall watchdog — fires on_stall if no token has arrived within timeout.
        stream_health.poll();
#endif

#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
        // Feed current P99 to the latency budget enforcer.
        // Callbacks fire on tier transitions; Breaker tier trips the circuit breaker.
        {
            int64_t p99_us_i = static_cast<int64_t>(stats.p99_latency.count());
            (void)latency_budget_enforcer.check(p99_us_i);
        }
#endif

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.check();
#endif

        // Colour the P99 value: green < 10μs, yellow < 50μs, red otherwise.
        auto p99 = stats.p99_latency.count();
        const char* p99_colour =
            (p99 < 10)  ? C("\033[32m") :
            (p99 < 50)  ? C("\033[33m") : C("\033[31m");

        // Colour the pressure bar.
        const char* press_colour =
            (pressure.composite < 0.5) ? C("\033[32m") :
            (pressure.composite < 0.8) ? C("\033[33m") : C("\033[31m");

#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
        stale_detector.check();
#endif

        // Use the LLMAdapter's own token counter for the stats bar — variance_n
        // is reset every 60 seconds (Welford reset) and would undercount after
        // the first reset interval.
        uint64_t tokens_total = llm_adapter.get_stats().tokens_processed;

        // Saturating addition for the BLOCK counter — prevent silent wrap-around.
        const auto& rs = risk_mgr.get_stats();
        uint64_t blocked = eng_stats.signals_suppressed.load();
        auto sat_add = [](uint64_t a, uint64_t b) -> uint64_t {
            return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
        };
        blocked = sat_add(blocked, rs.signals_blocked_magnitude.load());
        blocked = sat_add(blocked, rs.signals_blocked_confidence.load());
        blocked = sat_add(blocked, rs.signals_blocked_rate.load());
        blocked = sat_add(blocked, rs.signals_blocked_drawdown.load());
        blocked = sat_add(blocked, rs.signals_blocked_position.load());
        blocked = sat_add(blocked, rs.signals_blocked_pnl.load());

        // Welford periodic reset: avoid precision loss over time.
        // Reset the accumulators every 60 seconds (wall-clock) instead of
        // by sample count, preventing catastrophic cancellation while
        // bounding the reset interval to a predictable time window.
        // The mutex ensures the three stores are never interleaved with the
        // Welford update running in the token callback.
        {
            std::lock_guard<std::mutex> lk(variance_mutex);
            auto now_steady = std::chrono::steady_clock::now();
            if (now_steady - variance_last_reset > std::chrono::seconds{60}) {
                variance_n = 0;
                sentiment_mean_accum = 0.0;
                sentiment_variance_accum = 0.0;
                variance_last_reset = now_steady;
            }
        }

        // Update Prometheus snapshot (read once per second from the monitoring
        // thread; the scrape callback returns this cached string without
        // acquiring any latency-path locks).
#ifdef LLMQUANT_PROMETHEUS_ENABLED
        {
            std::ostringstream snap;
            snap << "# HELP llmquant_signals_generated_total Total trade signals generated\n"
                 << "# TYPE llmquant_signals_generated_total counter\n"
                 << "llmquant_signals_generated_total " << eng_stats.signals_generated.load() << "\n"
                 << "# HELP llmquant_signals_suppressed_total Signals with no callback or sink (fully suppressed)\n"
                 << "# TYPE llmquant_signals_suppressed_total counter\n"
                 << "llmquant_signals_suppressed_total " << eng_stats.signals_suppressed.load() << "\n"
                 << "# HELP llmquant_signals_aged_out_total Signals suppressed by the staleness guard\n"
                 << "# TYPE llmquant_signals_aged_out_total counter\n"
                 << "llmquant_signals_aged_out_total " << eng_stats.signals_aged_out.load() << "\n"
                 << "# HELP llmquant_signals_cooldown_suppressed_total Signals skipped because the signal cooldown had not elapsed\n"
                 << "# TYPE llmquant_signals_cooldown_suppressed_total counter\n"
                 << "llmquant_signals_cooldown_suppressed_total " << eng_stats.signals_suppressed_cooldown.load() << "\n"
                 << "# HELP llmquant_accumulator_clamped_total Times the bias accumulator cap was applied\n"
                 << "# TYPE llmquant_accumulator_clamped_total counter\n"
                 << "llmquant_accumulator_clamped_total " << eng_stats.accumulator_clamped.load() << "\n"
                 << "# HELP llmquant_memory_sink_size Current number of signals buffered in the in-memory sink\n"
                 << "# TYPE llmquant_memory_sink_size gauge\n"
                 << "llmquant_memory_sink_size " << memory_sink->size() << "\n"
                 << "# HELP llmquant_memory_sink_dropped_total Signals evicted from memory sink due to capacity cap\n"
                 << "# TYPE llmquant_memory_sink_dropped_total counter\n"
                 << "llmquant_memory_sink_dropped_total " << memory_sink->dropped_count() << "\n"
                 << "# HELP llmquant_signals_blocked_total Total trade signals blocked by risk\n"
                 << "# TYPE llmquant_signals_blocked_total counter\n"
                 << "llmquant_signals_blocked_total " << blocked << "\n"
                 << "# HELP llmquant_signals_blocked_magnitude_total Signals blocked: magnitude exceeded\n"
                 << "# TYPE llmquant_signals_blocked_magnitude_total counter\n"
                 << "llmquant_signals_blocked_magnitude_total " << rs.signals_blocked_magnitude.load() << "\n"
                 << "# HELP llmquant_signals_blocked_confidence_total Signals blocked: confidence below minimum\n"
                 << "# TYPE llmquant_signals_blocked_confidence_total counter\n"
                 << "llmquant_signals_blocked_confidence_total " << rs.signals_blocked_confidence.load() << "\n"
                 << "# HELP llmquant_signals_blocked_rate_total Signals blocked: rate limit exceeded\n"
                 << "# TYPE llmquant_signals_blocked_rate_total counter\n"
                 << "llmquant_signals_blocked_rate_total " << rs.signals_blocked_rate.load() << "\n"
                 << "# HELP llmquant_signals_blocked_drawdown_total Signals blocked: drawdown limit exceeded\n"
                 << "# TYPE llmquant_signals_blocked_drawdown_total counter\n"
                 << "llmquant_signals_blocked_drawdown_total " << rs.signals_blocked_drawdown.load() << "\n"
                 << "# HELP llmquant_signals_blocked_position_total Signals blocked: position limit breached\n"
                 << "# TYPE llmquant_signals_blocked_position_total counter\n"
                 << "llmquant_signals_blocked_position_total " << rs.signals_blocked_position.load() << "\n"
                 << "# HELP llmquant_signals_blocked_pnl_total Signals blocked: PnL limit breached\n"
                 << "# TYPE llmquant_signals_blocked_pnl_total counter\n"
                 << "llmquant_signals_blocked_pnl_total " << rs.signals_blocked_pnl.load() << "\n"
                 << "# HELP llmquant_latency_p99_us p99 token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_p99_us gauge\n"
                 << "llmquant_latency_p99_us " << p99 << "\n"
                 << "# HELP llmquant_latency_avg_us Average token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_avg_us gauge\n"
                 << "llmquant_latency_avg_us " << stats.avg_latency.count() << "\n"
                 << "# HELP llmquant_latency_p50_us p50 (median) token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_p50_us gauge\n"
                 << "llmquant_latency_p50_us " << stats.p50_latency.count() << "\n"
                 << "# HELP llmquant_latency_p95_us p95 token-to-signal latency in microseconds\n"
                 << "# TYPE llmquant_latency_p95_us gauge\n"
                 << "llmquant_latency_p95_us " << stats.p95_latency.count() << "\n"
                 << "# HELP llmquant_tokens_emitted_total Tokens emitted by simulator (0 in stream mode)\n"
                 << "# TYPE llmquant_tokens_emitted_total counter\n"
                 << "llmquant_tokens_emitted_total " << (!stream_mode ? token_sim.get_stats().tokens_emitted.load() : 0) << "\n"
                 << "# HELP llmquant_oms_update_count_total Total successful OMS position updates\n"
                 << "# TYPE llmquant_oms_update_count_total counter\n"
                 << "llmquant_oms_update_count_total "    << oms_adapter->update_count()    << "\n"
                 << "# HELP llmquant_oms_error_count_total Total OMS connection errors\n"
                 << "# TYPE llmquant_oms_error_count_total counter\n"
                 << "llmquant_oms_error_count_total "     << oms_adapter->error_count()     << "\n"
                 << "# HELP llmquant_oms_reconnect_count_total Total FIX session reconnect attempts\n"
                 << "# TYPE llmquant_oms_reconnect_count_total counter\n"
                 << "llmquant_oms_reconnect_count_total " << oms_adapter->reconnect_count() << "\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                 << "# HELP llmquant_dedup_novel_total Tokens processed as novel (not seen in TTL window)\n"
                 << "# TYPE llmquant_dedup_novel_total counter\n"
                 << "llmquant_dedup_novel_total " << dedup_backend->total_novel() << "\n"
                 << "# HELP llmquant_dedup_duplicates_total Tokens suppressed as duplicates within the TTL window\n"
                 << "# TYPE llmquant_dedup_duplicates_total counter\n"
                 << "llmquant_dedup_duplicates_total " << dedup_backend->total_duplicates() << "\n"
                 << "# HELP llmquant_dedup_redis_connected Whether a Redis dedup connection is active\n"
                 << "# TYPE llmquant_dedup_redis_connected gauge\n"
                 << "llmquant_dedup_redis_connected 0\n"
                 << "# HELP llmquant_dedup_redis_reconnect_attempts_total Total Redis reconnect attempts\n"
                 << "# TYPE llmquant_dedup_redis_reconnect_attempts_total counter\n"
                 << "llmquant_dedup_redis_reconnect_attempts_total 0\n"
#endif
                 << "# HELP llmquant_signals_passed_total Total trade signals that passed all risk gates\n"
                 << "# TYPE llmquant_signals_passed_total counter\n"
                 << "llmquant_signals_passed_total " << rs.signals_passed.load() << "\n"
                 << "# HELP llmquant_tokens_processed_total Total tokens processed since startup\n"
                 << "# TYPE llmquant_tokens_processed_total counter\n"
                 << "llmquant_tokens_processed_total " << llm_adapter.get_stats().tokens_processed << "\n"
                 << "# HELP llmquant_cache_hits_total Tokens resolved from the in-memory dictionary cache\n"
                 << "# TYPE llmquant_cache_hits_total counter\n"
                 << "llmquant_cache_hits_total " << llm_adapter.get_stats().cache_hits << "\n"
                 << "# HELP llmquant_cache_misses_total Tokens not found in the dictionary (neutral fallback)\n"
                 << "# TYPE llmquant_cache_misses_total counter\n"
                 << "llmquant_cache_misses_total " << llm_adapter.get_stats().cache_misses << "\n"
                 << "# HELP llmquant_adapter_cache_hit_rate Fraction of token lookups served from dictionary [0,1]\n"
                 << "# TYPE llmquant_adapter_cache_hit_rate gauge\n"
                 << "llmquant_adapter_cache_hit_rate " << std::fixed << std::setprecision(6) << llm_adapter.get_cache_hit_rate() << "\n"
                 << "# HELP llmquant_dictionary_size Number of entries in the LLMAdapter token dictionary\n"
                 << "# TYPE llmquant_dictionary_size gauge\n"
                 << "llmquant_dictionary_size " << llm_adapter.get_dictionary_size() << "\n"
                 << "# HELP llmquant_pressure_composite Current composite back-pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_composite gauge\n"
                 << "llmquant_pressure_composite " << std::fixed << std::setprecision(4) << pressure.composite << "\n"
                 << "# HELP llmquant_pressure_ingestion Current ingestion pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_ingestion gauge\n"
                 << "llmquant_pressure_ingestion " << pressure.ingestion_pressure << "\n"
                 << "# HELP llmquant_pressure_semantic Current semantic variance pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_semantic gauge\n"
                 << "llmquant_pressure_semantic " << pressure.semantic_pressure << "\n"
                 << "# HELP llmquant_pressure_queue Current signal queue pressure [0,1]\n"
                 << "# TYPE llmquant_pressure_queue gauge\n"
                 << "llmquant_pressure_queue " << pressure.queue_pressure << "\n"
                 << "# HELP llmquant_backoff_multiplier Current exponential backoff multiplier [1,5]\n"
                 << "# TYPE llmquant_backoff_multiplier gauge\n"
                 << "llmquant_backoff_multiplier " << std::setprecision(2) << backoff << "\n"
                 << "# HELP llmquant_latency_min_us Minimum observed token-to-signal latency\n"
                 << "# TYPE llmquant_latency_min_us gauge\n"
                 << "llmquant_latency_min_us " << stats.min_latency.count() << "\n"
                 << "# HELP llmquant_latency_max_us Maximum observed token-to-signal latency\n"
                 << "# TYPE llmquant_latency_max_us gauge\n"
                 << "llmquant_latency_max_us " << stats.max_latency.count() << "\n"
                 << "# HELP llmquant_latency_jitter_ms Latency standard deviation in milliseconds\n"
                 << "# TYPE llmquant_latency_jitter_ms gauge\n"
                 << "llmquant_latency_jitter_ms " << std::setprecision(4) << stats.jitter_ms << "\n"
                 << "# HELP llmquant_ring_buffer_drops_total Tokens dropped due to full simulator ring buffer\n"
                 << "# TYPE llmquant_ring_buffer_drops_total counter\n"
                 << "llmquant_ring_buffer_drops_total " << (!stream_mode ? token_sim.get_stats().ring_buffer_drops.load() : 0) << "\n"
                 << "# HELP llmquant_uptime_seconds Engine uptime since startup\n"
                 << "# TYPE llmquant_uptime_seconds gauge\n"
                 << "llmquant_uptime_seconds " << std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - engine_start_time).count() << "\n"
                 << "# HELP llmquant_dry_run Whether the engine is running in dry-run mode (1=yes)\n"
                 << "# TYPE llmquant_dry_run gauge\n"
                 << "llmquant_dry_run " << (dry_run ? 1 : 0) << "\n"
                 << "# HELP llmquant_shadow_mode_active 1 when RiskManager shadow/dry-run mode is active (gates evaluate but never block)\n"
                 << "# TYPE llmquant_shadow_mode_active gauge\n"
                 << "llmquant_shadow_mode_active " << (risk_mgr.get_config().dry_run_mode ? 1 : 0) << "\n"
                 << "# HELP llmquant_backtest_mode Whether the engine is running in backtest mode (1=yes)\n"
                 << "# TYPE llmquant_backtest_mode gauge\n"
                 << "llmquant_backtest_mode " << (backtest_mode ? 1 : 0) << "\n"
                 << "# HELP llmquant_version_info Engine version info (always 1; use labels for version string)\n"
                 << "# TYPE llmquant_version_info gauge\n"
                 << "llmquant_version_info{version=\"" LLMQUANT_VERSION "\"} 1\n"
                 << "# HELP llmquant_start_time_seconds Unix timestamp (seconds) when the engine process started\n"
                 << "# TYPE llmquant_start_time_seconds gauge\n"
                 << "llmquant_start_time_seconds " << engine_start_unix_s << "\n"
                 << "# HELP llmquant_process_rss_bytes Process resident set size in bytes\n"
                 << "# TYPE llmquant_process_rss_bytes gauge\n"
                 << "llmquant_process_rss_bytes " << get_process_rss_bytes() << "\n"
                 << "# HELP llmquant_process_cpu_fraction Process CPU usage fraction since last scrape (0=idle, 1=one full core)\n"
                 << "# TYPE llmquant_process_cpu_fraction gauge\n"
                 << "llmquant_process_cpu_fraction " << std::setprecision(4) << cpu_fraction << "\n"
                 << "# HELP llmquant_avg_signal_strength Running Welford mean of |delta_bias_shift|\n"
                 << "# TYPE llmquant_avg_signal_strength gauge\n"
                 << "llmquant_avg_signal_strength " << std::setprecision(6)
                     << eng_stats.avg_signal_strength.load() << "\n"
                 << "# HELP llmquant_latency_window_fill_ratio Fraction of sample window populated [0,1]\n"
                 << "# TYPE llmquant_latency_window_fill_ratio gauge\n"
                 << "llmquant_latency_window_fill_ratio " << std::fixed << std::setprecision(4) << latency_ctrl.get_window_fill_ratio() << "\n"
                 << "# HELP llmquant_latency_measurements_total Total latency samples recorded\n"
                 << "# TYPE llmquant_latency_measurements_total counter\n"
                 << "llmquant_latency_measurements_total " << stats.measurements << "\n"
                 << "# HELP llmquant_signal_age_threshold_us Configured staleness guard threshold (0=disabled)\n"
                 << "# TYPE llmquant_signal_age_threshold_us gauge\n"
                 << "llmquant_signal_age_threshold_us " << trade_engine.get_config().max_signal_age_us << "\n"
                 << "# HELP llmquant_min_bias_threshold Configured noise-filter minimum |bias| threshold (0=disabled)\n"
                 << "# TYPE llmquant_min_bias_threshold gauge\n"
                 << "llmquant_min_bias_threshold " << trade_engine.get_config().min_bias_threshold << "\n"
                 << "# HELP llmquant_min_vol_threshold Configured noise-filter minimum |vol| threshold (0=disabled)\n"
                 << "# TYPE llmquant_min_vol_threshold gauge\n"
                 << "llmquant_min_vol_threshold " << trade_engine.get_config().min_vol_threshold << "\n"
                 << "# HELP llmquant_max_accumulated_bias Configured accumulator cap (0=disabled)\n"
                 << "# TYPE llmquant_max_accumulated_bias gauge\n"
                 << "llmquant_max_accumulated_bias " << trade_engine.get_config().max_accumulated_bias << "\n"
                 << "# HELP llmquant_p5_latency_us 5th-percentile latency of the sample window (microseconds)\n"
                 << "# TYPE llmquant_p5_latency_us gauge\n"
                 << "llmquant_p5_latency_us " << stats.p5_latency.count() << "\n"
                 << "# HELP llmquant_p25_latency_us 25th-percentile (Q1) latency (microseconds)\n"
                 << "# TYPE llmquant_p25_latency_us gauge\n"
                 << "llmquant_p25_latency_us " << stats.p25_latency.count() << "\n"
                 << "# HELP llmquant_peak_bias Peak absolute value of the accumulated bias since last reset\n"
                 << "# TYPE llmquant_peak_bias gauge\n"
                 << "llmquant_peak_bias " << std::fixed << std::setprecision(6) << trade_engine.get_stats().peak_bias.load() << "\n"
                 << "# HELP llmquant_signal_efficiency Ratio of signals emitted to tokens processed [0,1]\n"
                 << "# TYPE llmquant_signal_efficiency gauge\n"
                 << "llmquant_signal_efficiency " << std::setprecision(6) << trade_engine.get_signal_efficiency() << "\n"
                 << "# HELP llmquant_tokens_per_second Token throughput (tokens processed per second)\n"
                 << "# TYPE llmquant_tokens_per_second gauge\n"
                 << "llmquant_tokens_per_second " << std::setprecision(2) << trade_engine.get_tokens_per_second() << "\n"
                 << "# HELP llmquant_avg_signal_quality Welford running mean of signal_quality [0,1]\n"
                 << "# TYPE llmquant_avg_signal_quality gauge\n"
                 << "llmquant_avg_signal_quality " << std::setprecision(6) << eng_stats.avg_signal_quality.load() << "\n"
                 << [&]() -> std::string {
                        double q_ema = trade_engine.get_signal_quality_ema();
                        if (q_ema < 0.0) return "";
                        std::ostringstream o;
                        o << "# HELP llmquant_signal_quality_ema EMA(alpha=0.1) of signal_quality [0,1]; omitted until first signal\n"
                          << "# TYPE llmquant_signal_quality_ema gauge\n"
                          << "llmquant_signal_quality_ema " << std::setprecision(6) << q_ema << "\n";
                        return o.str();
                    }()
#ifdef LLMQUANT_DEDUP_ENABLED
                 << "# HELP llmquant_dedup_duplicate_rate Fraction of checked tokens that were duplicates [0,1]\n"
                 << "# TYPE llmquant_dedup_duplicate_rate gauge\n"
                 << "llmquant_dedup_duplicate_rate " << [&]() -> double {
                        uint64_t dupes = dedup_backend->total_duplicates();
                        uint64_t novel = dedup_backend->total_novel();
                        uint64_t total = novel + dupes;
                        return (total > 0) ? (static_cast<double>(dupes) / static_cast<double>(total)) : 0.0;
                    }() << "\n"
#endif
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
                 << [&]() -> std::string {
                        if (!audit_log) return "";
                        std::ostringstream o;
                        o << "# HELP llmquant_audit_records_written_total Total signal audit records written to disk\n"
                          << "# TYPE llmquant_audit_records_written_total counter\n"
                          << "llmquant_audit_records_written_total " << audit_log->records_written() << "\n"
                          << "# HELP llmquant_audit_records_dropped_total Signal audit records dropped (queue full)\n"
                          << "# TYPE llmquant_audit_records_dropped_total counter\n"
                          << "llmquant_audit_records_dropped_total " << audit_log->records_dropped() << "\n";
                        return o.str();
                    }()
#endif
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
                 << "# HELP llmquant_circuit_breaker_state Circuit breaker state: 0=closed 1=open 2=half_open\n"
                 << "# TYPE llmquant_circuit_breaker_state gauge\n"
                 << "llmquant_circuit_breaker_state " << static_cast<int>(circuit_breaker.state()) << "\n"
                 << "# HELP llmquant_circuit_breaker_block_rate EMA block rate observed by circuit breaker [0,1]\n"
                 << "# TYPE llmquant_circuit_breaker_block_rate gauge\n"
                 << "llmquant_circuit_breaker_block_rate " << std::setprecision(4) << circuit_breaker.block_rate() << "\n"
                 << "# HELP llmquant_circuit_breaker_trips_total Times circuit has tripped to OPEN\n"
                 << "# TYPE llmquant_circuit_breaker_trips_total counter\n"
                 << "llmquant_circuit_breaker_trips_total " << circuit_breaker.trips() << "\n"
                 << "# HELP llmquant_circuit_breaker_recoveries_total Times circuit has recovered to CLOSED\n"
                 << "# TYPE llmquant_circuit_breaker_recoveries_total counter\n"
                 << "llmquant_circuit_breaker_recoveries_total " << circuit_breaker.recoveries() << "\n"
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
                 << "# HELP llmquant_stream_stale Whether the LLM token stream is currently silent (1=stale)\n"
                 << "# TYPE llmquant_stream_stale gauge\n"
                 << "llmquant_stream_stale " << (stale_detector.is_stale() ? 1 : 0) << "\n"
                 << "# HELP llmquant_stream_stale_events_total Times the token stream went silent\n"
                 << "# TYPE llmquant_stream_stale_events_total counter\n"
                 << "llmquant_stream_stale_events_total " << stale_detector.stale_events() << "\n"
                 << "# HELP llmquant_stream_ms_since_last_token Milliseconds since last token arrived\n"
                 << "# TYPE llmquant_stream_ms_since_last_token gauge\n"
                 << "llmquant_stream_ms_since_last_token " << stale_detector.ms_since_last_token() << "\n"
#endif
                ;
            // Signal quality histogram — per-bucket counts emitted as a Prometheus histogram.
            {
                auto qhb = trade_engine.get_quality_histogram();
                snap << "# HELP llmquant_signal_quality_histogram Distribution of emitted signal quality scores\n"
                     << "# TYPE llmquant_signal_quality_histogram histogram\n";
                uint64_t cumulative = 0;
                for (const auto& b : qhb) {
                    cumulative += b.count;
                    snap << "llmquant_signal_quality_histogram_bucket{le=\"" << b.upper_bound << "\"} " << cumulative << "\n";
                }
                snap << "llmquant_signal_quality_histogram_bucket{le=\"+Inf\"} " << cumulative << "\n";
                snap << "llmquant_signal_quality_histogram_count " << cumulative << "\n";
                double avg_q = eng_stats.avg_signal_quality.load();
                snap << "llmquant_signal_quality_histogram_sum " << std::setprecision(6)
                     << (std::isfinite(avg_q) ? avg_q * static_cast<double>(cumulative) : 0.0) << "\n";
            }
            // Prometheus native histogram — cumulative latency buckets.
            {
                auto hb = latency_ctrl.histogram_buckets();
                snap << "# HELP llmquant_token_latency_us Cumulative latency histogram of token-to-signal processing time (µs)\n"
                     << "# TYPE llmquant_token_latency_us histogram\n";
                uint64_t last_count = 0;
                for (const auto& b : hb) {
                    if (std::isinf(b.upper_bound_us)) {
                        snap << "llmquant_token_latency_us_bucket{le=\"+Inf\"} " << b.count << "\n";
                    } else {
                        snap << "llmquant_token_latency_us_bucket{le=\"" << b.upper_bound_us << "\"} " << b.count << "\n";
                    }
                    last_count = b.count;
                }
                snap << "llmquant_token_latency_us_count " << last_count << "\n"
                     << "llmquant_token_latency_us_sum "   << latency_ctrl.get_total_latency_us() << "\n";
            }
            {
                auto hs = latency_ctrl.get_health_state();
                snap << "# HELP llmquant_latency_warmed_up 1 when the sample window is >=50% populated, 0 during warmup\n"
                     << "# TYPE llmquant_latency_warmed_up gauge\n"
                     << "llmquant_latency_warmed_up " << (hs.warmed_up ? 1 : 0) << "\n"
                     << "# HELP llmquant_latency_budget_remaining_us Signed latency budget: target_us minus p99_us (negative = over budget)\n"
                     << "# TYPE llmquant_latency_budget_remaining_us gauge\n"
                     << "llmquant_latency_budget_remaining_us " << std::fixed << std::setprecision(1) << hs.budget_remaining_us << "\n";
            }
            snap << "# HELP llmquant_drawdown_cumulative_bias Current cumulative bias in the drawdown window\n"
                 << "# TYPE llmquant_drawdown_cumulative_bias gauge\n"
                 << "llmquant_drawdown_cumulative_bias " << std::fixed << std::setprecision(4) << risk_mgr.get_cumulative_bias() << "\n"
                 << "# HELP llmquant_risk_pass_rate_pct Percentage of signals that passed all risk gates (0-100)\n"
                 << "# TYPE llmquant_risk_pass_rate_pct gauge\n"
                 << "llmquant_risk_pass_rate_pct " << std::setprecision(4) << ((1.0 - risk_mgr.get_blocked_rate()) * 100.0) << "\n"
                 << "# HELP llmquant_slo_breach_rate Fraction of latency samples that exceeded the p99 target [0,1]\n"
                 << "# TYPE llmquant_slo_breach_rate gauge\n"
                 << "llmquant_slo_breach_rate " << std::setprecision(6) << latency_ctrl.get_slo_breach_rate() << "\n"
                 << "# HELP llmquant_drawdown_utilization Fraction of drawdown budget consumed in current window [0,1]\n"
                 << "# TYPE llmquant_drawdown_utilization gauge\n"
                 << "llmquant_drawdown_utilization " << std::setprecision(4) << risk_mgr.get_drawdown_utilization() << "\n"
                 << "# HELP llmquant_rate_limit_utilization Fraction of per-second rate cap consumed in current window [0,1]\n"
                 << "# TYPE llmquant_rate_limit_utilization gauge\n"
                 << "llmquant_rate_limit_utilization " << std::setprecision(4) << risk_mgr.get_rate_limit_utilization() << "\n"
                 << "# HELP llmquant_noise_filtered_total Tokens suppressed by the min-bias noise gate\n"
                 << "# TYPE llmquant_noise_filtered_total counter\n"
                 << "llmquant_noise_filtered_total " << eng_stats.noise_filtered.load() << "\n"
                 << "# HELP llmquant_bias_reversals_total Number of times accumulated_bias changed direction (sign reversal / momentum crossover)\n"
                 << "# TYPE llmquant_bias_reversals_total counter\n"
                 << "llmquant_bias_reversals_total " << eng_stats.bias_reversals.load() << "\n"
                 << "# HELP llmquant_risk_healthy Whether all risk gates are nominally healthy (1=yes)\n"
                 << "# TYPE llmquant_risk_healthy gauge\n"
                 << "llmquant_risk_healthy " << (risk_mgr.is_healthy() ? 1 : 0) << "\n"
#ifdef LLMQUANT_DEDUP_ENABLED
                 << "# HELP llmquant_dedup_dup_rate_pct Duplicate token rate as percentage [0,100]\n"
                 << "# TYPE llmquant_dedup_dup_rate_pct gauge\n"
                 << "llmquant_dedup_dup_rate_pct " << [&]() -> double {
                        uint64_t nov = dedup_backend->total_novel();
                        uint64_t dup = dedup_backend->total_duplicates();
                        uint64_t tot = nov + dup;
                        return (tot > 0) ? (static_cast<double>(dup) * 100.0 / static_cast<double>(tot)) : 0.0;
                    }() << "\n"
#endif
                ;
            // Top-5 influential tokens as labeled gauges for Grafana dashboards.
            {
                snap << "# HELP llmquant_top_influence_token Composite influence score (freq+bias blend) [0,1]\n"
                     << "# TYPE llmquant_top_influence_token gauge\n";
                for (const auto& [tok, score] : llm_adapter.export_hot_tokens(5)) {
                    // Prometheus text format requires escaping \, ", and \n in label values.
                    std::string safe_tok;
                    safe_tok.reserve(tok.size());
                    for (char c : tok) {
                        if (c == '\\')     { safe_tok += "\\\\"; }
                        else if (c == '"') { safe_tok += "\\\""; }
                        else if (c == '\n') { safe_tok += "\\n"; }
                        else               { safe_tok += c; }
                    }
                    snap << "llmquant_top_influence_token{token=\"" << safe_tok << "\"} "
                         << std::setprecision(4) << score << "\n";
                }
            }
            std::lock_guard<std::mutex> lk(prom_snapshot_mutex);
            prom_snapshot = snap.str();
        }
#endif // LLMQUANT_PROMETHEUS_ENABLED

        // Cache hit rate for LLMAdapter dictionary efficiency.
        auto adapter_stats = llm_adapter.get_stats();
        uint64_t hit_pct = (adapter_stats.tokens_processed > 0)
            ? (adapter_stats.cache_hits * 100 / adapter_stats.tokens_processed) : 0;

        // Log periodic latency snapshot and pipeline health (independent of --quiet).
        logger.log_latency_measurement(static_cast<uint64_t>(p99));
        {
            bool slo_healthy = (p99 <= sys_config.latency.target_latency_us);
            logger.log_pipeline_health(slo_healthy,
                                       latency_ctrl.get_slo_breach_rate(),
                                       backoff);
            if (!slo_healthy && last_tick != stats.measurements) {
                spdlog::warn("P99 latency {}us exceeds target {}us (breach rate {:.1f}%)",
                             p99, sys_config.latency.target_latency_us,
                             latency_ctrl.get_slo_breach_rate() * 100.0);
            }
        }
        // Log system resource usage once per second (memory RSS + CPU).
        // MetricsLogger::log_system_stats expects cpu_usage as percentage (0-100);
        // get_process_cpu_fraction() returns a fraction [0, N_cores], so multiply by 100.
        logger.log_system_stats(get_process_rss_bytes(), cpu_fraction * 100.0);

        // Compute rolling regime classification from sparkline ring (lag-1 AC).
        // mean > +0.05 && AC > 0.10  → BULL  mean < -0.05 && AC > 0.10  → BEAR
        // AC < -0.15 → CHOP  |mean| < 0.02 → FLAT  else → NOIS
        std::string current_regime;
        std::string current_regime_str;
        {
            int rgm_head = spark_head.load(std::memory_order_relaxed);
            int rgm_n    = std::min(rgm_head, kSparkSlots);
            if (rgm_n >= 4) {
                int rgm_start = (rgm_head >= kSparkSlots) ? (rgm_head % kSparkSlots) : 0;
                double rgm_sum = 0.0;
                for (int i = 0; i < rgm_n; ++i)
                    rgm_sum += spark_ring[(rgm_start + i) % kSparkSlots];
                double rgm_mean = rgm_sum / static_cast<double>(rgm_n);
                double rgm_cov = 0.0, rgm_var = 0.0;
                for (int i = 1; i < rgm_n; ++i) {
                    double x0 = spark_ring[(rgm_start + i - 1) % kSparkSlots] - rgm_mean;
                    double x1 = spark_ring[(rgm_start + i)     % kSparkSlots] - rgm_mean;
                    rgm_cov += x0 * x1;
                    rgm_var += x0 * x0;
                }
                double rgm_ac = (rgm_var > 1e-12) ? rgm_cov / rgm_var : 0.0;
                const char* rgm_col;
                if      (rgm_mean >  0.05 && rgm_ac >  0.10) { current_regime = "BULL"; rgm_col = "\033[32m"; }
                else if (rgm_mean < -0.05 && rgm_ac >  0.10) { current_regime = "BEAR"; rgm_col = "\033[31m"; }
                else if (rgm_ac < -0.15)                      { current_regime = "CHOP"; rgm_col = "\033[33m"; }
                else if (std::abs(rgm_mean) < 0.02)           { current_regime = "FLAT"; rgm_col = "\033[90m"; }
                else                                          { current_regime = "NOIS"; rgm_col = "\033[35m"; }
                current_regime_str = std::string("  RGM:") + C(rgm_col) + current_regime + C("\033[0m");
            }
        }

        // Signal morphology detector: scan the last 6 sparkline values for
        // named candlestick-like shapes in the bias stream.
        // Patterns: RALLY (4+ up), SELLOFF (4+ down), V_RVSL (3d+3u),
        //           INV_V (3u+3d), CONSOLIDATE (6 within ±0.02).
        std::string current_morphology;
        {
            int morph_head = spark_head.load(std::memory_order_relaxed);
            int morph_n    = std::min(morph_head, kSparkSlots);
            if (morph_n >= 6) {
                int morph_start = (morph_head >= kSparkSlots) ? (morph_head % kSparkSlots) : 0;
                // Extract the 6 most-recent values (indices morph_n-6 .. morph_n-1).
                double v[6];
                for (int i = 0; i < 6; ++i)
                    v[i] = spark_ring[(morph_start + morph_n - 6 + i) % kSparkSlots];

                // Compute differences
                int up6 = 0, dn6 = 0;
                for (int i = 1; i < 6; ++i) {
                    if (v[i] > v[i-1] + 0.005) ++up6;
                    else if (v[i] < v[i-1] - 0.005) ++dn6;
                }
                // First-half and second-half runs
                int up3a = (v[1]>v[0]+0.005)+(v[2]>v[1]+0.005);
                int dn3a = (v[1]<v[0]-0.005)+(v[2]<v[1]-0.005);
                int up3b = (v[4]>v[3]+0.005)+(v[5]>v[4]+0.005);
                int dn3b = (v[4]<v[3]-0.005)+(v[5]<v[4]-0.005);
                double band = *std::max_element(v, v+6) - *std::min_element(v, v+6);

                if      (up6 >= 4)                  current_morphology = "RALLY";
                else if (dn6 >= 4)                  current_morphology = "SELLOFF";
                else if (dn3a >= 2 && up3b >= 2)    current_morphology = "V_RVSL";
                else if (up3a >= 2 && dn3b >= 2)    current_morphology = "INV_V";
                else if (band < 0.02)               current_morphology = "CONSOL";

                if (!current_morphology.empty() && current_morphology != last_morphology) {
                    spdlog::info("[morphology] pattern detected: {}", current_morphology);
                }
                last_morphology = current_morphology;
            }
        }

        // Overwrite the stats line in-place. Suppressed in --quiet mode.
        if (!quiet) {
            std::cout << "\n  -- STATS "
                      << " TPS:"   << std::setw(4) << tps
                      << "  TOK:"  << std::setw(7) << tokens_total
                      << "  AVG:"  << std::setw(5) << stats.avg_latency.count() << "us"
                      << "  P99:"  << p99_colour
                                   << std::setw(5) << p99 << "us" << C("\033[0m")
                      << "  PRESS:" << press_colour
                                   << std::fixed << std::setprecision(2)
                                   << pressure.composite << C("\033[0m")
                      << "  BKOF:" << std::setprecision(1) << backoff << "x"
                      << "  HIT%:" << hit_pct
#ifdef LLMQUANT_DEDUP_ENABLED
                      << "  DEDUP:" << dedup_backend->total_duplicates()
#endif
                      << "  NOISE:" << trade_engine.get_stats().noise_filtered.load()
                      << "  PASS:" << risk_mgr.get_stats().signals_passed.load()
                      << "  BLOCK:" << blocked
                      << "  RATE%:" << [&]() -> uint64_t {
                            uint64_t passed = risk_mgr.get_stats().signals_passed.load();
                            uint64_t total  = (passed > UINT64_MAX - blocked) ? UINT64_MAX : passed + blocked;
                            return (total > 0) ? (passed * 100 / total) : 100;
                         }()
                      << (!stream_mode ? (std::string("  DROPS:") + std::to_string(token_sim.get_stats().ring_buffer_drops.load())) : "")
                      << [&]() -> std::string {
                            double q_ema = trade_engine.get_signal_quality_ema();
                            if (q_ema < 0.0) return "";
                            std::ostringstream o;
                            o << "  Q-EMA:" << std::fixed << std::setprecision(2) << q_ema;
                            return o.str();
                         }()
                      << [&]() -> std::string {
                            // Signal velocity: bias per second. Skip if no signal yet.
                            double vel = sig_velocity.load(std::memory_order_relaxed);
                            if (vel == 0.0) return "";
                            std::ostringstream o;
                            o << "  VEL:";
                            if (vel > 0.01)       o << C("\033[32m");
                            else if (vel < -0.01) o << C("\033[31m");
                            else                  o << C("\033[90m");
                            o << std::showpos << std::fixed << std::setprecision(3) << vel;
                            o << "/s" << C("\033[0m");
                            return o.str();
                         }()
                      << [&]() -> std::string {
                            // Render 24-slot sparkline of recent delta_bias_shift values.
                            // Values are clamped to [-1, 1] and mapped to 8 block levels.
                            // Slots not yet written (head < kSparkSlots) render as '·'.
                            static const char* kBlocks[] = {
                                "▁","▂","▃","▄","▅","▆","▇","█"
                            };
                            int head = spark_head.load(std::memory_order_relaxed);
                            if (head == 0) return "";
                            std::string out = "  BIAS:";
                            int filled = std::min(head, kSparkSlots);
                            int start  = (head >= kSparkSlots) ? (head % kSparkSlots) : 0;
                            for (int i = 0; i < kSparkSlots; ++i) {
                                if (i >= filled) { out += C("\033[90m"); out += "·"; out += C("\033[0m"); continue; }
                                int slot = (start + i) % kSparkSlots;
                                double v = spark_ring[slot];
                                // Map [-1,1] → [0,7]; neutral (0) → level 3 (▄)
                                double clamped = std::max(-1.0, std::min(1.0, v));
                                int level = static_cast<int>((clamped + 1.0) / 2.0 * 7.0 + 0.5);
                                // Colour: positive=green, negative=red, near-zero=yellow
                                if (v > 0.05)       out += C("\033[32m");
                                else if (v < -0.05) out += C("\033[31m");
                                else                out += C("\033[33m");
                                out += kBlocks[level];
                                out += C("\033[0m");
                            }
                            return out;
                         }()
                      << [&]() -> std::string { return current_regime_str; }()
                      << [&]() -> std::string {
                            if (current_morphology.empty()) return "";
                            // Colour by pattern type
                            const char* col;
                            if      (current_morphology == "RALLY")  col = "\033[32m";
                            else if (current_morphology == "SELLOFF") col = "\033[31m";
                            else if (current_morphology == "V_RVSL") col = "\033[36m";
                            else if (current_morphology == "INV_V")  col = "\033[35m";
                            else                                      col = "\033[90m";
                            return std::string("  MORPH:") + C(col) + current_morphology + C("\033[0m");
                         }()
                      << [&]() -> std::string {
                            // Bias Sharpe: mean(bias)/stddev(bias) over the sparkline window.
                            // Signal-to-noise ratio. |SHP|>1.0=directional, <0.5=noise.
                            int sh_n = std::min(spark_head.load(std::memory_order_relaxed), kSparkSlots);
                            if (sh_n < 4) return "";
                            int sh_start = (spark_head.load(std::memory_order_relaxed) >= kSparkSlots)
                                           ? (spark_head.load(std::memory_order_relaxed) % kSparkSlots) : 0;
                            double sh_mean = 0.0, sh_m2 = 0.0;
                            for (int i = 0; i < sh_n; ++i) {
                                double x = spark_ring[(sh_start + i) % kSparkSlots];
                                double d = x - sh_mean;
                                sh_mean += d / (i + 1);
                                sh_m2   += d * (x - sh_mean);
                            }
                            double sh_std = (sh_n > 1 && sh_m2 > 1e-12)
                                            ? std::sqrt(sh_m2 / (sh_n - 1)) : 0.0;
                            if (sh_std < 1e-9) return "";
                            double sharpe = sh_mean / sh_std;
                            std::ostringstream o;
                            o << "  SHP:";
                            if      (sharpe >  1.0) o << C("\033[32m");
                            else if (sharpe < -1.0) o << C("\033[31m");
                            else                    o << C("\033[90m");
                            o << std::showpos << std::fixed << std::setprecision(2) << sharpe << C("\033[0m");
                            return o.str();
                         }()
#ifdef LLMQUANT_ENTROPY_MONITOR_ENABLED
                      << [&]() -> std::string {
                            // Token entropy: 0=focused/repetitive, 1=uniform/noisy.
                            double h     = entropy_monitor.entropy();
                            bool focused = entropy_monitor.is_focused();
                            std::ostringstream o;
                            o << "  ENT:";
                            if      (focused)  o << C("\033[32m");   // green  = focused
                            else if (h > 0.75) o << C("\033[31m");   // red    = noisy
                            else               o << C("\033[33m");   // yellow = mixed
                            o << std::fixed << std::setprecision(2) << h << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_NARRATIVE_CHANGE_ENABLED
                      << [&]() -> std::string {
                            // Narrative similarity: 1=consistent, 0=topic break.
                            double sim  = narrative_detector.get_similarity();
                            bool brk    = narrative_detector.is_narrative_break();
                            std::ostringstream o;
                            o << "  NRR:";
                            if      (brk)        o << C("\033[31m");   // red    = break
                            else if (sim > 0.75)  o << C("\033[32m");   // green  = stable
                            else                  o << C("\033[33m");   // yellow = shifting
                            o << std::fixed << std::setprecision(2) << sim << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
                      << [&]() -> std::string {
                            // Hurst exponent: >0.55=trending, <0.45=mean-rev, else random.
                            double h = fractal_dim.hurst();
                            std::ostringstream o;
                            o << "  FRC:";
                            if      (h > 0.55) o << C("\033[36m");   // cyan   = trending
                            else if (h < 0.45) o << C("\033[35m");   // magenta= mean-rev
                            else               o << C("\033[90m");   // grey   = random
                            o << std::fixed << std::setprecision(2) << h << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
                      << [&]() -> std::string {
                            // Context fill fraction: green=ok, yellow=warn, red=critical.
                            double f = context_budget.fill_fraction();
                            std::ostringstream o;
                            o << "  CTX:";
                            if      (f >= 0.90) o << C("\033[31m");  // red    = critical
                            else if (f >= 0.70) o << C("\033[33m");  // yellow = warn
                            else                o << C("\033[32m");  // green  = normal
                            o << std::fixed << std::setprecision(0) << (f * 100.0) << "%" << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
                      << [&]() -> std::string {
                            // MACD histogram: positive=bullish momentum, negative=bearish.
                            double hist = signal_momentum_osc.histogram();
                            std::ostringstream o;
                            o << "  OSC:";
                            if      (hist >  0.005) o << C("\033[32m");  // green  = bullish
                            else if (hist < -0.005) o << C("\033[31m");  // red    = bearish
                            else                    o << C("\033[90m");  // grey   = flat
                            o << std::showpos << std::fixed << std::setprecision(3) << hist << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
                      << [&]() -> std::string {
                            // Dominant cycle period; highlighted when cyclic pattern detected.
                            int    p  = sentiment_cycle.dominant_period();
                            bool   cy = sentiment_cycle.is_cyclic();
                            std::ostringstream o;
                            o << "  CYC:";
                            if      (cy && p > 0)  o << C("\033[36m");  // cyan  = active cycle
                            else if (p > 0)         o << C("\033[33m");  // yellow= weak cycle
                            else                    o << C("\033[90m");  // grey  = none
                            if (p > 0) o << p;
                            else       o << "?";
                            o << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
                      << [&]() -> std::string {
                            // Normalized self-information: 0=expected, 1=maximally surprising.
                            double s   = signal_surprise.surprise();
                            bool   hi  = signal_surprise.is_high_surprise();
                            std::ostringstream o;
                            o << "  SUR:";
                            if      (hi)       o << C("\033[35m");   // magenta = surprise
                            else if (s > 0.5)  o << C("\033[33m");   // yellow  = moderate
                            else               o << C("\033[32m");   // green   = expected
                            o << std::fixed << std::setprecision(2) << s << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
                      << [&]() -> std::string {
                            // Stream health: green=healthy, red=stalled/flooded.
                            bool healthy = stream_health.is_healthy();
                            auto st = stream_health.status();
                            std::ostringstream o;
                            o << "  HLT:";
                            if (healthy) {
                                o << C("\033[32m") << "OK";
                            } else if (st == llmquant::TokenStreamHealthMonitor::Status::Stalled) {
                                o << C("\033[31m") << "STALL";
                            } else {
                                o << C("\033[31m") << "FLOOD";
                            }
                            o << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
                      << [&]() -> std::string {
                            // Bias velocity: green=fast-positive, red=fast-negative, grey=slow.
                            double v = velocity_tracker.velocity();
                            bool   fast = velocity_tracker.is_fast_move();
                            std::ostringstream o;
                            o << "  VEL:";
                            if      (fast && v > 0) o << C("\033[32m");   // green  = fast bullish
                            else if (fast && v < 0) o << C("\033[31m");   // red    = fast bearish
                            else                    o << C("\033[90m");   // grey   = slow
                            o << std::showpos << std::fixed << std::setprecision(3) << v << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
                      << [&]() -> std::string {
                            // Narrative clock quadrant: Q1=Rising/Q2=Fading/Q3=Falling/Q4=Recovering.
                            auto q = narrative_clock.quadrant();
                            using Q = llmquant::NarrativeMomentumClock::Quadrant;
                            std::ostringstream o;
                            o << "  CLK:";
                            switch (q) {
                                case Q::Rising:     o << C("\033[32m") << "Q1"; break;  // green
                                case Q::Fading:     o << C("\033[33m") << "Q2"; break;  // yellow
                                case Q::Falling:    o << C("\033[31m") << "Q3"; break;  // red
                                case Q::Recovering: o << C("\033[36m") << "Q4"; break;  // cyan
                            }
                            o << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
                      << [&]() -> std::string {
                            int cur  = sentiment_persistence.current_state();
                            int pred = sentiment_persistence.predicted_state();
                            std::ostringstream o;
                            o << "  MKV:";
                            if (cur < 0) { o << C("\033[90m") << "---" << C("\033[0m"); return o.str(); }
                            // colour: lower states bearish (red), upper states bullish (green)
                            const char* col = (cur >= 3) ? "\033[32m" : (cur <= 1) ? "\033[31m" : "\033[33m";
                            o << C(col) << cur << C("\033[0m");
                            if (pred >= 0) o << C("\033[90m") << "→" << pred << C("\033[0m");
                            return o.str();
                         }()
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
                      << [&]() -> std::string {
                            int r = sentiment_phase_portrait.current_row();
                            int c = sentiment_phase_portrait.current_col();
                            bool cyc = sentiment_phase_portrait.cycle_detected();
                            std::ostringstream o;
                            o << "  PHS:" << C("\033[36m") << r << "," << c << C("\033[0m");
                            if (cyc) o << C("\033[33m") << "~" << C("\033[0m");
                            return o.str();
                         }()
#endif
                      << std::flush;

            // Regime-change alert: log to spdlog when classified regime transitions.
            if (!current_regime.empty() && current_regime != last_regime && !last_regime.empty()) {
                spdlog::info("[regime] {} → {}", last_regime, current_regime);
            }
            last_regime = current_regime;

            // Alert if P99 exceeds budget.
            if (p99 > sys_config.latency.target_latency_us && last_tick != stats.measurements) {
                std::cout << "  " << C("\033[31m") << "[!] P99 > target" << C("\033[0m") << std::flush;
            }
        }
        last_tick = stats.measurements;
    }

    token_sim.stop();
#ifdef LLMQUANT_STREAM_CLIENT_ENABLED
    if (stream_client) stream_client->stop();
#endif
    oms_adapter->stop();
#ifdef LLMQUANT_PROMETHEUS_ENABLED
    prom_exporter.stop();
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    health_server.stop();
#endif
    // Flush all output sinks (CSV/JSON) before printing the session summary
    // so any buffered writes are visible if a crash follows.
    trade_engine.flush_sinks();
    config.stop_watching();

    auto final_stats = latency_ctrl.get_stats();
    std::cout << "\n\n  =========================================================\n";
    std::cout << "  SESSION SUMMARY\n";
    std::cout << "  ---------------------------------------------------------\n";
    // Use LLMAdapter's cumulative counter — variance_n resets every 60 s.
    std::cout << "  Tokens processed : " << llm_adapter.get_stats().tokens_processed << "\n";
    std::cout << "  Signals emitted  : " << trade_engine.get_stats().signals_generated.load() << "\n";
    {
        const auto& frs = risk_mgr.get_stats();
        auto fsat = [](uint64_t a, uint64_t b) -> uint64_t {
            return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
        };
        uint64_t fblocked = frs.signals_blocked_magnitude.load();
        fblocked = fsat(fblocked, frs.signals_blocked_confidence.load());
        fblocked = fsat(fblocked, frs.signals_blocked_rate.load());
        fblocked = fsat(fblocked, frs.signals_blocked_drawdown.load());
        fblocked = fsat(fblocked, frs.signals_blocked_position.load());
        fblocked = fsat(fblocked, frs.signals_blocked_pnl.load());
        std::cout << "  Signals blocked  : " << fblocked << "\n";
        std::cout << "  Blocked by gate  : " << risk_mgr.format_blocked_by_gate() << "\n";
    }
    std::cout << "  Most blocked gate: " << risk_mgr.get_most_blocked_gate() << "\n";
    std::cout << "  Memory sink size : " << memory_sink->get_signals().size() << "\n";
    std::cout << "  Avg latency      : " << final_stats.avg_latency.count() << "us\n";
    std::cout << "  Min latency      : " << final_stats.min_latency.count() << "us\n";
    std::cout << "  P50 latency      : " << final_stats.p50_latency.count() << "us\n";
    std::cout << "  P95 latency      : " << final_stats.p95_latency.count() << "us\n";
    std::cout << "  P99 latency      : " << final_stats.p99_latency.count() << "us\n";
    std::cout << "  Max latency      : " << final_stats.max_latency.count() << "us\n";
    std::cout << "  P5  latency      : " << final_stats.p5_latency.count()  << "us\n";
    std::cout << "  P25 latency      : " << final_stats.p25_latency.count() << "us\n";
    std::cout << "  Avg sig strength : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().avg_signal_strength.load() << "\n";
    std::cout << "  Avg sig quality  : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().avg_signal_quality.load() << "\n";
    {
        double ema = trade_engine.get_signal_quality_ema();
        if (ema >= 0.0) {
            std::cout << "  Quality EMA(0.1) : " << std::fixed << std::setprecision(4) << ema << "\n";
        }
    }
    std::cout << "  Noise filtered   : " << trade_engine.get_stats().noise_filtered.load() << "\n";
    std::cout << "  Cooldown skip    : " << trade_engine.get_stats().signals_suppressed_cooldown.load() << "\n";
    std::cout << "  Peak bias        : " << std::fixed << std::setprecision(4)
              << trade_engine.get_stats().peak_bias.load() << "\n";
    std::cout << "  SLO breach rate  : " << std::fixed << std::setprecision(2)
              << (latency_ctrl.get_slo_breach_rate() * 100.0) << "%\n";
    std::cout << "  Jitter           : " << std::fixed << std::setprecision(3)
              << final_stats.jitter_ms << "ms\n";
    {
        auto ads = llm_adapter.get_stats();
        uint64_t hit_pct2 = (ads.tokens_processed > 0)
            ? (ads.cache_hits * 100 / ads.tokens_processed) : 0;
        std::cout << "  Cache hit rate   : " << hit_pct2 << "% ("
                  << ads.cache_hits << "/" << ads.tokens_processed << ")\n";
    }
    std::cout << "  Signals aged out : " << trade_engine.get_stats().signals_aged_out.load() << "\n";
    std::cout << "  Accum. clamped   : " << trade_engine.get_stats().accumulator_clamped.load() << "\n";
    std::cout << "  Signals passed   : " << risk_mgr.get_stats().signals_passed.load() << "\n";
    std::cout << "  Latency warmup   : " << std::fixed << std::setprecision(0)
              << (latency_ctrl.get_window_fill_ratio() * 100.0) << "% window filled\n";
#ifdef LLMQUANT_DEDUP_ENABLED
    {
        auto ds = deduplicator.get_stats();
        uint64_t total_dedup = ds.total_novel + ds.total_duplicates;
        double dup_rate = (total_dedup > 0)
            ? (static_cast<double>(ds.total_duplicates) * 100.0 / static_cast<double>(total_dedup))
            : 0.0;
        std::cout << "  Dedup novel      : " << ds.total_novel << "\n";
        std::cout << "  Dedup duplicates : " << ds.total_duplicates
                  << "  (" << std::fixed << std::setprecision(1) << dup_rate << "% dup rate)\n";
    }
#endif
    {
        auto uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - engine_start_time).count();
        std::cout << "  Uptime           : " << uptime_s << "s\n";
    }
    std::cout << "  Log entries      : " << logger.get_log_entry_count() << "\n";
#ifdef LLMQUANT_AUDIT_LOG_ENABLED
    if (audit_log) {
        std::cout << "  Audit written    : " << audit_log->records_written()
                  << "  (dropped=" << audit_log->records_dropped()
                  << "  rot=" << audit_log->rotations() << ")\n";
        std::cout << "  Audit log file   : " << audit_log_path << "\n";
    }
#endif
#ifdef LLMQUANT_CIRCUIT_BREAKER_ENABLED
    std::cout << "  Circuit breaker  : " << circuit_breaker.state_name()
              << "  trips=" << circuit_breaker.trips()
              << "  recoveries=" << circuit_breaker.recoveries()
              << "  block_rate=" << std::fixed << std::setprecision(1)
              << (circuit_breaker.block_rate() * 100.0) << "%\n";
#endif
#ifdef LLMQUANT_HEALTH_SERVER_ENABLED
    std::cout << "  Health requests  : " << health_server.requests_served() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED
    std::cout << "  Adaptive cooldown: " << std::fixed << std::setprecision(0)
              << adaptive_cooldown.get_cooldown_us() << "µs"
              << "  expansions=" << adaptive_cooldown.pressure_expansions()
              << "  recoveries=" << adaptive_cooldown.recoveries() << "\n";
#endif
#ifdef LLMQUANT_STALE_DETECTOR_ENABLED
    std::cout << "  Stream stale evts: " << stale_detector.stale_events()
              << "  recoveries=" << stale_detector.recovery_events()
              << "  ms_since_last=" << stale_detector.ms_since_last_token() << "\n";
#endif
#ifdef LLMQUANT_REGIME_DETECTOR_ENABLED
    std::cout << "  Market regime    : " << regime_detector.current_regime_name()
              << "  transitions=" << regime_detector.total_transitions()
              << "  momentum=" << std::fixed << std::setprecision(3)
              << regime_detector.get_momentum() << "\n";
#endif
#ifdef LLMQUANT_TRADING_HOURS_ENABLED
    std::cout << "  Market hrs guard : "
              << (trading_hours_guard.is_market_open() ? "OPEN" : "CLOSED")
              << "  blocked=" << trading_hours_guard.signals_blocked()
              << "  transitions=" << trading_hours_guard.session_transitions()
              << "  et=" << trading_hours_guard.current_et_time_str() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CORRELATION_ENABLED
    std::cout << "  Signal corr      : "
              << "sources=" << signal_corr.source_names().size()
              << "  diverge_evts=" << signal_corr.divergence_events()
              << "  converge_evts=" << signal_corr.convergence_events() << "\n";
#endif
#ifdef LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED
    std::cout << "  Drawdown protect : "
              << "tier=" << drawdown_protector.current_tier()
              << "  drawdown=" << std::fixed << std::setprecision(1)
              << (drawdown_protector.current_drawdown_pct() * 100.0) << "%"
              << "  hwm=" << std::setprecision(6) << drawdown_protector.high_water_mark()
              << "  transitions=" << drawdown_protector.tier_transitions() << "\n";
#endif
#ifdef LLMQUANT_MULTI_TIMEFRAME_ENABLED
    std::cout << "  Multi-timeframe  : "
              << "consensus=" << std::fixed << std::setprecision(4) << multi_tf.consensus()
              << "  spread=" << multi_tf.timeframe_spread()
              << "  diverging=" << (multi_tf.is_diverging() ? "Y" : "N")
              << "  records=" << multi_tf.total_records() << "\n";
#endif
#ifdef LLMQUANT_VOLATILITY_FORECASTER_ENABLED
    std::cout << "  Vol forecast     : "
              << "cond_vol=" << std::fixed << std::setprecision(4) << vol_forecaster.conditional_vol()
              << "  high_vol=" << (vol_forecaster.is_high_vol() ? "Y" : "N")
              << "  events=" << vol_forecaster.high_vol_events() << "\n";
#endif
#ifdef LLMQUANT_BAYESIAN_FILTER_ENABLED
    std::cout << "  Bayes filter     : "
              << "bull=" << std::fixed << std::setprecision(3)
              << bayes_filter.posterior_confidence(true)
              << "  bear=" << bayes_filter.posterior_confidence(false)
              << "  signals=" << bayes_filter.total_signals() << "\n";
#endif
#ifdef LLMQUANT_ANOMALY_DETECTOR_ENABLED
    std::cout << "  Anomaly detect   : "
              << "soft=" << anomaly_detector.soft_anomalies()
              << "  hard=" << anomaly_detector.hard_anomalies()
              << "  last_z=" << std::fixed << std::setprecision(2)
              << anomaly_detector.last_z_score() << "\n";
#endif
#ifdef LLMQUANT_BURST_DETECTOR_ENABLED
    std::cout << "  Burst detector   : "
              << "rate=" << std::fixed << std::setprecision(1) << burst_detector.current_rate()
              << "tok/s  burst=" << (burst_detector.is_burst() ? "Y" : "N")
              << "  events=" << burst_detector.burst_events() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_PERSISTENCE_ENABLED
    std::cout << "  Persistence      : "
              << "streak=" << persistence_tracker.current_streak()
              << "  scale=" << std::fixed << std::setprecision(2)
              << persistence_tracker.conviction_scale()
              << "  reversals=" << persistence_tracker.total_reversals() << "\n";
#endif
#ifdef LLMQUANT_ROLLING_SHARPE_ENABLED
    std::cout << "  Rolling Sharpe   : "
              << "sharpe=" << std::fixed << std::setprecision(3) << rolling_sharpe.last_sharpe()
              << "  poor=" << (rolling_sharpe.is_poor_quality() ? "Y" : "N")
              << "  n=" << rolling_sharpe.sample_count() << "\n";
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
    std::cout << "  Context budget   : "
              << "used=" << context_budget.tokens_used()
              << "  fill=" << std::fixed << std::setprecision(1)
              << (context_budget.fill_fraction() * 100.0) << "%\n";
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
    std::cout << "  Fractal dim      : "
              << "hurst=" << std::fixed << std::setprecision(3) << fractal_dim.hurst()
              << "  "
              << (fractal_dim.is_trending() ? "TRENDING" :
                  fractal_dim.is_mean_reverting() ? "MEAN-REV" : "RANDOM")
              << "  n=" << fractal_dim.total_records() << "\n";
#endif
#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
    std::cout << "  Microstructure   : "
              << "half_spread=" << std::fixed << std::setprecision(5)
              << microstructure_filter.estimated_half_spread()
              << "  blocks=" << microstructure_filter.total_blocked()
              << "  passes=" << microstructure_filter.total_passed() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
    std::cout << "  Signal ensemble  : "
              << "output=" << std::fixed << std::setprecision(4) << signal_ensemble.ensemble_output()
              << "  outcomes=" << signal_ensemble.total_outcomes() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
    std::cout << "  Signal momentum  : "
              << "macd=" << std::showpos << std::fixed << std::setprecision(5)
              << signal_momentum_osc.macd()
              << "  hist=" << signal_momentum_osc.histogram()
              << std::noshowpos
              << "  " << (signal_momentum_osc.is_bullish() ? "BULL" :
                           signal_momentum_osc.is_bearish() ? "BEAR" : "FLAT")
              << "  crosses=" << signal_momentum_osc.total_crosses() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
    std::cout << "  Sentiment cycle  : "
              << "period=" << sentiment_cycle.dominant_period()
              << "  strength=" << std::fixed << std::setprecision(3) << sentiment_cycle.cycle_strength()
              << "  " << (sentiment_cycle.is_cyclic() ? "CYCLIC" : "none")
              << "  changes=" << sentiment_cycle.period_changes() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
    std::cout << "  Adaptive sampler : "
              << "interval=" << adaptive_sampler.recommended_interval_ms() << "ms"
              << "  accel=" << adaptive_sampler.accelerations()
              << "  decel=" << adaptive_sampler.decelerations()
              << "  " << (adaptive_sampler.is_at_min() ? "FAST" : adaptive_sampler.is_at_max() ? "SLOW" : "mid") << "\n";
#endif
#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
    std::cout << "  Mutual info      : "
              << "mi=" << std::fixed << std::setprecision(4) << mi_estimator.mi()
              << "  nmi=" << mi_estimator.normalized_mi()
              << "  n=" << mi_estimator.sample_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
    std::cout << "  Blind spots      : "
              << "flagged_slots=" << blind_spot.blind_spot_count()
              << "  outcomes=" << blind_spot.total_outcomes()
              << "  events=" << blind_spot.detection_events() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
    std::cout << "  Signal surprise  : "
              << "last=" << std::fixed << std::setprecision(3) << signal_surprise.surprise()
              << "  mean=" << signal_surprise.mean_surprise()
              << "  high=" << (signal_surprise.is_high_surprise() ? "YES" : "no")
              << "  events=" << signal_surprise.high_surprise_count() << "\n";
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
    {
        auto st = stream_health.status();
        const char* st_str = (st == llmquant::TokenStreamHealthMonitor::Status::Healthy) ? "HEALTHY"
                           : (st == llmquant::TokenStreamHealthMonitor::Status::Stalled) ? "STALLED"
                           : "FLOODED";
        std::cout << "  Stream health    : "
                  << st_str
                  << "  rate=" << std::fixed << std::setprecision(1) << stream_health.current_rate() << "tok/s"
                  << "  stalls=" << stream_health.stall_count()
                  << "  floods=" << stream_health.flood_count() << "\n";
    }
#endif
#ifdef LLMQUANT_REGIME_SIZER_ENABLED
    std::cout << "  Regime sizer     : "
              << "mult=" << std::fixed << std::setprecision(3) << regime_sizer.size_multiplier()
              << "  H=" << regime_sizer.current_hurst()
              << "  vol=" << regime_sizer.current_vol()
              << "  regime_f=" << regime_sizer.regime_factor()
              << "  vol_f=" << regime_sizer.vol_factor()
              << "  changes=" << regime_sizer.change_events() << "\n";
#endif
#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
    std::cout << "  Conf decay       : "
              << "half_life=" << std::fixed << std::setprecision(0) << conf_decay.half_life_ms() << "ms"
              << "  lambda=" << std::setprecision(5) << conf_decay.lambda()
              << "  fast=" << (conf_decay.is_fast_decay() ? "YES" : "no")
              << "  n=" << conf_decay.total_records() << "\n";
#endif
#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
    std::cout << "  Cross-asset corr : "
              << "assets=" << cross_asset_corr.asset_count()
              << "  bias~vol=" << std::fixed << std::setprecision(3)
              << cross_asset_corr.correlation("bias", "vol")
              << "  bias~conf=" << cross_asset_corr.correlation("bias", "confidence")
              << "  high_events=" << cross_asset_corr.high_corr_events()
              << "  low_events=" << cross_asset_corr.low_corr_events() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
    std::cout << "  Bias velocity    : "
              << "vel=" << std::showpos << std::fixed << std::setprecision(4)
              << velocity_tracker.velocity()
              << "  accel=" << velocity_tracker.acceleration()
              << "  fast_move=" << (velocity_tracker.is_fast_move() ? "YES" : "no")
              << "  n=" << velocity_tracker.total_records() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
    {
        using Q = llmquant::NarrativeMomentumClock::Quadrant;
        static const char* qnames[] = {"Rising(Q1)", "Fading(Q2)", "Falling(Q3)", "Recovering(Q4)"};
        std::cout << "  Narrative clock  : "
                  << "quadrant=" << qnames[static_cast<int>(narrative_clock.quadrant())]
                  << "  transitions=" << narrative_clock.quadrant_transitions()
                  << "  bias_ema=" << std::noshowpos << std::fixed << std::setprecision(5)
                  << narrative_clock.bias_ema()
                  << "  vel_ema=" << narrative_clock.velocity_ema()
                  << "  n=" << narrative_clock.total_records() << "\n";
    }
#endif
#ifdef LLMQUANT_ORDER_BOOK_SIM_ENABLED
    std::cout << "  Order book sim   : "
              << "mid=" << std::fixed << std::setprecision(4) << order_book_sim.mid_price()
              << "  bias=" << order_book_sim.cumulative_bias()
              << "  updates=" << order_book_sim.total_updates() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_HEATMAP_ENABLED
    std::cout << "  Sentiment heatmap: "
              << "tokens=" << sentiment_heatmap.token_count()
              << "  records=" << sentiment_heatmap.total_records() << "\n";
#endif
#ifdef LLMQUANT_CVAR_ENABLED
    std::cout << "  CVaR (ES α=0.95) : "
              << "cvar=" << std::showpos << std::fixed << std::setprecision(5) << cvar_calc.cvar()
              << "  var=" << cvar_calc.var()
              << std::noshowpos
              << "  breach=" << (cvar_calc.is_in_breach() ? "YES" : "no")
              << "  events=" << cvar_calc.breach_events() << "\n";
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
    std::cout << "  Phrase patterns  : "
              << "patterns=" << tpl.pattern_count()
              << "  tokens=" << tpl.total_tokens()
              << "  matches=" << tpl.total_matches() << "\n";
#endif
#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
    std::cout << "  Feedback loop    : "
              << "score=" << std::fixed << std::setprecision(3) << feedback_detector.feedback_score()
              << "  peak_lag=" << feedback_detector.peak_lag()
              << "  detected=" << (feedback_detector.feedback_detected() ? "YES" : "no")
              << "  events=" << feedback_detector.feedback_events() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
    std::cout << "  Vel breaker      : "
              << "open=" << (velocity_breaker.is_open() ? "YES" : "no")
              << "  trips=" << velocity_breaker.trip_count()
              << "  vel=" << std::fixed << std::setprecision(4) << velocity_breaker.smoothed_velocity() << "\n";
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
    std::cout << "  Order flow imb   : "
              << "imb=" << std::showpos << std::fixed << std::setprecision(3) << order_flow_detector.imbalance()
              << std::noshowpos
              << "  events=" << order_flow_detector.imbalance_events()
              << "  tokens=" << order_flow_detector.total_tokens() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
    {
        auto top = token_bias_heatmap.top_by_abs_contribution(3);
        std::cout << "  Token heatmap    : distinct=" << token_bias_heatmap.distinct_tokens()
                  << "  records=" << token_bias_heatmap.total_records();
        if (!top.empty()) {
            std::cout << "  top=[";
            for (size_t i = 0; i < top.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << top[i].token << ":" << std::showpos << std::fixed
                          << std::setprecision(3) << top[i].total_bias << std::noshowpos;
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
#endif
#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
    std::cout << "  Sig calibration  : "
              << "samples=" << signal_calibration.sample_count()
              << "  ece=" << std::fixed << std::setprecision(4) << signal_calibration.expected_calibration_error()
              << "  A=" << signal_calibration.platt_a()
              << "  B=" << signal_calibration.platt_b() << "\n";
#endif
#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
    std::cout << "  Cross-session    : session=" << cross_session_mem.session_number()
              << "  loaded=" << (cross_session_mem.has_loaded_state() ? "yes" : "no") << "\n";
#endif
#ifdef LLMQUANT_REGIME_PROB_ENABLED
    std::cout << "  Regime HMM       : "
              << "p_risk_on=" << std::fixed << std::setprecision(4) << regime_prob_est.prob_risk_on()
              << "  p_risk_off=" << regime_prob_est.prob_risk_off()
              << "  transitions=" << regime_prob_est.transition_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
    std::cout << "  Signal replay    : "
              << "retained=" << signal_replay.size()
              << "  total=" << signal_replay.total_pushed() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
    {
        auto top = ngram_profiler.top_by_frequency(3);
        std::cout << "  N-gram profiler  : "
                  << "distinct=" << ngram_profiler.distinct_ngrams()
                  << "  hot_events=" << ngram_profiler.hot_events();
        if (!top.empty()) {
            std::cout << "  top=[";
            for (size_t i = 0; i < top.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << "\"" << top[i].ngram << "\":" << top[i].count;
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
#endif
#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
    std::cout << "  Sent dispersion  : "
              << "sdi=" << std::fixed << std::setprecision(4) << sentiment_dispersion.sdi()
              << "  dispersed=" << (sentiment_dispersion.is_dispersed() ? "YES" : "no")
              << "  coherent=" << (sentiment_dispersion.is_coherent() ? "YES" : "no")
              << "  events=" << sentiment_dispersion.high_dispersion_events() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
    std::cout << "  Sent divergence  : "
              << "diverge=" << std::fixed << std::setprecision(4) << sentiment_divergence.divergence()
              << "  active=" << (sentiment_divergence.is_diverged() ? "YES" : "no")
              << "  events=" << sentiment_divergence.divergence_events()
              << "  sources=" << sentiment_divergence.source_count() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
    {
        std::cout << "  Token influence  : "
                  << "n=" << token_influence.total_recorded()
                  << "  window=" << token_influence.window_size();
        auto top = token_influence.attribute();
        if (!top.empty()) {
            std::cout << "  top=" << top[0].token
                      << "(inf=" << std::showpos << std::fixed << std::setprecision(4)
                      << top[0].influence << std::noshowpos << ")";
        }
        std::cout << "\n";
    }
#endif
#ifdef LLMQUANT_WALK_FORWARD_ENABLED
    std::cout << "  Walk-forward     : "
              << "folds=" << walk_forward.num_folds()
              << "  (tokens not loaded in live mode — offline use only)\n";
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
    std::cout << "  Adversarial det  : "
              << "armed=" << (adversarial_detector.is_armed() ? "YES" : "no")
              << "  anomalies=" << adversarial_detector.anomaly_count()
              << "  tokens=" << adversarial_detector.total_tokens() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CI_ENABLED
    std::cout << "  Signal CI (95%)  : "
              << "mean=" << std::showpos << std::fixed << std::setprecision(4) << signal_ci.mean()
              << "  hw=" << std::noshowpos << signal_ci.half_width()
              << "  [" << signal_ci.lower() << ", " << signal_ci.upper() << "]"
              << "  narrow=" << (signal_ci.is_narrow() ? "yes" : "no") << "\n";
#endif
#ifdef LLMQUANT_TOKEN_IB_ENABLED
    std::cout << "  Token IB         : "
              << "distinct=" << token_ib.distinct_tokens()
              << "  flagged=" << token_ib.flagged_count()
              << "  records=" << token_ib.total_records() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
    std::cout << "  Markov chain     : "
              << "state=" << sentiment_persistence.current_state()
              << "  predicted=" << sentiment_persistence.predicted_state()
              << "  stickiness=" << std::fixed << std::setprecision(4) << sentiment_persistence.stickiness()
              << "  transitions=" << sentiment_persistence.state_changes()
              << "  records=" << sentiment_persistence.total_records() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
    std::cout << "  Phase portrait   : "
              << "cell=(" << sentiment_phase_portrait.current_row() << "," << sentiment_phase_portrait.current_col() << ")"
              << "  attractor=(" << sentiment_phase_portrait.attractor_row() << "," << sentiment_phase_portrait.attractor_col() << ")"
              << "  cycle=" << (sentiment_phase_portrait.cycle_detected() ? "YES" : "no")
              << "  divergence=" << std::fixed << std::setprecision(4) << sentiment_phase_portrait.divergence_index()
              << "  transitions=" << sentiment_phase_portrait.cell_transitions()
              << "  records=" << sentiment_phase_portrait.total_records() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
    std::cout << "  Narrative topic  : "
              << "dominant=" << narrative_classifier.dominant_topic()
              << "  mult=" << std::fixed << std::setprecision(2) << narrative_classifier.dominant_multiplier()
              << "  classified=" << narrative_classifier.classified_count() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
    std::cout << "  Token clock      : "
              << "rate=" << std::fixed << std::setprecision(1) << token_clock.rate_hz() << "Hz"
              << "  budget_scale=" << std::setprecision(2) << token_clock.budget_scale() << "x"
              << "  jitter_cv=" << std::setprecision(3) << token_clock.jitter_cv()
              << "  tokens=" << token_clock.token_count()
              << "  rate_changes=" << token_clock.rate_change_count() << "\n";
#endif
#ifdef LLMQUANT_SHADOW_PORTFOLIO_ENABLED
    std::cout << "  Shadow portfolio : "
              << "shadow_pnl=" << std::fixed << std::setprecision(2) << shadow_portfolio.shadow_pnl()
              << "  live_pnl=" << shadow_portfolio.live_pnl()
              << "  drag=" << shadow_portfolio.constraint_drag()
              << "  drag_alerts=" << shadow_portfolio.drag_alert_count() << "\n";
#endif
#ifdef LLMQUANT_CONFIDENCE_BAND_ENABLED
    std::cout << "  Kalman band      : "
              << std::fixed << std::setprecision(4)
              << "[" << confidence_band.lower() << ", "
              << confidence_band.center() << ", "
              << confidence_band.upper() << "]"
              << "  hw=" << confidence_band.half_width()
              << "  obs=" << confidence_band.observation_count()
              << "  band_changes=" << confidence_band.band_change_count() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
    {
        auto tds_now = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        std::cout << "  Decay scheduler  : "
                  << "eff_sent=" << std::fixed << std::setprecision(4)
                  << decay_scheduler.effective_sentiment(tds_now)
                  << "  active=" << decay_scheduler.active_entries()
                  << "  total=" << decay_scheduler.total_recorded()
                  << "  flips=" << decay_scheduler.flip_count() << "\n";
    }
#endif
#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
    std::cout << "  Causal impact    : "
              << "cusum=" << std::fixed << std::setprecision(4) << causal_impact.cusum_stat()
              << "  break=" << (causal_impact.break_detected() ? "YES" : "no")
              << "  breaks=" << causal_impact.break_count()
              << "  obs=" << causal_impact.observation_count() << "\n";
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
    {
        const char* div_label = "none";
        auto dk = options_flow_bridge.last_divergence();
        if (dk == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBear)
            div_label = "SMART_MONEY_BEAR";
        else if (dk == llmquant::OptionsFlowSentimentBridge::DivergenceKind::SmartMoneyBull)
            div_label = "SMART_MONEY_BULL";
        std::cout << "  Options flow     : "
                  << "divergence=" << div_label
                  << "  score=" << std::fixed << std::setprecision(4) << options_flow_bridge.divergence_score()
                  << "  vel_ema=" << options_flow_bridge.sentiment_velocity_ema()
                  << "  skew_ema=" << options_flow_bridge.skew_ema()
                  << "  events=" << options_flow_bridge.divergence_count() << "\n";
    }
#endif
#ifdef LLMQUANT_REGIME_ROUTER_ENABLED
    std::cout << "  Regime router    : "
              << "regime=" << regime_router.regime_name()
              << "  vol=" << std::fixed << std::setprecision(4) << regime_router.smoothed_volatility()
              << "  mom=" << regime_router.smoothed_momentum()
              << "  changes=" << regime_router.regime_change_count() << "\n";
#endif
#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
    std::cout << "  Stream differ.   : "
              << "vel=" << std::fixed << std::setprecision(4) << stream_differencer.velocity_ema()
              << "  acc=" << stream_differencer.acceleration_ema()
              << "  jerk=" << stream_differencer.jerk_ema()
              << "  spikes=" << stream_differencer.jerk_spike_count()
              << "  obs=" << stream_differencer.observation_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_DRIFT_ENABLED
    std::cout << "  Signal drift     : "
              << "W1=" << std::fixed << std::setprecision(4) << signal_drift_monitor.last_w1()
              << "  drifting=" << (signal_drift_monitor.is_drifting() ? "YES" : "no")
              << "  events=" << signal_drift_monitor.drift_events()
              << "  obs=" << signal_drift_monitor.total_records() << "\n";
#endif
#ifdef LLMQUANT_LIFECYCLE_TRACKER_ENABLED
    std::cout << "  Lifecycle tracker: "
              << "active=" << lifecycle_tracker.active_signal_count()
              << "  dead=" << lifecycle_tracker.dead_count()
              << "  zombies=" << lifecycle_tracker.zombie_count()
              << "  mean_hl=" << std::fixed << std::setprecision(1) << lifecycle_tracker.mean_halflife_s() << "s\n";
#endif
#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
    std::cout << "  Token quantiser  : "
              << "error_ema=" << std::fixed << std::setprecision(5) << token_quantiser.quantisation_error()
              << "  clamp_rate=" << std::setprecision(3) << token_quantiser.clamp_rate()
              << "  total=" << token_quantiser.total_count() << "\n";
#endif
#ifdef LLMQUANT_POSITION_CONCENTRATION_ENABLED
    std::cout << "  Conc. guard HHI  : "
              << std::fixed << std::setprecision(4) << concentration_guard.hhi()
              << "  concentrated=" << (concentration_guard.is_concentrated() ? "YES" : "no")
              << "  dominant=" << concentration_guard.dominant_theme()
              << "  share=" << std::setprecision(2) << concentration_guard.dominant_share()
              << "  themes=" << concentration_guard.distinct_themes()
              << "  alerts=" << concentration_guard.concentration_events() << "\n";
#endif
#ifdef LLMQUANT_AUTOCORR_METER_ENABLED
    std::cout << "  Autocorrelation  : "
              << "lag1=" << std::fixed << std::setprecision(3) << autocorr_meter.autocorrelation(1)
              << "  lag2=" << autocorr_meter.autocorrelation(2)
              << "  trending=" << (autocorr_meter.is_trending() ? "YES" : "no")
              << "  obs=" << autocorr_meter.observation_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SSI_ENABLED
    std::cout << "  Signal SSI       : "
              << std::fixed << std::setprecision(1) << signal_ssi.ssi()
              << "  overbought=" << (signal_ssi.is_overbought() ? "YES" : "no")
              << "  oversold=" << (signal_ssi.is_oversold() ? "YES" : "no")
              << "  obs=" << signal_ssi.observation_count() << "\n";
#endif
#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
    std::cout << "  Flow Pressure    : "
              << std::fixed << std::setprecision(3) << flow_pressure.pressure()
              << "  spiking=" << (flow_pressure.is_spiking() ? "YES" : "no")
              << "  tokens=" << flow_pressure.token_count() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_FATIGUE_ENABLED
    std::cout << "  Signal Fatigue   : "
              << "score=" << std::fixed << std::setprecision(3) << signal_fatigue.fatigue_score()
              << "  streak=" << signal_fatigue.streak()
              << "  dir=" << signal_fatigue.direction()
              << "  fatigued=" << (signal_fatigue.is_fatigued() ? "YES" : "no")
              << "  obs=" << signal_fatigue.observation_count() << "\n";
#endif
#ifdef LLMQUANT_POLARIZATION_MONITOR_ENABLED
    std::cout << "  Polarization BC  : "
              << std::fixed << std::setprecision(3) << polarization_monitor.bimodality_coefficient()
              << "  polarized=" << (polarization_monitor.is_polarized() ? "YES" : "no")
              << "  skew=" << std::setprecision(3) << polarization_monitor.skewness()
              << "  kurt=" << polarization_monitor.excess_kurtosis()
              << "  events=" << polarization_monitor.polarization_events() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
    std::cout << "  Narrative Temp   : "
              << std::fixed << std::setprecision(3) << narrative_temperature.temperature()
              << "  hot=" << (narrative_temperature.is_hot() ? "YES" : "no")
              << "  ema=" << std::setprecision(4) << narrative_temperature.bias_ema()
              << "  sigma=" << narrative_temperature.bias_sigma()
              << "  heat_events=" << narrative_temperature.heat_events() << "\n";
#endif
#ifdef LLMQUANT_ECHO_SUPPRESSOR_ENABLED
    std::cout << "  Echo Suppressor  : "
              << std::fixed << std::setprecision(3) << echo_suppressor.echo_rate()
              << "  echoing=" << (echo_suppressor.is_echoing() ? "YES" : "no")
              << "  echo_count=" << echo_suppressor.echo_count()
              << "  events=" << echo_suppressor.echo_events() << "\n";
#endif
#ifdef LLMQUANT_HURST_ESTIMATOR_ENABLED
    std::cout << "  Hurst Exponent   : "
              << std::fixed << std::setprecision(3) << hurst_estimator.hurst()
              << "  trending=" << (hurst_estimator.is_trending() ? "YES" : "no")
              << "  reverting=" << (hurst_estimator.is_mean_reverting() ? "YES" : "no")
              << "  t_events=" << hurst_estimator.trend_events()
              << "  r_events=" << hurst_estimator.revert_events() << "\n";
#endif
#ifdef LLMQUANT_CHANGE_POINT_ENABLED
    std::cout << "  Change Point     : "
              << "C+=" << std::fixed << std::setprecision(3) << change_point_detector.cusum_plus()
              << "  C-=" << change_point_detector.cusum_minus()
              << "  upshifts=" << change_point_detector.upshift_count()
              << "  downshifts=" << change_point_detector.downshift_count() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
    std::cout << "  Velocity Breaker : "
              << std::fixed << std::setprecision(4) << bias_vbreaker.velocity()
              << "  tripped=" << (bias_vbreaker.is_tripped() ? "YES" : "no")
              << "  trips=" << bias_vbreaker.trip_count() << "\n";
#endif
#ifdef LLMQUANT_IR_TRACKER_ENABLED
    std::cout << "  Info Ratio       : "
              << std::fixed << std::setprecision(3) << ir_tracker.ir()
              << "  mu=" << ir_tracker.mean()
              << "  sigma=" << ir_tracker.stddev()
              << "  high=" << (ir_tracker.is_high_ir() ? "YES" : "no") << "\n";
#endif
#ifdef LLMQUANT_CONSISTENCY_METER_ENABLED
    std::cout << "  Consistency      : "
              << std::fixed << std::setprecision(3) << consistency_meter.consistency_score()
              << "  aad=" << consistency_meter.aad()
              << "  incon=" << (consistency_meter.is_inconsistent() ? "YES" : "no")
              << "  events=" << consistency_meter.inconsistency_events() << "\n";
#endif
#ifdef LLMQUANT_OSCILLATION_DETECTOR_ENABLED
    std::cout << "  Oscillation      : "
              << "zcr=" << std::fixed << std::setprecision(3) << oscillation_detector.zcr()
              << "  osc=" << (oscillation_detector.is_oscillating() ? "YES" : "no")
              << "  events=" << oscillation_detector.oscillation_events() << "\n";
#endif
#ifdef LLMQUANT_MOMENTUM_INDEX_ENABLED
    std::cout << "  Momentum (MACD)  : "
              << "hist=" << std::fixed << std::setprecision(4) << momentum_index.histogram()
              << "  macd=" << momentum_index.macd_line()
              << "  bull=" << (momentum_index.is_bullish() ? "YES" : "no")
              << "  cross_b=" << momentum_index.bullish_crossovers()
              << "  cross_e=" << momentum_index.bearish_crossovers() << "\n";
#endif
#ifdef LLMQUANT_GAIN_LOSS_RATIO_ENABLED
    std::cout << "  Gain/Loss        : "
              << "ratio=" << std::fixed << std::setprecision(3) << gain_loss_ratio.ratio()
              << "  gain=" << gain_loss_ratio.gain_sum()
              << "  loss=" << gain_loss_ratio.loss_sum()
              << "  rec=" << gain_loss_ratio.total_records() << "\n";
#endif
#ifdef LLMQUANT_REGIME_TRANSITION_MATRIX_ENABLED
    std::cout << "  Regime Matrix    : "
              << "cur=" << regime_transition.current_regime()
              << "  likely_next=" << regime_transition.most_likely_next()
              << "  prob=" << std::fixed << std::setprecision(3) << regime_transition.most_likely_prob()
              << "  rec=" << regime_transition.total_records() << "\n";
#endif
#ifdef LLMQUANT_REVERSAL_DETECTOR_ENABLED
    std::cout << "  Reversal         : "
              << "last=" << reversal_detector.last_reversal_direction()
              << "  bull=" << reversal_detector.bullish_reversals()
              << "  bear=" << reversal_detector.bearish_reversals()
              << "  rec=" << reversal_detector.total_records() << "\n";
#endif
#ifdef LLMQUANT_TSMI_ENABLED
    std::cout << "  TSMI             : "
              << std::fixed << std::setprecision(4) << tsmi.tsmi()
              << "  vel=" << tsmi.velocity()
              << "  acc=" << tsmi.acceleration()
              << "  ssi=" << std::setprecision(1) << tsmi.signal_strength()
              << "  rec=" << tsmi.total_records() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_THRESHOLD_ENABLED
    std::cout << "  Adaptive Thresh  : "
              << "OB=" << std::fixed << std::setprecision(1) << adaptive_threshold.overbought_threshold()
              << "  OS=" << adaptive_threshold.oversold_threshold()
              << "  sigma=" << std::setprecision(4) << adaptive_threshold.rolling_sigma()
              << "  changes=" << adaptive_threshold.change_events()
              << "  obs=" << adaptive_threshold.total_records() << "\n";
#endif
#ifdef LLMQUANT_CONDITIONAL_DIST_ENABLED
    std::cout << "  Cond Dist        : "
              << "tv=" << std::fixed << std::setprecision(3) << conditional_dist.tv_distance()
              << "  asym=" << (conditional_dist.is_asymmetric() ? "YES" : "no")
              << "  obs=" << conditional_dist.total_records() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_COMPRESSOR_ENABLED
    std::cout << "  LZ Compressor    : "
              << "lzc=" << signal_compressor.lz_complexity()
              << "  norm=" << std::fixed << std::setprecision(3) << signal_compressor.normalised_complexity()
              << "  obs=" << signal_compressor.total_records() << "\n";
#endif
#ifdef LLMQUANT_ROLLING_QUANTILE_ENABLED
    std::cout << "  Rolling Quantile : "
              << "P10=" << std::fixed << std::setprecision(3) << rolling_quantile.p10()
              << "  P50=" << rolling_quantile.p50()
              << "  P90=" << rolling_quantile.p90()
              << "  IQR=" << rolling_quantile.iqr()
              << "  skew=" << std::setprecision(3) << rolling_quantile.skew_ratio()
              << "  obs=" << rolling_quantile.total_records() << "\n";
#endif
#ifdef LLMQUANT_AUTOREGRESSOR_ENABLED
    std::cout << "  AR Model         : "
              << "pred=" << std::fixed << std::setprecision(4) << signal_ar.last_prediction()
              << "  err=" << std::setprecision(4) << signal_ar.last_prediction_error()
              << "  spikes=" << signal_ar.spike_events()
              << "  obs=" << signal_ar.total_records() << "\n";
#endif
#ifdef LLMQUANT_PHASE_SPACE_ENABLED
    {
        auto dom = phase_space.dominant_cell();
        std::cout << "  Phase Space      : "
                  << "dom=(" << dom.row << "," << dom.col << ")"
                  << "  entropy=" << std::fixed << std::setprecision(3) << phase_space.occupancy_entropy()
                  << "  shifts=" << phase_space.shift_events()
                  << "  obs=" << phase_space.total_records() << "\n";
    }
#endif
#ifdef LLMQUANT_TOPOLOGY_MAPPER_ENABLED
    std::cout << "  Topology         : "
              << "components=" << topology_mapper.component_count()
              << "  total_pers=" << std::fixed << std::setprecision(3) << topology_mapper.total_persistence()
              << "  changes=" << topology_mapper.topology_events()
              << "  obs=" << topology_mapper.total_records() << "\n";
#endif
#ifdef LLMQUANT_INFORMATION_GAIN_ENABLED
    std::cout << "  Info Gain        : "
              << "MI=" << std::fixed << std::setprecision(4) << info_gain.mutual_information()
              << "  NMI=" << std::setprecision(3) << info_gain.normalised_mi()
              << "  changes=" << info_gain.mi_change_events()
              << "  obs=" << info_gain.total_records() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_DRIFT_ENABLED
    std::cout << "  Narrative Drift  : "
              << "U=" << std::fixed << std::setprecision(4) << narrative_drift.upward_stat()
              << "  D=" << std::setprecision(4) << narrative_drift.downward_stat()
              << "  mean=" << narrative_drift.running_mean()
              << "  up_alarms=" << narrative_drift.upward_alarms()
              << "  dn_alarms=" << narrative_drift.downward_alarms()
              << "  obs=" << narrative_drift.total_records() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_GRAPH_ENABLED
    {
        auto hubs = sentiment_graph.top_hubs();
        std::cout << "  Sent Graph       : "
                  << "hubs=[" << hubs[0] << "," << hubs[1] << "," << hubs[2] << "]"
                  << "  obs=" << sentiment_graph.total_records() << "\n";
    }
#endif
#ifdef LLMQUANT_KALMAN_FILTER_ENABLED
    std::cout << "  Kalman Filter    : "
              << "x_hat=" << std::fixed << std::setprecision(4) << kalman_filter.filtered_bias()
              << "  innov=" << kalman_filter.innovation()
              << "  NIS=" << std::setprecision(3) << kalman_filter.nis()
              << "  K=" << kalman_filter.kalman_gain()
              << "  mismatches=" << kalman_filter.mismatch_events()
              << "  obs=" << kalman_filter.total_records() << "\n";
#endif
#ifdef LLMQUANT_SPECTRAL_ENTROPY_ENABLED
    std::cout << "  Spectral Entropy : "
              << "H_s=" << std::fixed << std::setprecision(4) << spectral_entropy.spectral_entropy()
              << "  norm=" << std::setprecision(3) << spectral_entropy.normalised_entropy()
              << "  changes=" << spectral_entropy.change_events()
              << "  obs=" << spectral_entropy.total_records() << "\n";
#endif
#ifdef LLMQUANT_BOOTSTRAP_CI_ENABLED
    std::cout << "  Bootstrap CI     : "
              << "[" << std::fixed << std::setprecision(3) << bootstrap_ci.ci_lo()
              << ", " << bootstrap_ci.ci_hi() << "]"
              << "  width=" << bootstrap_ci.ci_width()
              << "  mean=" << bootstrap_ci.rolling_mean()
              << "  wide_events=" << bootstrap_ci.wide_events()
              << "  obs=" << bootstrap_ci.total_records() << "\n";
#endif
#ifdef LLMQUANT_GARCH_ESTIMATOR_ENABLED
    std::cout << "  GARCH(1,1)       : "
              << "σ=" << std::fixed << std::setprecision(4) << garch_est.conditional_vol()
              << "  σ²=" << garch_est.conditional_var()
              << "  persist=" << std::setprecision(3) << garch_est.persistence()
              << "  spikes=" << garch_est.spike_events()
              << "  obs=" << garch_est.total_records() << "\n";
#endif
#ifdef LLMQUANT_REGIME_HMM_ENABLED
    std::cout << "  HMM Regime       : "
              << "P(bull)=" << std::fixed << std::setprecision(3) << regime_hmm.p_bullish()
              << "  state=" << regime_hmm.current_state()
              << "  changes=" << regime_hmm.regime_changes()
              << "  obs=" << regime_hmm.total_records() << "\n";
#endif
#ifdef LLMQUANT_POLARITY_INDEX_ENABLED
    std::cout << "  Polarity Index   : "
              << "NPI=" << std::fixed << std::setprecision(4) << polarity_idx.npi()
              << "  bull=" << polarity_idx.bull_ema()
              << "  bear=" << polarity_idx.bear_ema()
              << "  flips=" << polarity_idx.flip_events()
              << "  extremes=" << polarity_idx.extreme_events()
              << "  obs=" << polarity_idx.total_records() << "\n";
#endif
#ifdef LLMQUANT_RESIDUAL_ANALYSER_ENABLED
    std::cout << "  Residual (DW)    : "
              << "DW=" << std::fixed << std::setprecision(3) << residual_analyser.durbin_watson()
              << "  AR1=" << residual_analyser.ar1_coeff()
              << "  last_ε=" << std::setprecision(4) << residual_analyser.last_residual()
              << "  alarms=" << residual_analyser.alarm_events()
              << "  obs=" << residual_analyser.total_records() << "\n";
#endif
#ifdef LLMQUANT_SALIENCY_RANKER_ENABLED
    {
        auto top_b = saliency_ranker.top_bullish();
        auto top_r = saliency_ranker.top_bearish();
        std::cout << "  Saliency Ranker  : "
                  << "top_bull=[" << top_b[0] << "," << top_b[1] << "," << top_b[2] << "]"
                  << "  top_bear=[" << top_r[0] << "," << top_r[1] << "," << top_r[2] << "]"
                  << "  obs=" << saliency_ranker.total_records() << "\n";
    }
#endif
#ifdef LLMQUANT_TAIL_RISK_METER_ENABLED
    std::cout << "  Tail Risk ES95   : "
              << "VaR=" << std::fixed << std::setprecision(4) << tail_risk.var_alpha()
              << "  ES=" << tail_risk.expected_shortfall()
              << "  tail_events=" << tail_risk.tail_events()
              << "  obs=" << tail_risk.total_records() << "\n";
#endif
#ifdef LLMQUANT_LEVEL_CROSSING_ENABLED
    std::cout << "  Level Crossing   : "
              << "ZCR=" << std::fixed << std::setprecision(4) << level_crossing.zero_crossing_rate()
              << "  MCR=" << level_crossing.mean_crossing_rate()
              << "  changes=" << level_crossing.zcr_change_events()
              << "  obs=" << level_crossing.total_records() << "\n";
#endif
#ifdef LLMQUANT_CROSS_CORRELATOR_ENABLED
    std::cout << "  Cross Corr Lag   : "
              << "R(0)=" << std::fixed << std::setprecision(4) << cross_corr_lag.r_lag0()
              << "  R(1)=" << cross_corr_lag.r_lag1()
              << "  dom_lag=" << cross_corr_lag.dominant_lag()
              << "  changes=" << cross_corr_lag.lag_change_events()
              << "  obs=" << cross_corr_lag.total_records() << "\n";
#endif
#ifdef LLMQUANT_VOL_RATIO_ENABLED
    std::cout << "  Vol Ratio        : "
              << "σs=" << std::fixed << std::setprecision(4) << vol_ratio.short_vol()
              << "  σl=" << vol_ratio.long_vol()
              << "  ratio=" << std::setprecision(3) << vol_ratio.vol_ratio()
              << "  changes=" << vol_ratio.regime_change_events()
              << "  obs=" << vol_ratio.total_records() << "\n";
#endif
#ifdef LLMQUANT_PARABOLIC_SAR_ENABLED
    std::cout << "  Parabolic SAR    : "
              << "SAR=" << std::fixed << std::setprecision(4) << parabolic_sar.sar()
              << "  trend=" << parabolic_sar.trend()
              << "  AF=" << std::setprecision(3) << parabolic_sar.af()
              << "  reversals=" << parabolic_sar.reversal_events()
              << "  obs=" << parabolic_sar.total_records() << "\n";
#endif
#ifdef LLMQUANT_BOLLINGER_BANDS_ENABLED
    std::cout << "  Bollinger Bands  : "
              << "mid=" << std::fixed << std::setprecision(4) << bollinger_bands.middle()
              << "  up=" << bollinger_bands.upper()
              << "  lo=" << bollinger_bands.lower()
              << "  pct_b=" << std::setprecision(3) << bollinger_bands.pct_b()
              << "  squeezes=" << bollinger_bands.squeeze_events()
              << "  obs=" << bollinger_bands.total_records() << "\n";
#endif
#ifdef LLMQUANT_IMPULSE_DETECTOR_ENABLED
    std::cout << "  Impulse Detector : "
              << "last_z=" << std::fixed << std::setprecision(3) << impulse_det.last_z()
              << "  max_z=" << impulse_det.max_z()
              << "  impulses=" << impulse_det.impulse_events()
              << "  obs=" << impulse_det.total_records() << "\n";
#endif
#ifdef LLMQUANT_TREND_STRENGTH_INDEX_ENABLED
    std::cout << "  Trend Strength   : "
              << "tsi=" << std::fixed << std::setprecision(2) << trend_strength.tsi()
              << "  zero_crosses=" << trend_strength.zero_cross_events()
              << "  strong=" << trend_strength.strong_events()
              << "  obs=" << trend_strength.total_records() << "\n";
#endif
#ifdef LLMQUANT_MASS_INDEX_ENABLED
    std::cout << "  Mass Index       : "
              << "mi=" << std::fixed << std::setprecision(3) << mass_idx.mass_index()
              << "  in_bulge=" << (mass_idx.in_bulge() ? "Y" : "N")
              << "  bulges=" << mass_idx.bulge_events()
              << "  reversals=" << mass_idx.reversal_events()
              << "  obs=" << mass_idx.total_records() << "\n";
#endif
#ifdef LLMQUANT_CHOPPINESS_INDEX_ENABLED
    std::cout << "  Choppiness Idx   : "
              << "ci=" << std::fixed << std::setprecision(1) << choppiness.choppiness()
              << "  trending=" << (choppiness.is_trending() ? "Y" : "N")
              << "  choppy=" << (choppiness.is_choppy() ? "Y" : "N")
              << "  trend_events=" << choppiness.trending_events()
              << "  choppy_events=" << choppiness.choppy_events()
              << "  obs=" << choppiness.total_records() << "\n";
#endif
#ifdef LLMQUANT_ACCELERATION_METER_ENABLED
    std::cout << "  Accel Meter      : "
              << "vel=" << std::fixed << std::setprecision(5) << accel_meter.velocity()
              << "  acc=" << accel_meter.acceleration()
              << "  max_acc=" << accel_meter.max_accel()
              << "  inflections=" << accel_meter.inflection_events()
              << "  surges=" << accel_meter.surge_events()
              << "  obs=" << accel_meter.total_records() << "\n";
#endif
#ifdef LLMQUANT_FATIGUE_DETECTOR_ENABLED
    std::cout << "  Narrative Fatigue: "
              << "ratio=" << std::fixed << std::setprecision(4) << fatigue_det.ratio_delta()
              << "  fatigued=" << (fatigue_det.is_fatigued() ? "Y" : "N")
              << "  fatigue_ev=" << fatigue_det.fatigue_events()
              << "  recovery_ev=" << fatigue_det.recovery_events()
              << "  obs=" << fatigue_det.total_records() << "\n";
#endif
#ifdef LLMQUANT_SKEWNESS_TRACKER_ENABLED
    std::cout << "  Bias Skewness    : "
              << "skew=" << std::fixed << std::setprecision(4) << skewness_tracker.skewness()
              << "  pos_ev=" << skewness_tracker.positive_skew_events()
              << "  neg_ev=" << skewness_tracker.negative_skew_events()
              << "  obs=" << skewness_tracker.total_records() << "\n";
#endif
#ifdef LLMQUANT_ZERO_CROSS_RATE_ENABLED
    std::cout << "  Zero Cross Rate  : "
              << "zcr=" << std::fixed << std::setprecision(3) << zcr_meter.zcr()
              << "  high_ev=" << zcr_meter.high_zcr_events()
              << "  low_ev=" << zcr_meter.low_zcr_events()
              << "  obs=" << zcr_meter.total_records() << "\n";
#endif
#ifdef LLMQUANT_BIAS_CORRELOGRAM_ENABLED
    std::cout << "  Bias Correlogram : "
              << "dom_lag=" << bias_correlogram.dominant_lag()
              << "  peak_acf=" << std::fixed << std::setprecision(3) << bias_correlogram.peak_acf()
              << "  cycles=" << bias_correlogram.cycle_events()
              << "  obs=" << bias_correlogram.total_records() << "\n";
#endif
#ifdef LLMQUANT_KURTOSIS_TRACKER_ENABLED
    std::cout << "  Kurtosis         : "
              << "kurt=" << std::fixed << std::setprecision(3) << kurtosis_tracker.kurtosis()
              << "  fat_ev=" << kurtosis_tracker.fat_tail_events()
              << "  normal_ev=" << kurtosis_tracker.normal_tail_events()
              << "  obs=" << kurtosis_tracker.total_records() << "\n";
#endif
#ifdef LLMQUANT_PERSISTENCE_INDEX_ENABLED
    std::cout << "  Persistence Idx  : "
              << "pi=" << std::fixed << std::setprecision(3) << persistence_idx.persistence_index()
              << "  persistent=" << (persistence_idx.is_persistent() ? "Y" : "N")
              << "  reverting=" << (persistence_idx.is_reverting() ? "Y" : "N")
              << "  persist_ev=" << persistence_idx.persistent_events()
              << "  obs=" << persistence_idx.total_records() << "\n";
#endif
#ifdef LLMQUANT_BIAS_ENTROPY_RATE_ENABLED
    std::cout << "  Bias Entropy     : "
              << "H=" << std::fixed << std::setprecision(3) << bias_entropy.entropy()
              << "  high_ev=" << bias_entropy.high_entropy_events()
              << "  low_ev=" << bias_entropy.low_entropy_events()
              << "  obs=" << bias_entropy.total_records() << "\n";
#endif
#ifdef LLMQUANT_DRAWDOWN_METER_ENABLED
    std::cout << "  Drawdown Meter   : "
              << "cum=" << std::fixed << std::setprecision(4) << drawdown_meter.cum_bias()
              << "  peak=" << drawdown_meter.peak()
              << "  dd=" << drawdown_meter.drawdown()
              << "  max_dd=" << drawdown_meter.max_drawdown()
              << "  dd_ev=" << drawdown_meter.drawdown_events()
              << "  obs=" << drawdown_meter.total_records() << "\n";
#endif
#ifdef LLMQUANT_CADENCE_ANALYSER_ENABLED
    std::cout << "  Token Cadence    : "
              << "mean_iai=" << std::fixed << std::setprecision(2) << cadence_analyser.mean_iai_ms() << "ms"
              << "  cv=" << std::setprecision(3) << cadence_analyser.cv_iai()
              << "  bursts=" << cadence_analyser.burst_events()
              << "  gaps=" << cadence_analyser.gap_events()
              << "  obs=" << cadence_analyser.total_records() << "\n";
#endif
#ifdef LLMQUANT_MEAN_REVERSION_SPEED_ENABLED
    std::cout << "  Mean Rev Speed   : "
              << "theta=" << std::fixed << std::setprecision(4) << mean_rev_speed.theta()
              << "  kappa=" << mean_rev_speed.kappa()
              << "  half_life=" << mean_rev_speed.half_life()
              << "  fast_rev=" << mean_rev_speed.fast_reversion_events()
              << "  obs=" << mean_rev_speed.total_records() << "\n";
#endif
#ifdef LLMQUANT_CLUSTER_DETECTOR_ENABLED
    std::cout << "  Cluster Detector : "
              << "cluster=" << cluster_detector.active_cluster()
              << "  changes=" << cluster_detector.cluster_change_events()
              << "  obs=" << cluster_detector.total_records() << "\n";
#endif
#ifdef LLMQUANT_VOL_BREAKOUT_ENABLED
    std::cout << "  Vol Breakout     : "
              << "sv=" << std::fixed << std::setprecision(4) << vol_breakout.short_vol()
              << "  lv=" << vol_breakout.long_vol()
              << "  ratio=" << std::setprecision(3) << vol_breakout.vol_ratio()
              << "  exp=" << vol_breakout.expansion_events()
              << "  con=" << vol_breakout.contraction_events()
              << "  obs=" << vol_breakout.total_records() << "\n";
#endif
#ifdef LLMQUANT_STOCHASTIC_OSC_ENABLED
    std::cout << "  Stochastic Osc   : "
              << "%K=" << std::fixed << std::setprecision(1) << stochastic_osc.k_pct()
              << "  %D=" << stochastic_osc.d_pct()
              << "  ob=" << (stochastic_osc.is_overbought() ? "Y" : "N")
              << "  os=" << (stochastic_osc.is_oversold() ? "Y" : "N")
              << "  ob_ev=" << stochastic_osc.overbought_events()
              << "  os_ev=" << stochastic_osc.oversold_events()
              << "  obs=" << stochastic_osc.total_records() << "\n";
#endif
#ifdef LLMQUANT_BIAS_ACF_ENABLED
    std::cout << "  Bias ACF         : "
              << "r1=" << std::fixed << std::setprecision(3) << bias_acf.r1()
              << "  lag=" << bias_acf.dominant_lag()
              << "  acf=" << std::setprecision(3) << bias_acf.dominant_acf()
              << "  periodic=" << (bias_acf.is_periodic() ? "Y" : "N")
              << "  events=" << bias_acf.periodic_events()
              << "  rec=" << bias_acf.total_records() << "\n";
#endif
#ifdef LLMQUANT_ONLINE_GRANGER_ENABLED
    std::cout << "  Granger          : "
              << "F_xy=" << std::fixed << std::setprecision(2) << granger.f_xy()
              << "  F_yx=" << granger.f_yx()
              << "  x→y=" << (granger.x_causes_y() ? "Y" : "N")
              << "  y→x=" << (granger.y_causes_x() ? "Y" : "N")
              << "  xy_ev=" << granger.xy_causality_events()
              << "  rec=" << granger.total_records() << "\n";
#endif
#ifdef LLMQUANT_MACD_HISTOGRAM_ENABLED
    std::cout << "  MACD Histogram   : "
              << "macd=" << std::fixed << std::setprecision(6) << macd_hist.macd()
              << "  sig=" << macd_hist.signal_line()
              << "  hist=" << macd_hist.histogram()
              << "  x=" << macd_hist.zero_cross_events()
              << "  div=" << macd_hist.divergence_events()
              << "  rec=" << macd_hist.total_records() << "\n";
#endif
#ifdef LLMQUANT_REGIME_MARKOV_ENABLED
    std::cout << "  Regime Markov    : "
              << "state=" << regime_markov.current_state()
              << "  changes=" << regime_markov.regime_change_events()
              << "  rec=" << regime_markov.total_records() << "\n";
#endif
#ifdef LLMQUANT_CONCENTRATION_RISK_ENABLED
    std::cout << "  Concentration HHI: "
              << "hhi=" << std::fixed << std::setprecision(4) << conc_risk.hhi()
              << "  conc_ev=" << conc_risk.concentration_events()
              << "  disp_ev=" << conc_risk.dispersed_events()
              << "  rec=" << conc_risk.total_records() << "\n";
#endif
#ifdef LLMQUANT_WILLIAMS_R_ENABLED
    std::cout << "  Williams %%R      : "
              << "wr=" << std::fixed << std::setprecision(1) << williams_r.williams_r()
              << "  ob=" << (williams_r.is_overbought() ? "Y" : "N")
              << "  os=" << (williams_r.is_oversold() ? "Y" : "N")
              << "  ob_ev=" << williams_r.overbought_events()
              << "  os_ev=" << williams_r.oversold_events()
              << "  rec=" << williams_r.total_records() << "\n";
#endif
#ifdef LLMQUANT_INFLUENCE_DECAY_ENABLED
    std::cout << "  Influence Decay  : "
              << "pos=" << std::fixed << std::setprecision(4) << influence_decay.pos_influence()
              << "  neg=" << influence_decay.neg_influence()
              << "  net=" << influence_decay.net_influence()
              << "  +dom=" << (influence_decay.is_positive_dominant() ? "Y" : "N")
              << "  +ev=" << influence_decay.positive_dominant_events()
              << "  -ev=" << influence_decay.negative_dominant_events()
              << "  rec=" << influence_decay.total_records() << "\n";
#endif
#ifdef LLMQUANT_POLARITY_SHIFT_ENABLED
    std::cout << "  Polarity Shift   : "
              << "pos_frac=" << std::fixed << std::setprecision(3) << polarity_shift.positive_fraction()
              << "  bull=" << (polarity_shift.is_bullish() ? "Y" : "N")
              << "  bear=" << (polarity_shift.is_bearish() ? "Y" : "N")
              << "  bull_ev=" << polarity_shift.bullish_events()
              << "  bear_ev=" << polarity_shift.bearish_events()
              << "  rec=" << polarity_shift.total_records() << "\n";
#endif
#ifdef LLMQUANT_CHANDE_OSC_ENABLED
    std::cout << "  Chande Osc       : "
              << "cmo=" << std::fixed << std::setprecision(1) << chande_osc.value()
              << "  ob=" << (chande_osc.is_overbought() ? "Y" : "N")
              << "  os=" << (chande_osc.is_oversold() ? "Y" : "N")
              << "  ob_ev=" << chande_osc.overbought_events()
              << "  os_ev=" << chande_osc.oversold_events()
              << "  rec=" << chande_osc.total_records() << "\n";
#endif
#ifdef LLMQUANT_DONCHIAN_CHANNEL_ENABLED
    std::cout << "  Donchian Channel : "
              << "hi=" << std::fixed << std::setprecision(4) << donchian_ch.upper_band()
              << "  lo=" << donchian_ch.lower_band()
              << "  mid=" << donchian_ch.midline()
              << "  ub_ev=" << donchian_ch.upper_breakouts()
              << "  lb_ev=" << donchian_ch.lower_breakouts()
              << "  rec=" << donchian_ch.total_records() << "\n";
#endif
#ifdef LLMQUANT_BIAS_HISTOGRAM_ENABLED
    std::cout << "  Bias Histogram   : "
              << "peak_bin=" << bias_histogram.peak_bin()
              << "  peak_ctr=" << std::fixed << std::setprecision(4) << bias_histogram.peak_center()
              << "  entropy=" << bias_histogram.entropy()
              << "  mode_chg=" << bias_histogram.mode_changes()
              << "  rec=" << bias_histogram.total_records() << "\n";
#endif
#ifdef LLMQUANT_EXP_SMOOTHING_ENABLED
    std::cout << "  Exp Smoothing    : "
              << "level=" << std::fixed << std::setprecision(4) << exp_smooth.level()
              << "  trend=" << exp_smooth.trend()
              << "  fc=" << exp_smooth.forecast()
              << "  err=" << exp_smooth.last_error()
              << "  err_ev=" << exp_smooth.error_events()
              << "  rec=" << exp_smooth.total_records() << "\n";
#endif
#ifdef LLMQUANT_RELATIVE_VIGOR_ENABLED
    std::cout << "  RVI              : "
              << "rvi=" << std::fixed << std::setprecision(4) << rvi_signal.rvi()
              << "  sig=" << rvi_signal.signal_line()
              << "  bull=" << (rvi_signal.is_bullish() ? "Y" : "N")
              << "  bull_x=" << rvi_signal.bullish_crosses()
              << "  bear_x=" << rvi_signal.bearish_crosses()
              << "  rec=" << rvi_signal.total_records() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_VELOCITY_ENABLED
    std::cout << "  Sent Velocity    : "
              << "vel=" << std::fixed << std::setprecision(6) << sent_velocity.velocity()
              << "  accel=" << sent_velocity.acceleration()
              << "  pos_s=" << sent_velocity.positive_surges()
              << "  neg_s=" << sent_velocity.negative_surges()
              << "  rec=" << sent_velocity.total_records() << "\n";
#endif
#ifdef LLMQUANT_ZSCORE_NORMALISER_ENABLED
    std::cout << "  Z-Score Norm     : "
              << "z=" << std::fixed << std::setprecision(3) << zscore_norm.z_score()
              << "  mean=" << zscore_norm.rolling_mean()
              << "  std=" << zscore_norm.rolling_std()
              << "  pos_ext=" << zscore_norm.positive_extremes()
              << "  neg_ext=" << zscore_norm.negative_extremes()
              << "  rec=" << zscore_norm.total_records() << "\n";
#endif
#ifdef LLMQUANT_KELTNER_CHANNEL_ENABLED
    std::cout << "  Keltner Channel  : "
              << "ema=" << std::fixed << std::setprecision(4) << keltner_ch.ema()
              << "  atr=" << keltner_ch.atr()
              << "  ub_ev=" << keltner_ch.upper_breaks()
              << "  lb_ev=" << keltner_ch.lower_breaks()
              << "  rec=" << keltner_ch.total_records() << "\n";
#endif
#ifdef LLMQUANT_BURST_INTENSITY_ENABLED
    std::cout << "  Burst Intensity  : "
              << "ratio=" << std::fixed << std::setprecision(2) << burst_intensity.burst_ratio()
              << "  bursting=" << (burst_intensity.is_bursting() ? "Y" : "N")
              << "  ev=" << burst_intensity.burst_events()
              << "  rec=" << burst_intensity.total_records() << "\n";
#endif
#ifdef LLMQUANT_TRIPLE_EMA_ENABLED
    std::cout << "  TRIX Oscillator  : "
              << "trix=" << std::fixed << std::setprecision(6) << triple_ema.trix()
              << "  sig=" << triple_ema.signal_line()
              << "  zero_x=" << triple_ema.zero_cross_events()
              << "  sig_x=" << triple_ema.signal_cross_events()
              << "  rec=" << triple_ema.total_records() << "\n";
#endif
#ifdef LLMQUANT_COHERENCE_TRACKER_ENABLED
    std::cout << "  Coherence        : "
              << "coh=" << std::fixed << std::setprecision(3) << coherence_tracker.coherence()
              << "  hi_ev=" << coherence_tracker.high_coherence_events()
              << "  lo_ev=" << coherence_tracker.low_coherence_events()
              << "  rec=" << coherence_tracker.total_records() << "\n";
#endif
#ifdef LLMQUANT_LOCAL_EXTREMA_ENABLED
    std::cout << "  Local Extrema    : "
              << "peak=" << std::fixed << std::setprecision(4) << local_extrema.last_peak()
              << "  trough=" << local_extrema.last_trough()
              << "  seek=" << (local_extrema.seeking_peak() ? "peak" : "trough")
              << "  p_ev=" << local_extrema.peak_events()
              << "  t_ev=" << local_extrema.trough_events()
              << "  rec=" << local_extrema.total_records() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_FILTER_ENABLED
    std::cout << "  Adaptive Filter  : "
              << "atr=" << std::fixed << std::setprecision(4) << adaptive_filter.atr()
              << "  thresh=" << adaptive_filter.threshold()
              << "  ab=" << (adaptive_filter.is_above() ? "Y" : "N")
              << "  bel=" << (adaptive_filter.is_below() ? "Y" : "N")
              << "  ab_ev=" << adaptive_filter.above_events()
              << "  bel_ev=" << adaptive_filter.below_events()
              << "  rec=" << adaptive_filter.total_records() << "\n";
#endif
#ifdef LLMQUANT_PRESSURE_GAUGE_ENABLED
    std::cout << "  Pressure Gauge   : "
              << "press=" << std::fixed << std::setprecision(6) << pressure_gauge.pressure()
              << "  fast=" << pressure_gauge.fast_ema()
              << "  slow=" << pressure_gauge.slow_ema()
              << "  +ev=" << pressure_gauge.positive_pressure_events()
              << "  -ev=" << pressure_gauge.negative_pressure_events()
              << "  rec=" << pressure_gauge.total_records() << "\n";
#endif
#ifdef LLMQUANT_WAVELET_DECOMPOSER_ENABLED
    std::cout << "  Wavelet DWT      : "
              << "approx=" << std::fixed << std::setprecision(4) << wavelet_decomp.approx_mean()
              << "  max_detail=" << wavelet_decomp.max_detail_energy()
              << "  spiking=" << (wavelet_decomp.is_spiking() ? "YES" : "no")
              << "  spikes=" << wavelet_decomp.spike_events()
              << "  obs=" << wavelet_decomp.total_records() << "\n";
#endif
#ifdef LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED
    std::cout << "  RL Weighter      : "
              << "dominant='" << rl_weighter.dominant_arm() << "'"
              << "  changes=" << rl_weighter.dominant_arm_changes()
              << "  updates=" << rl_weighter.total_updates() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CONVEXITY_ENABLED
    std::cout << "  Convexity        : "
              << "d1=" << std::fixed << std::setprecision(4) << convexity_meter.first_derivative()
              << "  d2=" << convexity_meter.second_derivative()
              << "  accel=" << (convexity_meter.is_accelerating() ? "YES" : "no")
              << "  decel=" << (convexity_meter.is_decelerating() ? "YES" : "no")
              << "  changes=" << convexity_meter.regime_changes()
              << "  obs=" << convexity_meter.total_records() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED
    std::cout << "  Entropy Clock    : "
              << "kl=" << std::fixed << std::setprecision(4) << entropy_clock.accumulated_kl()
              << "  delta=" << entropy_clock.last_delta_kl()
              << "  exhaustions=" << entropy_clock.exhaustion_events()
              << "  rec=" << entropy_clock.total_records() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED
    std::cout << "  Decay Half-Life  : "
              << "hl=" << std::fixed << std::setprecision(2) << decay_halflife.half_life()
              << "  rate=" << std::setprecision(4) << decay_halflife.decay_rate()
              << "  r2=" << std::setprecision(3) << decay_halflife.r_squared()
              << "  fast=" << decay_halflife.fast_decay_events()
              << "  slow=" << decay_halflife.slow_decay_events() << "\n";
#endif
#ifdef LLMQUANT_BAYESIAN_SENTIMENT_ENABLED
    std::cout << "  Bayes Prior      : "
              << "mu=" << std::fixed << std::setprecision(4) << bayes_prior.posterior_mean()
              << "  std=" << bayes_prior.posterior_std()
              << "  shifted=" << (bayes_prior.is_shifted() ? "YES" : "no")
              << "  shifts=" << bayes_prior.belief_shifts()
              << "  updates=" << bayes_prior.total_updates() << "\n";
#endif
#ifdef LLMQUANT_WEIGHT_HISTOGRAM_ENABLED
    std::cout << "  Weight Histogram : "
              << "mode=" << weight_histogram.mode_bucket()
              << "  mode_val=" << std::fixed << std::setprecision(3) << weight_histogram.mode_value()
              << "  entropy=" << std::setprecision(3) << weight_histogram.entropy()
              << "  total=" << weight_histogram.total() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SLOPE_ENABLED
    std::cout << "  Signal Slope     : "
              << std::fixed << std::setprecision(5) << signal_slope.slope()
              << "  score=" << std::setprecision(3) << signal_slope.slope_score()
              << "  accel=" << (signal_slope.is_accelerating() ? "YES" : "no")
              << "  obs=" << signal_slope.observation_count() << "\n";
#endif
#ifdef LLMQUANT_RUN_LENGTH_ENABLED
    std::cout << "  Run Length       : "
              << "cur=" << run_length.current_run()
              << "  dir=" << run_length.current_direction()
              << "  max=" << run_length.max_run()
              << "  avg=" << std::fixed << std::setprecision(2) << run_length.avg_run()
              << "  runs=" << run_length.run_count() << "\n";
#endif
#ifdef LLMQUANT_COVERAGE_METER_ENABLED
    std::cout << "  Range Coverage   : "
              << std::fixed << std::setprecision(3) << coverage_meter.coverage()
              << "  min=" << coverage_meter.rolling_min()
              << "  max=" << coverage_meter.rolling_max()
              << "  expanded=" << (coverage_meter.is_expanded() ? "YES" : "no") << "\n";
#endif
#ifdef LLMQUANT_BIAS_HYSTERESIS_ENABLED
    std::cout << "  Hysteresis Gate  : "
              << "open=" << (hysteresis_gate.is_open() ? "YES" : "no")
              << "  ticks=" << hysteresis_gate.consecutive_ticks()
              << "  opens=" << hysteresis_gate.gate_open_events()
              << "  closes=" << hysteresis_gate.gate_close_events()
              << "  evals=" << hysteresis_gate.total_evaluated() << "\n";
#endif
#ifdef LLMQUANT_REALIZED_VOL_ENABLED
    std::cout << "  Realized Vol     : "
              << std::fixed << std::setprecision(4) << realized_vol.realized_vol()
              << "  high=" << (realized_vol.is_high_volatility() ? "YES" : "no")
              << "  alerts=" << realized_vol.alert_events()
              << "  obs=" << realized_vol.total_records() << "\n";
#endif
#ifdef LLMQUANT_CAUSAL_TRACER_ENABLED
    std::cout << "  Causal Tracer    : "
              << "max_contrib=" << std::fixed << std::setprecision(4) << causal_tracer.max_contribution()
              << "  strong=" << causal_tracer.strong_token_events()
              << "  tokens=" << causal_tracer.total_tokens() << "\n";
#endif
#ifdef LLMQUANT_DEPENDENCY_MAPPER_ENABLED
    std::cout << "  Dep Mapper       : "
              << "total_tok=" << dep_mapper.total_tokens()
              << "  clusters=" << dep_mapper.cluster_events() << "\n";
#endif
#ifdef LLMQUANT_FREQ_ANALYSER_ENABLED
    std::cout << "  Freq Analyser    : "
              << "dom_k=" << freq_analyser.dominant_k()
              << "  freq=" << std::fixed << std::setprecision(4) << freq_analyser.dominant_frequency()
              << "  power=" << std::setprecision(4) << freq_analyser.dominant_power()
              << "  obs=" << freq_analyser.total_records() << "\n";
#endif
#ifdef LLMQUANT_ENTROPY_RATCHET_ENABLED
    std::cout << "  Entropy Ratchet  : "
              << "h=" << std::fixed << std::setprecision(4) << entropy_ratchet.entropy()
              << "  floor=" << std::setprecision(4) << entropy_ratchet.entropy_floor()
              << "  spiked=" << (entropy_ratchet.is_spiked() ? "YES" : "no")
              << "  spikes=" << entropy_ratchet.spike_events()
              << "  obs=" << entropy_ratchet.total_records() << "\n";
#endif
#ifdef LLMQUANT_COHERENCE_SCORER_ENABLED
    std::cout << "  Coherence Scorer : "
              << std::fixed << std::setprecision(3) << coherence_scorer.coherence()
              << "  incoherent=" << (coherence_scorer.is_incoherent() ? "YES" : "no")
              << "  mu=" << std::setprecision(4) << coherence_scorer.rolling_mean()
              << "  sigma=" << std::setprecision(4) << coherence_scorer.rolling_stddev()
              << "  obs=" << coherence_scorer.total_records() << "\n";
#endif
#ifdef LLMQUANT_CROSS_TOKEN_CORR_ENABLED
    std::cout << "  Cross Corr       : "
              << "r(0,1)=" << std::fixed << std::setprecision(3) << cross_corr.correlation(0, 1)
              << "  obs=" << cross_corr.total_records() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_SIZER_ENABLED
    std::cout << "  Adaptive Sizer   : "
              << "mult=" << std::fixed << std::setprecision(3) << pos_sizer.multiplier()
              << "  changes=" << pos_sizer.change_events() << "\n";
#endif
#ifdef LLMQUANT_CLIP_MONITOR_ENABLED
    std::cout << "  Clip Monitor     : "
              << "rate=" << std::fixed << std::setprecision(3) << clip_monitor.clip_rate()
              << "  clips=" << clip_monitor.clip_count()
              << "  spiking=" << (clip_monitor.is_spiking() ? "YES" : "no")
              << "  obs=" << clip_monitor.observation_count() << "\n";
#endif
#ifdef LLMQUANT_INTENSITY_RAMP_ENABLED
    std::cout << "  Intensity Ramp   : "
              << "intensity=" << std::fixed << std::setprecision(4) << intensity_ramp.intensity()
              << "  ramp=" << std::setprecision(4) << intensity_ramp.ramp()
              << "  score=" << std::setprecision(3) << intensity_ramp.ramp_score()
              << "  surging=" << (intensity_ramp.is_surging() ? "YES" : "no")
              << "  fading=" << (intensity_ramp.is_fading() ? "YES" : "no") << "\n";
#endif
#ifdef LLMQUANT_ZSCORE_TRACKER_ENABLED
    std::cout << "  Z-Score Tracker  : "
              << "z=" << std::fixed << std::setprecision(3) << zscore_tracker.z_score()
              << "  mu=" << std::setprecision(4) << zscore_tracker.rolling_mean()
              << "  sigma=" << std::setprecision(4) << zscore_tracker.rolling_sigma()
              << "  extreme=" << (zscore_tracker.is_extreme() ? "YES" : "no")
              << "  obs=" << zscore_tracker.observation_count() << "\n";
#endif
#ifdef LLMQUANT_CONFLUENCE_DETECTOR_ENABLED
    std::cout << "  Confluence       : "
              << "score=" << std::fixed << std::setprecision(3) << confluence.confluence_score()
              << "  dir=" << confluence.dominant_direction()
              << "  confluent=" << (confluence.is_confluent() ? "YES" : "no")
              << "  obs=" << confluence.observation_count() << "\n";
#endif
    std::cout << "  Latency summary  : " << latency_ctrl.format_stats() << "\n";
    {
        std::cout << "  OMS adapter      : " << oms_adapter->description() << "\n";
        std::cout << "  OMS updates      : " << oms_adapter->update_count()
                  << "  errors=" << oms_adapter->error_count();
        if (oms_adapter->reconnect_count() > 0)
            std::cout << "  reconnects=" << oms_adapter->reconnect_count();
        std::cout << "\n";
    }
    {
        auto top = llm_adapter.top_tokens_by_frequency(5);
        if (!top.empty()) {
            std::cout << "  Top tokens (hits): ";
            for (size_t i = 0; i < top.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << top[i].first << "(" << top[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    {
        auto top_bias = llm_adapter.top_tokens_by_directional_bias(5);
        if (!top_bias.empty()) {
            std::cout << "  Top bias tokens  : ";
            for (size_t i = 0; i < top_bias.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << top_bias[i].first
                          << "(" << std::fixed << std::setprecision(3) << top_bias[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    {
        // Hot tokens: composite score = 0.5*(hit_rate) + 0.5*(|directional_bias|)
        auto hot = llm_adapter.export_hot_tokens(5);
        if (!hot.empty()) {
            std::cout << "  Hot tokens       : ";
            for (size_t i = 0; i < hot.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << hot[i].first
                          << "(" << std::fixed << std::setprecision(3) << hot[i].second << ")";
            }
            std::cout << "\n";
        }
    }
    std::cout << "  ---------------------------------------------------------\n\n";

#ifdef LLMQUANT_JSON_STATS_SUMMARY
    // Emit structured JSON summaries for all subsystems.
    if (!quiet) {
        std::cout << "  [json:risk]    " << risk_mgr.to_stats_json() << "\n";
        std::cout << "  [json:engine]  " << trade_engine.to_stats_json() << "\n";
        std::cout << "  [json:adapter] " << llm_adapter.to_stats_json() << "\n";
        std::cout << "  [json:latency] " << latency_ctrl.to_stats_json() << "\n";
#ifdef LLMQUANT_DEDUP_ENABLED
        std::cout << "  [json:dedup]   " << deduplicator.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_LATENCY_ENFORCER_ENABLED
        std::cout << "  [json:lbe]     " << latency_budget_enforcer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PNL_ATTRIBUTION_ENABLED
        std::cout << "  [json:pnl_attr] " << pnl_attribution.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PORTFOLIO_HEAT_ENABLED
        std::cout << "  [json:pheat]   " << portfolio_heat.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED
        std::cout << "  [json:ctx]     " << context_budget.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FRACTAL_DIMENSION_ENABLED
        std::cout << "  [json:fractal] " << fractal_dim.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED
        std::cout << "  [json:microstr]" << microstructure_filter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_ENSEMBLE_ENABLED
        std::cout << "  [json:ensemble]" << signal_ensemble.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED
        std::cout << "  [json:smo]     " << signal_momentum_osc.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CVAR_ENABLED
        std::cout << "  [json:cvar]    " << cvar_calc.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TEMPORAL_PATTERN_ENABLED
        std::cout << "  [json:tpl]     " << tpl.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FEEDBACK_LOOP_ENABLED
        std::cout << "  [json:fbl]     " << feedback_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_CYCLE_ENABLED
        std::cout << "  [json:cycle]   " << sentiment_cycle.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_SAMPLING_ENABLED
        std::cout << "  [json:sampler] " << adaptive_sampler.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MUTUAL_INFORMATION_ENABLED
        std::cout << "  [json:mi]      " << mi_estimator.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED
        std::cout << "  [json:bspot]   " << blind_spot.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SURPRISE_ENABLED
        std::cout << "  [json:surprise]" << signal_surprise.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_STREAM_HEALTH_ENABLED
        std::cout << "  [json:health]  " << stream_health.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_SIZER_ENABLED
        std::cout << "  [json:rsizer]  " << regime_sizer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONFIDENCE_DECAY_ENABLED
        std::cout << "  [json:cdecay]  " << conf_decay.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CROSS_ASSET_CORR_ENABLED
        std::cout << "  [json:xcorr]   " << cross_asset_corr.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_TRACKER_ENABLED
        std::cout << "  [json:vel]     " << velocity_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_CLOCK_ENABLED
        std::cout << "  [json:clock]   " << narrative_clock.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
        std::cout << "  [json:vbreaker] " << velocity_breaker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CALIBRATION_ENABLED
        std::cout << "  [json:sigcal]   " << signal_calibration.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED
        std::cout << "  [json:heatmap]  " << token_bias_heatmap.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED
        std::cout << "  [json:oflow]    " << order_flow_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CROSS_SESSION_MEMORY_ENABLED
        std::cout << "  [json:session]  " << cross_session_mem.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_PROB_ENABLED
        std::cout << "  [json:regime_hmm] " << regime_prob_est.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED
        std::cout << "  [json:replay]   " << signal_replay.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED
        std::cout << "  [json:ngram]    " << ngram_profiler.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_EXECUTION_QUALITY_ENABLED
        std::cout << "  [json:execqual] " << exec_quality.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_DISPERSION_ENABLED
        std::cout << "  [json:dispersion] " << sentiment_dispersion.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED
        std::cout << "  [json:divergence] " << sentiment_divergence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_INFLUENCE_ENABLED
        std::cout << "  [json:influence]  " << token_influence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADVERSARIAL_DETECT_ENABLED
        std::cout << "  [json:adversarial] " << adversarial_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CI_ENABLED
        std::cout << "  [json:signal_ci]  " << signal_ci.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_IB_ENABLED
        std::cout << "  [json:token_ib]   " << token_ib.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED
        std::cout << "  [json:markov]     " << sentiment_persistence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED
        std::cout << "  [json:phase]      " << sentiment_phase_portrait.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CAUSAL_IMPACT_ENABLED
        std::cout << "  [json:causal]     " << causal_impact.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED
        std::cout << "  [json:optflow]    " << options_flow_bridge.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
        std::cout << "  [json:topic]      " << narrative_classifier.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED
        std::cout << "  [json:tkclock]    " << token_clock.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SHADOW_PORTFOLIO_ENABLED
        std::cout << "  [json:shadow]     " << shadow_portfolio.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONFIDENCE_BAND_ENABLED
        std::cout << "  [json:confband]   " << confidence_band.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED
        {
            auto tds_now2 = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            std::cout << "  [json:decay]      " << decay_scheduler.to_stats_json(tds_now2) << "\n";
        }
#endif
#ifdef LLMQUANT_REGIME_ROUTER_ENABLED
        std::cout << "  [json:regime_rtr] " << regime_router.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_STREAM_DIFFERENCER_ENABLED
        std::cout << "  [json:differencer]" << stream_differencer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_DRIFT_ENABLED
        std::cout << "  [json:drift]      " << signal_drift_monitor.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_LIFECYCLE_TRACKER_ENABLED
        std::cout << "  [json:lifecycle]  " << lifecycle_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOKEN_QUANTISER_ENABLED
        std::cout << "  [json:quantiser]  " << token_quantiser.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_POSITION_CONCENTRATION_ENABLED
        std::cout << "  [json:conc_guard] " << concentration_guard.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_AUTOCORR_METER_ENABLED
        std::cout << "  [json:autocorr]   " << autocorr_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SSI_ENABLED
        std::cout << "  [json:ssi]        " << signal_ssi.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FLOW_PRESSURE_ENABLED
        std::cout << "  [json:flow_pressure] " << flow_pressure.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_FATIGUE_ENABLED
        std::cout << "  [json:fatigue]    " << signal_fatigue.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_POLARIZATION_MONITOR_ENABLED
        std::cout << "  [json:polarization] " << polarization_monitor.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED
        std::cout << "  [json:narrative_temp] " << narrative_temperature.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ECHO_SUPPRESSOR_ENABLED
        std::cout << "  [json:echo_suppressor] " << echo_suppressor.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_HURST_ESTIMATOR_ENABLED
        std::cout << "  [json:hurst] " << hurst_estimator.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CHANGE_POINT_ENABLED
        std::cout << "  [json:change_point] " << change_point_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VELOCITY_BREAKER_ENABLED
        std::cout << "  [json:velocity_breaker] " << bias_vbreaker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_IR_TRACKER_ENABLED
        std::cout << "  [json:ir_tracker] " << ir_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONSISTENCY_METER_ENABLED
        std::cout << "  [json:consistency] " << consistency_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_OSCILLATION_DETECTOR_ENABLED
        std::cout << "  [json:oscillation] " << oscillation_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_LATENCY_JITTER_ENABLED
        std::cout << "  [json:latency_jitter] " << latency_jitter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_WEIGHT_HISTOGRAM_ENABLED
        std::cout << "  [json:histogram]  " << weight_histogram.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_SLOPE_ENABLED
        std::cout << "  [json:slope]      " << signal_slope.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_RUN_LENGTH_ENABLED
        std::cout << "  [json:run_length] " << run_length.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_COVERAGE_METER_ENABLED
        std::cout << "  [json:coverage]   " << coverage_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BIAS_HYSTERESIS_ENABLED
        std::cout << "  [json:hysteresis] " << hysteresis_gate.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REALIZED_VOL_ENABLED
        std::cout << "  [json:realized_vol] " << realized_vol.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CAUSAL_TRACER_ENABLED
        std::cout << "  [json:causal]     " << causal_tracer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_DEPENDENCY_MAPPER_ENABLED
        std::cout << "  [json:dep_mapper] " << dep_mapper.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FREQ_ANALYSER_ENABLED
        std::cout << "  [json:freq]       " << freq_analyser.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ENTROPY_RATCHET_ENABLED
        std::cout << "  [json:entropy]    " << entropy_ratchet.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_COHERENCE_SCORER_ENABLED
        std::cout << "  [json:coherence]  " << coherence_scorer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CROSS_TOKEN_CORR_ENABLED
        std::cout << "  [json:cross_corr] " << cross_corr.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_SIZER_ENABLED
        std::cout << "  [json:pos_sizer]  " << pos_sizer.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CLIP_MONITOR_ENABLED
        std::cout << "  [json:clip]       " << clip_monitor.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_INTENSITY_RAMP_ENABLED
        std::cout << "  [json:intensity]  " << intensity_ramp.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ZSCORE_TRACKER_ENABLED
        std::cout << "  [json:zscore]     " << zscore_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONFLUENCE_DETECTOR_ENABLED
        std::cout << "  [json:confluence] " << confluence.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MULTI_FEED_AGGREGATOR_ENABLED
        std::cout << "  [json:multi_feed] " << multi_feed_agg.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CUSUM_ENABLED
        std::cout << "  [json:cusum]      " << signal_cusum.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MOMENTUM_INDEX_ENABLED
        std::cout << "  [json:momentum]   " << momentum_index.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_GAIN_LOSS_RATIO_ENABLED
        std::cout << "  [json:gain_loss]  " << gain_loss_ratio.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_TRANSITION_MATRIX_ENABLED
        std::cout << "  [json:regime_mat] " << regime_transition.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REVERSAL_DETECTOR_ENABLED
        std::cout << "  [json:reversal]   " << reversal_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TSMI_ENABLED
        std::cout << "  [json:tsmi]       " << tsmi.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_THRESHOLD_ENABLED
        std::cout << "  [json:adapt_thr]  " << adaptive_threshold.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONDITIONAL_DIST_ENABLED
        std::cout << "  [json:cond_dist]  " << conditional_dist.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_COMPRESSOR_ENABLED
        std::cout << "  [json:compressor] " << signal_compressor.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ROLLING_QUANTILE_ENABLED
        std::cout << "  [json:quantile]   " << rolling_quantile.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_AUTOREGRESSOR_ENABLED
        std::cout << "  [json:ar_model]   " << signal_ar.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PHASE_SPACE_ENABLED
        std::cout << "  [json:phase_space]" << phase_space.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TOPOLOGY_MAPPER_ENABLED
        std::cout << "  [json:topology]   " << topology_mapper.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_INFORMATION_GAIN_ENABLED
        std::cout << "  [json:info_gain]  " << info_gain.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_DRIFT_ENABLED
        std::cout << "  [json:narr_drift] " << narrative_drift.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_GRAPH_ENABLED
        std::cout << "  [json:sent_graph] " << sentiment_graph.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_KALMAN_FILTER_ENABLED
        std::cout << "  [json:kalman]     " << kalman_filter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SPECTRAL_ENTROPY_ENABLED
        std::cout << "  [json:spectral]   " << spectral_entropy.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BOOTSTRAP_CI_ENABLED
        std::cout << "  [json:bootstrap]  " << bootstrap_ci.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_GARCH_ESTIMATOR_ENABLED
        std::cout << "  [json:garch]      " << garch_est.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_HMM_ENABLED
        std::cout << "  [json:regime_hmm] " << regime_hmm.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_POLARITY_INDEX_ENABLED
        std::cout << "  [json:polarity]   " << polarity_idx.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_RESIDUAL_ANALYSER_ENABLED
        std::cout << "  [json:residual]   " << residual_analyser.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SALIENCY_RANKER_ENABLED
        std::cout << "  [json:saliency]   " << saliency_ranker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TAIL_RISK_METER_ENABLED
        std::cout << "  [json:tail_risk]  " << tail_risk.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_LEVEL_CROSSING_ENABLED
        std::cout << "  [json:level_xcr]  " << level_crossing.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CROSS_CORRELATOR_ENABLED
        std::cout << "  [json:xcorr]      " << cross_corr_lag.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VOL_RATIO_ENABLED
        std::cout << "  [json:vol_ratio]  " << vol_ratio.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PARABOLIC_SAR_ENABLED
        std::cout << "  [json:psar]       " << parabolic_sar.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BOLLINGER_BANDS_ENABLED
        std::cout << "  [json:bollinger]  " << bollinger_bands.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_IMPULSE_DETECTOR_ENABLED
        std::cout << "  [json:impulse]    " << impulse_det.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TREND_STRENGTH_INDEX_ENABLED
        std::cout << "  [json:tsi]        " << trend_strength.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MASS_INDEX_ENABLED
        std::cout << "  [json:mass_idx]   " << mass_idx.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CHOPPINESS_INDEX_ENABLED
        std::cout << "  [json:choppiness] " << choppiness.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ACCELERATION_METER_ENABLED
        std::cout << "  [json:accel]      " << accel_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_FATIGUE_DETECTOR_ENABLED
        std::cout << "  [json:fatigue]    " << fatigue_det.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SKEWNESS_TRACKER_ENABLED
        std::cout << "  [json:skewness]   " << skewness_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ZERO_CROSS_RATE_ENABLED
        std::cout << "  [json:zcr]        " << zcr_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BIAS_CORRELOGRAM_ENABLED
        std::cout << "  [json:correlogram]" << bias_correlogram.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_KURTOSIS_TRACKER_ENABLED
        std::cout << "  [json:kurtosis]   " << kurtosis_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PERSISTENCE_INDEX_ENABLED
        std::cout << "  [json:persistence]" << persistence_idx.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BIAS_ENTROPY_RATE_ENABLED
        std::cout << "  [json:entropy]    " << bias_entropy.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_DRAWDOWN_METER_ENABLED
        std::cout << "  [json:drawdown]   " << drawdown_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CADENCE_ANALYSER_ENABLED
        std::cout << "  [json:cadence]    " << cadence_analyser.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MEAN_REVERSION_SPEED_ENABLED
        std::cout << "  [json:mean_rev]   " << mean_rev_speed.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CLUSTER_DETECTOR_ENABLED
        std::cout << "  [json:cluster]    " << cluster_detector.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_VOL_BREAKOUT_ENABLED
        std::cout << "  [json:vol_break]  " << vol_breakout.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_STOCHASTIC_OSC_ENABLED
        std::cout << "  [json:stoch]      " << stochastic_osc.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BIAS_ACF_ENABLED
        std::cout << "  [json:bias_acf]   " << bias_acf.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ONLINE_GRANGER_ENABLED
        std::cout << "  [json:granger]    " << granger.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_MACD_HISTOGRAM_ENABLED
        std::cout << "  [json:macd]       " << macd_hist.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_REGIME_MARKOV_ENABLED
        std::cout << "  [json:regime_mkv] " << regime_markov.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CONCENTRATION_RISK_ENABLED
        std::cout << "  [json:conc_risk]  " << conc_risk.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_WILLIAMS_R_ENABLED
        std::cout << "  [json:williams_r] " << williams_r.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_INFLUENCE_DECAY_ENABLED
        std::cout << "  [json:inf_decay]  " << influence_decay.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_POLARITY_SHIFT_ENABLED
        std::cout << "  [json:polarity]   " << polarity_shift.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_CHANDE_OSC_ENABLED
        std::cout << "  [json:chande]     " << chande_osc.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_DONCHIAN_CHANNEL_ENABLED
        std::cout << "  [json:donchian]   " << donchian_ch.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BIAS_HISTOGRAM_ENABLED
        std::cout << "  [json:bias_hist]  " << bias_histogram.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_EXP_SMOOTHING_ENABLED
        std::cout << "  [json:exp_smooth] " << exp_smooth.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_RELATIVE_VIGOR_ENABLED
        std::cout << "  [json:rvi]        " << rvi_signal.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SENTIMENT_VELOCITY_ENABLED
        std::cout << "  [json:sent_vel]   " << sent_velocity.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ZSCORE_NORMALISER_ENABLED
        std::cout << "  [json:zscore]     " << zscore_norm.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_KELTNER_CHANNEL_ENABLED
        std::cout << "  [json:keltner]    " << keltner_ch.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BURST_INTENSITY_ENABLED
        std::cout << "  [json:burst]      " << burst_intensity.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_TRIPLE_EMA_ENABLED
        std::cout << "  [json:trix]       " << triple_ema.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_COHERENCE_TRACKER_ENABLED
        std::cout << "  [json:coherence]  " << coherence_tracker.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_LOCAL_EXTREMA_ENABLED
        std::cout << "  [json:extrema]    " << local_extrema.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_ADAPTIVE_FILTER_ENABLED
        std::cout << "  [json:adapt_filt] " << adaptive_filter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_PRESSURE_GAUGE_ENABLED
        std::cout << "  [json:pressure]   " << pressure_gauge.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_WAVELET_DECOMPOSER_ENABLED
        std::cout << "  [json:wavelet]    " << wavelet_decomp.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_RL_SIGNAL_WEIGHTER_ENABLED
        std::cout << "  [json:rl_weighter] " << rl_weighter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_CONVEXITY_ENABLED
        std::cout << "  [json:convexity]  " << convexity_meter.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_NARRATIVE_ENTROPY_CLOCK_ENABLED
        std::cout << "  [json:entropy_clock] " << entropy_clock.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_SIGNAL_DECAY_HALFLIFE_ENABLED
        std::cout << "  [json:halflife]   " << decay_halflife.to_stats_json() << "\n";
#endif
#ifdef LLMQUANT_BAYESIAN_SENTIMENT_ENABLED
        std::cout << "  [json:bayes]      " << bayes_prior.to_stats_json() << "\n";
#endif
    }
#endif // LLMQUANT_JSON_STATS_SUMMARY

    trade_engine.flush_sinks();
    logger.log_performance_summary();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "\n[FATAL] Unhandled exception: " << ex.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "\n[FATAL] Unknown exception" << std::endl;
    return 1;
  }
}
