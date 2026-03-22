cmake_minimum_required(VERSION 3.20)
project(LLMTokenStreamQuantEngine VERSION 1.2.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LLMQUANT_ENABLE_ASAN     "Build with AddressSanitizer + UBSan (Debug only)"  OFF)
option(LLMQUANT_ENABLE_TSAN     "Build with ThreadSanitizer (Debug only; mutually exclusive with ASAN)" OFF)
option(LLMQUANT_WARNINGS_AS_ERRORS "Treat all compiler warnings as errors"        ON)
option(LLMQUANT_ENABLE_CLANG_TIDY  "Run clang-tidy on library sources"            OFF)
option(LLMQUANT_ENABLE_FUZZING  "Build libFuzzer targets (requires clang + -fsanitize=fuzzer)" OFF)
option(LLMQUANT_ENABLE_COVERAGE "Build with gcov/lcov coverage instrumentation"   OFF)

# Feature flags — enable/disable optional subsystems for minimal or embedded builds.
option(LLMQUANT_ENABLE_PROMETHEUS  "Build the Prometheus /metrics HTTP scrape endpoint" ON)
option(LLMQUANT_ENABLE_FIX_OMS    "Build the FIX 4.2 OMS adapter (FixOmsAdapter)"      ON)
option(LLMQUANT_ENABLE_REST_OMS   "Build the REST OMS polling adapter (RestOmsAdapter)" ON)
option(LLMQUANT_ENABLE_PROFILING  "Compile latency profiling / percentile tracking"     ON)
option(LLMQUANT_ENABLE_DEDUP      "Build the token deduplication subsystem"             ON)
option(LLMQUANT_ENABLE_JSON_STATS_SUMMARY "Emit structured JSON stats lines to stdout on session exit" ON)
option(LLMQUANT_ENABLE_TLS        "Enable TLS support via OpenSSL in LLMStreamClient (auto-disabled if OpenSSL not found)" ON)
option(LLMQUANT_ENABLE_REDIS      "Enable Redis-backed deduplication via hiredis (auto-disabled if hiredis not found)"     ON)
option(LLMQUANT_ENABLE_HOT_RELOAD    "Enable config file hot-reload watcher thread"                                      ON)
option(LLMQUANT_ENABLE_STREAM_CLIENT "Build the LLMStreamClient live-streaming adapter (disabling also removes OpenSSL dep when TLS=OFF)" ON)
option(LLMQUANT_ENABLE_SIGNAL_TRACE  "Emit per-token spdlog::trace lines in the hot path (high verbosity, disable for production)" OFF)
option(LLMQUANT_ENABLE_SIMD          "Use SSE2 intrinsics in LLMAdapter::map_sequence_simd (disable for ARM/non-SSE2 targets or reproducibility testing)" ON)
option(LLMQUANT_ENABLE_MOCK_OMS      "Build MockOmsAdapter (deterministic test double); disable for production/minimal builds" ON)
option(LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY "Build SentimentTrajectoryAnalyzer (online linear-regression trend detector for token sentiment streams)" ON)
option(LLMQUANT_ENABLE_AUDIT_LOG         "Build the async NDJSON signal audit log (SignalAuditLog)" ON)
option(LLMQUANT_ENABLE_CIRCUIT_BREAKER   "Build the pipeline circuit-breaker (auto-pause on sustained high block rate)" ON)
option(LLMQUANT_ENABLE_BACKTEST          "Build BacktestRunner: full-pipeline token-sequence replay with PnL statistics" ON)
option(LLMQUANT_ENABLE_HEALTH_SERVER     "Build the HTTP /health JSON endpoint for Kubernetes liveness/readiness probes" ON)
option(LLMQUANT_ENABLE_KELLY_SIZER       "Build KellyPositionSizer: Kelly Criterion adaptive position sizing" ON)
option(LLMQUANT_ENABLE_ADAPTIVE_COOLDOWN "Build AdaptiveCooldownController: latency-aware dynamic signal cooldown" ON)
option(LLMQUANT_ENABLE_SIGNAL_BLEND      "Build SignalBlendLayer: multi-source confidence-weighted signal blending" ON)
option(LLMQUANT_ENABLE_STALE_DETECTOR    "Build StaleTokenDetector: watchdog that fires alerts when the LLM token stream goes silent" ON)
option(LLMQUANT_ENABLE_POSITION_TRACKER  "Build PositionTracker: real P&L accounting that auto-feeds outcomes into KellyPositionSizer" ON)
option(LLMQUANT_ENABLE_REGIME_DETECTOR   "Build RegimeDetector: EMA-based regime classifier (Bull/Bear/Volatile/RiskOff/Neutral)" ON)
option(LLMQUANT_ENABLE_TOKEN_REPLAY      "Build TokenReplayRecorder: binary token stream recorder and deterministic replay engine" ON)
option(LLMQUANT_ENABLE_PARAMETER_SWEEP   "Build ParameterSweep: grid-search optimiser over BacktestRunner configurations (requires BACKTEST)" ON)
option(LLMQUANT_ENABLE_ALERT_DISPATCHER  "Build AlertDispatcher: async non-blocking alert delivery with dedup window and pluggable sinks" ON)
option(LLMQUANT_ENABLE_ENTROPY_METER     "Build TokenEntropyMeter: Shannon entropy of the token sentiment stream for confidence scaling" ON)
option(LLMQUANT_ENABLE_LATENCY_ENFORCER  "Build LatencyBudgetEnforcer: tiered SLA enforcement (Warn/Throttle/Drop/Breaker) on p99 latency" ON)
option(LLMQUANT_ENABLE_SENTIMENT_MOMENTUM_FILTER "Build SentimentMomentumFilter: cross-validates trade signals against trajectory trend" ON)
option(LLMQUANT_ENABLE_ENTROPY_MONITOR   "Build TokenEntropyMonitor: real-time Shannon entropy of token type diversity in a sliding hash window" ON)
option(LLMQUANT_ENABLE_TRADING_HOURS        "Build TradingHoursGuard: NYSE/NASDAQ session-aware signal gating with DST-aware ET clock" ON)
option(LLMQUANT_ENABLE_SIGNAL_CORRELATION   "Build SignalCorrelationTracker: rolling Pearson correlation across named signal sources" ON)
option(LLMQUANT_ENABLE_LEAD_LAG            "Build LeadLagDetector: cross-correlation at multiple lags to find leading/lagging signal relationships" ON)
option(LLMQUANT_ENABLE_GRADIENT_OPTIMISER  "Build GradientOptimiser: gradient-ascent hyper-parameter search over BacktestRunner objectives" ON)
option(LLMQUANT_ENABLE_WARMUP_SEQUENCER          "Build WarmupSequencer: pre-seed pipeline EMA accumulators with synthetic tokens before live streaming" ON)
option(LLMQUANT_ENABLE_SIGNAL_DECAY              "Build SignalDecayEnvelope: exponential time-decay on accumulated bias so stale conviction fades" ON)
option(LLMQUANT_ENABLE_TOKEN_COST_BUDGET         "Build TokenCostBudget: real-time LLM API cost limiter with soft/hard budget tiers" ON)
option(LLMQUANT_ENABLE_DRAWDOWN_PROTECTOR        "Build DrawdownProtector: tiered risk scaling that tightens limits as cumulative losses grow" ON)
option(LLMQUANT_ENABLE_MULTI_TIMEFRAME              "Build MultiTimeframeAggregator: fuse signals across multiple EMA horizons (1s/5s/30s/5m)" ON)
option(LLMQUANT_ENABLE_REGIME_TRANSITION_MODEL      "Build RegimeTransitionModel: online Markov-chain learner for regime transition probabilities" ON)
option(LLMQUANT_ENABLE_SIGNAL_THROTTLE              "Build SignalThrottle: token-bucket rate limiter for signal emission to prevent OMS flooding" ON)
option(LLMQUANT_ENABLE_CONFIDENCE_CALIBRATOR        "Build ConfidenceCalibrator: Platt-scaling online calibration of confidence from trade outcomes" ON)
option(LLMQUANT_ENABLE_VOLATILITY_FORECASTER        "Build VolatilityForecaster: EWMA-based realized-vol forecast with regime-adaptive decay" ON)
option(LLMQUANT_ENABLE_BAYESIAN_SIGNAL_FILTER       "Build BayesianSignalFilter: Bayesian online update of signal prior from observed trade outcomes" ON)
option(LLMQUANT_ENABLE_ANOMALY_DETECTOR             "Build AnomalyDetector: z-score based spike detector for bias/confidence/volatility streams" ON)
option(LLMQUANT_ENABLE_BURST_DETECTOR               "Build TokenBurstDetector: sliding-window token arrival rate monitor with burst onset callback" ON)
option(LLMQUANT_ENABLE_SIGNAL_PERSISTENCE           "Build SignalPersistenceTracker: consecutive same-direction streak counter with conviction multiplier" ON)
option(LLMQUANT_ENABLE_ROLLING_SHARPE               "Build RollingSharpeBiasTracker: rolling Sharpe ratio of the bias stream for signal quality gating" ON)
option(LLMQUANT_ENABLE_NARRATIVE_CHANGE             "Build NarrativeChangeDetector: cosine-similarity break detector for LLM topic-switch events" ON)
option(LLMQUANT_ENABLE_PNL_ATTRIBUTION              "Build PnLAttributionEngine: retrospective P&L attribution across sentiment driver categories" ON)
option(LLMQUANT_ENABLE_PORTFOLIO_HEAT               "Build PortfolioHeatMonitor: cross-instrument risk heat aggregator with correlation burst detection" ON)
option(LLMQUANT_ENABLE_CONTEXT_WINDOW_BUDGET        "Build ContextWindowBudget: tracks LLM context fill fraction and throttles input when near capacity" ON)
option(LLMQUANT_ENABLE_FRACTAL_DIMENSION            "Build FractalDimensionEstimator: Higuchi fractal dimension of bias series as complexity metric" ON)
option(LLMQUANT_ENABLE_MARKET_MICROSTRUCTURE        "Build MarketMicrostructureFilter: models bid-ask bounce and short-term mean reversion to clean signals" ON)
option(LLMQUANT_ENABLE_SIGNAL_ENSEMBLE              "Build SignalEnsembleLayer: model-stacking layer averaging multiple sub-signal generators" ON)
option(LLMQUANT_ENABLE_SIGNAL_MOMENTUM_OSC          "Build SignalMomentumOscillator: MACD-style fast/slow EMA oscillator with histogram zero-cross signals" ON)
option(LLMQUANT_ENABLE_ORDER_BOOK_SIM               "Build OrderBookSimulator: sentiment-driven synthetic LOB for slippage-aware fill price estimation" ON)
option(LLMQUANT_ENABLE_SENTIMENT_HEATMAP            "Build TokenSentimentHeatmap: 2D rolling token×sentiment-bucket histogram for alpha attribution" ON)
option(LLMQUANT_ENABLE_CVAR                         "Build CVaRCalculator: rolling Conditional Value-at-Risk / Expected Shortfall risk metric" ON)
option(LLMQUANT_ENABLE_TEMPORAL_PATTERN             "Build TemporalPatternLibrary: trie-based multi-token pattern matcher with pre-computed sentiment boosts" ON)
option(LLMQUANT_ENABLE_FEEDBACK_LOOP                "Build FeedbackLoopDetector: cross-correlation detector for market-impact reflexivity in the LLM token stream" ON)
option(LLMQUANT_ENABLE_SENTIMENT_CYCLE              "Build SentimentCycleDetector: rolling ACF periodicity analysis to detect news-cycle patterns in the bias stream" ON)
option(LLMQUANT_ENABLE_ADAPTIVE_SAMPLING            "Build AdaptiveSamplingController: dynamically adjusts token-polling interval based on activity magnitude" ON)
option(LLMQUANT_ENABLE_MUTUAL_INFORMATION           "Build MutualInformationEstimator: discretized MI between LLM sentiment and returns via 2D joint histogram" ON)
option(LLMQUANT_ENABLE_SIGNAL_BLIND_SPOT            "Build SignalBlindSpotDetector: time-of-day win-rate tracker to identify low-predictability trading slots" ON)
option(LLMQUANT_ENABLE_SIGNAL_SURPRISE              "Build SignalSurpriseIndex: Shannon self-information of bias signals relative to their empirical distribution" ON)
option(LLMQUANT_ENABLE_STREAM_HEALTH                "Build TokenStreamHealthMonitor: watchdog detecting feed stalls and token floods via inter-arrival rate tracking" ON)
option(LLMQUANT_ENABLE_REGIME_SIZER                 "Build RegimeAwareSizer: integrates Hurst exponent and volatility to dynamically scale trade notional" ON)
option(LLMQUANT_ENABLE_CONFIDENCE_DECAY             "Build ConfidenceDecayTracker: fits exponential decay model to confidence series to estimate signal half-life" ON)
option(LLMQUANT_ENABLE_CROSS_ASSET_CORR             "Build CrossAssetCorrelationMonitor: rolling Pearson correlation between sentiment signals across asset symbols" ON)
option(LLMQUANT_ENABLE_VELOCITY_TRACKER             "Build TokenVelocityTracker: first and second time-derivatives of bias stream for momentum and regime detection" ON)
option(LLMQUANT_ENABLE_NARRATIVE_CLOCK              "Build NarrativeMomentumClock: four-quadrant investment-clock mapping (bias_ema, velocity_ema) to narrative rotation phases" ON)
option(LLMQUANT_ENABLE_VELOCITY_BREAKER             "Build AdaptiveVelocityBreaker: EMA-smoothed velocity circuit-breaker that trips on runaway bias oscillations" ON)
option(LLMQUANT_ENABLE_ORDER_FLOW_IMBALANCE        "Build OrderFlowImbalanceDetector: EMA buy/sell pressure from LLM token keywords" ON)
option(LLMQUANT_ENABLE_WALK_FORWARD                 "Build WalkForwardValidator: rolling walk-forward out-of-sample validation with optional ParameterSweep per fold" ON)
option(LLMQUANT_ENABLE_SIGNAL_CALIBRATION           "Build SignalCalibrationEngine: online Platt-scaling logistic calibration for raw signal confidence scores" ON)
option(LLMQUANT_ENABLE_TOKEN_BIAS_HEATMAP           "Build TokenBiasHeatmap: per-token cumulative bias contribution tracker with top-N query and capacity cap" ON)
option(LLMQUANT_ENABLE_CROSS_SESSION_MEMORY         "Build CrossSessionMemory: persist Kelly/drawdown state across process restarts for warm-start behaviour" ON)
option(LLMQUANT_ENABLE_REGIME_PROB                  "Build MarketRegimeProbabilityEstimator: online 2-state HMM Bayesian filter for risk-on/risk-off regime probability" ON)
option(LLMQUANT_ENABLE_SIGNAL_REPLAY_BUFFER         "Build SignalReplayBuffer: fixed-capacity ring buffer of recent TradeSignals for replay and post-hoc analysis" ON)
option(LLMQUANT_ENABLE_TOKEN_NGRAM_PROFILER         "Build TokenNgramProfiler: sliding-window n-gram frequency tracker for detecting repeated narrative patterns" ON)
option(LLMQUANT_ENABLE_EXECUTION_QUALITY            "Build ExecutionQualityMonitor: tracks fill latency and slippage distribution between signal emission and OMS ack" ON)
option(LLMQUANT_ENABLE_SENTIMENT_DISPERSION         "Build SentimentDispersionIndex: measures disagreement across sentiment dimensions (bias, vol, confidence)" ON)
option(LLMQUANT_ENABLE_TOKEN_INFLUENCE              "Build TokenInfluenceAttributor: Shapley-inspired per-token marginal contribution attribution at signal generation time" ON)
option(LLMQUANT_ENABLE_SENTIMENT_DIVERGENCE         "Build SentimentDivergenceDetector: multi-stream pairwise divergence detector that fires when sources disagree" ON)
option(LLMQUANT_ENABLE_ADVERSARIAL_DETECT           "Build AdversarialInputDetector: z-score + repetition + vocabulary-inflation adversarial token detector" ON)
option(LLMQUANT_ENABLE_SIGNAL_CI                    "Build SignalConfidenceInterval: rolling jackknife confidence interval estimator for trade signal uncertainty" ON)
option(LLMQUANT_ENABLE_TOKEN_RECORDER               "Build TokenStreamRecorder: append-only binary audit log with CRC-32 integrity and auto-rotation for regulatory compliance" ON)
option(LLMQUANT_ENABLE_ENSEMBLE_VOTER               "Build EnsembleSignalVoter: weighted vote aggregation (majority/confidence/rank) across multiple named signal sub-models" ON)
option(LLMQUANT_ENABLE_SENTIMENT_PERSISTENCE        "Build SentimentPersistenceMatrix: Markov-chain model of discretised bias states with transition-probability matrix and stationary distribution" ON)
option(LLMQUANT_ENABLE_CAUSAL_IMPACT                "Build CausalImpactEstimator: CUSUM structural-break detector that attributes return regime shifts to preceding sentiment events" ON)
option(LLMQUANT_ENABLE_OPTIONS_FLOW_BRIDGE          "Build OptionsFlowSentimentBridge: detects divergence between LLM sentiment velocity and options IV skew for smart-money signals" ON)
option(LLMQUANT_ENABLE_SENTIMENT_PHASE_PORTRAIT     "Build SentimentPhasePortrait: 2-D phase-plane analyser of (bias, velocity) dynamics with attractor and cycle detection" ON)
option(LLMQUANT_ENABLE_NARRATIVE_TOPIC_CLASSIFIER   "Build NarrativeTopicClassifier: lightweight topic-tagging of token streams for multi-theme signal routing" ON)
option(LLMQUANT_ENABLE_TOKEN_CLOCK_RECALIBRATOR     "Build TokenClockRecalibrator: detects drift between token-stream timestamps and wall-clock time, compensates token weights" ON)
option(LLMQUANT_ENABLE_SHADOW_PORTFOLIO             "Build SignalShadowPortfolio: paper-trades every signal in parallel to measure P&L drag from risk constraints" ON)
option(LLMQUANT_ENABLE_TOKEN_IB                     "Build TokenInformationBottleneck: per-token IB relevance/complexity scoring for online vocabulary pruning" ON)
option(LLMQUANT_ENABLE_CONFIDENCE_BAND              "Build LLMConfidenceBandTracker: Kalman-filter confidence bands around sentiment EMA for uncertainty-aware signal gating" ON)
option(LLMQUANT_ENABLE_TOKEN_DECAY_SCHEDULER        "Build TokenImportanceDecayScheduler: exponential token weight decay by token age and recency, for temporal relevance weighting" ON)
option(LLMQUANT_ENABLE_REGIME_ROUTER                "Build RegimeSwitchingSignalRouter: 3-state FSM (trending/ranging/crash) that routes signals to per-regime configurations" ON)
option(LLMQUANT_ENABLE_STREAM_DIFFERENCER           "Build TokenStreamDifferencer: computes velocity, acceleration, and jerk of the token weight series for early reversal detection" ON)
option(LLMQUANT_ENABLE_SIGNAL_DRIFT                 "Build SignalDriftMonitor: Wasserstein-1 drift detector comparing recent vs baseline bias-shift distributions" ON)
option(LLMQUANT_ENABLE_LIFECYCLE_TRACKER            "Build SignalLifecycleTracker: tracks signal birth/peak/decay/death and empirical half-lives, flags zombie signals" ON)
option(LLMQUANT_ENABLE_TOKEN_QUANTISER              "Build TokenWeightQuantiser: stochastic rounding to a fixed grid preserving E[w] exactly for integer-precision downstream" ON)
option(LLMQUANT_ENABLE_POSITION_CONCENTRATION       "Build PositionConcentrationGuard: HHI-based theme concentration monitor, fires when signal pipeline is too concentrated in one theme" ON)
option(LLMQUANT_ENABLE_AUTOCORR_METER               "Build SentimentAutocorrelationMeter: rolling lag-1..N autocorrelation of bias series for trending vs mean-reversion detection" ON)
option(LLMQUANT_ENABLE_SIGNAL_SSI                   "Build SignalStrengthIndexer: RSI applied to sentiment bias (Sentiment Strength Index), fires overbought/oversold callbacks" ON)
option(LLMQUANT_ENABLE_FLOW_PRESSURE                "Build TokenFlowPressureGauge: EMA of inter-token arrival intervals -> pressure score [0,1], fires on_pressure_spike" ON)
option(LLMQUANT_ENABLE_SIGNAL_FATIGUE               "Build SignalFatigueMeter: consecutive same-direction signal streak counter with fatigue score and on_fatigue callback" ON)
option(LLMQUANT_ENABLE_POLARIZATION_MONITOR         "Build SignalPolarizationMonitor: Sarle bimodality coefficient detector — fires when signal distribution splits into two opposing camps" ON)
option(LLMQUANT_ENABLE_NARRATIVE_TEMPERATURE        "Build NarrativeTemperatureGauge: combined |EMA(bias)|×σ(bias) temperature index for hot/volatile narrative detection" ON)
option(LLMQUANT_ENABLE_ECHO_SUPPRESSOR              "Build SignalEchoSuppressor: rolling fraction of near-duplicate consecutive signals; fires on rising/falling echo-rate edge" ON)
option(LLMQUANT_ENABLE_WEIGHT_HISTOGRAM             "Build TokenWeightHistogram: 20-bucket discretized histogram of bias values with entropy and mode tracking" ON)
option(LLMQUANT_ENABLE_SIGNAL_SLOPE                 "Build SignalSlopeMeter: OLS linear regression slope over rolling window of bias values; fires on_trend_acceleration" ON)
option(LLMQUANT_ENABLE_LATENCY_JITTER               "Build LatencyJitterMonitor: MAD-based latency jitter tracker; fires on_jitter_spike when consistency degrades beyond threshold" ON)
option(LLMQUANT_ENABLE_HURST_ESTIMATOR              "Build SignalHurstEstimator: R/S rescaled-range Hurst exponent on rolling bias window; H>0.6=trending, H<0.4=mean-reverting" ON)
option(LLMQUANT_ENABLE_CHANGE_POINT                 "Build BiasChangePointDetector: Page-CUSUM sequential change-point detector for LLM bias mean shifts; fires on_upshift/on_downshift" ON)
option(LLMQUANT_ENABLE_RUN_LENGTH                   "Build BiasRunLengthEncoder: run-length stats on bias sign sequence (current_run, max_run, avg_run); fires on_long_run callback" ON)
option(LLMQUANT_ENABLE_COVERAGE_METER               "Build SignalCoverageMeter: rolling min/max of bias values -> coverage fraction of configured range; fires on_range_expansion" ON)
option(LLMQUANT_ENABLE_VELOCITY_BREAKER             "Build BiasVelocityBreaker: EMA-smoothed rate-of-change circuit breaker; trips when |Δbias| EMA > max_velocity threshold" ON)
option(LLMQUANT_ENABLE_IR_TRACKER                   "Build SignalInformationRatioTracker: rolling IR = μ(bias)/σ(bias) quality index; fires on_high_ir when |IR| > threshold" ON)
option(LLMQUANT_ENABLE_CONSISTENCY_METER            "Build NarrativeConsistencyMeter: AAD-based smoothness score; fires on_inconsistent when erratic bias jumps dominate the window" ON)
option(LLMQUANT_ENABLE_OSCILLATION_DETECTOR         "Build SignalOscillationDetector: zero-crossing rate detector for alternating +/- bias flip-flop; fires on_oscillating callback" ON)
option(LLMQUANT_ENABLE_DEPENDENCY_MAPPER            "Build TokenDependencyMapper: sliding-window token co-occurrence matrix with clique-based cluster detection and on_cluster_detected callback" ON)
option(LLMQUANT_ENABLE_FREQ_ANALYSER                "Build BiasFrequencyAnalyser: DCT-II dominant oscillation frequency extractor; fires on_dominant_frequency_change when peak frequency shifts" ON)
option(LLMQUANT_ENABLE_ENTROPY_RATCHET              "Build SignalEntropyRatchet: one-way ratchet on Shannon entropy of bias distribution; floor drops only after sustained stabilisation" ON)
option(LLMQUANT_ENABLE_COHERENCE_SCORER             "Build NarrativeCoherenceScorer: |μ|/σ signal-to-noise coherence on bias series; fires on_coherence_drop/recover" ON)
option(LLMQUANT_ENABLE_CROSS_TOKEN_CORR             "Build CrossTokenCorrelationMatrix: rolling Pearson correlation between N bias channels; fires on_divergence/on_convergence" ON)
option(LLMQUANT_ENABLE_ADAPTIVE_SIZER               "Build AdaptivePositionSizer: synthesises coherence/confidence/vol/regime into a [0,1] position-size multiplier" ON)
option(LLMQUANT_ENABLE_CLIP_MONITOR                 "Build BiasClipMonitor: tracks fraction of bias values that hit configured clipping bounds; fires on_excessive_clipping" ON)
option(LLMQUANT_ENABLE_INTENSITY_RAMP               "Build NarrativeIntensityRamp: detects sustained upward or downward ramp in bias magnitude; fires on_ramp_start/stop" ON)
option(LLMQUANT_ENABLE_ZSCORE_TRACKER               "Build SentimentZScoreTracker: rolling Z-score of bias relative to expanding or rolling mean/std; fires on_extreme_zscore" ON)
option(LLMQUANT_ENABLE_CONFLUENCE_DETECTOR          "Build SignalConfluenceDetector: detects when multiple independent signal dimensions agree simultaneously; fires on_confluence" ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    set(_warn_flags /W4)
    if(LLMQUANT_WARNINGS_AS_ERRORS)
        list(APPEND _warn_flags /WX)
    endif()
    add_compile_options(${_warn_flags})
    # Suppress MSVC C4996 "deprecated" warnings for standard CRT functions
    # (fopen, strerror, etc.) that are valid C++ standard library calls.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Do NOT set CMAKE_CXX_FLAGS_RELEASE / CMAKE_CXX_FLAGS_DEBUG here — that
    # would overwrite CMake's default /MD CRT linkage flag, causing a CRT
    # mismatch with pre-built spdlog and yaml-cpp packages.  Per-target
    # optimisation flags are added via target_compile_options below instead.
else()
    set(_warn_flags -Wall -Wextra)
    if(LLMQUANT_WARNINGS_AS_ERRORS)
        list(APPEND _warn_flags -Werror)
    endif()
    add_compile_options(${_warn_flags})
    # -ffast-math intentionally removed: it permits FP reordering and
    # NaN/Inf mis-handling that are unsafe in a quant engine where
    # atomic accumulation results must be numerically reproducible.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    if(LLMQUANT_ENABLE_ASAN)
        set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")
    elseif(LLMQUANT_ENABLE_TSAN)
        # TSan uses -O1 so inlined atomics remain visible to the race detector.
        set(CMAKE_CXX_FLAGS_DEBUG   "-g -O1 -fsanitize=thread -fno-omit-frame-pointer")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")
    endif()
endif()

# ---------------------------------------------------------------------------
# clang-tidy integration (optional)
# ---------------------------------------------------------------------------
if(LLMQUANT_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-14 clang-tidy-15 clang-tidy-16)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY
            "${CLANG_TIDY_EXE}"
            "--extra-arg=-std=c++20"
            "--warnings-as-errors="
        )
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    else()
        message(WARNING "LLMQUANT_ENABLE_CLANG_TIDY requested but clang-tidy not found")
    endif()
endif()

# ---------------------------------------------------------------------------
# Build metadata: git commit hash and configure-time timestamp
# ---------------------------------------------------------------------------
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE LLMQUANT_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result
)
if(NOT _git_result EQUAL 0 OR NOT LLMQUANT_GIT_COMMIT)
    set(LLMQUANT_GIT_COMMIT "unknown")
endif()
string(TIMESTAMP LLMQUANT_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)
message(STATUS "Build metadata: git=${LLMQUANT_GIT_COMMIT}  ts=${LLMQUANT_BUILD_TIMESTAMP}")

# ---------------------------------------------------------------------------
# Generated version header
# ---------------------------------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/llmquant_version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/llmquant_version.h"
    @ONLY
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Dependency versions are pinned in vcpkg.json at the project root.
# To use vcpkg: cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
find_package(spdlog        REQUIRED)
# fmt required by spdlog; explicit linkage prevents version mismatch on some distros.
find_package(fmt CONFIG QUIET)
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
    src/PrometheusExporter.cpp
)

# SentimentTrajectoryAnalyzer: optional online trend-detection module.
if(LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY)
    target_sources(llmquant PRIVATE src/SentimentTrajectoryAnalyzer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_TRAJECTORY_ENABLED)
    message(STATUS "Feature: SentimentTrajectoryAnalyzer ENABLED (online linear-regression trend detector)")
else()
    message(STATUS "Feature: SentimentTrajectoryAnalyzer DISABLED (LLMQUANT_ENABLE_SENTIMENT_TRAJECTORY=OFF)")
endif()

# SignalAuditLog: async NDJSON audit log — feature-flagged for minimal builds.
if(LLMQUANT_ENABLE_AUDIT_LOG)
    target_sources(llmquant PRIVATE src/SignalAuditLog.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_AUDIT_LOG_ENABLED)
    message(STATUS "Feature: SignalAuditLog ENABLED (async NDJSON signal audit log)")
else()
    message(STATUS "Feature: SignalAuditLog DISABLED (LLMQUANT_ENABLE_AUDIT_LOG=OFF)")
endif()

# PipelineCircuitBreaker: auto-pause when sustained block rate exceeds threshold.
if(LLMQUANT_ENABLE_CIRCUIT_BREAKER)
    target_sources(llmquant PRIVATE src/PipelineCircuitBreaker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CIRCUIT_BREAKER_ENABLED)
    message(STATUS "Feature: PipelineCircuitBreaker ENABLED (auto-pause on high block rate)")
else()
    message(STATUS "Feature: PipelineCircuitBreaker DISABLED (LLMQUANT_ENABLE_CIRCUIT_BREAKER=OFF)")
endif()

# BacktestRunner: full-pipeline token-sequence replay with PnL statistics.
if(LLMQUANT_ENABLE_BACKTEST)
    target_sources(llmquant PRIVATE src/BacktestRunner.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_BACKTEST_ENABLED)
    message(STATUS "Feature: BacktestRunner ENABLED (full-pipeline replay with PnL stats)")
else()
    message(STATUS "Feature: BacktestRunner DISABLED (LLMQUANT_ENABLE_BACKTEST=OFF)")
endif()

# HealthServer: lightweight HTTP /health endpoint for k8s liveness/readiness probes.
if(LLMQUANT_ENABLE_HEALTH_SERVER)
    target_sources(llmquant PRIVATE src/HealthServer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_HEALTH_SERVER_ENABLED)
    message(STATUS "Feature: HealthServer ENABLED (HTTP /health endpoint on port 8080)")
else()
    message(STATUS "Feature: HealthServer DISABLED (LLMQUANT_ENABLE_HEALTH_SERVER=OFF)")
endif()

# KellyPositionSizer: Kelly Criterion adaptive position sizing.
if(LLMQUANT_ENABLE_KELLY_SIZER)
    target_sources(llmquant PRIVATE src/KellyPositionSizer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_KELLY_SIZER_ENABLED)
    message(STATUS "Feature: KellyPositionSizer ENABLED (Kelly Criterion position sizing)")
else()
    message(STATUS "Feature: KellyPositionSizer DISABLED (LLMQUANT_ENABLE_KELLY_SIZER=OFF)")
endif()

# AdaptiveCooldownController: latency-aware dynamic signal emission cooldown.
if(LLMQUANT_ENABLE_ADAPTIVE_COOLDOWN)
    target_sources(llmquant PRIVATE src/AdaptiveCooldownController.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ADAPTIVE_COOLDOWN_ENABLED)
    message(STATUS "Feature: AdaptiveCooldownController ENABLED (latency-aware cooldown)")
else()
    message(STATUS "Feature: AdaptiveCooldownController DISABLED (LLMQUANT_ENABLE_ADAPTIVE_COOLDOWN=OFF)")
endif()

# SignalBlendLayer: multi-source confidence-weighted signal blending.
if(LLMQUANT_ENABLE_SIGNAL_BLEND)
    target_sources(llmquant PRIVATE src/SignalBlendLayer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_BLEND_ENABLED)
    message(STATUS "Feature: SignalBlendLayer ENABLED (multi-source signal blending)")
else()
    message(STATUS "Feature: SignalBlendLayer DISABLED (LLMQUANT_ENABLE_SIGNAL_BLEND=OFF)")
endif()

# RegimeDetector: EMA-based signal pipeline regime classifier.
if(LLMQUANT_ENABLE_REGIME_DETECTOR)
    target_sources(llmquant PRIVATE src/RegimeDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REGIME_DETECTOR_ENABLED)
    message(STATUS "Feature: RegimeDetector ENABLED (EMA-based regime classifier)")
else()
    message(STATUS "Feature: RegimeDetector DISABLED (LLMQUANT_ENABLE_REGIME_DETECTOR=OFF)")
endif()

# TokenReplayRecorder: binary token stream recorder and deterministic replay.
if(LLMQUANT_ENABLE_TOKEN_REPLAY)
    target_sources(llmquant PRIVATE src/TokenReplayRecorder.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_REPLAY_ENABLED)
    message(STATUS "Feature: TokenReplayRecorder ENABLED (binary token replay)")
else()
    message(STATUS "Feature: TokenReplayRecorder DISABLED (LLMQUANT_ENABLE_TOKEN_REPLAY=OFF)")
endif()

# StaleTokenDetector: watchdog that alerts when the LLM token stream goes silent.
if(LLMQUANT_ENABLE_STALE_DETECTOR)
    target_sources(llmquant PRIVATE src/StaleTokenDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_STALE_DETECTOR_ENABLED)
    message(STATUS "Feature: StaleTokenDetector ENABLED (LLM stream silence watchdog)")
else()
    message(STATUS "Feature: StaleTokenDetector DISABLED (LLMQUANT_ENABLE_STALE_DETECTOR=OFF)")
endif()

# PositionTracker: real-time P&L accounting with automatic Kelly feedback.
if(LLMQUANT_ENABLE_POSITION_TRACKER)
    target_sources(llmquant PRIVATE src/PositionTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_POSITION_TRACKER_ENABLED)
    message(STATUS "Feature: PositionTracker ENABLED (real P&L + Kelly feedback)")
else()
    message(STATUS "Feature: PositionTracker DISABLED (LLMQUANT_ENABLE_POSITION_TRACKER=OFF)")
endif()

# ParameterSweep: grid-search optimiser over BacktestRunner configs.
if(LLMQUANT_ENABLE_PARAMETER_SWEEP AND LLMQUANT_ENABLE_BACKTEST)
    target_sources(llmquant PRIVATE src/ParameterSweep.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PARAMETER_SWEEP_ENABLED)
    message(STATUS "Feature: ParameterSweep ENABLED (grid-search over BacktestRunner)")
elseif(LLMQUANT_ENABLE_PARAMETER_SWEEP AND NOT LLMQUANT_ENABLE_BACKTEST)
    message(STATUS "Feature: ParameterSweep SKIPPED (requires LLMQUANT_ENABLE_BACKTEST=ON)")
else()
    message(STATUS "Feature: ParameterSweep DISABLED (LLMQUANT_ENABLE_PARAMETER_SWEEP=OFF)")
endif()

# MockOmsAdapter is only compiled when LLMQUANT_ENABLE_MOCK_OMS=ON (default).
# Disable for production builds to strip the test double from the binary.
if(LLMQUANT_ENABLE_MOCK_OMS)
    target_sources(llmquant PRIVATE src/MockOmsAdapter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_MOCK_OMS_ENABLED)
    message(STATUS "Feature: MockOmsAdapter ENABLED (test double included)")
else()
    message(STATUS "Feature: MockOmsAdapter DISABLED (LLMQUANT_ENABLE_MOCK_OMS=OFF)")
endif()

target_include_directories(llmquant
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# MSVC optimisation flags: appended to the target rather than overriding
# CMAKE_CXX_FLAGS_RELEASE so that CMake's default /MD CRT linkage is preserved.
if(MSVC)
    target_compile_options(llmquant PRIVATE
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
    )
    # Prevent windows.h from defining min/max macros that conflict with std::max/std::min.
    target_compile_definitions(llmquant PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(llmquant
    PUBLIC
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        nlohmann_json::nlohmann_json
        Threads::Threads
        # fmt::fmt explicitly linked to prevent version mismatch with spdlog on some distros.
        $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>
)

if(WIN32)
    target_link_libraries(llmquant PUBLIC ws2_32 psapi)
endif()

if(LLMQUANT_ENABLE_TLS AND (OpenSSL_FOUND OR OPENSSL_FOUND))
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TLS_ENABLED)
    target_link_libraries(llmquant PUBLIC OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "Feature: TLS (OpenSSL) ENABLED")
elseif(LLMQUANT_ENABLE_TLS)
    message(STATUS "Feature: TLS requested but OpenSSL not found — TLS disabled")
else()
    message(STATUS "Feature: TLS disabled (LLMQUANT_ENABLE_TLS=OFF)")
endif()

if(LLMQUANT_ENABLE_REDIS AND (hiredis_FOUND OR HIREDIS_FOUND))
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REDIS_ENABLED)
    if(TARGET hiredis::hiredis)
        target_link_libraries(llmquant PUBLIC hiredis::hiredis)
    else()
        target_link_libraries(llmquant PUBLIC hiredis)
    endif()
    message(STATUS "Feature: Redis (hiredis) ENABLED")
elseif(LLMQUANT_ENABLE_REDIS)
    message(STATUS "Feature: Redis requested but hiredis not found — Redis disabled")
else()
    message(STATUS "Feature: Redis disabled (LLMQUANT_ENABLE_REDIS=OFF)")
endif()

# Wire feature flags as compile-time preprocessor symbols.
if(LLMQUANT_ENABLE_PROMETHEUS)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PROMETHEUS_ENABLED)
    message(STATUS "Feature: Prometheus scrape endpoint ENABLED")
else()
    message(STATUS "Feature: Prometheus scrape endpoint DISABLED (LLMQUANT_ENABLE_PROMETHEUS=OFF)")
endif()
if(LLMQUANT_ENABLE_FIX_OMS)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_FIX_OMS_ENABLED)
    message(STATUS "Feature: FIX 4.2 OMS adapter ENABLED")
else()
    message(STATUS "Feature: FIX 4.2 OMS adapter DISABLED (LLMQUANT_ENABLE_FIX_OMS=OFF)")
endif()
if(LLMQUANT_ENABLE_REST_OMS)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REST_OMS_ENABLED)
    message(STATUS "Feature: REST OMS adapter ENABLED")
else()
    message(STATUS "Feature: REST OMS adapter DISABLED (LLMQUANT_ENABLE_REST_OMS=OFF)")
endif()
if(LLMQUANT_ENABLE_PROFILING)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PROFILING_ENABLED)
    message(STATUS "Feature: Latency profiling ENABLED")
else()
    message(STATUS "Feature: Latency profiling DISABLED (LLMQUANT_ENABLE_PROFILING=OFF)")
endif()
if(LLMQUANT_ENABLE_DEDUP)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_DEDUP_ENABLED)
    message(STATUS "Feature: Token deduplication ENABLED")
else()
    message(STATUS "Feature: Token deduplication DISABLED (LLMQUANT_ENABLE_DEDUP=OFF)")
endif()
if(LLMQUANT_ENABLE_JSON_STATS_SUMMARY)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_JSON_STATS_SUMMARY)
    message(STATUS "Feature: JSON stats summary on exit ENABLED")
else()
    message(STATUS "Feature: JSON stats summary on exit DISABLED (LLMQUANT_ENABLE_JSON_STATS_SUMMARY=OFF)")
endif()
if(LLMQUANT_ENABLE_HOT_RELOAD)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_HOT_RELOAD_ENABLED)
    message(STATUS "Feature: Config hot-reload watcher ENABLED")
else()
    message(STATUS "Feature: Config hot-reload watcher DISABLED (LLMQUANT_ENABLE_HOT_RELOAD=OFF)")
endif()
if(LLMQUANT_ENABLE_STREAM_CLIENT)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_STREAM_CLIENT_ENABLED)
    message(STATUS "Feature: LLMStreamClient live-streaming adapter ENABLED")
else()
    message(STATUS "Feature: LLMStreamClient live-streaming adapter DISABLED (LLMQUANT_ENABLE_STREAM_CLIENT=OFF)")
endif()
if(LLMQUANT_ENABLE_SIGNAL_TRACE)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_TRACE_ENABLED)
    message(STATUS "Feature: Per-token signal trace logging ENABLED (high verbosity)")
else()
    message(STATUS "Feature: Per-token signal trace logging DISABLED (LLMQUANT_ENABLE_SIGNAL_TRACE=OFF, default)")
endif()
if(NOT LLMQUANT_ENABLE_SIMD)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIMD_DISABLED)
    message(STATUS "Feature: SSE2 SIMD batch processing DISABLED (LLMQUANT_ENABLE_SIMD=OFF)")
else()
    message(STATUS "Feature: SSE2 SIMD batch processing ENABLED (hardware-conditional)")
endif()
if(LLMQUANT_ENABLE_ALERT_DISPATCHER)
    target_sources(llmquant PRIVATE src/AlertDispatcher.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ALERT_DISPATCHER_ENABLED)
    message(STATUS "Feature: AlertDispatcher ENABLED (async alert delivery)")
else()
    message(STATUS "Feature: AlertDispatcher DISABLED (LLMQUANT_ENABLE_ALERT_DISPATCHER=OFF)")
endif()
if(LLMQUANT_ENABLE_ENTROPY_METER)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ENTROPY_METER_ENABLED)
    message(STATUS "Feature: TokenEntropyMeter ENABLED (header-only; Shannon entropy confidence scaling)")
else()
    message(STATUS "Feature: TokenEntropyMeter DISABLED (LLMQUANT_ENABLE_ENTROPY_METER=OFF)")
endif()
if(LLMQUANT_ENABLE_LATENCY_ENFORCER)
    target_sources(llmquant PRIVATE src/LatencyBudgetEnforcer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_LATENCY_ENFORCER_ENABLED)
    message(STATUS "Feature: LatencyBudgetEnforcer ENABLED (tiered SLA enforcement)")
else()
    message(STATUS "Feature: LatencyBudgetEnforcer DISABLED (LLMQUANT_ENABLE_LATENCY_ENFORCER=OFF)")
endif()
if(LLMQUANT_ENABLE_ENTROPY_MONITOR)
    target_sources(llmquant PRIVATE src/TokenEntropyMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ENTROPY_MONITOR_ENABLED)
    message(STATUS "Feature: TokenEntropyMonitor ENABLED (real-time token diversity entropy)")
else()
    message(STATUS "Feature: TokenEntropyMonitor DISABLED (LLMQUANT_ENABLE_ENTROPY_MONITOR=OFF)")
endif()
if(LLMQUANT_ENABLE_TRADING_HOURS)
    target_sources(llmquant PRIVATE src/TradingHoursGuard.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TRADING_HOURS_ENABLED)
    message(STATUS "Feature: TradingHoursGuard ENABLED (NYSE/NASDAQ session-aware signal gating)")
else()
    message(STATUS "Feature: TradingHoursGuard DISABLED (LLMQUANT_ENABLE_TRADING_HOURS=OFF)")
endif()
if(LLMQUANT_ENABLE_SENTIMENT_MOMENTUM_FILTER)
    target_sources(llmquant PRIVATE src/SentimentMomentumFilter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_MOMENTUM_FILTER_ENABLED)
    message(STATUS "Feature: SentimentMomentumFilter ENABLED (trajectory-validated signal gating)")
else()
    message(STATUS "Feature: SentimentMomentumFilter DISABLED (LLMQUANT_ENABLE_SENTIMENT_MOMENTUM_FILTER=OFF)")
endif()

# SignalCorrelationTracker: rolling Pearson correlation across named signal sources.
if(LLMQUANT_ENABLE_SIGNAL_CORRELATION)
    target_sources(llmquant PRIVATE src/SignalCorrelationTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_CORRELATION_ENABLED)
    message(STATUS "Feature: SignalCorrelationTracker ENABLED (rolling Pearson correlation)")
else()
    message(STATUS "Feature: SignalCorrelationTracker DISABLED (LLMQUANT_ENABLE_SIGNAL_CORRELATION=OFF)")
endif()

# LeadLagDetector: cross-correlation at multiple lags to find leading indicators.
if(LLMQUANT_ENABLE_LEAD_LAG)
    target_sources(llmquant PRIVATE src/LeadLagDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_LEAD_LAG_ENABLED)
    message(STATUS "Feature: LeadLagDetector ENABLED (cross-correlation lead-lag analysis)")
else()
    message(STATUS "Feature: LeadLagDetector DISABLED (LLMQUANT_ENABLE_LEAD_LAG=OFF)")
endif()

# WarmupSequencer: pre-seed pipeline EMA accumulators with synthetic tokens.
if(LLMQUANT_ENABLE_WARMUP_SEQUENCER)
    target_sources(llmquant PRIVATE src/WarmupSequencer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_WARMUP_SEQUENCER_ENABLED)
    message(STATUS "Feature: WarmupSequencer ENABLED (pre-seed EMA with synthetic tokens)")
else()
    message(STATUS "Feature: WarmupSequencer DISABLED (LLMQUANT_ENABLE_WARMUP_SEQUENCER=OFF)")
endif()
if(LLMQUANT_ENABLE_SIGNAL_DECAY)
    target_sources(llmquant PRIVATE src/SignalDecayEnvelope.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_DECAY_ENABLED)
    message(STATUS "Feature: SignalDecayEnvelope ENABLED (exponential bias decay)")
else()
    message(STATUS "Feature: SignalDecayEnvelope DISABLED (LLMQUANT_ENABLE_SIGNAL_DECAY=OFF)")
endif()
if(LLMQUANT_ENABLE_TOKEN_COST_BUDGET)
    target_sources(llmquant PRIVATE src/TokenCostBudget.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_COST_BUDGET_ENABLED)
    message(STATUS "Feature: TokenCostBudget ENABLED (LLM API cost limiter)")
else()
    message(STATUS "Feature: TokenCostBudget DISABLED (LLMQUANT_ENABLE_TOKEN_COST_BUDGET=OFF)")
endif()
if(LLMQUANT_ENABLE_GRADIENT_OPTIMISER AND LLMQUANT_ENABLE_BACKTEST)
    target_sources(llmquant PRIVATE src/GradientOptimiser.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_GRADIENT_OPTIMISER_ENABLED)
    message(STATUS "Feature: GradientOptimiser ENABLED (gradient-ascent hyper-parameter search)")
elseif(LLMQUANT_ENABLE_GRADIENT_OPTIMISER AND NOT LLMQUANT_ENABLE_BACKTEST)
    message(STATUS "Feature: GradientOptimiser SKIPPED (requires LLMQUANT_ENABLE_BACKTEST=ON)")
else()
    message(STATUS "Feature: GradientOptimiser DISABLED (LLMQUANT_ENABLE_GRADIENT_OPTIMISER=OFF)")
endif()

# DrawdownProtector: tiered dynamic risk tightening on cumulative drawdown.
if(LLMQUANT_ENABLE_DRAWDOWN_PROTECTOR)
    target_sources(llmquant PRIVATE src/DrawdownProtector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_DRAWDOWN_PROTECTOR_ENABLED)
    message(STATUS "Feature: DrawdownProtector ENABLED (tiered risk scaling on drawdown)")
else()
    message(STATUS "Feature: DrawdownProtector DISABLED (LLMQUANT_ENABLE_DRAWDOWN_PROTECTOR=OFF)")
endif()

# MultiTimeframeAggregator: multi-horizon EMA signal fusion.
if(LLMQUANT_ENABLE_MULTI_TIMEFRAME)
    target_sources(llmquant PRIVATE src/MultiTimeframeAggregator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_MULTI_TIMEFRAME_ENABLED)
    message(STATUS "Feature: MultiTimeframeAggregator ENABLED (1s/5s/30s/5m EMA fusion)")
else()
    message(STATUS "Feature: MultiTimeframeAggregator DISABLED (LLMQUANT_ENABLE_MULTI_TIMEFRAME=OFF)")
endif()

# RegimeTransitionModel: online Markov-chain learner for regime transitions.
if(LLMQUANT_ENABLE_REGIME_TRANSITION_MODEL AND LLMQUANT_ENABLE_REGIME_DETECTOR)
    target_sources(llmquant PRIVATE src/RegimeTransitionModel.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REGIME_TRANSITION_MODEL_ENABLED)
    message(STATUS "Feature: RegimeTransitionModel ENABLED (Markov regime transition predictor)")
elseif(LLMQUANT_ENABLE_REGIME_TRANSITION_MODEL AND NOT LLMQUANT_ENABLE_REGIME_DETECTOR)
    message(STATUS "Feature: RegimeTransitionModel SKIPPED (requires LLMQUANT_ENABLE_REGIME_DETECTOR=ON)")
else()
    message(STATUS "Feature: RegimeTransitionModel DISABLED (LLMQUANT_ENABLE_REGIME_TRANSITION_MODEL=OFF)")
endif()
if(LLMQUANT_ENABLE_SIGNAL_THROTTLE)
    target_sources(llmquant PRIVATE src/SignalThrottle.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_THROTTLE_ENABLED)
    message(STATUS "Feature: SignalThrottle ENABLED (token-bucket rate limiter)")
else()
    message(STATUS "Feature: SignalThrottle DISABLED (LLMQUANT_ENABLE_SIGNAL_THROTTLE=OFF)")
endif()
if(LLMQUANT_ENABLE_CONFIDENCE_CALIBRATOR)
    target_sources(llmquant PRIVATE src/ConfidenceCalibrator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CONFIDENCE_CALIBRATOR_ENABLED)
    message(STATUS "Feature: ConfidenceCalibrator ENABLED (Platt-scaling online confidence calibration)")
else()
    message(STATUS "Feature: ConfidenceCalibrator DISABLED (LLMQUANT_ENABLE_CONFIDENCE_CALIBRATOR=OFF)")
endif()

# VolatilityForecaster: EWMA-based realized-vol forecast.
if(LLMQUANT_ENABLE_VOLATILITY_FORECASTER)
    target_sources(llmquant PRIVATE src/VolatilityForecaster.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_VOLATILITY_FORECASTER_ENABLED)
    message(STATUS "Feature: VolatilityForecaster ENABLED (EWMA realized-vol forecast)")
else()
    message(STATUS "Feature: VolatilityForecaster DISABLED (LLMQUANT_ENABLE_VOLATILITY_FORECASTER=OFF)")
endif()

# BayesianSignalFilter: Bayesian online prior update from trade outcomes.
if(LLMQUANT_ENABLE_BAYESIAN_SIGNAL_FILTER)
    target_sources(llmquant PRIVATE src/BayesianSignalFilter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_BAYESIAN_FILTER_ENABLED)
    message(STATUS "Feature: BayesianSignalFilter ENABLED (Bayesian online confidence update)")
else()
    message(STATUS "Feature: BayesianSignalFilter DISABLED (LLMQUANT_ENABLE_BAYESIAN_SIGNAL_FILTER=OFF)")
endif()

# AnomalyDetector: z-score spike detector across signal streams.
if(LLMQUANT_ENABLE_ANOMALY_DETECTOR)
    target_sources(llmquant PRIVATE src/AnomalyDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ANOMALY_DETECTOR_ENABLED)
    message(STATUS "Feature: AnomalyDetector ENABLED (z-score spike detection)")
else()
    message(STATUS "Feature: AnomalyDetector DISABLED (LLMQUANT_ENABLE_ANOMALY_DETECTOR=OFF)")
endif()

# NarrativeChangeDetector: cosine-similarity break detector for LLM topic-switch.
if(LLMQUANT_ENABLE_NARRATIVE_CHANGE)
    target_sources(llmquant PRIVATE src/NarrativeChangeDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_NARRATIVE_CHANGE_ENABLED)
    message(STATUS "Feature: NarrativeChangeDetector ENABLED (cosine-similarity narrative break)")
else()
    message(STATUS "Feature: NarrativeChangeDetector DISABLED (LLMQUANT_ENABLE_NARRATIVE_CHANGE=OFF)")
endif()

# PnLAttributionEngine: retrospective P&L attribution across signal driver categories.
if(LLMQUANT_ENABLE_PNL_ATTRIBUTION)
    target_sources(llmquant PRIVATE src/PnLAttributionEngine.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PNL_ATTRIBUTION_ENABLED)
    message(STATUS "Feature: PnLAttributionEngine ENABLED (P&L attribution by sentiment driver)")
else()
    message(STATUS "Feature: PnLAttributionEngine DISABLED (LLMQUANT_ENABLE_PNL_ATTRIBUTION=OFF)")
endif()

# PortfolioHeatMonitor: cross-instrument risk heat aggregator.
if(LLMQUANT_ENABLE_PORTFOLIO_HEAT)
    target_sources(llmquant PRIVATE src/PortfolioHeatMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_PORTFOLIO_HEAT_ENABLED)
    message(STATUS "Feature: PortfolioHeatMonitor ENABLED (cross-instrument risk heat)")
else()
    message(STATUS "Feature: PortfolioHeatMonitor DISABLED (LLMQUANT_ENABLE_PORTFOLIO_HEAT=OFF)")
endif()

# TokenBurstDetector: sliding-window token arrival rate burst detector.
if(LLMQUANT_ENABLE_BURST_DETECTOR)
    target_sources(llmquant PRIVATE src/TokenBurstDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_BURST_DETECTOR_ENABLED)
    message(STATUS "Feature: TokenBurstDetector ENABLED (token arrival rate burst detection)")
else()
    message(STATUS "Feature: TokenBurstDetector DISABLED (LLMQUANT_ENABLE_BURST_DETECTOR=OFF)")
endif()

# SignalPersistenceTracker: consecutive same-direction streak with conviction scaling.
if(LLMQUANT_ENABLE_SIGNAL_PERSISTENCE)
    target_sources(llmquant PRIVATE src/SignalPersistenceTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_PERSISTENCE_ENABLED)
    message(STATUS "Feature: SignalPersistenceTracker ENABLED (directional streak + conviction scale)")
else()
    message(STATUS "Feature: SignalPersistenceTracker DISABLED (LLMQUANT_ENABLE_SIGNAL_PERSISTENCE=OFF)")
endif()

# RollingSharpeBiasTracker: rolling Sharpe ratio of the bias stream.
if(LLMQUANT_ENABLE_ROLLING_SHARPE)
    target_sources(llmquant PRIVATE src/RollingSharpeBiasTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ROLLING_SHARPE_ENABLED)
    message(STATUS "Feature: RollingSharpeBiasTracker ENABLED (rolling Sharpe signal quality)")
else()
    message(STATUS "Feature: RollingSharpeBiasTracker DISABLED (LLMQUANT_ENABLE_ROLLING_SHARPE=OFF)")
endif()

# ContextWindowBudget: LLM context fill fraction tracker.
if(LLMQUANT_ENABLE_CONTEXT_WINDOW_BUDGET)
    target_sources(llmquant PRIVATE src/ContextWindowBudget.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CONTEXT_WINDOW_BUDGET_ENABLED)
    message(STATUS "Feature: ContextWindowBudget ENABLED (context fill throttle)")
else()
    message(STATUS "Feature: ContextWindowBudget DISABLED (LLMQUANT_ENABLE_CONTEXT_WINDOW_BUDGET=OFF)")
endif()

# FractalDimensionEstimator: Higuchi fractal dimension.
if(LLMQUANT_ENABLE_FRACTAL_DIMENSION)
    target_sources(llmquant PRIVATE src/FractalDimensionEstimator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_FRACTAL_DIMENSION_ENABLED)
    message(STATUS "Feature: FractalDimensionEstimator ENABLED (Higuchi fractal dimension)")
else()
    message(STATUS "Feature: FractalDimensionEstimator DISABLED (LLMQUANT_ENABLE_FRACTAL_DIMENSION=OFF)")
endif()

# MarketMicrostructureFilter: bid-ask + mean-reversion signal cleaning.
if(LLMQUANT_ENABLE_MARKET_MICROSTRUCTURE)
    target_sources(llmquant PRIVATE src/MarketMicrostructureFilter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_MARKET_MICROSTRUCTURE_ENABLED)
    message(STATUS "Feature: MarketMicrostructureFilter ENABLED (microstructure noise filter)")
else()
    message(STATUS "Feature: MarketMicrostructureFilter DISABLED (LLMQUANT_ENABLE_MARKET_MICROSTRUCTURE=OFF)")
endif()

# SignalEnsembleLayer: model-stacking for multiple sub-signal generators.
if(LLMQUANT_ENABLE_SIGNAL_ENSEMBLE)
    target_sources(llmquant PRIVATE src/SignalEnsembleLayer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_ENSEMBLE_ENABLED)
    message(STATUS "Feature: SignalEnsembleLayer ENABLED (ensemble stacking)")
else()
    message(STATUS "Feature: SignalEnsembleLayer DISABLED (LLMQUANT_ENABLE_SIGNAL_ENSEMBLE=OFF)")
endif()

# SignalMomentumOscillator: MACD-style fast/slow EMA oscillator on bias stream.
if(LLMQUANT_ENABLE_SIGNAL_MOMENTUM_OSC)
    target_sources(llmquant PRIVATE src/SignalMomentumOscillator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_MOMENTUM_OSC_ENABLED)
    message(STATUS "Feature: SignalMomentumOscillator ENABLED (MACD oscillator)")
else()
    message(STATUS "Feature: SignalMomentumOscillator DISABLED (LLMQUANT_ENABLE_SIGNAL_MOMENTUM_OSC=OFF)")
endif()

# OrderBookSimulator: sentiment-driven synthetic LOB for slippage-aware sizing.
if(LLMQUANT_ENABLE_ORDER_BOOK_SIM)
    target_sources(llmquant PRIVATE src/OrderBookSimulator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ORDER_BOOK_SIM_ENABLED)
    message(STATUS "Feature: OrderBookSimulator ENABLED (synthetic LOB + fill price estimation)")
else()
    message(STATUS "Feature: OrderBookSimulator DISABLED (LLMQUANT_ENABLE_ORDER_BOOK_SIM=OFF)")
endif()

# TokenSentimentHeatmap: 2D token×bucket rolling histogram for alpha attribution.
if(LLMQUANT_ENABLE_SENTIMENT_HEATMAP)
    target_sources(llmquant PRIVATE src/TokenSentimentHeatmap.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_HEATMAP_ENABLED)
    message(STATUS "Feature: TokenSentimentHeatmap ENABLED (token-level sentiment attribution)")
else()
    message(STATUS "Feature: TokenSentimentHeatmap DISABLED (LLMQUANT_ENABLE_SENTIMENT_HEATMAP=OFF)")
endif()

# CVaRCalculator: rolling Conditional Value-at-Risk / Expected Shortfall.
if(LLMQUANT_ENABLE_CVAR)
    target_sources(llmquant PRIVATE src/CVaRCalculator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CVAR_ENABLED)
    message(STATUS "Feature: CVaRCalculator ENABLED (rolling CVaR / Expected Shortfall)")
else()
    message(STATUS "Feature: CVaRCalculator DISABLED (LLMQUANT_ENABLE_CVAR=OFF)")
endif()

# TemporalPatternLibrary: trie-based multi-token pattern matcher.
if(LLMQUANT_ENABLE_TEMPORAL_PATTERN)
    target_sources(llmquant PRIVATE src/TemporalPatternLibrary.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TEMPORAL_PATTERN_ENABLED)
    message(STATUS "Feature: TemporalPatternLibrary ENABLED (trie pattern matcher)")
else()
    message(STATUS "Feature: TemporalPatternLibrary DISABLED (LLMQUANT_ENABLE_TEMPORAL_PATTERN=OFF)")
endif()

# FeedbackLoopDetector: cross-correlation reflexivity detector.
if(LLMQUANT_ENABLE_FEEDBACK_LOOP)
    target_sources(llmquant PRIVATE src/FeedbackLoopDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_FEEDBACK_LOOP_ENABLED)
    message(STATUS "Feature: FeedbackLoopDetector ENABLED (market-impact reflexivity detection)")
else()
    message(STATUS "Feature: FeedbackLoopDetector DISABLED (LLMQUANT_ENABLE_FEEDBACK_LOOP=OFF)")
endif()

# SentimentCycleDetector: rolling ACF periodicity analysis on bias stream.
if(LLMQUANT_ENABLE_SENTIMENT_CYCLE)
    target_sources(llmquant PRIVATE src/SentimentCycleDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_CYCLE_ENABLED)
    message(STATUS "Feature: SentimentCycleDetector ENABLED (ACF news-cycle analysis)")
else()
    message(STATUS "Feature: SentimentCycleDetector DISABLED (LLMQUANT_ENABLE_SENTIMENT_CYCLE=OFF)")
endif()

# AdaptiveSamplingController: multiplicative interval adaptation for token polling.
if(LLMQUANT_ENABLE_ADAPTIVE_SAMPLING)
    target_sources(llmquant PRIVATE src/AdaptiveSamplingController.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ADAPTIVE_SAMPLING_ENABLED)
    message(STATUS "Feature: AdaptiveSamplingController ENABLED (adaptive token-poll interval)")
else()
    message(STATUS "Feature: AdaptiveSamplingController DISABLED (LLMQUANT_ENABLE_ADAPTIVE_SAMPLING=OFF)")
endif()

# MutualInformationEstimator: discretized MI between LLM sentiment and returns.
if(LLMQUANT_ENABLE_MUTUAL_INFORMATION)
    target_sources(llmquant PRIVATE src/MutualInformationEstimator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_MUTUAL_INFORMATION_ENABLED)
    message(STATUS "Feature: MutualInformationEstimator ENABLED (sentiment-return MI)")
else()
    message(STATUS "Feature: MutualInformationEstimator DISABLED (LLMQUANT_ENABLE_MUTUAL_INFORMATION=OFF)")
endif()

# SignalBlindSpotDetector: time-of-day win-rate slot analysis.
if(LLMQUANT_ENABLE_SIGNAL_BLIND_SPOT)
    target_sources(llmquant PRIVATE src/SignalBlindSpotDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_BLIND_SPOT_ENABLED)
    message(STATUS "Feature: SignalBlindSpotDetector ENABLED (time-of-day blind-spot analysis)")
else()
    message(STATUS "Feature: SignalBlindSpotDetector DISABLED (LLMQUANT_ENABLE_SIGNAL_BLIND_SPOT=OFF)")
endif()

# SignalSurpriseIndex: Shannon self-information of bias signals.
if(LLMQUANT_ENABLE_SIGNAL_SURPRISE)
    target_sources(llmquant PRIVATE src/SignalSurpriseIndex.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_SURPRISE_ENABLED)
    message(STATUS "Feature: SignalSurpriseIndex ENABLED (surprise / self-information)")
else()
    message(STATUS "Feature: SignalSurpriseIndex DISABLED (LLMQUANT_ENABLE_SIGNAL_SURPRISE=OFF)")
endif()

# TokenStreamHealthMonitor: watchdog for feed stalls and token floods.
if(LLMQUANT_ENABLE_STREAM_HEALTH)
    target_sources(llmquant PRIVATE src/TokenStreamHealthMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_STREAM_HEALTH_ENABLED)
    message(STATUS "Feature: TokenStreamHealthMonitor ENABLED (stall/flood detection)")
else()
    message(STATUS "Feature: TokenStreamHealthMonitor DISABLED (LLMQUANT_ENABLE_STREAM_HEALTH=OFF)")
endif()

# RegimeAwareSizer: integrates Hurst exponent + vol forecast into notional scaling.
if(LLMQUANT_ENABLE_REGIME_SIZER)
    target_sources(llmquant PRIVATE src/RegimeAwareSizer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REGIME_SIZER_ENABLED)
    message(STATUS "Feature: RegimeAwareSizer ENABLED (regime + vol position sizing)")
else()
    message(STATUS "Feature: RegimeAwareSizer DISABLED (LLMQUANT_ENABLE_REGIME_SIZER=OFF)")
endif()

# ConfidenceDecayTracker: exponential decay fitting for signal confidence half-life.
if(LLMQUANT_ENABLE_CONFIDENCE_DECAY)
    target_sources(llmquant PRIVATE src/ConfidenceDecayTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CONFIDENCE_DECAY_ENABLED)
    message(STATUS "Feature: ConfidenceDecayTracker ENABLED (signal confidence half-life)")
else()
    message(STATUS "Feature: ConfidenceDecayTracker DISABLED (LLMQUANT_ENABLE_CONFIDENCE_DECAY=OFF)")
endif()

# CrossAssetCorrelationMonitor: rolling Pearson correlation across asset pairs.
if(LLMQUANT_ENABLE_CROSS_ASSET_CORR)
    target_sources(llmquant PRIVATE src/CrossAssetCorrelationMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CROSS_ASSET_CORR_ENABLED)
    message(STATUS "Feature: CrossAssetCorrelationMonitor ENABLED (multi-asset signal correlation)")
else()
    message(STATUS "Feature: CrossAssetCorrelationMonitor DISABLED (LLMQUANT_ENABLE_CROSS_ASSET_CORR=OFF)")
endif()

# TokenVelocityTracker: first and second time-derivatives of the bias stream.
if(LLMQUANT_ENABLE_VELOCITY_TRACKER)
    target_sources(llmquant PRIVATE src/TokenVelocityTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_VELOCITY_TRACKER_ENABLED)
    message(STATUS "Feature: TokenVelocityTracker ENABLED (bias velocity + acceleration)")
else()
    message(STATUS "Feature: TokenVelocityTracker DISABLED (LLMQUANT_ENABLE_VELOCITY_TRACKER=OFF)")
endif()

# NarrativeMomentumClock: four-quadrant investment-clock on smoothed bias+velocity.
if(LLMQUANT_ENABLE_NARRATIVE_CLOCK)
    target_sources(llmquant PRIVATE src/NarrativeMomentumClock.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_NARRATIVE_CLOCK_ENABLED)
    message(STATUS "Feature: NarrativeMomentumClock ENABLED (four-quadrant narrative rotation)")
else()
    message(STATUS "Feature: NarrativeMomentumClock DISABLED (LLMQUANT_ENABLE_NARRATIVE_CLOCK=OFF)")
endif()

# AdaptiveVelocityBreaker: EMA-smoothed velocity circuit breaker.
if(LLMQUANT_ENABLE_VELOCITY_BREAKER)
    target_sources(llmquant PRIVATE src/AdaptiveVelocityBreaker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_VELOCITY_BREAKER_ENABLED)
    message(STATUS "Feature: AdaptiveVelocityBreaker ENABLED (velocity circuit breaker)")
else()
    message(STATUS "Feature: AdaptiveVelocityBreaker DISABLED (LLMQUANT_ENABLE_VELOCITY_BREAKER=OFF)")
endif()

# WalkForwardValidator: rolling OOS validation with optional parameter sweep.
if(LLMQUANT_ENABLE_WALK_FORWARD)
    target_sources(llmquant PRIVATE src/WalkForwardValidator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_WALK_FORWARD_ENABLED)
    message(STATUS "Feature: WalkForwardValidator ENABLED (walk-forward OOS validation)")
else()
    message(STATUS "Feature: WalkForwardValidator DISABLED (LLMQUANT_ENABLE_WALK_FORWARD=OFF)")
endif()

# SignalCalibrationEngine: online Platt-scaling logistic calibration.
if(LLMQUANT_ENABLE_SIGNAL_CALIBRATION)
    target_sources(llmquant PRIVATE src/SignalCalibrationEngine.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_CALIBRATION_ENABLED)
    message(STATUS "Feature: SignalCalibrationEngine ENABLED (Platt confidence calibration)")
else()
    message(STATUS "Feature: SignalCalibrationEngine DISABLED (LLMQUANT_ENABLE_SIGNAL_CALIBRATION=OFF)")
endif()

# OrderFlowImbalanceDetector: EMA of buy/sell pressure from token keywords.
if(LLMQUANT_ENABLE_ORDER_FLOW_IMBALANCE)
    target_sources(llmquant PRIVATE src/OrderFlowImbalanceDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ORDER_FLOW_IMBALANCE_ENABLED)
    message(STATUS "Feature: OrderFlowImbalanceDetector ENABLED")
else()
    message(STATUS "Feature: OrderFlowImbalanceDetector DISABLED (LLMQUANT_ENABLE_ORDER_FLOW_IMBALANCE=OFF)")
endif()

# TokenBiasHeatmap: per-token cumulative bias contribution tracker.
if(LLMQUANT_ENABLE_TOKEN_BIAS_HEATMAP)
    target_sources(llmquant PRIVATE src/TokenBiasHeatmap.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_BIAS_HEATMAP_ENABLED)
    message(STATUS "Feature: TokenBiasHeatmap ENABLED (per-token bias heatmap)")
else()
    message(STATUS "Feature: TokenBiasHeatmap DISABLED (LLMQUANT_ENABLE_TOKEN_BIAS_HEATMAP=OFF)")
endif()

# CrossSessionMemory: persist Kelly / drawdown state across process restarts.
if(LLMQUANT_ENABLE_CROSS_SESSION_MEMORY)
    target_sources(llmquant PRIVATE src/CrossSessionMemory.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CROSS_SESSION_MEMORY_ENABLED)
    message(STATUS "Feature: CrossSessionMemory ENABLED (warm-start state persistence)")
else()
    message(STATUS "Feature: CrossSessionMemory DISABLED (LLMQUANT_ENABLE_CROSS_SESSION_MEMORY=OFF)")
endif()

# MarketRegimeProbabilityEstimator: online 2-state HMM Bayesian filter.
if(LLMQUANT_ENABLE_REGIME_PROB)
    target_sources(llmquant PRIVATE src/MarketRegimeProbabilityEstimator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REGIME_PROB_ENABLED)
    message(STATUS "Feature: MarketRegimeProbabilityEstimator ENABLED")
else()
    message(STATUS "Feature: MarketRegimeProbabilityEstimator DISABLED (LLMQUANT_ENABLE_REGIME_PROB=OFF)")
endif()

# SignalReplayBuffer: fixed-capacity ring buffer for recent TradeSignals.
if(LLMQUANT_ENABLE_SIGNAL_REPLAY_BUFFER)
    target_sources(llmquant PRIVATE src/SignalReplayBuffer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_REPLAY_BUFFER_ENABLED)
    message(STATUS "Feature: SignalReplayBuffer ENABLED")
else()
    message(STATUS "Feature: SignalReplayBuffer DISABLED (LLMQUANT_ENABLE_SIGNAL_REPLAY_BUFFER=OFF)")
endif()

# TokenNgramProfiler: n-gram frequency tracker on the token stream.
if(LLMQUANT_ENABLE_TOKEN_NGRAM_PROFILER)
    target_sources(llmquant PRIVATE src/TokenNgramProfiler.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_NGRAM_PROFILER_ENABLED)
    message(STATUS "Feature: TokenNgramProfiler ENABLED")
else()
    message(STATUS "Feature: TokenNgramProfiler DISABLED (LLMQUANT_ENABLE_TOKEN_NGRAM_PROFILER=OFF)")
endif()

# ExecutionQualityMonitor: fill latency and slippage distribution tracker.
if(LLMQUANT_ENABLE_EXECUTION_QUALITY)
    target_sources(llmquant PRIVATE src/ExecutionQualityMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_EXECUTION_QUALITY_ENABLED)
    message(STATUS "Feature: ExecutionQualityMonitor ENABLED")
else()
    message(STATUS "Feature: ExecutionQualityMonitor DISABLED (LLMQUANT_ENABLE_EXECUTION_QUALITY=OFF)")
endif()

# SentimentDispersionIndex: disagreement index across sentiment dimensions.
if(LLMQUANT_ENABLE_SENTIMENT_DISPERSION)
    target_sources(llmquant PRIVATE src/SentimentDispersionIndex.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_DISPERSION_ENABLED)
    message(STATUS "Feature: SentimentDispersionIndex ENABLED")
else()
    message(STATUS "Feature: SentimentDispersionIndex DISABLED (LLMQUANT_ENABLE_SENTIMENT_DISPERSION=OFF)")
endif()

# TokenInfluenceAttributor: Shapley-inspired per-token marginal contribution attribution.
if(LLMQUANT_ENABLE_TOKEN_INFLUENCE)
    target_sources(llmquant PRIVATE src/TokenInfluenceAttributor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_INFLUENCE_ENABLED)
    message(STATUS "Feature: TokenInfluenceAttributor ENABLED (per-token Shapley attribution)")
else()
    message(STATUS "Feature: TokenInfluenceAttributor DISABLED (LLMQUANT_ENABLE_TOKEN_INFLUENCE=OFF)")
endif()

# SentimentDivergenceDetector: multi-stream pairwise divergence detector.
if(LLMQUANT_ENABLE_SENTIMENT_DIVERGENCE)
    target_sources(llmquant PRIVATE src/SentimentDivergenceDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_DIVERGENCE_ENABLED)
    message(STATUS "Feature: SentimentDivergenceDetector ENABLED (multi-stream divergence)")
else()
    message(STATUS "Feature: SentimentDivergenceDetector DISABLED (LLMQUANT_ENABLE_SENTIMENT_DIVERGENCE=OFF)")
endif()

# AdversarialInputDetector: z-score + repetition + vocabulary-inflation anomaly detection.
if(LLMQUANT_ENABLE_ADVERSARIAL_DETECT)
    target_sources(llmquant PRIVATE src/AdversarialInputDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ADVERSARIAL_DETECT_ENABLED)
    message(STATUS "Feature: AdversarialInputDetector ENABLED (adversarial token detection)")
else()
    message(STATUS "Feature: AdversarialInputDetector DISABLED (LLMQUANT_ENABLE_ADVERSARIAL_DETECT=OFF)")
endif()

# SignalConfidenceInterval: rolling jackknife CI for signal uncertainty quantification.
if(LLMQUANT_ENABLE_SIGNAL_CI)
    target_sources(llmquant PRIVATE src/SignalConfidenceInterval.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_CI_ENABLED)
    message(STATUS "Feature: SignalConfidenceInterval ENABLED (jackknife CI estimation)")
else()
    message(STATUS "Feature: SignalConfidenceInterval DISABLED (LLMQUANT_ENABLE_SIGNAL_CI=OFF)")
endif()

# TokenStreamRecorder: append-only binary audit log with CRC-32 and auto-rotation.
if(LLMQUANT_ENABLE_TOKEN_RECORDER)
    target_sources(llmquant PRIVATE src/TokenStreamRecorder.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_RECORDER_ENABLED)
    message(STATUS "Feature: TokenStreamRecorder ENABLED (audit log)")
else()
    message(STATUS "Feature: TokenStreamRecorder DISABLED (LLMQUANT_ENABLE_TOKEN_RECORDER=OFF)")
endif()

# EnsembleSignalVoter: weighted vote aggregation across multiple sub-models.
if(LLMQUANT_ENABLE_ENSEMBLE_VOTER)
    target_sources(llmquant PRIVATE src/EnsembleSignalVoter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ENSEMBLE_VOTER_ENABLED)
    message(STATUS "Feature: EnsembleSignalVoter ENABLED (multi-model consensus)")
else()
    message(STATUS "Feature: EnsembleSignalVoter DISABLED (LLMQUANT_ENABLE_ENSEMBLE_VOTER=OFF)")
endif()

# SentimentPersistenceMatrix: Markov-chain transition model for bias states.
if(LLMQUANT_ENABLE_SENTIMENT_PERSISTENCE)
    target_sources(llmquant PRIVATE src/SentimentPersistenceMatrix.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_PERSISTENCE_ENABLED)
    message(STATUS "Feature: SentimentPersistenceMatrix ENABLED (Markov sentiment-state transitions)")
else()
    message(STATUS "Feature: SentimentPersistenceMatrix DISABLED (LLMQUANT_ENABLE_SENTIMENT_PERSISTENCE=OFF)")
endif()

# CausalImpactEstimator: CUSUM structural-break detector for return series.
if(LLMQUANT_ENABLE_CAUSAL_IMPACT)
    target_sources(llmquant PRIVATE src/CausalImpactEstimator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CAUSAL_IMPACT_ENABLED)
    message(STATUS "Feature: CausalImpactEstimator ENABLED (CUSUM sentiment→return causality)")
else()
    message(STATUS "Feature: CausalImpactEstimator DISABLED (LLMQUANT_ENABLE_CAUSAL_IMPACT=OFF)")
endif()

if(LLMQUANT_ENABLE_OPTIONS_FLOW_BRIDGE)
    target_sources(llmquant PRIVATE src/OptionsFlowSentimentBridge.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_OPTIONS_FLOW_BRIDGE_ENABLED)
    message(STATUS "Feature: OptionsFlowSentimentBridge ENABLED (IV skew vs sentiment velocity divergence)")
else()
    message(STATUS "Feature: OptionsFlowSentimentBridge DISABLED (LLMQUANT_ENABLE_OPTIONS_FLOW_BRIDGE=OFF)")
endif()

# SentimentPhasePortrait: 2-D phase-plane (bias, velocity) attractor/cycle analysis.
if(LLMQUANT_ENABLE_SENTIMENT_PHASE_PORTRAIT)
    target_sources(llmquant PRIVATE src/SentimentPhasePortrait.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SENTIMENT_PHASE_PORTRAIT_ENABLED)
    message(STATUS "Feature: SentimentPhasePortrait ENABLED (2-D phase-plane dynamics)")
else()
    message(STATUS "Feature: SentimentPhasePortrait DISABLED (LLMQUANT_ENABLE_SENTIMENT_PHASE_PORTRAIT=OFF)")
endif()

# NarrativeTopicClassifier: lightweight topic tagging for multi-theme signal routing.
if(LLMQUANT_ENABLE_NARRATIVE_TOPIC_CLASSIFIER)
    target_sources(llmquant PRIVATE src/NarrativeTopicClassifier.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED)
    message(STATUS "Feature: NarrativeTopicClassifier ENABLED (multi-theme token routing)")
else()
    message(STATUS "Feature: NarrativeTopicClassifier DISABLED (LLMQUANT_ENABLE_NARRATIVE_TOPIC_CLASSIFIER=OFF)")
endif()

# TokenClockRecalibrator: detects token-stream clock drift and compensates.
if(LLMQUANT_ENABLE_TOKEN_CLOCK_RECALIBRATOR)
    target_sources(llmquant PRIVATE src/TokenClockRecalibrator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_CLOCK_RECALIBRATOR_ENABLED)
    message(STATUS "Feature: TokenClockRecalibrator ENABLED (clock drift compensation)")
else()
    message(STATUS "Feature: TokenClockRecalibrator DISABLED (LLMQUANT_ENABLE_TOKEN_CLOCK_RECALIBRATOR=OFF)")
endif()

# SignalShadowPortfolio: paper-trades every signal in parallel for P&L drag attribution.
if(LLMQUANT_ENABLE_SHADOW_PORTFOLIO)
    target_sources(llmquant PRIVATE src/SignalShadowPortfolio.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SHADOW_PORTFOLIO_ENABLED)
    message(STATUS "Feature: SignalShadowPortfolio ENABLED (shadow P&L vs live attribution)")
else()
    message(STATUS "Feature: SignalShadowPortfolio DISABLED (LLMQUANT_ENABLE_SHADOW_PORTFOLIO=OFF)")
endif()

# TokenInformationBottleneck: per-token IB relevance/complexity score for vocabulary pruning.
if(LLMQUANT_ENABLE_TOKEN_IB)
    target_sources(llmquant PRIVATE src/TokenInformationBottleneck.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_IB_ENABLED)
    message(STATUS "Feature: TokenInformationBottleneck ENABLED (IB vocabulary pruning)")
else()
    message(STATUS "Feature: TokenInformationBottleneck DISABLED (LLMQUANT_ENABLE_TOKEN_IB=OFF)")
endif()

# LLMConfidenceBandTracker: Kalman-filter confidence bands for uncertainty-aware signal gating.
if(LLMQUANT_ENABLE_CONFIDENCE_BAND)
    target_sources(llmquant PRIVATE src/LLMConfidenceBandTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CONFIDENCE_BAND_ENABLED)
    message(STATUS "Feature: LLMConfidenceBandTracker ENABLED (Kalman confidence bands)")
else()
    message(STATUS "Feature: LLMConfidenceBandTracker DISABLED (LLMQUANT_ENABLE_CONFIDENCE_BAND=OFF)")
endif()

# TokenImportanceDecayScheduler: exponential decay of token weights by age.
if(LLMQUANT_ENABLE_TOKEN_DECAY_SCHEDULER)
    target_sources(llmquant PRIVATE src/TokenImportanceDecayScheduler.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_DECAY_SCHEDULER_ENABLED)
    message(STATUS "Feature: TokenImportanceDecayScheduler ENABLED (temporal weight decay)")
else()
    message(STATUS "Feature: TokenImportanceDecayScheduler DISABLED (LLMQUANT_ENABLE_TOKEN_DECAY_SCHEDULER=OFF)")
endif()

# RegimeSwitchingSignalRouter: 3-state FSM signal router per market regime.
if(LLMQUANT_ENABLE_REGIME_ROUTER)
    target_sources(llmquant PRIVATE src/RegimeSwitchingSignalRouter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REGIME_ROUTER_ENABLED)
    message(STATUS "Feature: RegimeSwitchingSignalRouter ENABLED (trending/ranging/crash signal routing)")
else()
    message(STATUS "Feature: RegimeSwitchingSignalRouter DISABLED (LLMQUANT_ENABLE_REGIME_ROUTER=OFF)")
endif()

# TokenStreamDifferencer: velocity/acceleration/jerk of token weight series.
if(LLMQUANT_ENABLE_STREAM_DIFFERENCER)
    target_sources(llmquant PRIVATE src/TokenStreamDifferencer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_STREAM_DIFFERENCER_ENABLED)
    message(STATUS "Feature: TokenStreamDifferencer ENABLED (jerk-based reversal detection)")
else()
    message(STATUS "Feature: TokenStreamDifferencer DISABLED (LLMQUANT_ENABLE_STREAM_DIFFERENCER=OFF)")
endif()

# SignalDriftMonitor: Wasserstein-1 distribution-drift detection.
if(LLMQUANT_ENABLE_SIGNAL_DRIFT)
    target_sources(llmquant PRIVATE src/SignalDriftMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_DRIFT_ENABLED)
    message(STATUS "Feature: SignalDriftMonitor ENABLED (W1 distribution drift)")
else()
    message(STATUS "Feature: SignalDriftMonitor DISABLED (LLMQUANT_ENABLE_SIGNAL_DRIFT=OFF)")
endif()

# SignalLifecycleTracker: birth/peak/decay/death lifecycle tracking.
if(LLMQUANT_ENABLE_LIFECYCLE_TRACKER)
    target_sources(llmquant PRIVATE src/SignalLifecycleTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_LIFECYCLE_TRACKER_ENABLED)
    message(STATUS "Feature: SignalLifecycleTracker ENABLED (signal lifecycle + zombie detection)")
else()
    message(STATUS "Feature: SignalLifecycleTracker DISABLED (LLMQUANT_ENABLE_LIFECYCLE_TRACKER=OFF)")
endif()

# TokenWeightQuantiser: stochastic rounding for integer-precision downstream.
if(LLMQUANT_ENABLE_TOKEN_QUANTISER)
    target_sources(llmquant PRIVATE src/TokenWeightQuantiser.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_TOKEN_QUANTISER_ENABLED)
    message(STATUS "Feature: TokenWeightQuantiser ENABLED (stochastic rounding, E[w] preserved)")
else()
    message(STATUS "Feature: TokenWeightQuantiser DISABLED (LLMQUANT_ENABLE_TOKEN_QUANTISER=OFF)")
endif()

# PositionConcentrationGuard: HHI theme concentration monitor.
if(LLMQUANT_ENABLE_POSITION_CONCENTRATION)
    target_sources(llmquant PRIVATE src/PositionConcentrationGuard.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_POSITION_CONCENTRATION_ENABLED)
    message(STATUS "Feature: PositionConcentrationGuard ENABLED (HHI theme concentration)")
else()
    message(STATUS "Feature: PositionConcentrationGuard DISABLED (LLMQUANT_ENABLE_POSITION_CONCENTRATION=OFF)")
endif()

if(LLMQUANT_ENABLE_AUTOCORR_METER)
    target_sources(llmquant PRIVATE src/SentimentAutocorrelationMeter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_AUTOCORR_METER_ENABLED)
    message(STATUS "Feature: SentimentAutocorrelationMeter ENABLED (lag-1..N AC of bias series)")
else()
    message(STATUS "Feature: SentimentAutocorrelationMeter DISABLED (LLMQUANT_ENABLE_AUTOCORR_METER=OFF)")
endif()

if(LLMQUANT_ENABLE_SIGNAL_SSI)
    target_sources(llmquant PRIVATE src/SignalStrengthIndexer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_SSI_ENABLED)
    message(STATUS "Feature: SignalStrengthIndexer ENABLED (RSI-based Sentiment Strength Index)")
else()
    message(STATUS "Feature: SignalStrengthIndexer DISABLED (LLMQUANT_ENABLE_SIGNAL_SSI=OFF)")
endif()

# TokenFlowPressureGauge: EMA of inter-token arrival intervals -> normalized pressure score.
if(LLMQUANT_ENABLE_FLOW_PRESSURE)
    target_sources(llmquant PRIVATE src/TokenFlowPressureGauge.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_FLOW_PRESSURE_ENABLED)
    message(STATUS "Feature: TokenFlowPressureGauge ENABLED (arrival pressure score, spike callback)")
else()
    message(STATUS "Feature: TokenFlowPressureGauge DISABLED (LLMQUANT_ENABLE_FLOW_PRESSURE=OFF)")
endif()

# SignalFatigueMeter: consecutive same-direction streak counter with fatigue score.
if(LLMQUANT_ENABLE_SIGNAL_FATIGUE)
    target_sources(llmquant PRIVATE src/SignalFatigueMeter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_FATIGUE_ENABLED)
    message(STATUS "Feature: SignalFatigueMeter ENABLED (streak tracking, fatigue score, fatigue callback)")
else()
    message(STATUS "Feature: SignalFatigueMeter DISABLED (LLMQUANT_ENABLE_SIGNAL_FATIGUE=OFF)")
endif()

# SignalPolarizationMonitor: Sarle bimodality coefficient for distribution polarization.
if(LLMQUANT_ENABLE_POLARIZATION_MONITOR)
    target_sources(llmquant PRIVATE src/SignalPolarizationMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_POLARIZATION_MONITOR_ENABLED)
    message(STATUS "Feature: SignalPolarizationMonitor ENABLED (bimodality coefficient, polarization detection)")
else()
    message(STATUS "Feature: SignalPolarizationMonitor DISABLED (LLMQUANT_ENABLE_POLARIZATION_MONITOR=OFF)")
endif()

# NarrativeTemperatureGauge: |EMA(bias)| × σ(bias) "temperature" index.
if(LLMQUANT_ENABLE_NARRATIVE_TEMPERATURE)
    target_sources(llmquant PRIVATE src/NarrativeTemperatureGauge.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_NARRATIVE_TEMPERATURE_ENABLED)
    message(STATUS "Feature: NarrativeTemperatureGauge ENABLED (intensity × volatility temperature index)")
else()
    message(STATUS "Feature: NarrativeTemperatureGauge DISABLED (LLMQUANT_ENABLE_NARRATIVE_TEMPERATURE=OFF)")
endif()

# SignalEchoSuppressor: rolling echo-rate detector; fires on rising/falling edge.
if(LLMQUANT_ENABLE_ECHO_SUPPRESSOR)
    target_sources(llmquant PRIVATE src/SignalEchoSuppressor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ECHO_SUPPRESSOR_ENABLED)
    message(STATUS "Feature: SignalEchoSuppressor ENABLED (rolling near-duplicate signal rate detector)")
else()
    message(STATUS "Feature: SignalEchoSuppressor DISABLED (LLMQUANT_ENABLE_ECHO_SUPPRESSOR=OFF)")
endif()

# TokenWeightHistogram: discretized histogram of bias values with entropy and mode.
if(LLMQUANT_ENABLE_WEIGHT_HISTOGRAM)
    target_sources(llmquant PRIVATE src/TokenWeightHistogram.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_WEIGHT_HISTOGRAM_ENABLED)
    message(STATUS "Feature: TokenWeightHistogram ENABLED (20-bucket bias histogram, entropy, mode tracking)")
else()
    message(STATUS "Feature: TokenWeightHistogram DISABLED (LLMQUANT_ENABLE_WEIGHT_HISTOGRAM=OFF)")
endif()

# SignalSlopeMeter: OLS regression slope over rolling window for trend acceleration.
if(LLMQUANT_ENABLE_SIGNAL_SLOPE)
    target_sources(llmquant PRIVATE src/SignalSlopeMeter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_SIGNAL_SLOPE_ENABLED)
    message(STATUS "Feature: SignalSlopeMeter ENABLED (OLS slope, trend acceleration detection)")
else()
    message(STATUS "Feature: SignalSlopeMeter DISABLED (LLMQUANT_ENABLE_SIGNAL_SLOPE=OFF)")
endif()

# LatencyJitterMonitor: MAD-based latency jitter tracking.
if(LLMQUANT_ENABLE_LATENCY_JITTER)
    target_sources(llmquant PRIVATE src/LatencyJitterMonitor.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_LATENCY_JITTER_ENABLED)
    message(STATUS "Feature: LatencyJitterMonitor ENABLED (MAD jitter tracking, spike/clear callbacks)")
else()
    message(STATUS "Feature: LatencyJitterMonitor DISABLED (LLMQUANT_ENABLE_LATENCY_JITTER=OFF)")
endif()

# SignalHurstEstimator: R/S rescaled-range Hurst exponent for persistence classification.
if(LLMQUANT_ENABLE_HURST_ESTIMATOR)
    target_sources(llmquant PRIVATE src/SignalHurstEstimator.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_HURST_ESTIMATOR_ENABLED)
    message(STATUS "Feature: SignalHurstEstimator ENABLED (R/S Hurst: H>0.6 trending, H<0.4 reverting)")
else()
    message(STATUS "Feature: SignalHurstEstimator DISABLED (LLMQUANT_ENABLE_HURST_ESTIMATOR=OFF)")
endif()

# BiasChangePointDetector: Page-CUSUM sequential change-point detection.
if(LLMQUANT_ENABLE_CHANGE_POINT)
    target_sources(llmquant PRIVATE src/BiasChangePointDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CHANGE_POINT_ENABLED)
    message(STATUS "Feature: BiasChangePointDetector ENABLED (Page-CUSUM upshift/downshift detection)")
else()
    message(STATUS "Feature: BiasChangePointDetector DISABLED (LLMQUANT_ENABLE_CHANGE_POINT=OFF)")
endif()

# BiasRunLengthEncoder: run-length statistics on bias sign sequence.
if(LLMQUANT_ENABLE_RUN_LENGTH)
    target_sources(llmquant PRIVATE src/BiasRunLengthEncoder.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_RUN_LENGTH_ENABLED)
    message(STATUS "Feature: BiasRunLengthEncoder ENABLED (run-length stats, on_long_run callback)")
else()
    message(STATUS "Feature: BiasRunLengthEncoder DISABLED (LLMQUANT_ENABLE_RUN_LENGTH=OFF)")
endif()

# SignalCoverageMeter: rolling min/max -> coverage fraction of configured bias range.
if(LLMQUANT_ENABLE_COVERAGE_METER)
    target_sources(llmquant PRIVATE src/SignalCoverageMeter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_COVERAGE_METER_ENABLED)
    message(STATUS "Feature: SignalCoverageMeter ENABLED (bias range coverage, on_range_expansion callback)")
else()
    message(STATUS "Feature: SignalCoverageMeter DISABLED (LLMQUANT_ENABLE_COVERAGE_METER=OFF)")
endif()

# BiasVelocityBreaker: EMA rate-of-change circuit breaker.
if(LLMQUANT_ENABLE_VELOCITY_BREAKER)
    target_sources(llmquant PRIVATE src/BiasVelocityBreaker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_VELOCITY_BREAKER_ENABLED)
    message(STATUS "Feature: BiasVelocityBreaker ENABLED (EMA velocity trip/clear)")
else()
    message(STATUS "Feature: BiasVelocityBreaker DISABLED (LLMQUANT_ENABLE_VELOCITY_BREAKER=OFF)")
endif()

# SignalInformationRatioTracker: rolling IR = μ(bias)/σ(bias).
if(LLMQUANT_ENABLE_IR_TRACKER)
    target_sources(llmquant PRIVATE src/SignalInformationRatioTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_IR_TRACKER_ENABLED)
    message(STATUS "Feature: SignalInformationRatioTracker ENABLED (rolling IR quality index)")
else()
    message(STATUS "Feature: SignalInformationRatioTracker DISABLED (LLMQUANT_ENABLE_IR_TRACKER=OFF)")
endif()

# NarrativeConsistencyMeter: AAD-based bias smoothness score.
if(LLMQUANT_ENABLE_CONSISTENCY_METER)
    target_sources(llmquant PRIVATE src/NarrativeConsistencyMeter.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CONSISTENCY_METER_ENABLED)
    message(STATUS "Feature: NarrativeConsistencyMeter ENABLED (AAD consistency, on_inconsistent callback)")
else()
    message(STATUS "Feature: NarrativeConsistencyMeter DISABLED (LLMQUANT_ENABLE_CONSISTENCY_METER=OFF)")
endif()

# SignalOscillationDetector: zero-crossing rate for +/- flip-flop detection.
if(LLMQUANT_ENABLE_OSCILLATION_DETECTOR)
    target_sources(llmquant PRIVATE src/SignalOscillationDetector.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_OSCILLATION_DETECTOR_ENABLED)
    message(STATUS "Feature: SignalOscillationDetector ENABLED (ZCR oscillation detection)")
else()
    message(STATUS "Feature: SignalOscillationDetector DISABLED (LLMQUANT_ENABLE_OSCILLATION_DETECTOR=OFF)")
endif()

# BiasHysteresisGate: debounce / time-persistence gate for bias direction changes.
option(LLMQUANT_ENABLE_BIAS_HYSTERESIS "Build BiasHysteresisGate: N-tick debounce that requires bias to sustain above threshold before opening, preventing whipsaw" ON)
if(LLMQUANT_ENABLE_BIAS_HYSTERESIS)
    target_sources(llmquant PRIVATE src/BiasHysteresisGate.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_BIAS_HYSTERESIS_ENABLED)
    message(STATUS "Feature: BiasHysteresisGate ENABLED (N-tick debounce, entry/exit hysteresis)")
else()
    message(STATUS "Feature: BiasHysteresisGate DISABLED (LLMQUANT_ENABLE_BIAS_HYSTERESIS=OFF)")
endif()

# RealizedVolatilityTracker: rolling realized-volatility estimator for bias signal.
option(LLMQUANT_ENABLE_REALIZED_VOL "Build RealizedVolatilityTracker: rolling sqrt(mean squared return) with high-vol alerts for dynamic position sizing" ON)
if(LLMQUANT_ENABLE_REALIZED_VOL)
    target_sources(llmquant PRIVATE src/RealizedVolatilityTracker.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_REALIZED_VOL_ENABLED)
    message(STATUS "Feature: RealizedVolatilityTracker ENABLED (rolling RV, high/low-vol callbacks)")
else()
    message(STATUS "Feature: RealizedVolatilityTracker DISABLED (LLMQUANT_ENABLE_REALIZED_VOL=OFF)")
endif()

# SignalCausalChainTracer: traces top-N token contributions that caused a signal.
option(LLMQUANT_ENABLE_CAUSAL_TRACER "Build SignalCausalChainTracer: rolling (token,contribution) ring ranked by impact for signal explainability" ON)
if(LLMQUANT_ENABLE_CAUSAL_TRACER)
    target_sources(llmquant PRIVATE src/SignalCausalChainTracer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CAUSAL_TRACER_ENABLED)
    message(STATUS "Feature: SignalCausalChainTracer ENABLED (token causal chain, strong-token callback)")
else()
    message(STATUS "Feature: SignalCausalChainTracer DISABLED (LLMQUANT_ENABLE_CAUSAL_TRACER=OFF)")
endif()

# TokenDependencyMapper: sliding-window co-occurrence matrix with clique cluster detection.
if(LLMQUANT_ENABLE_DEPENDENCY_MAPPER)
    target_sources(llmquant PRIVATE src/TokenDependencyMapper.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_DEPENDENCY_MAPPER_ENABLED)
    message(STATUS "Feature: TokenDependencyMapper ENABLED (token co-occurrence, cluster detection)")
else()
    message(STATUS "Feature: TokenDependencyMapper DISABLED (LLMQUANT_ENABLE_DEPENDENCY_MAPPER=OFF)")
endif()

# BiasFrequencyAnalyser: DCT-II dominant oscillation frequency extractor.
if(LLMQUANT_ENABLE_FREQ_ANALYSER)
    target_sources(llmquant PRIVATE src/BiasFrequencyAnalyser.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_FREQ_ANALYSER_ENABLED)
    message(STATUS "Feature: BiasFrequencyAnalyser ENABLED (DCT dominant frequency, on_change callback)")
else()
    message(STATUS "Feature: BiasFrequencyAnalyser DISABLED (LLMQUANT_ENABLE_FREQ_ANALYSER=OFF)")
endif()

# SignalEntropyRatchet: one-way ratchet on Shannon entropy of bias distribution.
if(LLMQUANT_ENABLE_ENTROPY_RATCHET)
    target_sources(llmquant PRIVATE src/SignalEntropyRatchet.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ENTROPY_RATCHET_ENABLED)
    message(STATUS "Feature: SignalEntropyRatchet ENABLED (entropy floor ratchet, spike/floor-drop callbacks)")
else()
    message(STATUS "Feature: SignalEntropyRatchet DISABLED (LLMQUANT_ENABLE_ENTROPY_RATCHET=OFF)")
endif()

# NarrativeCoherenceScorer: |μ|/σ signal-to-noise coherence on rolling bias window.
if(LLMQUANT_ENABLE_COHERENCE_SCORER)
    target_sources(llmquant PRIVATE src/NarrativeCoherenceScorer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_COHERENCE_SCORER_ENABLED)
    message(STATUS "Feature: NarrativeCoherenceScorer ENABLED (|mu|/sigma coherence, drop/recover callbacks)")
else()
    message(STATUS "Feature: NarrativeCoherenceScorer DISABLED (LLMQUANT_ENABLE_COHERENCE_SCORER=OFF)")
endif()

# CrossTokenCorrelationMatrix: rolling Pearson correlation across N bias channels.
if(LLMQUANT_ENABLE_CROSS_TOKEN_CORR)
    target_sources(llmquant PRIVATE src/CrossTokenCorrelationMatrix.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_CROSS_TOKEN_CORR_ENABLED)
    message(STATUS "Feature: CrossTokenCorrelationMatrix ENABLED (pairwise rolling correlation, divergence/convergence callbacks)")
else()
    message(STATUS "Feature: CrossTokenCorrelationMatrix DISABLED (LLMQUANT_ENABLE_CROSS_TOKEN_CORR=OFF)")
endif()

# AdaptivePositionSizer: synthesises coherence/confidence/vol/regime into position multiplier.
if(LLMQUANT_ENABLE_ADAPTIVE_SIZER)
    target_sources(llmquant PRIVATE src/AdaptivePositionSizer.cpp)
    target_compile_definitions(llmquant PUBLIC LLMQUANT_ADAPTIVE_SIZER_ENABLED)
    message(STATUS "Feature: AdaptivePositionSizer ENABLED (multi-factor [0,1] multiplier, on_size_change callback)")
else()
    message(STATUS "Feature: AdaptivePositionSizer DISABLED (LLMQUANT_ENABLE_ADAPTIVE_SIZER=OFF)")
endif()

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(LLMTokenStreamQuantEngine src/main.cpp)
target_link_libraries(LLMTokenStreamQuantEngine PRIVATE llmquant)

# ---------------------------------------------------------------------------
# Coverage instrumentation (GCC/Clang, requires LLMQUANT_ENABLE_COVERAGE=ON)
# ---------------------------------------------------------------------------
if(LLMQUANT_ENABLE_COVERAGE)
    if(MSVC)
        message(WARNING "Coverage instrumentation is not supported with MSVC; ignoring LLMQUANT_ENABLE_COVERAGE")
    else()
        target_compile_options(llmquant PUBLIC --coverage -O0 -g)
        target_link_options(llmquant PUBLIC --coverage)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# LibFuzzer targets (LLMQUANT_ENABLE_FUZZING=ON, clang only)
# fuzz/ subdirectory contains root-level harnesses (fuzz_deduplicator,
# fuzz_token_stream); tests/fuzz/ contains the original pipeline harnesses.
# ---------------------------------------------------------------------------
if(LLMQUANT_ENABLE_FUZZING)
    # fuzz/ builds fuzz_deduplicator and fuzz_token_stream.
    add_subdirectory(fuzz)

    # tests/fuzz/fuzz_pipeline.cpp is the only target not covered by fuzz/.
    # Do NOT add fuzz_deduplicator here — it is already defined above and CMake
    # forbids two add_executable() calls with the same target name.
    add_executable(fuzz_pipeline tests/fuzz/fuzz_pipeline.cpp)
    target_include_directories(fuzz_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(fuzz_pipeline PRIVATE llmquant)
    target_compile_options(fuzz_pipeline PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(   fuzz_pipeline PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

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
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.h.in" EXCLUDE
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/llmquant_version.h"
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
