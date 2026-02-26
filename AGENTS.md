# AGENTS.md — Multi-Agent Workflow

## Work Decomposition

Modules are independent after Config and LLMAdapter are stable.

## Parallel vs Sequential

Sequential (must complete in order):
1. Config + headers first (everything depends on these)
2. LLMAdapter (TradeSignalEngine depends on SemanticWeight)
3. All other modules can parallelize

Parallel (safe to build simultaneously):
- TokenStreamSimulator + LatencyController
- MetricsLogger + OutputSink
- All unit test files (one per module)

## Agent Ownership

| Agent   | Module(s)                        | Test Files                                                     |
|---------|----------------------------------|----------------------------------------------------------------|
| Agent-1 | Config, headers                  | tests/unit/test_config.cpp                                     |
| Agent-2 | LLMAdapter                       | tests/unit/test_llm_adapter.cpp                                |
| Agent-3 | LatencyController, pressure system | tests/unit/test_latency_controller.cpp                       |
| Agent-4 | MetricsLogger, OutputSink        | tests/unit/test_metrics_logger.cpp, test_output_sink.cpp       |
| Agent-5 | TokenStreamSimulator             | tests/unit/test_token_stream_simulator.cpp                     |
| Agent-6 | TradeSignalEngine                | tests/unit/test_trade_signal_engine.cpp                        |
| Agent-7 | Integration + performance tests  | tests/integration/, tests/performance/                         |

## Cross-Module Dependencies

- TradeSignalEngine depends on LLMAdapter (SemanticWeight type)
- OutputSink depends on TradeSignalEngine (TradeSignal type)
- main.cpp depends on all modules
- Tests depend on their module + any mocked dependencies

## Progress Communication

Each agent updates the Module Status table in CLAUDE.md when a module is done.

## Definition of Done Per Module

1. Header fully declared and doc-commented
2. Implementation compiles with zero warnings (-Wall -Wextra -Werror)
3. Unit tests pass
4. At least one integration test exercises this module
5. CLAUDE.md status table updated

## Integration Verification

Run `ctest` from the build directory. All tests must pass before any push.
Integration tests in tests/integration/ verify cross-module correctness.

## Build Isolation (multiple agents on the same machine)

Set a unique CARGO_TARGET_DIR equivalent (CMAKE_BINARY_DIR) per agent:

```bash
mkdir build_agent1 && cd build_agent1
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

Each agent uses its own named build directory to avoid file-lock contention.
