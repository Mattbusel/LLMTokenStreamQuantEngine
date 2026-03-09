#pragma once

#include "OmsAdapter.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace llmquant {

/// Deterministic mock OMS adapter for unit and integration testing.
///
/// Emits a pre-loaded sequence of PositionState updates at a configurable
/// interval when started. Stops automatically when the sequence is exhausted.
///
/// ## Typical Usage
/// ```cpp
/// MockOmsAdapter mock;
/// mock.load_states({{0.1, 1.0, 0.5, -10.0}, {-0.2, 1.0, -0.1, -10.0}});
/// mock.set_position_callback([&](const RiskManager::PositionState& s) {
///     risk_mgr.update_position(s);
/// });
/// mock.start();
/// // wait ...
/// mock.stop();
/// ```
///
/// ## Thread Safety
/// load_states and set_position_callback must be called before start().
/// is_running and emitted_count are safe from any thread at all times.
class MockOmsAdapter : public OmsAdapter {
public:
    /// Configuration for the mock emitter.
    struct Config {
        /// Delay between consecutive state emissions.
        std::chrono::milliseconds emit_interval{10};
    };

    /// Construct the mock adapter with the given configuration.
    ///
    /// # Arguments
    /// * `config` — Emission timing parameters (default: 10 ms interval).
    explicit MockOmsAdapter(Config config = {});

    /// Destructor: calls stop() to join the background thread.
    ~MockOmsAdapter() override;

    /// Pre-load the sequence of PositionState values that will be emitted.
    ///
    /// # Arguments
    /// * `states` — Ordered list of states; emitted one per emit_interval.
    void load_states(std::vector<RiskManager::PositionState> states);

    /// Register the callback that receives each emitted PositionState.
    ///
    /// # Arguments
    /// * `cb` — Invoked on the emitter thread for each state in the sequence.
    void set_position_callback(PositionCallback cb) override;

    /// Start the background emitter thread.
    ///
    /// # Returns
    /// `false` if already running.
    bool start() override;

    /// Signal the emitter to stop and block until the thread exits.
    void stop() override;

    /// True while the background emitter thread is active.
    bool is_running() const override { return running_.load(); }

    /// Returns the string "MockOmsAdapter".
    std::string description() const override { return "MockOmsAdapter"; }

    /// Return the number of states emitted so far.
    uint64_t emitted_count() const { return emitted_.load(); }

private:
    /// Main loop executed on the emitter thread.
    void emitter_thread();

    Config config_;
    PositionCallback callback_;
    std::vector<RiskManager::PositionState> states_;
    std::mutex states_mutex_;
    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> emitted_{0};
    std::thread thread_;
};

} // namespace llmquant