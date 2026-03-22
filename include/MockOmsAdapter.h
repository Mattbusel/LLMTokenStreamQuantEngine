#pragma once

#include "OmsAdapter.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace llmquant {

/**
 * @brief Deterministic mock OMS adapter for unit and integration testing.
 *
 * Emits a pre-loaded sequence of PositionState updates at a configurable
 * interval when started. Stops automatically when the sequence is exhausted.
 *
 * Typical usage:
 * @code
 * MockOmsAdapter mock;
 * mock.load_states({{0.1, 1.0, 0.5, -10.0}, {-0.2, 1.0, -0.1, -10.0}});
 * mock.set_position_callback([&](const RiskManager::PositionState& s) {
 *     risk_mgr.update_position(s);
 * });
 * mock.start();
 * // wait ...
 * mock.stop();
 * @endcode
 *
 * Thread safety: load_states and set_position_callback must be called before
 * start(). is_running and emitted_count are safe from any thread at all times.
 */
class MockOmsAdapter : public OmsAdapter {
public:
    /**
     * @brief Configuration for the mock emitter.
     */
    struct Config {
        /** @brief Delay between consecutive state emissions. */
        std::chrono::milliseconds emit_interval{10};
    };

    /**
     * @brief Construct the mock adapter with default configuration (10 ms interval).
     */
    explicit MockOmsAdapter();

    /**
     * @brief Construct the mock adapter with the given configuration.
     *
     * @param config Emission timing parameters.
     */
    explicit MockOmsAdapter(Config config);

    /**
     * @brief Destructor calls stop() to join the background thread.
     */
    ~MockOmsAdapter() override;

    /**
     * @brief Pre-load the sequence of PositionState values that will be emitted.
     *
     * @param states Ordered list of states; emitted one per emit_interval.
     */
    void load_states(std::vector<RiskManager::PositionState> states);

    /**
     * @brief Register the callback that receives each emitted PositionState.
     *
     * @param cb Invoked on the emitter thread for each state in the sequence.
     */
    void set_position_callback(PositionCallback cb) override;

    /**
     * @brief Start the background emitter thread.
     *
     * @return false if already running.
     */
    [[nodiscard]] bool start() override;

    /**
     * @brief Signal the emitter to stop and block until the thread exits.
     */
    void stop() override;

    /**
     * @brief Returns true while the background emitter thread is active.
     *
     * @return Running state.
     */
    [[nodiscard]] bool is_running() const override { return running_.load(); }

    /**
     * @brief Returns the string "MockOmsAdapter".
     *
     * @return Adapter description.
     */
    [[nodiscard]] std::string description() const override { return "MockOmsAdapter"; }

    /**
     * @brief Return the number of states emitted so far.
     *
     * @return Emitted state count.
     */
    [[nodiscard]] uint64_t emitted_count() const { return emitted_.load(); }

private:
    /** @brief Main loop executed on the emitter thread. */
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
