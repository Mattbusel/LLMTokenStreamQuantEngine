#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "RiskManager.h"

namespace llmquant {

/**
 * @brief Abstract interface for all OMS position-feed adapters.
 *
 * Concrete implementations: RestOmsAdapter (HTTP polling), FixOmsAdapter
 * (minimal FIX 4.2 session reader), MockOmsAdapter (for tests).
 *
 * Lifecycle: construct -> set_position_callback -> start() -> (running) -> stop()
 *
 * Thread safety: start/stop may be called from any thread. The position
 * callback is invoked from the adapter's background thread.
 */
class OmsAdapter {
public:
    /**
     * @brief Invoked on the adapter thread each time a new position snapshot arrives.
     */
    using PositionCallback =
        std::function<void(const RiskManager::PositionState& state)>;

    virtual ~OmsAdapter() = default;

    /**
     * @brief Register the callback that receives position updates.
     *
     * Must be called before start().
     *
     * @param cb Callable invoked on each position update from the adapter thread.
     */
    virtual void set_position_callback(PositionCallback cb) = 0;

    /**
     * @brief Start the adapter (open connection, begin background thread).
     *
     * @return false if already running or the connection cannot be established.
     */
    virtual bool start() = 0;

    /**
     * @brief Stop the adapter and block until the background thread exits.
     */
    virtual void stop() = 0;

    /**
     * @brief Returns true if the adapter is currently running.
     *
     * @return Running state.
     */
    virtual bool is_running() const = 0;

    /**
     * @brief Human-readable description of the adapter type and endpoint.
     *
     * @return Description string.
     */
    virtual std::string description() const = 0;
};

} // namespace llmquant
