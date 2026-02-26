#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "RiskManager.h"

namespace llmquant {

/// Abstract interface for all OMS position-feed adapters.
///
/// Concrete implementations: RestOmsAdapter (HTTP polling), FixOmsAdapter
/// (minimal FIX 4.2 session reader), MockOmsAdapter (for tests).
///
/// ## Lifecycle
/// construct → set_position_callback → start() → (running) → stop()
///
/// ## Thread Safety
/// start/stop may be called from any thread. The position callback is invoked
/// from the adapter's background thread.
class OmsAdapter {
public:
    /// Invoked on the adapter thread each time a new position snapshot arrives.
    using PositionCallback =
        std::function<void(const RiskManager::PositionState& state)>;

    virtual ~OmsAdapter() = default;

    /// Register the callback that receives position updates.
    ///
    /// # Arguments
    /// * `cb` — Callable invoked on each position update from the adapter thread.
    ///
    /// Must be called before start().
    virtual void set_position_callback(PositionCallback cb) = 0;

    /// Start the adapter (open connection, begin background thread).
    ///
    /// # Returns
    /// `false` if already running or the connection cannot be established.
    virtual bool start() = 0;

    /// Stop the adapter and block until the background thread exits.
    virtual void stop() = 0;

    /// True if the adapter is currently running.
    virtual bool is_running() const = 0;

    /// Human-readable description of the adapter type and endpoint.
    virtual std::string description() const = 0;
};

} // namespace llmquant
