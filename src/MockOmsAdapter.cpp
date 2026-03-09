#include "MockOmsAdapter.h"

namespace llmquant {

MockOmsAdapter::MockOmsAdapter() : MockOmsAdapter(Config{}) {}

MockOmsAdapter::MockOmsAdapter(Config config) : config_(std::move(config)) {}

MockOmsAdapter::~MockOmsAdapter() {
    stop();
}

void MockOmsAdapter::load_states(std::vector<RiskManager::PositionState> states) {
    std::lock_guard<std::mutex> lock(states_mutex_);
    states_ = std::move(states);
}

void MockOmsAdapter::set_position_callback(PositionCallback cb) {
    callback_ = std::move(cb);
}

bool MockOmsAdapter::start() {
    if (running_.load()) return false;
    running_ = true;
    thread_ = std::thread(&MockOmsAdapter::emitter_thread, this);
    return true;
}

void MockOmsAdapter::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void MockOmsAdapter::emitter_thread() {
    // Take a local copy so the caller can call load_states() safely during
    // emission without data races (we don't re-read states_ after this point).
    std::vector<RiskManager::PositionState> local;
    {
        std::lock_guard<std::mutex> lock(states_mutex_);
        local = states_;
    }

    for (const auto& state : local) {
        if (!running_.load()) break;
        if (callback_) callback_(state);
        emitted_++;
        std::this_thread::sleep_for(config_.emit_interval);
    }

    running_ = false;
}

} // namespace llmquant