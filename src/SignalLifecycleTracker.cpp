#include "SignalLifecycleTracker.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <vector>

namespace llmquant {

using Clock = std::chrono::steady_clock;

SignalLifecycleTracker::SignalLifecycleTracker(Config cfg)
    : cfg_(std::move(cfg))
{}

void SignalLifecycleTracker::record_signal(const std::string& id, double bias) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = signals_.find(id);
    if (it == signals_.end()) {
        SignalEntry e;
        e.birth_bias       = bias;
        e.peak_bias        = std::abs(bias);
        e.current_bias     = std::abs(bias);
        e.birth_time       = Clock::now();
        e.halflife_time    = Clock::now();
        e.halflife_recorded = false;
        e.zombie_fired     = false;
        signals_.emplace(id, std::move(e));
        active_count_.store(signals_.size(), std::memory_order_relaxed);
    } else {
        double ab = std::abs(bias);
        it->second.current_bias = ab;
        if (ab > it->second.peak_bias) {
            it->second.peak_bias = ab;
        }
    }
}

void SignalLifecycleTracker::tick() {
    std::vector<std::string> to_remove;

    // Callbacks collected inside lock.
    struct DeathEvent { std::string id; double hl; double peak; };
    struct ZombieEvent { std::string id; double age; };
    std::vector<DeathEvent>  deaths;
    std::vector<ZombieEvent> zombies;

    std::function<void(const std::string&, double, double)> death_cb;
    std::function<void(const std::string&, double)>         zombie_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        death_cb  = cfg_.on_death;
        zombie_cb = cfg_.on_zombie;

        auto now = Clock::now();

        for (auto& [id, e] : signals_) {
            double age_s = std::chrono::duration<double>(now - e.birth_time).count();

            // Half-life recording.
            if (!e.halflife_recorded &&
                e.current_bias <= e.peak_bias * cfg_.halflife_fraction) {
                e.halflife_recorded = true;
                e.halflife_time     = now;
                double hl = std::chrono::duration<double>(now - e.birth_time).count();
                halflife_sum_ += hl;
                ++halflife_n_;
                if (halflife_n_ > 0)
                    mean_halflife_.store(halflife_sum_ / static_cast<double>(halflife_n_),
                                        std::memory_order_relaxed);
            }

            // Death detection.
            if (e.current_bias <= e.peak_bias * cfg_.decay_fraction) {
                double hl = std::chrono::duration<double>(now - e.birth_time).count();
                deaths.push_back({id, hl, e.peak_bias});
                to_remove.push_back(id);
                dead_count_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Zombie detection.
            if (!e.zombie_fired && age_s > cfg_.max_age_s) {
                e.zombie_fired = true;
                zombies.push_back({id, age_s});
                zombie_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        for (auto& k : to_remove) signals_.erase(k);
        active_count_.store(signals_.size(), std::memory_order_relaxed);
    }

    // Fire callbacks outside lock.
    for (auto& d : deaths)
        if (death_cb) death_cb(d.id, d.hl, d.peak);
    for (auto& z : zombies)
        if (zombie_cb) zombie_cb(z.id, z.age);
}

std::string SignalLifecycleTracker::to_stats_json() const {
    std::size_t ac;
    uint64_t dc, zc;
    double mh;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        ac = active_count_.load(std::memory_order_relaxed);
        dc = dead_count_.load(std::memory_order_relaxed);
        zc = zombie_count_.load(std::memory_order_relaxed);
        mh = mean_halflife_.load(std::memory_order_relaxed);
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"active_signal_count\":" << ac << ","
       << "\"dead_count\":"          << dc << ","
       << "\"zombie_count\":"        << zc << ","
       << "\"mean_halflife_s\":"     << mh
       << "}";
    return ss.str();
}

void SignalLifecycleTracker::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    signals_.clear();
    halflife_sum_ = 0.0;
    halflife_n_   = 0;
    active_count_.store(0, std::memory_order_relaxed);
    dead_count_.store(0, std::memory_order_relaxed);
    zombie_count_.store(0, std::memory_order_relaxed);
    mean_halflife_.store(0.0, std::memory_order_relaxed);
}

void SignalLifecycleTracker::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    signals_.clear();
    halflife_sum_ = 0.0;
    halflife_n_   = 0;
    active_count_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
