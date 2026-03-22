#include "SignalCUSUMController.h"

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace llmquant {

SignalCUSUMController::SignalCUSUMController(Config cfg)
    : cfg_(std::move(cfg)) {}

void SignalCUSUMController::update(double x) {
    std::function<void(double)> up_cb, dn_cb;
    double sp_out = 0.0, sn_out = 0.0;
    bool fire_up = false, fire_dn = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        total_.fetch_add(1, std::memory_order_relaxed);

        // Two-sided CUSUM update.
        cur_sp_ = std::max(0.0, cur_sp_ + (x - cfg_.target_mean - cfg_.allowance));
        cur_sn_ = std::max(0.0, cur_sn_ + (cfg_.target_mean - x - cfg_.allowance));

        bool was_alarmed = alarmed_.load(std::memory_order_relaxed);

        // Check alarms.
        if (cur_sp_ > cfg_.threshold) {
            fire_up = true;
            up_alarms_.fetch_add(1, std::memory_order_relaxed);
            up_cb = cfg_.on_upward_shift;
            sp_out = cur_sp_;
            if (cfg_.reset_on_alarm) { cur_sp_ = 0.0; cur_sn_ = 0.0; }
        }
        if (cur_sn_ > cfg_.threshold) {
            fire_dn = true;
            down_alarms_.fetch_add(1, std::memory_order_relaxed);
            dn_cb = cfg_.on_downward_shift;
            sn_out = cur_sn_;
            if (cfg_.reset_on_alarm) { cur_sp_ = 0.0; cur_sn_ = 0.0; }
        }

        bool now_alarmed = (cur_sp_ > cfg_.threshold || cur_sn_ > cfg_.threshold);
        alarmed_.store(fire_up || fire_dn || now_alarmed, std::memory_order_relaxed);
        (void)was_alarmed;

        sp_.store(cur_sp_, std::memory_order_relaxed);
        sn_.store(cur_sn_, std::memory_order_relaxed);
    }

    if (fire_up && up_cb) try { up_cb(sp_out); } catch (...) {}
    if (fire_dn && dn_cb) try { dn_cb(sn_out); } catch (...) {}
}

void SignalCUSUMController::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    cur_sp_ = cur_sn_ = 0.0;
    sp_.store(0.0, std::memory_order_relaxed);
    sn_.store(0.0, std::memory_order_relaxed);
    alarmed_.store(false, std::memory_order_relaxed);
    up_alarms_.store(0, std::memory_order_relaxed);
    down_alarms_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void SignalCUSUMController::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    cur_sp_ = cur_sn_ = 0.0;
    sp_.store(0.0, std::memory_order_relaxed);
    sn_.store(0.0, std::memory_order_relaxed);
    alarmed_.store(false, std::memory_order_relaxed);
    up_alarms_.store(0, std::memory_order_relaxed);
    down_alarms_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

std::string SignalCUSUMController::to_stats_json() const {
    double sp = sp_.load(std::memory_order_relaxed);
    double sn = sn_.load(std::memory_order_relaxed);
    bool   al = alarmed_.load(std::memory_order_relaxed);
    uint64_t u = up_alarms_.load(std::memory_order_relaxed);
    uint64_t d = down_alarms_.load(std::memory_order_relaxed);
    uint64_t t = total_.load(std::memory_order_relaxed);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"cusum_pos\":%.6f,\"cusum_neg\":%.6f,\"is_alarmed\":%s,"
        "\"up_alarms\":%llu,\"down_alarms\":%llu,\"total\":%llu}",
        sp, sn, al ? "true" : "false",
        static_cast<unsigned long long>(u),
        static_cast<unsigned long long>(d),
        static_cast<unsigned long long>(t));
    return buf;
}

} // namespace llmquant
