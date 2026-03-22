#include "ReinforcementSignalWeighter.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llmquant {

ReinforcementSignalWeighter::ReinforcementSignalWeighter(Config cfg)
    : cfg_(std::move(cfg))
{
    for (const auto& spec : cfg_.arms) {
        ArmState s;
        s.name   = spec.name;
        s.weight = spec.initial_weight;
        arms_.push_back(std::move(s));
    }
    if (!arms_.empty()) dominant_idx_ = 0;
}

int ReinforcementSignalWeighter::find_arm(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(arms_.size()); ++i) {
        if (arms_[static_cast<std::size_t>(i)].name == name) return i;
    }
    return -1;
}

void ReinforcementSignalWeighter::recompute_weights_locked() {
    uint64_t T = total_.load(std::memory_order_relaxed);
    if (T == 0) return;

    double log_T = std::log(static_cast<double>(T));

    // Compute UCB scores
    for (auto& arm : arms_) {
        if (arm.pulls == 0) {
            arm.ucb_score = 1e9;  // Unseen arm → infinite optimism
        } else {
            double expl = cfg_.exploration_c *
                          std::sqrt(log_T / static_cast<double>(arm.pulls));
            arm.ucb_score = arm.mean_reward + expl;
        }
    }

    // Softmax over UCB scores
    double temp = std::max(1e-9, cfg_.softmax_temp);
    double max_ucb = arms_[0].ucb_score;
    for (const auto& arm : arms_) max_ucb = std::max(max_ucb, arm.ucb_score);

    double sum_exp = 0.0;
    for (auto& arm : arms_) {
        arm.weight = std::exp((arm.ucb_score - max_ucb) / temp);
        sum_exp   += arm.weight;
    }
    if (sum_exp > 0.0) {
        for (auto& arm : arms_) arm.weight /= sum_exp;
    }
}

void ReinforcementSignalWeighter::update(const std::string& arm_name, double reward) {
    std::string new_dominant;
    bool        fire_change = false;
    std::function<void(const std::string&)> cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        cb = cfg_.on_dominant_arm_change;

        int idx = find_arm(arm_name);
        if (idx < 0) return;

        total_.fetch_add(1, std::memory_order_relaxed);
        auto& arm = arms_[static_cast<std::size_t>(idx)];
        ++arm.pulls;
        arm.sum_reward  += reward;
        arm.mean_reward  = arm.sum_reward / static_cast<double>(arm.pulls);

        recompute_weights_locked();

        // Find new dominant arm (highest UCB score)
        int new_dom = 0;
        for (int i = 1; i < static_cast<int>(arms_.size()); ++i) {
            if (arms_[static_cast<std::size_t>(i)].ucb_score >
                arms_[static_cast<std::size_t>(new_dom)].ucb_score)
                new_dom = i;
        }
        if (new_dom != dominant_idx_) {
            dominant_idx_ = new_dom;
            dom_changes_.fetch_add(1, std::memory_order_relaxed);
            new_dominant = arms_[static_cast<std::size_t>(new_dom)].name;
            fire_change  = true;
        }
    }

    if (fire_change && cb) try { cb(new_dominant); } catch (...) {}
}

double ReinforcementSignalWeighter::weight(const std::string& arm_name) const {
    std::lock_guard<std::mutex> lk(mutex_);
    int idx = find_arm(arm_name);
    if (idx < 0) return 0.0;
    return arms_[static_cast<std::size_t>(idx)].weight;
}

std::string ReinforcementSignalWeighter::dominant_arm() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (dominant_idx_ < 0 || dominant_idx_ >= static_cast<int>(arms_.size()))
        return "";
    return arms_[static_cast<std::size_t>(dominant_idx_)].name;
}

std::string ReinforcementSignalWeighter::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{"
       << "\"total_updates\":"      << total_.load(std::memory_order_relaxed) << ","
       << "\"dominant_arm_changes\":" << dom_changes_.load(std::memory_order_relaxed) << ","
       << "\"dominant_arm\":\""
       << (dominant_idx_ >= 0 ? arms_[static_cast<std::size_t>(dominant_idx_)].name : "") << "\","
       << "\"arms\":[";
    for (std::size_t i = 0; i < arms_.size(); ++i) {
        if (i > 0) ss << ",";
        const auto& arm = arms_[i];
        ss << "{\"name\":\"" << arm.name << "\","
           << "\"weight\":"      << arm.weight      << ","
           << "\"mean_reward\":" << arm.mean_reward  << ","
           << "\"pulls\":"       << arm.pulls        << "}";
    }
    ss << "]}";
    return ss.str();
}

void ReinforcementSignalWeighter::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& arm : arms_) {
        arm.mean_reward = arm.sum_reward = arm.ucb_score = 0.0;
        arm.pulls  = 0;
        arm.weight = 1.0 / static_cast<double>(arms_.size() > 0 ? arms_.size() : 1);
    }
    dominant_idx_ = arms_.empty() ? -1 : 0;
    total_.store(0, std::memory_order_relaxed);
    dom_changes_.store(0, std::memory_order_relaxed);
}

void ReinforcementSignalWeighter::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    arms_.clear();
    for (const auto& spec : cfg_.arms) {
        ArmState s;
        s.name   = spec.name;
        s.weight = spec.initial_weight;
        arms_.push_back(std::move(s));
    }
    dominant_idx_ = arms_.empty() ? -1 : 0;
    total_.store(0, std::memory_order_relaxed);
    dom_changes_.store(0, std::memory_order_relaxed);
}

} // namespace llmquant
