#include "TokenInfluenceDecay.h"
#include <sstream>
#include <iomanip>

namespace llmquant {

TokenInfluenceDecay::TokenInfluenceDecay(Config cfg)
    : cfg_(std::move(cfg))
{}

void TokenInfluenceDecay::record(double bias) {
    double new_pos = 0.0, new_neg = 0.0, new_net = 0.0;
    bool pos_ev = false, neg_ev = false;
    bool new_pos_dom = false, new_neg_dom = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        if (bias >= 0.0) {
            ema_pos_ += cfg_.alpha_pos * (bias - ema_pos_);
            // Negative EMA decays toward zero
            ema_neg_ *= (1.0 - cfg_.alpha_neg);
        } else {
            ema_neg_ += cfg_.alpha_neg * (-bias - ema_neg_);
            ema_pos_ *= (1.0 - cfg_.alpha_pos);
        }

        new_pos = ema_pos_;
        new_neg = ema_neg_;
        new_net = new_pos - new_neg;

        if (total_.load(std::memory_order_relaxed) + 1 >= static_cast<std::size_t>(cfg_.min_samples)) {
            new_pos_dom = (new_net > cfg_.dominance_threshold);
            new_neg_dom = (new_net < -cfg_.dominance_threshold);

            if (new_pos_dom && !prev_pos_dom_) {
                pos_ev = true;
                pos_dom_events_.fetch_add(1, std::memory_order_relaxed);
            }
            if (new_neg_dom && !prev_neg_dom_) {
                neg_ev = true;
                neg_dom_events_.fetch_add(1, std::memory_order_relaxed);
            }
            prev_pos_dom_ = new_pos_dom;
            prev_neg_dom_ = new_neg_dom;
        }
    }

    pos_inf_.store(new_pos, std::memory_order_relaxed);
    neg_inf_.store(new_neg, std::memory_order_relaxed);
    net_inf_.store(new_net, std::memory_order_relaxed);
    pos_dom_.store(new_pos_dom, std::memory_order_relaxed);
    neg_dom_.store(new_neg_dom, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (pos_ev && cfg_.on_positive_dominant) cfg_.on_positive_dominant(new_net);
    if (neg_ev && cfg_.on_negative_dominant) cfg_.on_negative_dominant(new_net);
}

double TokenInfluenceDecay::pos_influence()  const noexcept { return pos_inf_.load(std::memory_order_relaxed); }
double TokenInfluenceDecay::neg_influence()  const noexcept { return neg_inf_.load(std::memory_order_relaxed); }
double TokenInfluenceDecay::net_influence()  const noexcept { return net_inf_.load(std::memory_order_relaxed); }

bool TokenInfluenceDecay::is_positive_dominant() const noexcept { return pos_dom_.load(std::memory_order_relaxed); }
bool TokenInfluenceDecay::is_negative_dominant() const noexcept { return neg_dom_.load(std::memory_order_relaxed); }

std::size_t TokenInfluenceDecay::positive_dominant_events() const noexcept { return pos_dom_events_.load(std::memory_order_relaxed); }
std::size_t TokenInfluenceDecay::negative_dominant_events() const noexcept { return neg_dom_events_.load(std::memory_order_relaxed); }
std::size_t TokenInfluenceDecay::total_records()            const noexcept { return total_.load(std::memory_order_relaxed); }

std::string TokenInfluenceDecay::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"pos_influence\":" << pos_inf_.load(std::memory_order_relaxed)
       << ",\"neg_influence\":" << neg_inf_.load(std::memory_order_relaxed)
       << ",\"net_influence\":" << net_inf_.load(std::memory_order_relaxed)
       << ",\"positive_dominant_events\":" << pos_dom_events_.load(std::memory_order_relaxed)
       << ",\"negative_dominant_events\":" << neg_dom_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void TokenInfluenceDecay::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    ema_pos_ = 0.0; ema_neg_ = 0.0;
    prev_pos_dom_ = false; prev_neg_dom_ = false;
    pos_inf_.store(0.0, std::memory_order_relaxed);
    neg_inf_.store(0.0, std::memory_order_relaxed);
    net_inf_.store(0.0, std::memory_order_relaxed);
    pos_dom_.store(false, std::memory_order_relaxed);
    neg_dom_.store(false, std::memory_order_relaxed);
    pos_dom_events_.store(0, std::memory_order_relaxed);
    neg_dom_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void TokenInfluenceDecay::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
}

} // namespace llmquant
