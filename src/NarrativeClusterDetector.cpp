#include "NarrativeClusterDetector.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace llmquant {

NarrativeClusterDetector::NarrativeClusterDetector(Config cfg)
    : cfg_(std::move(cfg))
    , centroids_(cfg_.k, 0.0)
{}

void NarrativeClusterDetector::record(double bias) {
    int  new_cluster  = 0;
    bool change_event = false;

    {
        std::lock_guard<std::mutex> lk(mutex_);

        // Initialise centroids by spreading over first k samples
        if (!initialized_) {
            if (init_count_ < cfg_.k) {
                centroids_[init_count_] = bias;
                ++init_count_;
            }
            if (init_count_ >= cfg_.k) {
                initialized_ = true;
                prev_cluster_ = 0;
            }
        }

        if (initialized_) {
            // Find nearest centroid
            double best_dist = std::numeric_limits<double>::max();
            new_cluster = 0;
            for (int c = 0; c < cfg_.k; ++c) {
                double d = std::fabs(bias - centroids_[c]);
                if (d < best_dist) { best_dist = d; new_cluster = c; }
            }

            // Update centroid via EMA
            centroids_[new_cluster] += cfg_.alpha * (bias - centroids_[new_cluster]);

            // Check for cluster change
            bool has_min = static_cast<std::size_t>(cfg_.min_samples) <= total_.load(std::memory_order_relaxed) + 1;
            if (has_min && new_cluster != prev_cluster_) {
                change_event = true;
                change_events_.fetch_add(1, std::memory_order_relaxed);
                prev_cluster_ = new_cluster;
            } else if (prev_cluster_ == -1) {
                prev_cluster_ = new_cluster;
            }
        }
    }

    active_cluster_.store(new_cluster, std::memory_order_relaxed);
    total_.fetch_add(1, std::memory_order_relaxed);

    if (change_event && cfg_.on_cluster_change) {
        double centroid_val = 0.0;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            centroid_val = centroids_[new_cluster];
        }
        cfg_.on_cluster_change(prev_cluster_, new_cluster, centroid_val);
    }
}

int NarrativeClusterDetector::active_cluster() const noexcept {
    return active_cluster_.load(std::memory_order_relaxed);
}

double NarrativeClusterDetector::centroid(int cluster_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (cluster_id < 0 || cluster_id >= cfg_.k) return 0.0;
    return centroids_[cluster_id];
}

std::size_t NarrativeClusterDetector::cluster_change_events() const noexcept {
    return change_events_.load(std::memory_order_relaxed);
}
std::size_t NarrativeClusterDetector::total_records() const noexcept {
    return total_.load(std::memory_order_relaxed);
}

std::string NarrativeClusterDetector::to_stats_json() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\"active_cluster\":" << active_cluster_.load(std::memory_order_relaxed)
       << ",\"cluster_change_events\":" << change_events_.load(std::memory_order_relaxed)
       << ",\"total_records\":" << total_.load(std::memory_order_relaxed)
       << "}";
    return ss.str();
}

void NarrativeClusterDetector::reset() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::fill(centroids_.begin(), centroids_.end(), 0.0);
    initialized_ = false; prev_cluster_ = -1; init_count_ = 0;
    active_cluster_.store(0, std::memory_order_relaxed);
    change_events_.store(0, std::memory_order_relaxed);
    total_.store(0, std::memory_order_relaxed);
}

void NarrativeClusterDetector::update_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_ = cfg;
    centroids_.assign(cfg_.k, 0.0);
    initialized_ = false; prev_cluster_ = -1; init_count_ = 0;
}

} // namespace llmquant
