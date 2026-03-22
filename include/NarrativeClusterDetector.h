#pragma once
/// NarrativeClusterDetector — online k-means-like bias regime classifier.
/// Maintains k centroids updated via EMA. Each new sample is assigned to the
/// nearest centroid. Fires on_cluster_change when the active cluster changes.
///
/// Feature flag: LLMQUANT_ENABLE_CLUSTER_DETECTOR / LLMQUANT_CLUSTER_DETECTOR_ENABLED

#ifdef LLMQUANT_CLUSTER_DETECTOR_ENABLED

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class NarrativeClusterDetector {
public:
    struct Config {
        int    k              = 3;     ///< number of clusters
        double alpha          = 0.1;   ///< EMA smoothing for centroid update
        int    min_samples    = 10;    ///< minimum before firing events

        /// Fires when active cluster assignment changes.
        /// Arguments: old_cluster, new_cluster, centroid_value
        std::function<void(int old_cluster, int new_cluster, double centroid)> on_cluster_change;
    };

    explicit NarrativeClusterDetector(Config cfg = {});

    void record(double bias);

    int    active_cluster()                   const noexcept;
    double centroid(int cluster_id)           const;  ///< centroid value for cluster
    std::size_t cluster_change_events()       const noexcept;
    std::size_t total_records()               const noexcept;

    std::string to_stats_json() const;
    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    std::vector<double> centroids_;
    bool   initialized_ = false;
    int    prev_cluster_ = -1;
    int    init_count_   = 0;

    std::atomic<int>         active_cluster_{0};
    std::atomic<std::size_t> change_events_{0};
    std::atomic<std::size_t> total_{0};
};

} // namespace llmquant

#endif // LLMQUANT_CLUSTER_DETECTOR_ENABLED
