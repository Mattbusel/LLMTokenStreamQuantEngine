#include <gtest/gtest.h>

#ifdef LLMQUANT_ALERT_DISPATCHER_ENABLED

#include "AlertDispatcher.h"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace llmquant;

namespace {

TEST(AlertDispatcher, DefaultConstructsNotRunning) {
    AlertDispatcher ad;
    EXPECT_FALSE(ad.is_running());
    EXPECT_EQ(ad.total_posted(), 0u);
}

TEST(AlertDispatcher, StartAndStop) {
    AlertDispatcher ad;
    EXPECT_TRUE(ad.start());
    EXPECT_TRUE(ad.is_running());
    ad.stop();
    EXPECT_FALSE(ad.is_running());
}

TEST(AlertDispatcher, DoubleStartReturnsFalse) {
    AlertDispatcher ad;
    EXPECT_TRUE(ad.start());
    EXPECT_FALSE(ad.start());
    ad.stop();
}

TEST(AlertDispatcher, StopWithoutStartIsNoOp) {
    AlertDispatcher ad;
    EXPECT_NO_FATAL_FAILURE(ad.stop());
}

TEST(AlertDispatcher, PostDeliversToSink) {
    AlertDispatcher ad;
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert& a) {
        EXPECT_EQ(a.category, "test");
        EXPECT_EQ(a.message, "hello");
        fired.fetch_add(1);
    });
    (void)ad.start();
    ad.post(AlertDispatcher::Severity::Info, "test", "hello");
    // Give background thread time to dispatch.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ad.stop();
    EXPECT_EQ(fired.load(), 1);
    EXPECT_EQ(ad.dispatched_count(), 1u);
}

TEST(AlertDispatcher, ConvenienceMethodsPost) {
    AlertDispatcher ad;
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
    (void)ad.start();
    ad.info("cat", "info msg");
    ad.warn("cat", "warn msg");
    ad.critical("cat", "crit msg");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ad.stop();
    EXPECT_EQ(fired.load(), 3);
}

TEST(AlertDispatcher, SeverityFilterDropsBelow) {
    AlertDispatcher::Config cfg;
    cfg.min_severity = AlertDispatcher::Severity::Warn;
    AlertDispatcher ad(cfg);
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
    (void)ad.start();
    ad.post(AlertDispatcher::Severity::Debug, "cat", "debug");  // filtered
    ad.post(AlertDispatcher::Severity::Info,  "cat", "info");   // filtered
    ad.post(AlertDispatcher::Severity::Warn,  "cat", "warn");   // passes
    ad.post(AlertDispatcher::Severity::Critical, "cat", "crit"); // passes
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ad.stop();
    EXPECT_EQ(fired.load(), 2);
}

TEST(AlertDispatcher, DedupWindowSuppressesDuplicates) {
    AlertDispatcher::Config cfg;
    cfg.dedup_window_ms = 5000;  // 5 s dedup window
    AlertDispatcher ad(cfg);
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
    (void)ad.start();
    // Post same alert 3 times — only the first should be dispatched.
    ad.post(AlertDispatcher::Severity::Info, "storm", "same msg");
    ad.post(AlertDispatcher::Severity::Info, "storm", "same msg");
    ad.post(AlertDispatcher::Severity::Info, "storm", "same msg");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ad.stop();
    EXPECT_EQ(fired.load(), 1);
    EXPECT_GE(ad.dedup_count(), 2u);
}

TEST(AlertDispatcher, DedupDoesNotSuppressDifferentMessages) {
    AlertDispatcher::Config cfg;
    cfg.dedup_window_ms = 5000;
    AlertDispatcher ad(cfg);
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
    (void)ad.start();
    ad.post(AlertDispatcher::Severity::Info, "cat", "msg A");
    ad.post(AlertDispatcher::Severity::Info, "cat", "msg B");
    ad.post(AlertDispatcher::Severity::Info, "cat", "msg C");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ad.stop();
    EXPECT_EQ(fired.load(), 3);
}

TEST(AlertDispatcher, QueueFullDropsAlert) {
    AlertDispatcher::Config cfg;
    cfg.max_queue_depth = 2;
    AlertDispatcher ad(cfg);
    // Don't start — all posts go to a stopped queue and should count as total_posted.
    // Actually with queue not started, running_ is false so posts are dropped without queueing.
    // Start but hold the sink so queue fills.
    std::mutex hold;
    hold.lock();
    ad.add_sink([&](const AlertDispatcher::Alert&) {
        // Block until we release.
        std::lock_guard<std::mutex> lk(hold);
    });
    (void)ad.start();

    // Post 10 alerts; only 2 fit in queue.
    for (int i = 0; i < 10; ++i)
        ad.post(AlertDispatcher::Severity::Info, "cat", "msg");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    hold.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ad.stop();
    EXPECT_GT(ad.dropped_count(), 0u);
    EXPECT_EQ(ad.total_posted(), 10u);
}

TEST(AlertDispatcher, RemoveAllSinksStopsDelivery) {
    AlertDispatcher ad;
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
    (void)ad.start();
    ad.post(AlertDispatcher::Severity::Info, "cat", "msg1");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    ad.remove_all_sinks();
    ad.post(AlertDispatcher::Severity::Info, "cat", "msg2");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    ad.stop();
    EXPECT_EQ(fired.load(), 1);  // only msg1 was dispatched to a sink
}

TEST(AlertDispatcher, SeverityNameReturnsString) {
    EXPECT_STREQ(AlertDispatcher::severity_name(AlertDispatcher::Severity::Debug),    "DEBUG");
    EXPECT_STREQ(AlertDispatcher::severity_name(AlertDispatcher::Severity::Info),     "INFO");
    EXPECT_STREQ(AlertDispatcher::severity_name(AlertDispatcher::Severity::Warn),     "WARN");
    EXPECT_STREQ(AlertDispatcher::severity_name(AlertDispatcher::Severity::Critical), "CRITICAL");
}

TEST(AlertDispatcher, ToStatsJsonContainsKeys) {
    AlertDispatcher ad;
    (void)ad.start();
    ad.post(AlertDispatcher::Severity::Info, "test", "hello");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    ad.stop();
    std::string json = ad.to_stats_json();
    EXPECT_NE(json.find("total_posted"), std::string::npos);
    EXPECT_NE(json.find("dropped"), std::string::npos);
    EXPECT_NE(json.find("dispatched"), std::string::npos);
    EXPECT_NE(json.find("running"), std::string::npos);
}

TEST(AlertDispatcher, AddSpdlogSinkDoesNotCrash) {
    AlertDispatcher ad;
    ad.add_spdlog_sink();
    (void)ad.start();
    ad.info("unit_test", "spdlog sink smoke test");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    ad.stop();
    EXPECT_GE(ad.dispatched_count(), 1u);
}

TEST(AlertDispatcher, DestructorDrainsQueue) {
    std::atomic<int> fired{0};
    {
        AlertDispatcher ad;
        ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
        (void)ad.start();
        for (int i = 0; i < 5; ++i)
            ad.post(AlertDispatcher::Severity::Info, "cat", std::to_string(i));
        // Destructor should drain and stop.
    }
    EXPECT_EQ(fired.load(), 5);
}

TEST(AlertDispatcher, UpdateConfigChangesSeverityFilter) {
    AlertDispatcher ad;
    std::atomic<int> fired{0};
    ad.add_sink([&](const AlertDispatcher::Alert&) { fired.fetch_add(1); });
    (void)ad.start();

    // Initially accept all.
    ad.info("cat", "pre-update");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Now raise threshold to CRITICAL.
    AlertDispatcher::Config cfg;
    cfg.min_severity = AlertDispatcher::Severity::Critical;
    ad.update_config(cfg);
    ad.info("cat", "filtered out");  // should be filtered
    ad.critical("cat", "passes");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    ad.stop();
    EXPECT_EQ(fired.load(), 2);  // pre-update + critical
}

} // namespace

#else  // LLMQUANT_ALERT_DISPATCHER_ENABLED not defined

TEST(AlertDispatcher, DisabledAtBuildTime) {
    SUCCEED();
}

#endif  // LLMQUANT_ALERT_DISPATCHER_ENABLED
