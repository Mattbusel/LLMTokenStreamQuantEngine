#include <gtest/gtest.h>

#ifdef LLMQUANT_NARRATIVE_TOPIC_CLASSIFIER_ENABLED
#include "NarrativeTopicClassifier.h"
#include <atomic>
#include <thread>
using namespace llmquant;

TEST(NarrativeTopicClassifierTest, InitialState) {
    NarrativeTopicClassifier c;
    EXPECT_TRUE(c.dominant_topic().empty());
    EXPECT_NEAR(c.dominant_multiplier(), 1.0, 1e-9);
    EXPECT_EQ(c.classified_count(), 0u);
}

TEST(NarrativeTopicClassifierTest, RegisterAndClassify) {
    NarrativeTopicClassifier c;
    NarrativeTopicClassifier::TopicDef earn{"earnings", 0.8, 1.5};
    NarrativeTopicClassifier::TopicDef macro{"macro",   -0.3, 0.8};
    EXPECT_TRUE(c.register_topic(earn));
    EXPECT_TRUE(c.register_topic(macro));

    // High-weight token → should land in earnings (centroid 0.8).
    int idx = c.classify("beat", 0.9);
    EXPECT_EQ(idx, 0);

    // Low/negative weight → macro (centroid -0.3).
    idx = c.classify("recession", -0.4);
    EXPECT_EQ(idx, 1);
}

TEST(NarrativeTopicClassifierTest, DominantTopicEmergence) {
    NarrativeTopicClassifier::Config cfg;
    cfg.min_warmup = 5;
    NarrativeTopicClassifier c(cfg);
    NarrativeTopicClassifier::TopicDef earn{"earnings", 0.8, 1.5};
    NarrativeTopicClassifier::TopicDef macro{"macro",  -0.3, 0.8};
    c.register_topic(earn);
    c.register_topic(macro);

    for (int i = 0; i < 20; ++i) c.classify("tok", 0.9);  // all → earnings
    EXPECT_EQ(c.dominant_topic(), "earnings");
    EXPECT_NEAR(c.dominant_multiplier(), 1.5, 1e-9);
}

TEST(NarrativeTopicClassifierTest, TopicChangeCallbackFires) {
    NarrativeTopicClassifier::Config cfg;
    cfg.min_warmup = 3;
    cfg.freq_alpha = 0.5;
    std::atomic<int> fires{0};
    cfg.on_topic_change = [&](const std::string&, const std::string&, double) {
        ++fires;
    };
    NarrativeTopicClassifier c(cfg);
    NarrativeTopicClassifier::TopicDef t1{"a", 1.0, 1.0};
    NarrativeTopicClassifier::TopicDef t2{"b", -1.0, 1.0};
    c.register_topic(t1);
    c.register_topic(t2);

    for (int i = 0; i < 10; ++i) c.classify("x", 1.0);   // a dominant
    for (int i = 0; i < 10; ++i) c.classify("x", -1.0);  // b dominant
    EXPECT_GE(fires.load(), 1);
}

TEST(NarrativeTopicClassifierTest, SignalMultiplierByName) {
    NarrativeTopicClassifier c;
    c.register_topic({"geo", 0.0, 2.0});
    EXPECT_NEAR(c.signal_multiplier("geo"), 2.0, 1e-9);
    EXPECT_NEAR(c.signal_multiplier("unknown"), 1.0, 1e-9);
}

TEST(NarrativeTopicClassifierTest, TopicFrequencyNonNegative) {
    NarrativeTopicClassifier c;
    c.register_topic({"x", 0.0, 1.0});
    for (int i = 0; i < 10; ++i) c.classify("t", 0.0);
    EXPECT_GE(c.topic_frequency(0), 0.0);
}

TEST(NarrativeTopicClassifierTest, MaxTopicsLimit) {
    NarrativeTopicClassifier c;
    for (int i = 0; i < NarrativeTopicClassifier::kMaxTopics; ++i)
        EXPECT_TRUE(c.register_topic({std::to_string(i), 0.0, 1.0}));
    EXPECT_FALSE(c.register_topic({"overflow", 0.0, 1.0}));
}

TEST(NarrativeTopicClassifierTest, ResetKeepsTopics) {
    NarrativeTopicClassifier::Config cfg;
    cfg.min_warmup = 2;
    NarrativeTopicClassifier c(cfg);
    c.register_topic({"t", 0.0, 1.0});
    for (int i = 0; i < 10; ++i) c.classify("tok", 0.0);
    c.reset();
    EXPECT_EQ(c.classified_count(), 0u);
    EXPECT_TRUE(c.dominant_topic().empty());
    // Topics still accessible.
    EXPECT_NEAR(c.signal_multiplier("t"), 1.0, 1e-9);
}

TEST(NarrativeTopicClassifierTest, StatsJsonNonEmpty) {
    NarrativeTopicClassifier c;
    c.register_topic({"earnings", 0.5, 1.2});
    c.classify("tok", 0.5);
    std::string j = c.to_stats_json();
    EXPECT_NE(j.find("dominant_topic"), std::string::npos);
    EXPECT_NE(j.find("topics"), std::string::npos);
}

TEST(NarrativeTopicClassifierTest, ConcurrentClassifySafe) {
    NarrativeTopicClassifier c;
    c.register_topic({"a", 1.0, 1.0});
    c.register_topic({"b", -1.0, 1.0});
    std::atomic<bool> go{false};
    auto worker = [&] {
        while (!go.load()) {}
        for (int i = 0; i < 200; ++i)
            c.classify("t", (i % 2 == 0) ? 1.0 : -1.0);
    };
    std::thread t1(worker), t2(worker);
    go.store(true);
    t1.join(); t2.join();
    EXPECT_EQ(c.classified_count(), 400u);
}

#else
TEST(NarrativeTopicClassifierTest, DisabledAtBuildTime) { SUCCEED(); }
#endif
