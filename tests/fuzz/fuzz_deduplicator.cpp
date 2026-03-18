// LibFuzzer target for the token deduplication hot path.
//
// Exercises:
//   1. DedupKey::from_token (FNV-1a hash, token + context)
//   2. InProcessDeduplicator::check_and_register (TTL lookup, emplace, expiry)
//   3. Deduplicator facade check() and evict()
//
// Build:
//   cmake -S . -B build-fuzz \
//     -DLLMQUANT_ENABLE_FUZZING=ON \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer,address,undefined"
//   cmake --build build-fuzz --target fuzz_deduplicator

#include "Deduplicator.h"

#include <chrono>
#include <cstdint>
#include <memory>

using namespace llmquant;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) return 0;

    // Byte 0 is a control byte that steers the TTL and split point.
    const uint8_t control = data[0];
    const size_t  payload = size - 1;

    // Split fuzz bytes into a token string and a context string.
    const size_t mid = (payload == 0) ? 0 : (control % (payload + 1));

    const std::string token  (reinterpret_cast<const char*>(data + 1),         mid);
    const std::string context(reinterpret_cast<const char*>(data + 1 + mid),
                               payload - mid);

    // Vary TTL between 1 ms and 1600 ms using the upper nibble.
    const uint8_t ttl_nibble = (control >> 4) & 0x0F;
    const auto    ttl = std::chrono::milliseconds{(ttl_nibble == 0) ? 1 : ttl_nibble * 100};

    // --- DedupKey::from_token must not crash on any byte sequence ---------------
    const DedupKey key = DedupKey::from_token(token, context);
    (void)key;

    // --- InProcessDeduplicator --------------------------------------------------
    auto backend = std::make_shared<InProcessDeduplicator>();
    Deduplicator dedup(backend, ttl);

    // First check: Novel (key not yet registered).
    (void)dedup.check(token, context);

    // Second check within TTL: Duplicate.
    (void)dedup.check(token, context);

    // Evict then re-check: Novel again.
    dedup.evict(token, context);
    (void)dedup.check(token, context);

    // purge_expired must not crash on a live or empty table.
    dedup.purge_expired();

    // size() must return a valid value.
    (void)backend->size();

    // check_with_ttl on the pre-built key.
    (void)dedup.check_with_ttl(key, std::chrono::milliseconds{1});

    // background purge start/stop exercises the thread destructor path.
    backend->start_background_purge(1);
    backend->stop_background_purge();

    return 0;
}
