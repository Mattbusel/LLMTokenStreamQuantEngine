#include "KVCacheManager.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llmquant {

// ─── CONSTRUCTOR / DESTRUCTOR ────────────────────────────────────────────────

KVCacheManager::KVCacheManager(int num_layers, int block_size, int max_blocks,
                               size_t bytes_per_block)
    : num_layers_(num_layers),
      block_size_(block_size),
      max_blocks_(max_blocks),
      bytes_per_block_(bytes_per_block),
      buffer_(nullptr) {

    if (max_blocks_ <= 0) {
        throw std::invalid_argument("KVCacheManager: max_blocks must be > 0");
    }
    if (bytes_per_block_ == 0) {
        throw std::invalid_argument("KVCacheManager: bytes_per_block must be > 0");
    }

    // Allocate the flat backing buffer.
    const size_t total_bytes = static_cast<size_t>(max_blocks_) * bytes_per_block_;
    buffer_ = static_cast<uint8_t*>(std::malloc(total_bytes));
    if (!buffer_) {
        throw std::bad_alloc();
    }
    std::memset(buffer_, 0, total_bytes);

    // Initialise block metadata.
    blocks_.resize(static_cast<size_t>(max_blocks_));
    for (int i = 0; i < max_blocks_; ++i) {
        auto& b = blocks_[static_cast<size_t>(i)];
        b.block_id       = i;
        b.seq_id         = -1;
        b.layer_id       = -1;
        b.is_free        = true;
        b.data_size_bytes = bytes_per_block_;
        b.data_ptr       = buffer_ + static_cast<std::ptrdiff_t>(i) *
                                     static_cast<std::ptrdiff_t>(bytes_per_block_);
    }
}

KVCacheManager::~KVCacheManager() {
    std::free(buffer_);
    buffer_ = nullptr;
}

// ─── ALLOCATION ──────────────────────────────────────────────────────────────

int KVCacheManager::allocate_block(int seq_id, int layer_id) {
    for (auto& b : blocks_) {
        if (b.is_free) {
            b.is_free  = false;
            b.seq_id   = seq_id;
            b.layer_id = layer_id;
            return b.block_id;
        }
    }
    return -1; // cache full
}

void KVCacheManager::free_block(int block_id) {
    if (block_id < 0 || block_id >= max_blocks_) {
        return;
    }
    auto& b = blocks_[static_cast<std::size_t>(block_id)];
    b.is_free  = true;
    b.seq_id   = -1;
    b.layer_id = -1;
}

void KVCacheManager::free_sequence(int seq_id) {
    for (auto& b : blocks_) {
        if (!b.is_free && b.seq_id == seq_id) {
            b.is_free  = true;
            b.seq_id   = -1;
            b.layer_id = -1;
        }
    }
}

// ─── ACCESSORS ───────────────────────────────────────────────────────────────

CacheBlock* KVCacheManager::get_block(int block_id) {
    if (block_id < 0 || block_id >= max_blocks_) {
        return nullptr;
    }
    return &blocks_[static_cast<std::size_t>(block_id)];
}

const CacheBlock* KVCacheManager::get_block(int block_id) const {
    if (block_id < 0 || block_id >= max_blocks_) {
        return nullptr;
    }
    return &blocks_[static_cast<std::size_t>(block_id)];
}

KVCacheStats KVCacheManager::stats() const {
    int used = 0;
    for (const auto& b : blocks_) {
        if (!b.is_free) ++used;
    }
    int free_count = max_blocks_ - used;
    float frag = (max_blocks_ > 0)
        ? static_cast<float>(free_count) / static_cast<float>(max_blocks_)
        : 0.0f;
    return KVCacheStats{max_blocks_, used, free_count, frag};
}

std::vector<int> KVCacheManager::get_sequence_blocks(int seq_id) const {
    std::vector<int> result;
    for (const auto& b : blocks_) {
        if (!b.is_free && b.seq_id == seq_id) {
            result.push_back(b.block_id);
        }
    }
    // Sort by layer_id for a deterministic, layer-ordered result.
    std::sort(result.begin(), result.end(),
              [this](int a, int b_id) {
                  return blocks_[static_cast<std::size_t>(a)].layer_id <
                         blocks_[static_cast<std::size_t>(b_id)].layer_id;
              });
    return result;
}

} // namespace llmquant
