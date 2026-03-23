#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llmquant {

/// A single allocated KV-cache block.
struct CacheBlock {
    int    block_id;        ///< Unique block identifier.
    int    seq_id;          ///< Sequence that owns this block (-1 if free).
    int    layer_id;        ///< Transformer layer index this block holds KV for.
    bool   is_free;         ///< True when this block is available for allocation.
    size_t data_size_bytes; ///< Size of the payload in bytes.
    void*  data_ptr;        ///< Pointer into the pre-allocated flat buffer.
};

/// Aggregate statistics returned by KVCacheManager::stats().
struct KVCacheStats {
    int   total_blocks;          ///< Total number of blocks in the pool.
    int   used_blocks;           ///< Currently allocated blocks.
    int   free_blocks;           ///< Currently free blocks.
    float fragmentation_ratio;   ///< free_blocks / total_blocks (0 = full, 1 = empty).
};

/**
 * @brief Pre-allocated KV-cache block manager for attention key/value tensors.
 *
 * Manages a flat byte buffer split into fixed-size blocks.  Each block is
 * associated with a (seq_id, layer_id) pair.  Callers request blocks via
 * allocate_block() and release them via free_block() or free_sequence().
 *
 * ## Memory layout
 *
 * A single contiguous buffer of size `max_blocks * bytes_per_block` is
 * allocated in the constructor.  Block i occupies bytes
 *   [i * bytes_per_block, (i+1) * bytes_per_block).
 *
 * ## Thread safety
 *
 * This implementation is NOT thread-safe.  External locking is required
 * when multiple threads call allocation methods concurrently.
 *
 * ## Feature flag
 * Controlled by `LLMQUANT_ENABLE_KV_CACHE_MANAGER` (default ON).
 */
class KVCacheManager {
public:
    /**
     * @brief Construct the cache manager and pre-allocate the backing buffer.
     *
     * @param num_layers      Number of transformer layers.
     * @param block_size      Tokens per block (metadata; not used for byte sizing).
     * @param max_blocks      Maximum number of blocks in the pool.
     * @param bytes_per_block Size in bytes of each block's KV tensor data.
     */
    KVCacheManager(int num_layers, int block_size, int max_blocks,
                   size_t bytes_per_block);

    /// Destructor: releases the backing buffer.
    ~KVCacheManager();

    // Non-copyable.
    KVCacheManager(const KVCacheManager&)            = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    // Movable.
    KVCacheManager(KVCacheManager&&)            = default;
    KVCacheManager& operator=(KVCacheManager&&) = default;

    // ----------------------------------------------------------------
    // Allocation
    // ----------------------------------------------------------------

    /**
     * @brief Allocate a free block for a given sequence and layer.
     *
     * Finds the first free block in the pool (linear scan), marks it in-use,
     * and records the seq_id and layer_id.
     *
     * @param seq_id   Sequence identifier.
     * @param layer_id Transformer layer index.
     * @return block_id on success, or -1 if the cache is full.
     */
    int allocate_block(int seq_id, int layer_id);

    /**
     * @brief Free a single block by block_id.
     *
     * Marks the block as free and resets its metadata.  No-op if block_id
     * is out of range or the block is already free.
     */
    void free_block(int block_id);

    /**
     * @brief Free all blocks belonging to a sequence.
     *
     * Iterates the entire pool and releases every block whose seq_id matches.
     */
    void free_sequence(int seq_id);

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    /**
     * @brief Return a pointer to a CacheBlock by block_id.
     *
     * @return Pointer to the CacheBlock, or nullptr if block_id is invalid.
     */
    CacheBlock* get_block(int block_id);

    /**
     * @brief Return a const pointer to a CacheBlock by block_id.
     */
    const CacheBlock* get_block(int block_id) const;

    /**
     * @brief Return aggregate usage statistics.
     */
    KVCacheStats stats() const;

    /**
     * @brief Return all block_ids for a sequence, sorted by layer_id.
     *
     * @param seq_id Sequence to query.
     * @return Sorted vector of block_ids; empty if the sequence has no blocks.
     */
    std::vector<int> get_sequence_blocks(int seq_id) const;

    // ----------------------------------------------------------------
    // Metadata accessors
    // ----------------------------------------------------------------

    int    num_layers()      const { return num_layers_; }
    int    block_size()      const { return block_size_; }
    int    max_blocks()      const { return max_blocks_; }
    size_t bytes_per_block() const { return bytes_per_block_; }

private:
    int    num_layers_;
    int    block_size_;
    int    max_blocks_;
    size_t bytes_per_block_;

    /// Pre-allocated flat buffer.
    uint8_t* buffer_;

    /// Block metadata table.
    std::vector<CacheBlock> blocks_;
};

} // namespace llmquant
