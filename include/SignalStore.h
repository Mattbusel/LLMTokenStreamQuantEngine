#pragma once

/**
 * @file include/SignalStore.h
 * @brief SQLite-backed persistence store for generated trade signals.
 *
 * ## Purpose
 *
 * Provides durable persistence for every `TradeSignal` produced by the
 * engine, enabling:
 *
 *  - **Replay**: re-run the downstream risk model against historical signals.
 *  - **Analysis**: query signal distribution, directional bias, confidence
 *    histograms, and latency percentiles from the stored database.
 *  - **Training data**: export the signal stream as a labelled dataset for
 *    ML model training.
 *
 * ## Database schema
 *
 * ```sql
 * CREATE TABLE signals (
 *     id              INTEGER PRIMARY KEY AUTOINCREMENT,
 *     timestamp_ns    INTEGER NOT NULL,
 *     delta_bias      REAL    NOT NULL,
 *     volatility_adj  REAL    NOT NULL,
 *     spread_modifier REAL    NOT NULL,
 *     confidence      REAL    NOT NULL,
 *     latency_us      REAL    NOT NULL,
 *     strategy_toggle INTEGER NOT NULL,
 *     strategy_weight REAL    NOT NULL,
 *     signal_quality  REAL    NOT NULL
 * );
 * CREATE INDEX idx_signals_ts ON signals(timestamp_ns);
 * ```
 *
 * ## Write path
 *
 * `insert()` serialises each `TradeSignal` into a prepared `INSERT` statement
 * and executes it synchronously.  For high-throughput scenarios, call
 * `begin_batch()` / `commit_batch()` to wrap multiple inserts in a single
 * SQLite transaction (10–100x throughput improvement).
 *
 * ## Thread safety
 *
 * `SignalStore` serialises all SQLite access through an internal mutex.
 * Multiple threads may call `insert()` concurrently; writes are serialised
 * internally.
 *
 * ## SQLite dependency
 *
 * Uses the SQLite amalgamation (single-file C library) bundled in
 * `third_party/sqlite/sqlite3.h` / `sqlite3.c`.  No external library
 * dependency is required.
 *
 * ## Feature flag
 * `LLMQUANT_ENABLE_SIGNAL_STORE` (default ON).
 */

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declare sqlite3 to avoid pulling in the full SQLite header
// from every translation unit that includes this header.
struct sqlite3;
struct sqlite3_stmt;

#include "TradeSignalEngine.h"  // TradeSignal

namespace llmquant {

/**
 * @brief A stored signal record (mirrors the `signals` table row).
 */
struct StoredSignal {
    int64_t  id{0};               ///< Auto-incremented row id.
    uint64_t timestamp_ns{0};
    double   delta_bias{0.0};
    double   volatility_adj{0.0};
    double   spread_modifier{0.0};
    double   confidence{0.0};
    double   latency_us{0.0};
    int      strategy_toggle{0};
    double   strategy_weight{0.0};
    double   signal_quality{0.0};
};

/**
 * @brief Aggregate statistics over the stored signal set.
 */
struct SignalStoreStats {
    uint64_t total_signals{0};       ///< Total rows in the signals table.
    double   mean_confidence{0.0};   ///< Average confidence value.
    double   mean_latency_us{0.0};   ///< Average token-to-signal latency.
    double   mean_quality{0.0};      ///< Average signal quality.
    double   bullish_fraction{0.0};  ///< Fraction with delta_bias > 0.
    uint64_t inserts_total{0};       ///< In-process insert counter.
};

/**
 * @brief SQLite-backed persistence store for `TradeSignal` records.
 */
class SignalStore {
public:
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    struct Config {
        /**
         * @brief Path to the SQLite database file.
         *
         * Use `:memory:` for an in-memory database (testing / benchmarks).
         * Default: `"signals.db"`.
         */
        std::string db_path{"signals.db"};

        /**
         * @brief Enable WAL (Write-Ahead Logging) mode for better concurrent
         * read performance.  Default: true.
         */
        bool wal_mode{true};

        /**
         * @brief SQLite synchronous mode.
         * 0=OFF, 1=NORMAL, 2=FULL.  Default: 1 (NORMAL).
         * Lower values reduce durability but improve throughput.
         */
        int synchronous_mode{1};
    };

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    /**
     * @brief Open (or create) the signal database.
     *
     * Creates the `signals` table and index if they do not already exist.
     *
     * @param config  Store configuration.
     * @throws std::runtime_error if the database cannot be opened or
     *         the schema cannot be created.
     */
    explicit SignalStore(Config config = Config{});

    ~SignalStore();

    SignalStore(const SignalStore&)            = delete;
    SignalStore& operator=(const SignalStore&) = delete;

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /**
     * @brief Persist a `TradeSignal` to the database.
     *
     * Thread-safe.  Blocks until the INSERT completes.
     *
     * @param signal  Signal to persist.
     * @throws std::runtime_error on SQLite error.
     */
    void insert(const TradeSignal& signal);

    /**
     * @brief Begin a batch transaction.
     *
     * All subsequent `insert()` calls are grouped into a single SQLite
     * transaction until `commit_batch()` is called.  Dramatically improves
     * throughput for bulk inserts.
     *
     * Thread-safe (only one batch transaction may be open at a time).
     *
     * @throws std::runtime_error if a batch is already open or on SQLite error.
     */
    void begin_batch();

    /**
     * @brief Commit the open batch transaction.
     *
     * Thread-safe.
     *
     * @throws std::runtime_error if no batch is open or on SQLite error.
     */
    void commit_batch();

    /**
     * @brief Roll back the open batch transaction without persisting.
     *
     * Thread-safe.
     */
    void rollback_batch();

    // -----------------------------------------------------------------------
    // Read
    // -----------------------------------------------------------------------

    /**
     * @brief Return the last `n` signals ordered by timestamp_ns DESC.
     *
     * Thread-safe.
     *
     * @param n  Maximum number of rows to return.
     * @return   Vector of StoredSignal, newest first.
     */
    [[nodiscard]] std::vector<StoredSignal> last_n(uint64_t n) const;

    /**
     * @brief Return the single most recently inserted signal.
     *
     * Thread-safe.
     *
     * @return The latest signal, or a zero-initialised record if the table
     *         is empty.
     */
    [[nodiscard]] StoredSignal latest() const;

    /**
     * @brief Return signals in the half-open time range [from_ns, to_ns).
     *
     * Thread-safe.
     *
     * @param from_ns  Lower bound (inclusive), nanoseconds since Unix epoch.
     * @param to_ns    Upper bound (exclusive), nanoseconds since Unix epoch.
     * @return         Matching signals ordered by timestamp_ns ASC.
     */
    [[nodiscard]] std::vector<StoredSignal>
    range(uint64_t from_ns, uint64_t to_ns) const;

    /**
     * @brief Return aggregate statistics over all stored signals.
     *
     * Thread-safe.
     */
    [[nodiscard]] SignalStoreStats stats() const;

    /**
     * @brief Total number of rows in the signals table.
     *
     * Thread-safe.
     */
    [[nodiscard]] uint64_t count() const;

    /**
     * @brief Delete all rows from the signals table.
     *
     * Thread-safe.
     *
     * @throws std::runtime_error on SQLite error.
     */
    void clear();

    /**
     * @brief Serialise in-process statistics to JSON.
     *
     * Thread-safe.
     */
    [[nodiscard]] std::string to_stats_json() const;

private:
    void exec(const char* sql);
    void prepare_statements();

    mutable std::mutex   mutex_;
    Config               config_;
    sqlite3*             db_{nullptr};
    sqlite3_stmt*        insert_stmt_{nullptr};
    bool                 in_batch_{false};
    uint64_t             inserts_total_{0};
};

} // namespace llmquant
