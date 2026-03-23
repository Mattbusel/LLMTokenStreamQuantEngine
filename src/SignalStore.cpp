/**
 * @file src/SignalStore.cpp
 * @brief SQLite-backed persistence for TradeSignal records.
 *
 * Uses the SQLite amalgamation C API directly for zero-dependency persistence.
 * If SQLite is not available, all operations compile to stubs that throw
 * std::runtime_error with a clear "SQLite not available" message — controlled
 * by the LLMQUANT_HAVE_SQLITE preprocessor guard set in CMakeLists.txt when
 * SQLite3 is found via find_package(SQLite3).
 */

#include "SignalStore.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

// Pull in SQLite3 — available either as system library or the bundled amalgamation.
#if __has_include(<sqlite3.h>)
#  include <sqlite3.h>
#  define LLMQUANT_HAVE_SQLITE 1
#else
#  define LLMQUANT_HAVE_SQLITE 0
#endif

namespace llmquant {

// ===========================================================================
// Helper: throw on SQLite error
// ===========================================================================

#if LLMQUANT_HAVE_SQLITE

static void sqlite_check(int rc, const char* context, sqlite3* db = nullptr) {
    if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_ROW) return;
    std::string msg = std::string(context) + ": SQLite error " + std::to_string(rc);
    if (db) {
        const char* errmsg = sqlite3_errmsg(db);
        if (errmsg) msg += " — ";
        if (errmsg) msg += errmsg;
    }
    throw std::runtime_error(msg);
}

#endif // LLMQUANT_HAVE_SQLITE

// ===========================================================================
// Schema DDL
// ===========================================================================

static constexpr const char* kCreateTable = R"sql(
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ns    INTEGER NOT NULL,
    delta_bias      REAL    NOT NULL,
    volatility_adj  REAL    NOT NULL,
    spread_modifier REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    latency_us      REAL    NOT NULL,
    strategy_toggle INTEGER NOT NULL,
    strategy_weight REAL    NOT NULL,
    signal_quality  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp_ns);
)sql";

static constexpr const char* kInsertSQL = R"sql(
INSERT INTO signals
    (timestamp_ns, delta_bias, volatility_adj, spread_modifier,
     confidence, latency_us, strategy_toggle, strategy_weight, signal_quality)
VALUES (?,?,?,?,?,?,?,?,?);
)sql";

// ===========================================================================
// Construction
// ===========================================================================

SignalStore::SignalStore(Config config)
    : config_(std::move(config))
{
#if LLMQUANT_HAVE_SQLITE
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
              | SQLITE_OPEN_NOMUTEX;   // serialise via our own mutex
    sqlite_check(
        sqlite3_open_v2(config_.db_path.c_str(), &db_, flags, nullptr),
        "SignalStore::open"
    );

    if (config_.wal_mode) {
        exec("PRAGMA journal_mode=WAL;");
    }
    std::string sync_pragma = "PRAGMA synchronous="
                            + std::to_string(config_.synchronous_mode) + ";";
    exec(sync_pragma.c_str());
    exec(kCreateTable);
    prepare_statements();
#else
    (void)config_;
    // SQLite not available: store will throw on every write.
#endif
}

SignalStore::~SignalStore() {
#if LLMQUANT_HAVE_SQLITE
    if (insert_stmt_) sqlite3_finalize(insert_stmt_);
    if (db_)          sqlite3_close(db_);
#endif
}

// ===========================================================================
// Internal helpers
// ===========================================================================

void SignalStore::exec(const char* sql) {
#if LLMQUANT_HAVE_SQLITE
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string msg = "SignalStore::exec: ";
        if (errmsg) { msg += errmsg; sqlite3_free(errmsg); }
        throw std::runtime_error(msg);
    }
#else
    (void)sql;
    throw std::runtime_error("SignalStore: SQLite not available in this build");
#endif
}

void SignalStore::prepare_statements() {
#if LLMQUANT_HAVE_SQLITE
    sqlite_check(
        sqlite3_prepare_v2(db_, kInsertSQL, -1, &insert_stmt_, nullptr),
        "SignalStore::prepare insert", db_
    );
#endif
}

// ===========================================================================
// insert
// ===========================================================================

void SignalStore::insert(const TradeSignal& sig) {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    sqlite3_reset(insert_stmt_);
    sqlite3_bind_int64(insert_stmt_, 1, static_cast<sqlite3_int64>(sig.timestamp_ns));
    sqlite3_bind_double(insert_stmt_, 2, sig.delta_bias_shift);
    sqlite3_bind_double(insert_stmt_, 3, sig.volatility_adjustment);
    sqlite3_bind_double(insert_stmt_, 4, sig.spread_modifier);
    sqlite3_bind_double(insert_stmt_, 5, sig.confidence);
    sqlite3_bind_double(insert_stmt_, 6, sig.latency_us);
    sqlite3_bind_int(insert_stmt_,    7, sig.strategy_toggle);
    sqlite3_bind_double(insert_stmt_, 8, sig.strategy_weight);
    sqlite3_bind_double(insert_stmt_, 9, sig.signal_quality);
    sqlite_check(sqlite3_step(insert_stmt_), "SignalStore::insert step", db_);
    ++inserts_total_;
#else
    (void)sig;
    throw std::runtime_error("SignalStore: SQLite not available in this build");
#endif
}

// ===========================================================================
// Batch transaction
// ===========================================================================

void SignalStore::begin_batch() {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    if (in_batch_) throw std::runtime_error("SignalStore::begin_batch: batch already open");
    exec("BEGIN TRANSACTION;");
    in_batch_ = true;
#else
    throw std::runtime_error("SignalStore: SQLite not available in this build");
#endif
}

void SignalStore::commit_batch() {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    if (!in_batch_) throw std::runtime_error("SignalStore::commit_batch: no batch open");
    exec("COMMIT;");
    in_batch_ = false;
#else
    throw std::runtime_error("SignalStore: SQLite not available in this build");
#endif
}

void SignalStore::rollback_batch() {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    if (!in_batch_) return;
    try { exec("ROLLBACK;"); } catch (...) {}
    in_batch_ = false;
#endif
}

// ===========================================================================
// Read: last_n
// ===========================================================================

std::vector<StoredSignal> SignalStore::last_n(uint64_t n) const {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<StoredSignal> results;
    results.reserve(static_cast<std::size_t>(n));

    const std::string sql =
        "SELECT id,timestamp_ns,delta_bias,volatility_adj,spread_modifier,"
        "confidence,latency_us,strategy_toggle,strategy_weight,signal_quality "
        "FROM signals ORDER BY timestamp_ns DESC LIMIT " + std::to_string(n) + ";";

    sqlite3_stmt* stmt = nullptr;
    sqlite_check(sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr),
                 "SignalStore::last_n prepare", db_);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        StoredSignal s;
        s.id              = sqlite3_column_int64(stmt, 0);
        s.timestamp_ns    = static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));
        s.delta_bias      = sqlite3_column_double(stmt, 2);
        s.volatility_adj  = sqlite3_column_double(stmt, 3);
        s.spread_modifier = sqlite3_column_double(stmt, 4);
        s.confidence      = sqlite3_column_double(stmt, 5);
        s.latency_us      = sqlite3_column_double(stmt, 6);
        s.strategy_toggle = sqlite3_column_int(stmt, 7);
        s.strategy_weight = sqlite3_column_double(stmt, 8);
        s.signal_quality  = sqlite3_column_double(stmt, 9);
        results.push_back(s);
    }
    sqlite3_finalize(stmt);
    return results;
#else
    (void)n;
    return {};
#endif
}

// ===========================================================================
// Read: latest
// ===========================================================================

StoredSignal SignalStore::latest() const {
    auto v = last_n(1);
    return v.empty() ? StoredSignal{} : v[0];
}

// ===========================================================================
// Read: range
// ===========================================================================

std::vector<StoredSignal> SignalStore::range(uint64_t from_ns, uint64_t to_ns) const {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<StoredSignal> results;

    const char* sql =
        "SELECT id,timestamp_ns,delta_bias,volatility_adj,spread_modifier,"
        "confidence,latency_us,strategy_toggle,strategy_weight,signal_quality "
        "FROM signals WHERE timestamp_ns >= ? AND timestamp_ns < ? "
        "ORDER BY timestamp_ns ASC;";

    sqlite3_stmt* stmt = nullptr;
    sqlite_check(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr),
                 "SignalStore::range prepare", db_);
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(from_ns));
    sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(to_ns));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        StoredSignal s;
        s.id              = sqlite3_column_int64(stmt, 0);
        s.timestamp_ns    = static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));
        s.delta_bias      = sqlite3_column_double(stmt, 2);
        s.volatility_adj  = sqlite3_column_double(stmt, 3);
        s.spread_modifier = sqlite3_column_double(stmt, 4);
        s.confidence      = sqlite3_column_double(stmt, 5);
        s.latency_us      = sqlite3_column_double(stmt, 6);
        s.strategy_toggle = sqlite3_column_int(stmt, 7);
        s.strategy_weight = sqlite3_column_double(stmt, 8);
        s.signal_quality  = sqlite3_column_double(stmt, 9);
        results.push_back(s);
    }
    sqlite3_finalize(stmt);
    return results;
#else
    (void)from_ns; (void)to_ns;
    return {};
#endif
}

// ===========================================================================
// stats / count / clear
// ===========================================================================

SignalStoreStats SignalStore::stats() const {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    SignalStoreStats s;
    s.inserts_total = inserts_total_;

    const char* sql =
        "SELECT COUNT(*), AVG(confidence), AVG(latency_us), AVG(signal_quality),"
        "       SUM(CASE WHEN delta_bias > 0 THEN 1 ELSE 0 END)"
        " FROM signals;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            s.total_signals    = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
            s.mean_confidence  = sqlite3_column_double(stmt, 1);
            s.mean_latency_us  = sqlite3_column_double(stmt, 2);
            s.mean_quality     = sqlite3_column_double(stmt, 3);
            uint64_t bull      = static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
            s.bullish_fraction = (s.total_signals > 0)
                                 ? static_cast<double>(bull) / static_cast<double>(s.total_signals)
                                 : 0.0;
        }
        sqlite3_finalize(stmt);
    }
    return s;
#else
    return {};
#endif
}

uint64_t SignalStore::count() const {
#if LLMQUANT_HAVE_SQLITE
    std::lock_guard<std::mutex> lk(mutex_);
    sqlite3_stmt* stmt = nullptr;
    uint64_t n = 0;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM signals;", -1, &stmt, nullptr)
        == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW)
            n = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        sqlite3_finalize(stmt);
    }
    return n;
#else
    return 0;
#endif
}

void SignalStore::clear() {
    exec("DELETE FROM signals;");
}

std::string SignalStore::to_stats_json() const {
    auto s = stats();
    std::ostringstream os;
    os << "{"
       << "\"total_signals\":"   << s.total_signals    << ","
       << "\"mean_confidence\":" << s.mean_confidence  << ","
       << "\"mean_latency_us\":" << s.mean_latency_us  << ","
       << "\"mean_quality\":"    << s.mean_quality      << ","
       << "\"bullish_fraction\":"<< s.bullish_fraction  << ","
       << "\"inserts_total\":"   << s.inserts_total
       << "}";
    return os.str();
}

} // namespace llmquant
