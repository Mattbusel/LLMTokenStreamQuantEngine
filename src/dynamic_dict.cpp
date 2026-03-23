/// @file src/dynamic_dict.cpp
/// @brief Implementation of DynamicTokenDictionary, DictionaryLoader, and
///        TokenLearner.

#include "dynamic_dict.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace llmquant {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// @brief Trim leading/trailing ASCII whitespace from a string view.
std::string_view trim(std::string_view s) noexcept {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '\r' || s.front() == '\n'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                          s.back() == '\r' || s.back() == '\n'))
        s.remove_suffix(1);
    return s;
}

/// @brief Case-insensitive ASCII compare.
bool iequal(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }
    return true;
}

/// @brief Read an entire file into a string.  Returns empty string on error.
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// @brief Get file mtime as nanoseconds since epoch (best-effort).
int64_t file_mtime_ns(const std::string& path) noexcept {
#if defined(_WIN32)
    struct _stat64 st{};
    if (::_stat64(path.c_str(), &st) != 0) return 0;
    // st_mtime is seconds on MSVC; multiply to nanoseconds.
    return static_cast<int64_t>(st.st_mtime) * 1'000'000'000LL;
#else
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return 0;
#  if defined(__APPLE__)
    return st.st_mtimespec.tv_sec * 1'000'000'000LL + st.st_mtimespec.tv_nsec;
#  else
    return st.st_mtim.tv_sec * 1'000'000'000LL + st.st_mtim.tv_nsec;
#  endif
#endif
}

/// @brief Clamp a double to [lo, hi].
constexpr double clamp_d(double v, double lo, double hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// Minimal JSON parser helpers
// ---------------------------------------------------------------------------
// We implement a lightweight, zero-dependency JSON parser that handles the
// subset of JSON used by the dictionary format.  This avoids pulling in
// nlohmann/json or similar.

/// @brief Skip whitespace and return the current character.
char peek(const char*& p, const char* end) noexcept {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
        ++p;
    return (p < end) ? *p : '\0';
}

/// @brief Parse a JSON string (assumes opening " has been consumed).
std::string parse_json_string(const char*& p, const char* end) {
    std::string out;
    out.reserve(32);
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (p < end) {
                switch (*p) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    default:   out += *p;   break;
                }
            }
        } else {
            out += *p;
        }
        ++p;
    }
    if (p < end) ++p; // consume closing "
    return out;
}

/// @brief Parse a JSON number.
double parse_json_number(const char*& p, const char* end) {
    const char* start = p;
    while (p < end && (*p == '-' || *p == '+' || (*p >= '0' && *p <= '9') ||
                       *p == '.' || *p == 'e' || *p == 'E'))
        ++p;
    // Use std::stod for correctness.
    char buf[64]{};
    std::size_t len = static_cast<std::size_t>(p - start);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    std::memcpy(buf, start, len);
    char* endptr{};
    double v = std::strtod(buf, &endptr);
    return v;
}

/// @brief Skip a JSON value (any type) without parsing it.
void skip_json_value(const char*& p, const char* end);

void skip_json_object(const char*& p, const char* end) {
    int depth = 1;
    ++p; // consume '{'
    while (p < end && depth > 0) {
        char c = *p++;
        if (c == '{') ++depth;
        else if (c == '}') --depth;
        else if (c == '"') { // skip string
            while (p < end && *p != '"') {
                if (*p == '\\') ++p;
                ++p;
            }
            if (p < end) ++p;
        }
    }
}

void skip_json_array(const char*& p, const char* end) {
    int depth = 1;
    ++p; // consume '['
    while (p < end && depth > 0) {
        char c = *p++;
        if (c == '[') ++depth;
        else if (c == ']') --depth;
        else if (c == '"') {
            while (p < end && *p != '"') {
                if (*p == '\\') ++p;
                ++p;
            }
            if (p < end) ++p;
        }
    }
}

void skip_json_value(const char*& p, const char* end) {
    char c = peek(p, end);
    if (c == '{') { skip_json_object(p, end); }
    else if (c == '[') { skip_json_array(p, end); }
    else if (c == '"') { ++p; parse_json_string(p, end); }
    else { parse_json_number(p, end); }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// TokenCategory helpers
// ---------------------------------------------------------------------------

const char* token_category_to_string(TokenCategory cat) noexcept {
    switch (cat) {
        case TokenCategory::Fear:       return "fear";
        case TokenCategory::Bullish:    return "bullish";
        case TokenCategory::Bearish:    return "bearish";
        case TokenCategory::Volatility: return "volatility";
        case TokenCategory::Corporate:  return "corporate";
        case TokenCategory::Macro:      return "macro";
        case TokenCategory::Neutral:    return "neutral";
        default:                        return "neutral";
    }
}

TokenCategory token_category_from_string(std::string_view s) noexcept {
    if (iequal(s, "fear"))       return TokenCategory::Fear;
    if (iequal(s, "bullish"))    return TokenCategory::Bullish;
    if (iequal(s, "bearish"))    return TokenCategory::Bearish;
    if (iequal(s, "volatility")) return TokenCategory::Volatility;
    if (iequal(s, "corporate"))  return TokenCategory::Corporate;
    if (iequal(s, "macro"))      return TokenCategory::Macro;
    return TokenCategory::Neutral;
}

// ---------------------------------------------------------------------------
// CategoryWeights
// ---------------------------------------------------------------------------

double CategoryWeights::get(TokenCategory cat) const noexcept {
    switch (cat) {
        case TokenCategory::Fear:       return fear;
        case TokenCategory::Bullish:    return bullish;
        case TokenCategory::Bearish:    return bearish;
        case TokenCategory::Volatility: return volatility;
        case TokenCategory::Corporate:  return corporate;
        case TokenCategory::Macro:      return macro;
        case TokenCategory::Neutral:    return neutral;
        default:                        return 1.0;
    }
}

// ---------------------------------------------------------------------------
// TokenEntry
// ---------------------------------------------------------------------------

SemanticWeight TokenEntry::to_semantic_weight(const CategoryWeights& cw) const noexcept {
    const double m = cw.get(category);
    SemanticWeight sw;
    sw.sentiment_score  = clamp_d(sentiment_score  * m, -1.0, 1.0);
    sw.confidence_score = clamp_d(confidence,            0.0, 1.0);
    sw.volatility_score = clamp_d(volatility_weight * m,  0.0, 1.0);
    sw.directional_bias = clamp_d(bias_weight       * m, -1.0, 1.0);
    return sw;
}

SemanticWeight TokenEntry::to_semantic_weight() const noexcept {
    return to_semantic_weight(CategoryWeights{});
}

// ---------------------------------------------------------------------------
// DynamicTokenDictionary
// ---------------------------------------------------------------------------

DynamicTokenDictionary::DynamicTokenDictionary()
    : snapshot_(std::make_shared<Snapshot>()) {}

SemanticWeight
DynamicTokenDictionary::map_token_to_weight(const std::string& token) const noexcept {
    // Lock-free read via atomic load.
    auto snap = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    if (!snap) return SemanticWeight{0.0, 0.5, 0.1, 0.0};
    auto it = snap->entries.find(token);
    if (it == snap->entries.end()) return SemanticWeight{0.0, 0.5, 0.1, 0.0};
    return it->second.to_semantic_weight(snap->category_weights);
}

SemanticWeight
DynamicTokenDictionary::map_sequence_to_weight(
    const std::vector<std::string>& tokens) const noexcept
{
    if (tokens.empty()) return SemanticWeight{};
    auto snap = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    if (!snap) return SemanticWeight{};

    double sum_sent = 0, sum_vol = 0, sum_bias = 0, sum_conf = 0;
    double total_w = 0;

    for (const auto& tok : tokens) {
        auto it = snap->entries.find(tok);
        SemanticWeight sw;
        if (it != snap->entries.end()) {
            sw = it->second.to_semantic_weight(snap->category_weights);
        } else {
            sw = SemanticWeight{0.0, 0.5, 0.1, 0.0};
        }
        const double w = sw.confidence_score;
        sum_sent  += w * sw.sentiment_score;
        sum_vol   += w * sw.volatility_score;
        sum_bias  += w * sw.directional_bias;
        sum_conf  += w;
        total_w   += w;
    }

    if (total_w < 1e-12) return SemanticWeight{};

    SemanticWeight out;
    out.sentiment_score  = sum_sent  / total_w;
    out.volatility_score = sum_vol   / total_w;
    out.directional_bias = sum_bias  / total_w;
    out.confidence_score = sum_conf  / static_cast<double>(tokens.size());
    return out;
}

void DynamicTokenDictionary::publish_snapshot(std::shared_ptr<Snapshot> snap) {
    std::atomic_store_explicit(&snapshot_, snap, std::memory_order_release);
}

std::shared_ptr<const DynamicTokenDictionary::Snapshot>
DynamicTokenDictionary::current_snapshot() const noexcept {
    return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

std::size_t DynamicTokenDictionary::size() const noexcept {
    auto snap = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    return snap ? snap->entries.size() : 0u;
}

uint64_t DynamicTokenDictionary::version() const noexcept {
    auto snap = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    return snap ? snap->version : 0u;
}

std::shared_ptr<DynamicTokenDictionary::Snapshot>
DynamicTokenDictionary::clone_snapshot() const {
    auto cur = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    auto clone = std::make_shared<Snapshot>();
    if (cur) *clone = *cur; // deep copy
    return clone;
}

void DynamicTokenDictionary::upsert(const TokenEntry& entry) {
    std::lock_guard lock(mutation_mtx_);
    auto snap = clone_snapshot();
    snap->entries[entry.text] = entry;
    snap->version++;
    std::atomic_store_explicit(&snapshot_, snap, std::memory_order_release);
}

void DynamicTokenDictionary::set_category_weights(const CategoryWeights& weights) {
    std::lock_guard lock(mutation_mtx_);
    auto snap = clone_snapshot();
    snap->category_weights = weights;
    snap->version++;
    std::atomic_store_explicit(&snapshot_, snap, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// DictionaryLoader
// ---------------------------------------------------------------------------

DictionaryLoader::DictionaryLoader(const std::string&        filepath,
                                   DynamicTokenDictionary&   dict,
                                   std::chrono::milliseconds poll_ms)
    : filepath_(filepath), dict_(dict), poll_ms_(poll_ms)
{
    // Perform initial synchronous load.
    auto snap = parse_file();
    if (!snap) {
        throw std::runtime_error(
            "DictionaryLoader: failed to parse '" + filepath_ + "' on first load");
    }
    last_mtime_ns_.store(get_file_mtime_ns(), std::memory_order_relaxed);
    dict_.publish_snapshot(snap);

    // Start background watcher thread.
    watcher_ = std::thread([this] { watch_loop(); });
}

DictionaryLoader::~DictionaryLoader() {
    stop_.store(true, std::memory_order_relaxed);
    if (watcher_.joinable()) watcher_.join();
}

void DictionaryLoader::set_reload_callback(ReloadCallback cb) {
    std::lock_guard lock(cb_mtx_);
    on_reload_ = std::move(cb);
}

bool DictionaryLoader::reload_now() {
    auto snap = parse_file();
    if (!snap) return false;
    last_mtime_ns_.store(get_file_mtime_ns(), std::memory_order_relaxed);
    dict_.publish_snapshot(snap);
    uint64_t cnt = reload_count_.fetch_add(1, std::memory_order_relaxed) + 1;

    ReloadCallback cb;
    {
        std::lock_guard lock(cb_mtx_);
        cb = on_reload_;
    }
    if (cb) cb(snap->version, snap->entries.size());
    return true;
}

uint64_t DictionaryLoader::reload_count() const noexcept {
    return reload_count_.load(std::memory_order_relaxed);
}

const std::string& DictionaryLoader::filepath() const noexcept {
    return filepath_;
}

void DictionaryLoader::watch_loop() {
    while (!stop_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(poll_ms_);
        if (stop_.load(std::memory_order_relaxed)) break;

        const int64_t mtime = get_file_mtime_ns();
        const int64_t prev  = last_mtime_ns_.load(std::memory_order_relaxed);
        if (mtime != 0 && mtime != prev) {
            auto snap = parse_file();
            if (snap) {
                last_mtime_ns_.store(mtime, std::memory_order_relaxed);
                dict_.publish_snapshot(snap);
                uint64_t cnt = reload_count_.fetch_add(1, std::memory_order_relaxed) + 1;

                ReloadCallback cb;
                {
                    std::lock_guard lock(cb_mtx_);
                    cb = on_reload_;
                }
                if (cb) cb(snap->version, snap->entries.size());
            }
        }
    }
}

int64_t DictionaryLoader::get_file_mtime_ns() const noexcept {
    return file_mtime_ns(filepath_);
}

// ---------------------------------------------------------------------------
// DictionaryLoader — file parsing
// ---------------------------------------------------------------------------

std::shared_ptr<DynamicTokenDictionary::Snapshot>
DictionaryLoader::parse_file() const {
    std::string content = slurp(filepath_);
    if (content.empty()) return nullptr;

    // Detect format by extension and content.
    bool is_json = false, is_yaml = false;
    if (filepath_.size() >= 5) {
        std::string ext = filepath_.substr(filepath_.size() - 5);
        std::transform(ext.begin(), ext.end(), ext.begin(),
            [](unsigned char c){ return std::tolower(c); });
        if (ext == ".json") is_json = true;
        if (ext == ".yaml" || ext.substr(1) == ".yml") is_yaml = true;
    }
    // Content sniffing: if first non-whitespace character is '{' → JSON.
    for (char c : content) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        if (c == '{') is_json = true;
        break;
    }

    if (is_json)      return parse_json(content, filepath_);
    if (is_yaml)      return parse_yaml(content, filepath_);
    return parse_tsv(content, filepath_);
}

// --- JSON parser -------------------------------------------------------------

std::shared_ptr<DynamicTokenDictionary::Snapshot>
DictionaryLoader::parse_json(const std::string& content,
                              const std::string& source) const
{
    auto snap = std::make_shared<DynamicTokenDictionary::Snapshot>();
    snap->source_path = source;

    const char* p   = content.data();
    const char* end = p + content.size();

    // Expect top-level object.
    if (peek(p, end) != '{') return nullptr;
    ++p; // consume '{'

    while (true) {
        char c = peek(p, end);
        if (c == '}' || c == '\0') break;
        if (c == ',') { ++p; continue; }
        if (c != '"') { ++p; continue; } // skip unexpected chars
        ++p; // consume opening "

        std::string key = parse_json_string(p, end);
        if (peek(p, end) != ':') continue;
        ++p; // consume ':'

        if (key == "category_weights") {
            // Parse category_weights object.
            if (peek(p, end) != '{') { skip_json_value(p, end); continue; }
            ++p; // consume '{'
            while (true) {
                char cc = peek(p, end);
                if (cc == '}' || cc == '\0') { if (cc == '}') ++p; break; }
                if (cc == ',') { ++p; continue; }
                if (cc != '"') { ++p; continue; }
                ++p;
                std::string wkey = parse_json_string(p, end);
                if (peek(p, end) != ':') continue;
                ++p;
                double val = parse_json_number(p, end);
                TokenCategory cat = token_category_from_string(wkey);
                switch (cat) {
                    case TokenCategory::Fear:       snap->category_weights.fear       = val; break;
                    case TokenCategory::Bullish:    snap->category_weights.bullish    = val; break;
                    case TokenCategory::Bearish:    snap->category_weights.bearish    = val; break;
                    case TokenCategory::Volatility: snap->category_weights.volatility = val; break;
                    case TokenCategory::Corporate:  snap->category_weights.corporate  = val; break;
                    case TokenCategory::Macro:      snap->category_weights.macro      = val; break;
                    default: break;
                }
            }
        } else if (key == "tokens") {
            // Parse tokens array.
            if (peek(p, end) != '[') { skip_json_value(p, end); continue; }
            ++p; // consume '['
            while (true) {
                char cc = peek(p, end);
                if (cc == ']' || cc == '\0') { if (cc == ']') ++p; break; }
                if (cc == ',') { ++p; continue; }
                if (cc != '{') { skip_json_value(p, end); continue; }
                ++p; // consume '{'

                TokenEntry entry;
                entry.category = TokenCategory::Neutral;
                entry.source   = "json";

                while (true) {
                    char fc = peek(p, end);
                    if (fc == '}' || fc == '\0') { if (fc == '}') ++p; break; }
                    if (fc == ',') { ++p; continue; }
                    if (fc != '"') { ++p; continue; }
                    ++p;
                    std::string fkey = parse_json_string(p, end);
                    if (peek(p, end) != ':') continue;
                    ++p;

                    if (fkey == "text" || fkey == "token") {
                        if (peek(p, end) == '"') { ++p; entry.text = parse_json_string(p, end); }
                        else skip_json_value(p, end);
                    } else if (fkey == "bias") {
                        entry.bias_weight = clamp_d(parse_json_number(p, end), -1.0, 1.0);
                    } else if (fkey == "volatility") {
                        entry.volatility_weight = clamp_d(parse_json_number(p, end), 0.0, 1.0);
                    } else if (fkey == "sentiment") {
                        entry.sentiment_score = clamp_d(parse_json_number(p, end), -1.0, 1.0);
                    } else if (fkey == "confidence") {
                        entry.confidence = clamp_d(parse_json_number(p, end), 0.0, 1.0);
                    } else if (fkey == "category") {
                        if (peek(p, end) == '"') {
                            ++p;
                            std::string catstr = parse_json_string(p, end);
                            entry.category = token_category_from_string(catstr);
                        } else skip_json_value(p, end);
                    } else if (fkey == "source") {
                        if (peek(p, end) == '"') { ++p; entry.source = parse_json_string(p, end); }
                        else skip_json_value(p, end);
                    } else {
                        skip_json_value(p, end);
                    }
                }

                if (!entry.text.empty()) {
                    snap->entries[entry.text] = entry;
                }
            }
        } else {
            skip_json_value(p, end);
        }
    }

    snap->version = 1;
    return snap;
}

// --- YAML parser (line-oriented subset) ------------------------------------

std::shared_ptr<DynamicTokenDictionary::Snapshot>
DictionaryLoader::parse_yaml(const std::string& content,
                              const std::string& source) const
{
    auto snap = std::make_shared<DynamicTokenDictionary::Snapshot>();
    snap->source_path = source;

    // States for the simple line-based YAML FSM.
    enum class State { TopLevel, CategoryWeights, TokensList, TokenEntry } state = State::TopLevel;
    TokenEntry current;
    bool in_token = false;

    auto flush_token = [&] {
        if (in_token && !current.text.empty()) {
            snap->entries[current.text] = current;
        }
        current = TokenEntry{};
        current.category = TokenCategory::Neutral;
        current.source   = "yaml";
        in_token = false;
    };

    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        // Remove trailing CR.
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Compute indent level.
        std::size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') ++indent;

        std::string_view sv = trim(std::string_view(line));
        if (sv.empty() || sv.front() == '#') continue; // skip comments

        // Remove leading '- ' for list items.
        bool is_list_item = false;
        if (sv.size() >= 2 && sv[0] == '-' && sv[1] == ' ') {
            sv.remove_prefix(2);
            is_list_item = true;
        }

        // Split on first ':'.
        auto colon = sv.find(':');
        if (colon == std::string_view::npos) continue;
        std::string_view key = trim(sv.substr(0, colon));
        std::string_view val = (colon + 1 < sv.size()) ?
                               trim(sv.substr(colon + 1)) :
                               std::string_view{};

        // Top-level keys.
        if (indent == 0 && !is_list_item) {
            flush_token();
            if (key == "category_weights") { state = State::CategoryWeights; }
            else if (key == "tokens") { state = State::TokensList; }
            else { state = State::TopLevel; }
            continue;
        }

        if (state == State::CategoryWeights) {
            double v = val.empty() ? 1.0 : std::strtod(std::string(val).c_str(), nullptr);
            TokenCategory cat = token_category_from_string(key);
            switch (cat) {
                case TokenCategory::Fear:       snap->category_weights.fear       = v; break;
                case TokenCategory::Bullish:    snap->category_weights.bullish    = v; break;
                case TokenCategory::Bearish:    snap->category_weights.bearish    = v; break;
                case TokenCategory::Volatility: snap->category_weights.volatility = v; break;
                case TokenCategory::Corporate:  snap->category_weights.corporate  = v; break;
                case TokenCategory::Macro:      snap->category_weights.macro      = v; break;
                default: break;
            }
            continue;
        }

        if (state == State::TokensList || state == State::TokenEntry) {
            if (is_list_item) {
                // New token entry starts.
                flush_token();
                in_token = true;
                state = State::TokenEntry;
            }
            // Parse field.
            if (key == "text" || key == "token") {
                current.text = std::string(val);
            } else if (key == "bias") {
                current.bias_weight = clamp_d(
                    std::strtod(std::string(val).c_str(), nullptr), -1.0, 1.0);
            } else if (key == "volatility") {
                current.volatility_weight = clamp_d(
                    std::strtod(std::string(val).c_str(), nullptr), 0.0, 1.0);
            } else if (key == "sentiment") {
                current.sentiment_score = clamp_d(
                    std::strtod(std::string(val).c_str(), nullptr), -1.0, 1.0);
            } else if (key == "confidence") {
                current.confidence = clamp_d(
                    std::strtod(std::string(val).c_str(), nullptr), 0.0, 1.0);
            } else if (key == "category") {
                current.category = token_category_from_string(val);
            } else if (key == "source") {
                current.source = std::string(val);
            }
        }
    }
    flush_token(); // flush last token if any.

    snap->version = 1;
    return snap;
}

// --- TSV (legacy) parser ----------------------------------------------------

std::shared_ptr<DynamicTokenDictionary::Snapshot>
DictionaryLoader::parse_tsv(const std::string& content,
                             const std::string& source) const
{
    auto snap = std::make_shared<DynamicTokenDictionary::Snapshot>();
    snap->source_path = source;

    std::istringstream iss(content);
    std::string line;
    bool header = true;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (header) { header = false; continue; } // skip header row

        std::istringstream ls(line);
        std::string text;
        double sentiment = 0, confidence = 0.5, volatility = 0, bias = 0;
        ls >> text >> sentiment >> confidence >> volatility >> bias;
        if (text.empty() || text[0] == '#') continue;

        TokenEntry entry;
        entry.text             = text;
        entry.sentiment_score  = clamp_d(sentiment,  -1.0, 1.0);
        entry.confidence       = clamp_d(confidence,  0.0, 1.0);
        entry.volatility_weight= clamp_d(volatility,  0.0, 1.0);
        entry.bias_weight      = clamp_d(bias,        -1.0, 1.0);
        entry.category         = TokenCategory::Neutral;
        entry.source           = "tsv";
        snap->entries[entry.text] = entry;
    }

    snap->version = 1;
    return snap;
}

// ---------------------------------------------------------------------------
// TokenLearner
// ---------------------------------------------------------------------------

TokenLearner::TokenLearner(DynamicTokenDictionary&        shadow_dict,
                           const DynamicTokenDictionary&  source_dict,
                           Config                         cfg)
    : shadow_dict_(shadow_dict), source_dict_(source_dict), cfg_(cfg) {}

void TokenLearner::record_outcome(const Outcome& outcome) {
    if (outcome.active_tokens.empty()) return;

    std::lock_guard lock(mtx_);
    obs_count_.fetch_add(1, std::memory_order_relaxed);

    // Compute total confidence-weighted bias contribution.
    double total_conf = 0;
    for (const auto& tok : outcome.active_tokens) {
        SemanticWeight sw = source_dict_.map_token_to_weight(tok);
        total_conf += sw.confidence_score;
    }
    if (total_conf < 1e-12) return;

    const double pnl_sign = (outcome.realised_pnl > 0) ? 1.0
                          : (outcome.realised_pnl < 0) ? -1.0 : 0.0;
    if (pnl_sign == 0.0) return;

    // Get a mutable snapshot of the shadow dictionary.
    auto snap = const_cast<DynamicTokenDictionary&>(shadow_dict_)
                    .current_snapshot();
    // We need to build a new snapshot for mutation.
    auto new_snap = std::make_shared<DynamicTokenDictionary::Snapshot>();
    if (snap) {
        *new_snap = *snap; // deep copy
    }
    new_snap->version = snap ? snap->version + 1 : 1;

    for (const auto& tok : outcome.active_tokens) {
        // Track observation count.
        token_obs_[tok]++;
        if (token_obs_[tok] < cfg_.min_observations) continue;

        SemanticWeight sw = source_dict_.map_token_to_weight(tok);
        const double   frac = (total_conf > 0) ? sw.confidence_score / total_conf : 0.0;
        const double   delta = cfg_.alpha * pnl_sign * frac;
        const double   clamped_delta = std::clamp(delta,
                                                  -cfg_.max_weight_delta,
                                                   cfg_.max_weight_delta);

        // Find or create the token entry in shadow dict.
        auto it = new_snap->entries.find(tok);
        if (it == new_snap->entries.end()) {
            // Insert a new entry cloned from the source dict snapshot.
            auto src_snap = source_dict_.current_snapshot();
            if (src_snap) {
                auto src_it = src_snap->entries.find(tok);
                if (src_it != src_snap->entries.end()) {
                    new_snap->entries[tok] = src_it->second;
                    it = new_snap->entries.find(tok);
                }
            }
        }
        if (it == new_snap->entries.end()) continue;

        // Apply EMA update.
        it->second.bias_weight = clamp_d(
            it->second.bias_weight + clamped_delta, -1.0, 1.0);
        it->second.confidence  = std::max(
            cfg_.min_confidence,
            it->second.confidence + 0.5 * std::abs(clamped_delta));
        it->second.source = "learned";
    }

    shadow_dict_.publish_snapshot(new_snap);
}

uint64_t TokenLearner::observation_count() const noexcept {
    return obs_count_.load(std::memory_order_relaxed);
}

const TokenLearner::Config& TokenLearner::config() const noexcept {
    return cfg_;
}

std::unordered_map<std::string, uint64_t>
TokenLearner::observation_counts() const {
    std::lock_guard lock(mtx_);
    return token_obs_;
}

} // namespace llmquant
