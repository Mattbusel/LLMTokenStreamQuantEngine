/**
 * @file src/LanguageRouter.cpp
 * @brief Implementation of LanguageRouter and built-in weight tables.
 *
 * Language detection uses Unicode code-point range checks on UTF-8 bytes.
 * Weight tables are implemented as static unordered_maps of lowercase token
 * strings to f64 bias weights calibrated for financial sentiment in each
 * language.
 */

#include "LanguageRouter.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace llmquant {

// ---------------------------------------------------------------------------
// language_name
// ---------------------------------------------------------------------------

const char* language_name(Language lang) noexcept {
    switch (lang) {
        case Language::English: return "English";
        case Language::Spanish: return "Spanish";
        case Language::Chinese: return "Chinese";
        case Language::Arabic:  return "Arabic";
        case Language::Russian: return "Russian";
        case Language::Other:   return "Other";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// to_lower
// ---------------------------------------------------------------------------

/* static */
std::string LanguageRouter::to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) out.push_back(static_cast<char>(std::tolower(c)));
    return out;
}

// ---------------------------------------------------------------------------
// Built-in English weight table
// ---------------------------------------------------------------------------

namespace {

class EnglishWeightTable final : public ISemanticWeightTable {
public:
    EnglishWeightTable() {
        // Bullish terms (positive weight)
        weights_ = {
            {"bullish",       0.90}, {"bull",           0.85},
            {"rally",         0.80}, {"surge",          0.82},
            {"gain",          0.70}, {"gains",          0.70},
            {"upside",        0.75}, {"outperform",     0.78},
            {"buy",           0.72}, {"long",           0.68},
            {"positive",      0.65}, {"growth",         0.70},
            {"beat",          0.75}, {"strong",         0.65},
            {"upgrade",       0.80}, {"breakout",       0.82},
            {"momentum",      0.60}, {"optimistic",     0.72},
            {"recover",       0.65}, {"recovery",       0.65},
            {"record",        0.60}, {"high",           0.55},
            {"expansion",     0.60}, {"accelerate",     0.65},
            {"profit",        0.70}, {"profitable",     0.70},
            // Bearish terms (negative weight)
            {"bearish",      -0.90}, {"bear",          -0.85},
            {"sell",         -0.72}, {"short",         -0.68},
            {"decline",      -0.75}, {"drop",          -0.72},
            {"fall",         -0.70}, {"loss",          -0.75},
            {"losses",       -0.75}, {"downside",      -0.75},
            {"underperform", -0.78}, {"downgrade",     -0.80},
            {"negative",     -0.65}, {"weak",          -0.65},
            {"crash",        -0.90}, {"recession",     -0.85},
            {"slowdown",     -0.70}, {"contraction",   -0.72},
            {"risk",         -0.45}, {"uncertainty",   -0.50},
            {"concern",      -0.40}, {"warning",       -0.55},
            {"miss",         -0.70}, {"disappointing", -0.72},
            {"default",      -0.80}, {"bankruptcy",    -0.90},
            {"inflation",    -0.50}, {"volatility",    -0.35},
        };
    }

    [[nodiscard]] double lookup(const std::string& token) const override {
        auto it = weights_.find(token);
        return (it != weights_.end()) ? it->second : 0.0;
    }

    [[nodiscard]] Language language() const override { return Language::English; }

private:
    std::unordered_map<std::string, double> weights_;
};

// ---------------------------------------------------------------------------
// Spanish weight table
// ---------------------------------------------------------------------------

class SpanishWeightTable final : public ISemanticWeightTable {
public:
    SpanishWeightTable() {
        weights_ = {
            // Bullish
            {"alcista",      0.90}, {"alza",          0.80},
            {"rebote",       0.72}, {"subida",        0.75},
            {"ganancia",     0.70}, {"ganancias",     0.70},
            {"compra",       0.72}, {"largo",         0.68},
            {"positivo",     0.65}, {"crecimiento",   0.70},
            {"recuperar",    0.65}, {"recuperacion",  0.65},
            {"beneficio",    0.70}, {"rentable",      0.70},
            {"fuerte",       0.60}, {"optimista",     0.72},
            {"superar",      0.75}, {"mejora",        0.68},
            // Bearish
            {"bajista",     -0.90}, {"baja",         -0.80},
            {"caida",       -0.75}, {"venta",        -0.72},
            {"perdida",     -0.75}, {"perdidas",     -0.75},
            {"negativo",    -0.65}, {"debil",        -0.65},
            {"recesion",    -0.85}, {"quiebra",      -0.90},
            {"riesgo",      -0.45}, {"incertidumbre",-0.50},
            {"crisis",      -0.82}, {"descenso",     -0.75},
            {"inflacion",   -0.50}, {"volatilidad",  -0.35},
            {"deterioro",   -0.70}, {"preocupacion", -0.40},
        };
    }

    [[nodiscard]] double lookup(const std::string& token) const override {
        auto it = weights_.find(token);
        return (it != weights_.end()) ? it->second : 0.0;
    }

    [[nodiscard]] Language language() const override { return Language::Spanish; }

private:
    std::unordered_map<std::string, double> weights_;
};

// ---------------------------------------------------------------------------
// Chinese weight table (Pinyin romanisation + key characters)
// ---------------------------------------------------------------------------

class ChineseWeightTable final : public ISemanticWeightTable {
public:
    ChineseWeightTable() {
        weights_ = {
            // Bullish (Pinyin romanisation)
            {"duotou",      0.90}, {"shangzhang",    0.80},
            {"fanzhang",    0.72}, {"zuoyue",        0.68},
            {"yingli",      0.70}, {"zengzhang",     0.72},
            {"qiangshi",    0.65}, {"leiguan",       0.70},
            {"jishang",     0.75}, {"tupo",          0.80},
            // Bearish
            {"kongfang",   -0.90}, {"xiadie",       -0.80},
            {"kuisun",     -0.75}, {"ruoshi",       -0.65},
            {"shuaitui",   -0.72}, {"fengxian",     -0.45},
            {"weiji",      -0.80}, {"biaodie",      -0.72},
            {"tongzuo",    -0.70}, {"baodie",       -0.85},
            // Common characters (UTF-8 sequences rendered as escape strings)
            {"\xe6\xb6\xa8", 0.80},  // 涨 (rise)
            {"\xe8\xb7\x8c",-0.80},  // 跌 (fall)
            {"\xe5\x88\xa9\xe6\xb6\xa6", 0.72}, // 利润 (profit)
            {"\xe4\xba\x8f\xe6\x8d\x9f",-0.75}, // 亏损 (loss)
        };
    }

    [[nodiscard]] double lookup(const std::string& token) const override {
        auto it = weights_.find(token);
        return (it != weights_.end()) ? it->second : 0.0;
    }

    [[nodiscard]] Language language() const override { return Language::Chinese; }

private:
    std::unordered_map<std::string, double> weights_;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

std::unique_ptr<ISemanticWeightTable> make_english_weight_table() {
    return std::make_unique<EnglishWeightTable>();
}

std::unique_ptr<ISemanticWeightTable> make_spanish_weight_table() {
    return std::make_unique<SpanishWeightTable>();
}

std::unique_ptr<ISemanticWeightTable> make_chinese_weight_table() {
    return std::make_unique<ChineseWeightTable>();
}

// ---------------------------------------------------------------------------
// LanguageRouter construction
// ---------------------------------------------------------------------------

LanguageRouter::LanguageRouter(Config config)
    : config_(std::move(config))
{}

// ---------------------------------------------------------------------------
// Table management
// ---------------------------------------------------------------------------

void LanguageRouter::register_table(std::unique_ptr<ISemanticWeightTable> table) {
    std::lock_guard<std::mutex> lk(mutex_);
    tables_[static_cast<int>(table->language())] = std::move(table);
}

void LanguageRouter::unregister_table(Language lang) {
    std::lock_guard<std::mutex> lk(mutex_);
    tables_.erase(static_cast<int>(lang));
}

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

Language LanguageRouter::detect(const std::string& token) const noexcept {
    if (token.empty()) return config_.default_language;

    // Scan bytes for Unicode block membership.
    bool has_cjk     = false;
    bool has_arabic  = false;
    bool has_cyrillic= false;
    bool has_spanish = false;
    bool has_non_ascii = false;

    const unsigned char* p = reinterpret_cast<const unsigned char*>(token.data());
    const unsigned char* end = p + token.size();

    while (p < end) {
        unsigned char b0 = *p;

        if (b0 < 0x80) {
            // ASCII — check for Spanish-specific punctuation.
            if (b0 == 0xBF || b0 == 0xA1) { // ¿ ¡
                has_spanish = true;
            }
            ++p;
            continue;
        }

        has_non_ascii = true;

        if ((b0 & 0xE0) == 0xC0 && p + 1 < end) {
            // 2-byte sequence: U+0080–U+07FF
            uint32_t cp = ((b0 & 0x1F) << 6) | (p[1] & 0x3F);
            if (cp >= 0x0400 && cp <= 0x04FF) has_cyrillic = true;
            if (cp >= 0x0600 && cp <= 0x06FF) has_arabic   = true;
            // Spanish diacritics: á é í ó ú ü ñ Á É etc.
            if ((cp >= 0x00C0 && cp <= 0x00FF)
             || cp == 0x00F1 || cp == 0x00D1    // ñ Ñ
             || cp == 0x00FC || cp == 0x00DC) {  // ü Ü
                has_spanish = true;
            }
            p += 2;
        } else if ((b0 & 0xF0) == 0xE0 && p + 2 < end) {
            // 3-byte sequence: U+0800–U+FFFF
            uint32_t cp = ((b0 & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
            if (cp >= 0x4E00 && cp <= 0x9FFF) has_cjk     = true; // CJK Unified
            if (cp >= 0x3040 && cp <= 0x30FF) has_cjk     = true; // Hiragana/Katakana
            if (cp >= 0xAC00 && cp <= 0xD7AF) has_cjk     = true; // Hangul
            p += 3;
        } else if ((b0 & 0xF8) == 0xF0 && p + 3 < end) {
            // 4-byte sequence: U+10000+
            p += 4;
        } else {
            ++p; // malformed, skip
        }
    }

    // Priority: CJK > Arabic > Cyrillic > Spanish > English > Other.
    if (has_cjk)      return Language::Chinese;
    if (has_arabic)   return Language::Arabic;
    if (has_cyrillic) return Language::Russian;
    if (has_spanish)  return Language::Spanish;
    if (!has_non_ascii) return Language::English;
    return config_.default_language;
}

// ---------------------------------------------------------------------------
// route
// ---------------------------------------------------------------------------

LanguageRouter::RouteResult LanguageRouter::route(const std::string& token) const {
    tokens_routed_.fetch_add(1, std::memory_order_relaxed);

    Language lang = detect(token);
    std::string lower = to_lower(token);

    double weight   = 0.0;
    bool   fallback = false;

    auto lookup_in = [&](Language l) -> bool {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = tables_.find(static_cast<int>(l));
        if (it != tables_.end()) {
            weight = it->second->lookup(lower);
            return true;
        }
        return false;
    };

    if (!lookup_in(lang)) {
        // Try default language.
        if (lang != config_.default_language) {
            fallback = lookup_in(config_.default_language);
        }
        if (!fallback) {
            weight   = 0.0;
            fallback = true;
        }
        fallback_count_.fetch_add(1, std::memory_order_relaxed);
    }

    if (config_.verbose && config_.log_fn) {
        std::ostringstream os;
        os << "[LanguageRouter] token=\"" << token
           << "\" lang=" << language_name(lang)
           << " weight=" << weight
           << (fallback ? " (fallback)" : "");
        config_.log_fn(os.str());
    }

    return RouteResult{ lang, weight, fallback };
}

// ---------------------------------------------------------------------------
// Statistics JSON
// ---------------------------------------------------------------------------

std::string LanguageRouter::to_stats_json() const {
    uint64_t routed   = tokens_routed_.load(std::memory_order_relaxed);
    uint64_t fallback = fallback_count_.load(std::memory_order_relaxed);
    std::size_t table_count{0};
    {
        std::lock_guard<std::mutex> lk(mutex_);
        table_count = tables_.size();
    }
    std::ostringstream os;
    os << "{"
       << "\"tokens_routed\":"  << routed   << ","
       << "\"fallback_count\":" << fallback << ","
       << "\"tables_registered\":" << table_count
       << "}";
    return os.str();
}

} // namespace llmquant
