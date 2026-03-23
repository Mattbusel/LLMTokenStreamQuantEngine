/**
 * @file python/bindings.cpp
 * @brief Python pybind11 bindings for LLMTokenStreamQuantEngine.
 *
 * Exposes a high-level `TokenSignalEngine` class that wraps the C++ pipeline
 * (LLMAdapter + TradeSignalEngine) for use from Python.
 *
 * ## Build
 * See python/setup.py or use CMake with -DLLMQUANT_ENABLE_PYTHON_BINDINGS=ON.
 *
 * ## Python usage
 * ```python
 * import llmquant
 * engine = llmquant.TokenSignalEngine()
 * sig = engine.process_token("bullish")
 * print(sig.direction, sig.strength, sig.confidence)
 * ```
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

#include "LLMAdapter.h"
#include "TradeSignalEngine.h"

namespace py = pybind11;
using namespace llmquant;

// ── Signal: simplified Python-facing signal datatype ─────────────────────────

/**
 * @brief Simplified signal datatype exposed to Python.
 *
 * Derived from TradeSignal with Python-friendly field names:
 *   - `direction`:  +1 (bullish), 0 (neutral), -1 (bearish)
 *   - `strength`:   magnitude of the directional bias shift in [-1, +1]
 *   - `confidence`: signal confidence in [0, 1]
 *
 * All other TradeSignal fields are available via the `raw_json()` method.
 */
struct Signal {
    /// +1 = bullish, 0 = neutral, -1 = bearish
    int direction{0};
    /// Magnitude of the bias shift (signed), clamped to [-1, +1].
    double strength{0.0};
    /// Model confidence in [0.0, 1.0].
    double confidence{0.0};
    /// Volatility adjustment from this signal.
    double volatility{0.0};
    /// Token-to-signal latency in microseconds.
    double latency_us{0.0};
    /// Full JSON serialisation of the underlying TradeSignal.
    std::string raw_json{};

    /// Construct from a TradeSignal.
    explicit Signal(const TradeSignal& ts)
        : direction(ts.strategy_toggle)
        , strength(std::max(-1.0, std::min(1.0, ts.delta_bias_shift)))
        , confidence(ts.confidence)
        , volatility(ts.volatility_adjustment)
        , latency_us(ts.latency_us)
        , raw_json(ts.to_json())
    {}

    /// Neutral (no signal) constructor.
    Signal() = default;

    std::string repr() const {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "Signal(direction=%+d, strength=%.4f, confidence=%.4f, latency_us=%.2f)",
            direction, strength, confidence, latency_us);
        return buf;
    }
};

// ── TokenSignalEngine: the main Python-facing engine class ────────────────────

/**
 * @brief Python-facing engine that maps LLM tokens to quantitative signals.
 *
 * Wraps the LLMAdapter (token -> SemanticWeight) and TradeSignalEngine
 * (SemanticWeight -> TradeSignal) in a single object.
 *
 * Thread safety: Not thread-safe. Use one engine per thread or protect with
 * an external lock.
 */
class TokenSignalEngine {
public:
    /**
     * @brief Construct a new engine with default configuration.
     *
     * Loads the built-in ~130-token dictionary. Additional tokens can be
     * added via add_token_mapping().
     */
    explicit TokenSignalEngine()
        : adapter_()
        , engine_()
    {}

    /**
     * @brief Process a single token and return the resulting signal.
     *
     * @param token  Raw token string (case-sensitive, e.g. "bullish", "crash").
     * @return Signal with direction, strength, and confidence. Returns a
     *         neutral Signal (direction=0, strength=0, confidence=0.5) when
     *         the engine is in cooldown or the token is unknown.
     */
    Signal process_token(const std::string& token) {
        SemanticWeight w = adapter_.map_token_to_weight(token);
        auto opt_signal = engine_.process_semantic_weight(w);
        if (opt_signal.has_value()) {
            return Signal(*opt_signal);
        }
        // In cooldown or no signal: return a neutral signal.
        Signal neutral;
        neutral.confidence = 0.5;
        return neutral;
    }

    /**
     * @brief Process a sequence of tokens and return the aggregate signal.
     *
     * Tokens are processed in order. Only the last emitted TradeSignal (if any)
     * is returned. Returns a neutral Signal if the sequence produces no emission.
     *
     * @param tokens  Ordered list of token strings.
     */
    Signal process_tokens(const std::vector<std::string>& tokens) {
        Signal last;
        last.confidence = 0.5;
        for (const auto& tok : tokens) {
            Signal s = process_token(tok);
            if (s.direction != 0 || s.strength != 0.0) {
                last = s;
            }
        }
        return last;
    }

    /**
     * @brief Add a custom token -> signal mapping to the dictionary.
     *
     * @param token      Token string.
     * @param sentiment  Sentiment score in [-1, 1].
     * @param confidence Confidence in [0, 1].
     * @param volatility Volatility score in [0, 1].
     * @param bias       Directional bias in [-1, 1].
     */
    void add_token_mapping(const std::string& token,
                           double sentiment,
                           double confidence,
                           double volatility,
                           double bias) {
        SemanticWeight w;
        w.sentiment_score  = sentiment;
        w.confidence_score = confidence;
        w.volatility_score = volatility;
        w.directional_bias = bias;
        adapter_.add_token_mapping(token, w);
    }

    /**
     * @brief Reset the engine accumulators (bias, volatility, cooldown).
     *
     * Call this between independent token streams to avoid state leakage.
     */
    void reset() {
        engine_.reset();
    }

    /**
     * @brief Return a JSON statistics summary of the engine state.
     */
    std::string get_stats_json() const {
        // format_stats returns a human-readable string; wrap in simple JSON.
        std::string stats = engine_.format_stats();
        return "{\"stats\": \"" + stats + "\"}";
    }

private:
    LLMAdapter        adapter_;
    TradeSignalEngine engine_;
};

// ── pybind11 module definition ────────────────────────────────────────────────

PYBIND11_MODULE(llmquant, m) {
    m.doc() = R"doc(
LLMTokenStreamQuantEngine Python bindings.

Maps LLM token streams to quantitative trading signals using the C++20
production engine with sub-10us token-to-signal P99 latency.

Quick start::

    import llmquant
    engine = llmquant.TokenSignalEngine()

    # Process a single token
    sig = engine.process_token("bullish")
    print(f"direction={sig.direction} strength={sig.strength:.4f} confidence={sig.confidence:.4f}")

    # Process a token sequence
    tokens = ["economy", "expanding", "bullish", "growth", "positive"]
    sig = engine.process_tokens(tokens)

    # Add a custom token
    engine.add_token_mapping("moon", sentiment=0.9, confidence=0.8, volatility=0.7, bias=0.9)
)doc";

    // ── Signal ────────────────────────────────────────────────────────────────
    py::class_<Signal>(m, "Signal",
        "Quantitative trade signal derived from one or more LLM tokens.")
        .def(py::init<>())
        .def_readonly("direction",   &Signal::direction,
            "Signal direction: +1 (bullish), 0 (neutral), -1 (bearish).")
        .def_readonly("strength",    &Signal::strength,
            "Signed bias magnitude, clamped to [-1, +1].")
        .def_readonly("confidence",  &Signal::confidence,
            "Model confidence in [0.0, 1.0].")
        .def_readonly("volatility",  &Signal::volatility,
            "Implied volatility adjustment from this signal.")
        .def_readonly("latency_us",  &Signal::latency_us,
            "Token-to-signal processing latency in microseconds.")
        .def_readonly("raw_json",    &Signal::raw_json,
            "Full JSON serialisation of the underlying TradeSignal.")
        .def("__repr__",             &Signal::repr)
        .def("is_bullish",  [](const Signal& s) { return s.direction > 0; },
            "True if direction is bullish (+1).")
        .def("is_bearish",  [](const Signal& s) { return s.direction < 0; },
            "True if direction is bearish (-1).")
        .def("is_neutral",  [](const Signal& s) { return s.direction == 0; },
            "True if direction is neutral (0).");

    // ── TokenSignalEngine ─────────────────────────────────────────────────────
    py::class_<TokenSignalEngine>(m, "TokenSignalEngine",
        R"doc(
Main engine class. Maps LLM tokens to quantitative signals.

The engine wraps the C++ LLMAdapter (token dictionary lookup) and
TradeSignalEngine (exponential-decay accumulator, cooldown gate, risk gating).

Example::

    engine = llmquant.TokenSignalEngine()
    sig = engine.process_token("recession")
    if sig.is_bearish():
        print(f"Sell signal: strength={sig.strength:.3f}")
)doc")
        .def(py::init<>(),
            "Construct with default configuration and built-in token dictionary.")
        .def("process_token",        &TokenSignalEngine::process_token,
            py::arg("token"),
            "Process a single token and return the resulting Signal.")
        .def("process_tokens",       &TokenSignalEngine::process_tokens,
            py::arg("tokens"),
            "Process a list of tokens in order and return the last emitted Signal.")
        .def("add_token_mapping",    &TokenSignalEngine::add_token_mapping,
            py::arg("token"),
            py::arg("sentiment"),
            py::arg("confidence"),
            py::arg("volatility"),
            py::arg("bias"),
            "Register a custom token -> SemanticWeight mapping.")
        .def("reset",                &TokenSignalEngine::reset,
            "Reset engine accumulators (bias, volatility, cooldown).")
        .def("get_stats_json",       &TokenSignalEngine::get_stats_json,
            "Return a JSON string with engine runtime statistics.");
}
