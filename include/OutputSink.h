#pragma once

#include <string>

namespace llmquant {

// Forward declaration — breaks the circular include with TradeSignalEngine.h.
struct TradeSignal;

/// Abstract output sink for routing trade signals to a destination.
///
/// Concrete sinks are provided in OutputSinkImpl.h (CsvOutputSink,
/// JsonOutputSink, MemoryOutputSink).  This header exposes only the pure
/// interface so that TradeSignalEngine.h can include it without introducing
/// a circular dependency.
///
/// Thread safety: individual sink implementations are NOT thread-safe.
/// External synchronisation is required if multiple threads call emit()
/// on the same sink instance concurrently.
class OutputSink {
public:
    virtual ~OutputSink() = default;

    /// Emit a single trade signal to this sink.
    ///
    /// # Arguments
    /// * `sig` — Fully populated TradeSignal to write.
    virtual void emit(const TradeSignal& sig) = 0;

    /// Flush any internally buffered output to the underlying destination.
    ///
    /// The default implementation is a no-op; file sinks override this.
    virtual void flush() {}
};

} // namespace llmquant
