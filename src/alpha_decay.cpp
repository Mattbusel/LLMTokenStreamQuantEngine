/// @file src/alpha_decay.cpp
/// @brief Alpha Decay Model — compilation unit.
///
/// All logic is header-only in include/alpha_decay.hpp.  This file exists
/// to provide a translation unit for the AlphaDecay feature flag in CMakeLists
/// and to allow future non-inlineable logic without ABI breakage.

#include "alpha_decay.hpp"

namespace llmquant {

// All definitions are header-only (inline / constexpr).
// Future non-trivial implementations (e.g. SIMD-optimised batch decay) can
// be added here without changing the public API.

} // namespace llmquant
