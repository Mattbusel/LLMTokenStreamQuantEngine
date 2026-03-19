# ADR-001: 128-bit Deduplication Hash

**Status:** Accepted
**Date:** 2026-03-19

## Context

The deduplicator uses a hash of (token, context) as a lookup key. A single 64-bit
FNV-1a hash has a birthday-bound collision risk: after approximately 2^32 unique
inputs (~4 billion), the probability of at least one collision exceeds 50%. In a
high-throughput production deployment this threshold can be approached over weeks
of continuous operation.

## Decision

Perform two independent FNV-1a passes over the same input using different seeds,
producing a (hi, lo) 128-bit key. Both halves must match for a collision to be
silently accepted, giving collision resistance equivalent to 2^128 rather than 2^64.

## Alternatives Considered

- **xxHash128** (external dep): faster and well-studied, but introduces a new
  vcpkg dependency. Rejected to keep the dependency footprint minimal.
- **SHA-256**: cryptographically secure but ~10x slower on the hot path. Rejected
  — security beyond collision resistance is not required here.
- **Single 64-bit FNV-1a**: original implementation. Rejected due to birthday-bound
  collision risk over the expected system lifetime.

## Consequences

- No external dependency added; the two-pass implementation is self-contained in
  `src/Deduplicator.cpp`.
- Hash computation cost approximately doubles (~2 ns vs ~1 ns per token) — acceptable
  given that the hash is off the critical signal path.
- Collision probability is negligible for any realistic lifetime of continuous operation.
- The `DedupKey::value` field retains the low 64-bit half for backward compatibility
  with any consumers that only inspect the original field.
