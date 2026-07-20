# SemanticLLMCacheKit

**Every LLM feature you ship has a meter running.** Users ask the same questions in slightly different words all day — "track my order", "where's my order?", "order status please" — and a naive client pays full token price for every one of them. This package is the client-side answer: a **semantic response cache** (embedding-similarity keyed, not exact-string keyed) composed with a **per-feature token-budget ledger** (reserve-then-settle, sliding window) and **single-flight request coalescing**, behind one gateway actor that a feature calls instead of calling the model directly.

This is the "design a client-side AI gateway" system-design interview question — semantic caching, budget enforcement, request deduplication — as real, tested, runnable Swift instead of a whiteboard sketch.

> **Demo app:** [semantic-llm-cache-kit-demo-app](https://github.com/rajatslakhina/semantic-llm-cache-kit-demo-app) — a separate repo that consumes this package as a remote SPM dependency, the way any real consumer would.

## Why this matters

An Engineering Lead putting an LLM feature into a consumer app owns three problems on day one:

1. **Cost.** Token spend scales with raw request volume, but a large fraction of consumer queries are semantic repeats. An exact-string cache catches almost none of them; a similarity cache catches most.
2. **Blast radius.** One runaway feature (a retry loop, a viral screen) can burn the whole app's inference budget. Budgets must be *per-feature*, enforced client-side before the request leaves the device, and fail with a *retry-after*, not a silent drop.
3. **Stampedes.** The same prompt fired concurrently (double-tap, list refresh, multiple views sharing a question) must cost exactly one generation.

The three problems interact — which is why this is one composed system, not three utilities. The interactions are where the design decisions live.

## Architecture

```
                       SemanticGateway (actor)
                             │
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                     ▼
  Embedder (seam)    SemanticResponseCache   FeatureTokenLedger
  hash-based sim ·     (actor)                 (actor)
  real encoder drops   similarity lookup       sliding-window spend
  in without changes   TTL · LRU+byte evict    reserve → settle/release
                       near-dup admission      per-feature budgets
                       versioned snapshots     retry-after on denial
        ▼
  ResponseGenerator (seam) ← your real model client goes here
```

**Request pipeline (order is load-bearing):** normalize → embed → **cache lookup → budget check** → coalesce-or-register → generate → settle → admit.

The cache is checked *before* the budget on purpose: cache hits are free, so a feature that has exhausted its budget degrades to "answers it already paid for" instead of turning off. That single ordering decision is the difference between a feature that gets slower to update and a feature that errors out.

## Design decisions (and the alternatives they beat)

| Decision | Rejected alternative | Why |
|---|---|---|
| **Exact linear similarity scan** over live candidates | ANN index (IVF/HNSW) | A response cache is bounded and small (hundreds of entries, enforced). Exact scan is microseconds, always correct, zero index maintenance. ANN buys speed the cache doesn't need and pays with *false negatives* — missed hits are re-spent tokens. |
| **Reserve-then-settle budget accounting** | Advisory check-then-record | Advisory accounting lets N concurrent requests all pass the same check and collectively overshoot by N−1 requests. Reservations hold the estimate at decision time; actual spend replaces it on success, full release on failure. |
| **Cache-before-budget pipeline order** | Budget-first | Budget-first makes an exhausted feature return errors for questions it already answered. Strictly worse for users; saves nothing. |
| **Near-duplicate admission rejection** | Admit everything | Two entries answering the same neighborhood burn capacity without adding hit rate. Admission is a policy, not an insert. |
| **Coalescing on *exact* normalized key** | Coalescing on semantic similarity | Joining a flight is only safe when callers provably asked the same thing. Semantic "same-ish" belongs to the cache, where a threshold is tunable; in coalescing it would hand caller A an answer to caller B's different question mid-flight. |
| **Template version in the cache identity** | Time-based invalidation only | A prompt-template rollout changes what a correct answer *is*. Version bump makes stale entries unreachable immediately — the same discipline as schema versioning. |
| **Whole-snapshot discard on schema mismatch** | Best-effort migration | A cache is a performance optimization. Rehydrating entries whose semantics changed risks correctness to save a cold start. |
| **Fractional `trim(toFraction:)` under memory pressure** | `removeAll()` | A memory warning should cost some hit rate, not all of it. Trim keeps the warmest working set. |
| **chars/4 token estimator behind a seam** | Bundling a real BPE tokenizer | Estimates only need to be proportionate — settle-time correction against the provider's actual counts does the rest. A vocab table is hundreds of KB to improve a number that gets replaced anyway. |

## Failure modes, as designed states

- **Generation fails** → budget reservation released in full, nothing cached, every caller of that flight gets the error, flight deregistered so the *next* identical request generates fresh. Failures never poison future requests (pinned by a dedicated test).
- **Budget exhausted** → `budgetExhausted(feature:retryAfter:)` with the ledger's estimate of when the oldest spend record ages out; cached answers still serve.
- **Embedding fails** → request fails fast, wrapped; nothing charged, nothing cached.
- **Mid-flight duplicate registration race** → structurally impossible: check-then-register is synchronous inside the gateway actor, and budget authorization runs *inside* the flight task so it cannot reopen the race.
- **Slow completion vs. successor flight** → deregistration is identity-checked; a stale completion can never evict its successor's registration.
- **Relaunch** → versioned snapshot restore drops expired entries, respects capacity warmest-first, and discards wholesale on schema mismatch.

## What the simulation is (honestly)

The bundled `DeterministicHashEmbedder` (FNV-1a trigram hashing, L2-normalized) and `SimulatedAssistantGenerator` are labeled simulations, chosen so the *system* — thresholds, TTL, eviction, budgets, coalescing — is what gets tested, deterministically, on any platform. A real encoder (Core ML sentence encoder, Foundation Models embedding API) and a real model client drop into the `Embedder` and `ResponseGenerator` seams without touching any other code. The hash embedder rewards wording overlap, not true paraphrase similarity — that limitation is inherent to the simulation, not the design.

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/rajatslakhina/semantic-llm-cache-kit.git", branch: "main")
]
```

## Usage

```swift
import SemanticLLMCacheKit

let cache = SemanticResponseCache(
    configuration: SemanticCacheConfiguration(similarityThreshold: 0.90, ttl: 3600)
)
let ledger = FeatureTokenLedger(
    configuration: TokenBudgetConfiguration(budgets: ["support-chat": 20_000])
)
let gateway = SemanticGateway(
    cache: cache,
    ledger: ledger,
    embedder: DeterministicHashEmbedder(),        // swap in your real encoder
    generator: SimulatedAssistantGenerator()      // swap in your real model client
)

let response = try await gateway.respond(
    to: SemanticQuery(feature: "support-chat", text: "Where is my order?", templateVersion: 1)
)
// response.source: .semanticCache(similarity:) | .generated | .coalesced
// response.tokensCharged: 0 for cache hits and coalesced joins
```

## Running the tests

```bash
swift build
swift test
```

65 tests cover the crash-and-correctness edges: exact threshold-boundary hits (built from bit-identical floating-point expressions, not rounded constants), TTL expiry at the exact boundary, LRU + byte-budget eviction, zero-capacity configs, near-duplicate admission, a metrics **conservation law** (every admitted entry provably leaves through exactly one exit door), reserve/settle/release budget races, retry-after computation, window slide, N-concurrent-request coalescing into one generation, failure delivery to all coalesced callers with no poisoning of the next request, cache-hits-despite-exhausted-budget, template-version bumps, and snapshot schema mismatch/expiry/capacity on restore.

## Verification (honest)

`swift build` and `swift test` were run for real on Linux (Swift 6 toolchain, strict concurrency language mode) in this repo's CI-equivalent environment before pushing; the full suite passed. SwiftUI does not exist on Linux, so nothing here imports it — this package is pure Foundation by design. The companion demo app repo documents its own (separately honest) verification tier.
