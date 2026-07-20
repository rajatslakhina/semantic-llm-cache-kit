import Foundation

// MARK: - Query

/// A request for an LLM-backed feature, as seen by the gateway.
///
/// `templateVersion` is part of the cache identity on purpose: when a team
/// ships a new prompt template for a feature, every previously cached response
/// was produced under different instructions and must never be served again.
/// Bumping the version makes stale entries unreachable immediately (they are
/// swept lazily), which is the same invalidation discipline schema versions
/// give a database migration.
public struct SemanticQuery: Sendable, Equatable {
    public let feature: String
    public let text: String
    public let templateVersion: Int

    public init(feature: String, text: String, templateVersion: Int = 1) {
        self.feature = feature
        self.text = text
        // Negative versions have no meaning; clamp rather than trap.
        self.templateVersion = max(0, templateVersion)
    }
}

// MARK: - Generation output

/// A complete, successful model response with its token accounting.
///
/// The cache stores only complete responses — a partial (mid-stream failed)
/// generation is never admitted, because serving a truncated answer from cache
/// would silently repeat the failure forever.
public struct GeneratedResponse: Sendable, Codable, Equatable {
    public let text: String
    public let promptTokens: Int
    public let completionTokens: Int

    public var totalTokens: Int { promptTokens + completionTokens }

    public init(text: String, promptTokens: Int, completionTokens: Int) {
        self.text = text
        // Token counts come from external providers; never trust them to be
        // non-negative. Clamping keeps every downstream budget computation sane.
        self.promptTokens = max(0, promptTokens)
        self.completionTokens = max(0, completionTokens)
    }
}

// MARK: - Cache entry

/// One admitted cache entry. `normalizedText` and `embedding` are both kept:
/// the embedding drives similarity lookup, the normalized text drives
/// duplicate-admission checks and human-readable inspection in tooling.
public struct CachedEntry: Sendable, Codable, Identifiable, Equatable {
    public let id: UUID
    public let feature: String
    public let templateVersion: Int
    public let normalizedText: String
    public let embedding: [Double]
    public let response: GeneratedResponse
    public let createdAt: Date
    public let expiresAt: Date?
    public internal(set) var lastAccessedAt: Date

    /// Approximate in-memory footprint used for byte-budget accounting.
    /// Deliberately an estimate: exact malloc-level accounting buys nothing
    /// here, while a stable, deterministic estimate makes eviction testable.
    public var approximateByteCost: Int {
        normalizedText.utf8.count
            + response.text.utf8.count
            + embedding.count * MemoryLayout<Double>.size
            + 128 // fixed overhead for identity, dates and bookkeeping
    }

    public init(
        id: UUID,
        feature: String,
        templateVersion: Int,
        normalizedText: String,
        embedding: [Double],
        response: GeneratedResponse,
        createdAt: Date,
        expiresAt: Date?,
        lastAccessedAt: Date
    ) {
        self.id = id
        self.feature = feature
        self.templateVersion = max(0, templateVersion)
        self.normalizedText = normalizedText
        self.embedding = embedding
        self.response = response
        self.createdAt = createdAt
        self.expiresAt = expiresAt
        self.lastAccessedAt = lastAccessedAt
    }
}

// MARK: - Lookup / admission outcomes

/// The result of a semantic lookup. A lookup lands in exactly one of these
/// two states — expired entries encountered during the scan are removed and
/// counted separately, but the *lookup itself* still resolves to hit or miss,
/// which is what makes the metrics conservation law checkable.
public enum SemanticLookupResult: Sendable, Equatable {
    /// Best candidate met the similarity threshold.
    case hit(CachedEntry, similarity: Double)
    /// No candidate met the threshold. `bestSimilarity` is reported for
    /// observability (it is how you tune the threshold in production);
    /// nil when there were no live candidates at all.
    case miss(bestSimilarity: Double?)
}

/// Why an admission did or did not enter the cache.
public enum AdmissionOutcome: Sendable, Equatable {
    case admitted(evictedForCapacity: Int)
    /// Entry alone exceeds the byte budget — admitting it would evict the
    /// whole cache and still not fit.
    case rejectedOversized
    /// Cache configured with zero capacity (entries or bytes).
    case rejectedZeroCapacity
    /// A live entry for the same feature+version is already within the
    /// similarity threshold — admitting a near-duplicate would burn capacity
    /// without improving hit rate.
    case rejectedNearDuplicate(existing: UUID, similarity: Double)
    /// The embedding was empty or dimensionally inconsistent with the entry set.
    case rejectedInvalidEmbedding
}

/// Result of restoring a persisted snapshot.
public struct SnapshotRestoreOutcome: Sendable, Equatable {
    public let restored: Int
    public let droppedExpired: Int
    public let droppedForCapacity: Int
    /// True when the snapshot's schema version did not match and the whole
    /// snapshot was discarded — a version-skewed cache is worse than a cold one.
    public let discardedSchemaMismatch: Bool
}

// MARK: - Metrics

/// Point-in-time cache accounting. The invariant these counters exist to
/// prove (and that the test suite pins):
///
///     admitted == liveCount + expiredRemovals + capacityEvictions
///                 + trimRemovals + invalidatedRemovals
///
/// Every admitted entry leaves the cache through exactly one door. If that
/// equation ever breaks, entries are leaking or being double-counted.
public struct CacheMetricsSnapshot: Sendable, Equatable {
    public var lookups: Int = 0
    public var hits: Int = 0
    public var misses: Int = 0
    public var admitted: Int = 0
    public var admissionRejections: Int = 0
    public var expiredRemovals: Int = 0
    public var capacityEvictions: Int = 0
    public var trimRemovals: Int = 0
    public var invalidatedRemovals: Int = 0
    /// Tokens that a hit avoided re-spending (the cached response's total).
    public var tokensSaved: Int = 0

    public init() {}
}

// MARK: - Gateway outcomes

/// Where a gateway response came from.
public enum GatewayResponseSource: Sendable, Equatable {
    /// Served from the semantic cache at the given similarity. Zero tokens charged.
    case semanticCache(similarity: Double)
    /// Freshly generated; the caller's feature budget was charged.
    case generated
    /// Joined an identical in-flight generation started by another caller.
    /// Zero tokens charged — the owning request pays exactly once.
    case coalesced
}

public struct GatewayResponse: Sendable, Equatable {
    public let text: String
    public let source: GatewayResponseSource
    public let tokensCharged: Int

    public init(text: String, source: GatewayResponseSource, tokensCharged: Int) {
        self.text = text
        self.source = source
        self.tokensCharged = max(0, tokensCharged)
    }
}

/// Gateway-level counters (cache and ledger keep their own detailed books).
public struct GatewayStats: Sendable, Equatable {
    public var cacheHits: Int = 0
    public var generated: Int = 0
    public var coalesced: Int = 0
    public var budgetDenials: Int = 0
    public var generationFailures: Int = 0

    public init() {}
}

/// Errors the gateway surfaces to feature code. String payloads (rather than
/// nested `any Error`) keep this Equatable and therefore directly assertable
/// in tests.
public enum GatewayError: Error, Sendable, Equatable {
    /// Query text was empty after normalization.
    case emptyQuery
    /// The embedder failed; the request cannot be cache-checked or keyed.
    case embeddingFailed(reason: String)
    /// The feature's token budget cannot cover the estimated spend.
    /// `retryAfter` is the ledger's estimate of when budget frees up
    /// (nil when the request could never fit the configured budget).
    case budgetExhausted(feature: String, retryAfter: TimeInterval?)
    /// Generation failed after budget was reserved; the reservation has been
    /// released and nothing was cached.
    case generationFailed(feature: String, reason: String)
}
