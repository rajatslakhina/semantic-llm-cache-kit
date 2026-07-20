import Foundation

// MARK: - Configuration

public struct SemanticCacheConfiguration: Sendable {
    /// Maximum live entries. Clamped to >= 0; zero means "cache disabled"
    /// (every admission rejected, every lookup a miss) rather than a trap.
    public let maxEntries: Int
    /// Maximum total approximate bytes. Clamped to >= 0.
    public let maxTotalBytes: Int
    /// Cosine similarity a candidate must meet (>=) to count as a hit.
    /// Clamped into [0, 1]. This is THE tuning knob of a semantic cache:
    /// too low and users get wrong answers to different questions (false
    /// hits), too high and the cache never fires. The cache reports
    /// `bestSimilarity` on misses precisely so this can be tuned from data.
    public let similarityThreshold: Double
    /// Time-to-live for entries. nil = entries never expire by age.
    /// Non-positive values are normalized to nil at init.
    public let ttl: TimeInterval?

    public init(
        maxEntries: Int = 256,
        maxTotalBytes: Int = 4_000_000,
        similarityThreshold: Double = 0.90,
        ttl: TimeInterval? = 60 * 60
    ) {
        self.maxEntries = max(0, maxEntries)
        self.maxTotalBytes = max(0, maxTotalBytes)
        self.similarityThreshold = min(1, max(0, similarityThreshold))
        if let ttl, ttl > 0 {
            self.ttl = ttl
        } else {
            self.ttl = nil
        }
    }
}

// MARK: - Snapshot

/// Versioned persistence format. Schema version mismatches discard the whole
/// snapshot: a cache is a performance optimization, and rehydrating entries
/// whose layout or semantics changed risks correctness for a cold-start save.
public struct CacheSnapshot: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let exportedAt: Date
    public let entries: [CachedEntry]

    public init(schemaVersion: Int = CacheSnapshot.currentSchemaVersion, exportedAt: Date, entries: [CachedEntry]) {
        self.schemaVersion = schemaVersion
        self.exportedAt = exportedAt
        self.entries = entries
    }
}

// MARK: - Cache actor

/// Embedding-similarity keyed response cache with TTL, LRU + byte-budget
/// eviction, near-duplicate admission control, and versioned snapshots.
///
/// ## Concurrency & ordering guarantees
/// The actor is the single mutation boundary. Every public method completes
/// its mutation without interior `await`s, so each call is atomic with
/// respect to other cache operations — there is no window where a lookup can
/// observe a half-applied eviction.
///
/// ## Why exact linear scan, not an ANN index
/// Lookup scans all live candidates for the feature+version and computes
/// exact cosine similarity. A response cache is *bounded and small* (hundreds
/// of entries, enforced by `maxEntries`) — at that size an exact O(n) scan is
/// microseconds, always correct, and has zero index-maintenance cost.
/// Rejected alternative: an IVF/HNSW approximate index (as used for
/// thousand-to-million-vector RAG corpora) — here it would add partition
/// tuning and *false negatives* (missed hits = re-spent tokens) to shave
/// microseconds off a scan that is already effectively free at cache scale.
///
/// ## Why array-based LRU, not a linked list
/// Recency is a UUID array (least-recent first). Moves are O(n) in cache
/// size; with n bounded in the hundreds this is nanoseconds-to-microseconds,
/// and the array is trivially inspectable in tests. Rejected alternative: an
/// intrusive doubly-linked list (O(1) moves) — measurably better only when a
/// cache holds tens of thousands of entries, which this one, by design and by
/// clamp, never does.
public actor SemanticResponseCache {
    private let configuration: SemanticCacheConfiguration
    private let clock: any CacheClock

    private var entries: [UUID: CachedEntry] = [:]
    /// Least-recently-used first. Invariant: exactly the keys of `entries`.
    private var recency: [UUID] = []
    private var totalBytes: Int = 0
    private var metrics = CacheMetricsSnapshot()

    public init(configuration: SemanticCacheConfiguration, clock: any CacheClock = SystemCacheClock()) {
        self.configuration = configuration
        self.clock = clock
    }

    // MARK: Lookup

    /// Finds the best live candidate for `feature` + `templateVersion` at or
    /// above the similarity threshold. Expired entries encountered during the
    /// scan are removed and counted as `expiredRemovals` — lazily, so no
    /// timer machinery exists to drift or fire in the background.
    public func lookup(feature: String, templateVersion: Int, embedding: [Double]) -> SemanticLookupResult {
        metrics.lookups += 1
        let now = clock.now

        removeExpiredEntries(asOf: now)

        var best: (id: UUID, similarity: Double)?
        for (id, entry) in entries where entry.feature == feature && entry.templateVersion == templateVersion {
            guard let similarity = CosineSimilarity.between(entry.embedding, embedding) else { continue }
            if let current = best {
                if similarity > current.similarity { best = (id, similarity) }
            } else {
                best = (id, similarity)
            }
        }

        guard let best, best.similarity >= configuration.similarityThreshold,
              var entry = entries[best.id] else {
            metrics.misses += 1
            return .miss(bestSimilarity: best?.similarity)
        }

        entry.lastAccessedAt = now
        entries[best.id] = entry
        touchRecency(best.id)
        metrics.hits += 1
        metrics.tokensSaved += entry.response.totalTokens
        return .hit(entry, similarity: best.similarity)
    }

    // MARK: Admission

    /// Admits a freshly generated response. Admission is a *policy*, not an
    /// unconditional insert: oversized entries, zero-capacity configs and
    /// near-duplicates are rejected, and capacity pressure evicts LRU-first.
    public func admit(
        feature: String,
        templateVersion: Int,
        normalizedText: String,
        embedding: [Double],
        response: GeneratedResponse
    ) -> AdmissionOutcome {
        guard !embedding.isEmpty else {
            metrics.admissionRejections += 1
            return .rejectedInvalidEmbedding
        }
        guard configuration.maxEntries > 0, configuration.maxTotalBytes > 0 else {
            metrics.admissionRejections += 1
            return .rejectedZeroCapacity
        }

        let now = clock.now
        removeExpiredEntries(asOf: now)

        // Near-duplicate control: if a live entry already answers queries in
        // this neighborhood, a second one burns capacity without adding hits.
        for (id, entry) in entries where entry.feature == feature && entry.templateVersion == templateVersion {
            if let similarity = CosineSimilarity.between(entry.embedding, embedding),
               similarity >= configuration.similarityThreshold {
                metrics.admissionRejections += 1
                return .rejectedNearDuplicate(existing: id, similarity: similarity)
            }
        }

        let entry = CachedEntry(
            id: UUID(),
            feature: feature,
            templateVersion: templateVersion,
            normalizedText: normalizedText,
            embedding: embedding,
            response: response,
            createdAt: now,
            expiresAt: configuration.ttl.map { now.addingTimeInterval($0) },
            lastAccessedAt: now
        )

        guard entry.approximateByteCost <= configuration.maxTotalBytes else {
            metrics.admissionRejections += 1
            return .rejectedOversized
        }

        var evicted = 0
        while entries.count + 1 > configuration.maxEntries
            || totalBytes + entry.approximateByteCost > configuration.maxTotalBytes {
            guard evictLeastRecentlyUsed() else { break } // empty cache; nothing left to evict
            evicted += 1
        }

        insert(entry)
        metrics.admitted += 1
        return .admitted(evictedForCapacity: evicted)
    }

    // MARK: Invalidation & trimming

    /// Removes every live entry for a feature (e.g. its knowledge source changed).
    public func invalidate(feature: String) -> Int {
        let ids = entries.values.filter { $0.feature == feature }.map(\.id)
        for id in ids { remove(id) }
        metrics.invalidatedRemovals += ids.count
        return ids.count
    }

    /// Removes entries for a feature whose template version is older than
    /// `currentVersion`. Lookup already never matches other versions; this
    /// reclaims their capacity eagerly after a prompt-template rollout.
    public func pruneTemplateVersions(olderThan currentVersion: Int, feature: String) -> Int {
        let ids = entries.values
            .filter { $0.feature == feature && $0.templateVersion < currentVersion }
            .map(\.id)
        for id in ids { remove(id) }
        metrics.invalidatedRemovals += ids.count
        return ids.count
    }

    public func invalidateAll() -> Int {
        let count = entries.count
        for id in Array(entries.keys) { remove(id) }
        metrics.invalidatedRemovals += count
        return count
    }

    /// Memory-pressure response: keep only the most recently used fraction.
    /// A fractional trim (vs `removeAll`) preserves the working set, so a
    /// memory warning costs some hit rate instead of all of it.
    /// Fraction is clamped into [0, 1].
    public func trim(toFraction fraction: Double) -> Int {
        let clamped = min(1, max(0, fraction))
        let keepCount = Int((Double(entries.count) * clamped).rounded(.down))
        let removeCount = entries.count - keepCount
        guard removeCount > 0 else { return 0 }
        // recency is LRU-first, so the first `removeCount` are the coldest.
        let victims = Array(recency.prefix(removeCount))
        for id in victims { remove(id) }
        metrics.trimRemovals += victims.count
        return victims.count
    }

    // MARK: Snapshots

    public func exportSnapshot() -> CacheSnapshot {
        CacheSnapshot(exportedAt: clock.now, entries: Array(entries.values))
    }

    /// Restores from a snapshot. Whole-snapshot discard on schema mismatch;
    /// expired entries are dropped; capacity limits are enforced (surplus
    /// entries beyond capacity are dropped coldest-first, not force-fit).
    public func restore(from snapshot: CacheSnapshot) -> SnapshotRestoreOutcome {
        guard snapshot.schemaVersion == CacheSnapshot.currentSchemaVersion else {
            return SnapshotRestoreOutcome(
                restored: 0,
                droppedExpired: 0,
                droppedForCapacity: 0,
                discardedSchemaMismatch: true
            )
        }
        guard configuration.maxEntries > 0, configuration.maxTotalBytes > 0 else {
            return SnapshotRestoreOutcome(
                restored: 0,
                droppedExpired: 0,
                droppedForCapacity: snapshot.entries.count,
                discardedSchemaMismatch: false
            )
        }

        let now = clock.now
        var restored = 0
        var droppedExpired = 0
        var droppedForCapacity = 0

        // Warmest-first so that, if capacity runs out, the entries left
        // behind are the coldest ones.
        let ordered = snapshot.entries.sorted { $0.lastAccessedAt > $1.lastAccessedAt }
        for entry in ordered {
            if let expiresAt = entry.expiresAt, now >= expiresAt {
                droppedExpired += 1
                continue
            }
            if entries[entry.id] != nil {
                droppedForCapacity += 1 // duplicate ID; snapshot corruption guard
                continue
            }
            if entries.count + 1 > configuration.maxEntries
                || totalBytes + entry.approximateByteCost > configuration.maxTotalBytes {
                droppedForCapacity += 1
                continue
            }
            insertColdest(entry)
            restored += 1
        }
        metrics.admitted += restored
        return SnapshotRestoreOutcome(
            restored: restored,
            droppedExpired: droppedExpired,
            droppedForCapacity: droppedForCapacity,
            discardedSchemaMismatch: false
        )
    }

    // MARK: Introspection

    public var count: Int { entries.count }
    public var approximateTotalBytes: Int { totalBytes }
    public func metricsSnapshot() -> CacheMetricsSnapshot { metrics }
    public func liveEntries() -> [CachedEntry] {
        // Warmest-first for inspector UIs.
        recency.reversed().compactMap { entries[$0] }
    }

    // MARK: Private plumbing

    private func removeExpiredEntries(asOf now: Date) {
        let expired = entries.values.filter { entry in
            guard let expiresAt = entry.expiresAt else { return false }
            return now >= expiresAt
        }
        for entry in expired { remove(entry.id) }
        metrics.expiredRemovals += expired.count
    }

    /// Returns false when there was nothing to evict.
    private func evictLeastRecentlyUsed() -> Bool {
        guard let victim = recency.first else { return false }
        remove(victim)
        metrics.capacityEvictions += 1
        return true
    }

    private func insert(_ entry: CachedEntry) {
        entries[entry.id] = entry
        recency.append(entry.id) // most-recent position
        totalBytes += entry.approximateByteCost
    }

    /// Insert at the cold end (used during ordered snapshot restore, where
    /// warmer entries are inserted first and must stay warmer).
    private func insertColdest(_ entry: CachedEntry) {
        entries[entry.id] = entry
        recency.insert(entry.id, at: 0)
        totalBytes += entry.approximateByteCost
    }

    private func remove(_ id: UUID) {
        guard let entry = entries.removeValue(forKey: id) else { return }
        totalBytes -= entry.approximateByteCost
        if totalBytes < 0 { totalBytes = 0 } // defensive; estimator is stable so this should not fire
        if let index = recency.firstIndex(of: id) {
            recency.remove(at: index)
        }
    }

    private func touchRecency(_ id: UUID) {
        if let index = recency.firstIndex(of: id) {
            recency.remove(at: index)
        }
        recency.append(id)
    }
}
