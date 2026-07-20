import XCTest
@testable import SemanticLLMCacheKit

final class SemanticResponseCacheTests: XCTestCase {

    private func makeCache(
        maxEntries: Int = 16,
        maxTotalBytes: Int = 1_000_000,
        threshold: Double = 0.90,
        ttl: TimeInterval? = nil,
        clock: ManualClock = ManualClock()
    ) -> SemanticResponseCache {
        SemanticResponseCache(
            configuration: SemanticCacheConfiguration(
                maxEntries: maxEntries,
                maxTotalBytes: maxTotalBytes,
                similarityThreshold: threshold,
                ttl: ttl
            ),
            clock: clock
        )
    }

    @discardableResult
    private func admit(
        _ cache: SemanticResponseCache,
        feature: String = "support",
        version: Int = 1,
        text: String = "q",
        vector: [Double],
        tokens: Int = 40
    ) async -> AdmissionOutcome {
        await cache.admit(
            feature: feature,
            templateVersion: version,
            normalizedText: text,
            embedding: vector,
            response: GeneratedResponse(text: "r", promptTokens: tokens / 2, completionTokens: tokens - tokens / 2)
        )
    }

    /// Unit vector along axis `i` in 8 dimensions — mutually orthogonal fixtures.
    private func axis(_ i: Int) -> [Double] {
        var v = [Double](repeating: 0, count: 8)
        v[min(max(0, i), 7)] = 1
        return v
    }

    // MARK: Lookup semantics

    func testHitAboveThreshold() async {
        let cache = makeCache()
        await admit(cache, vector: Vectors.base)
        let result = await cache.lookup(feature: "support", templateVersion: 1, embedding: Vectors.similar95)
        guard case .hit(_, let similarity) = result else {
            return XCTFail("expected hit, got \(result)")
        }
        XCTAssertEqual(similarity, 0.95, accuracy: 0.01)
    }

    func testExactThresholdBoundaryIsHit() async {
        // Threshold and candidate similarity are built from the *identical*
        // floating-point expression (1/√2), so the >= comparison at the
        // boundary is deterministic rather than at the mercy of rounding.
        let boundary = 1.0 / (1.0.squareRoot() * 2.0.squareRoot())
        let cache = makeCache(threshold: boundary)
        await admit(cache, vector: [1, 0, 0, 0])
        let result = await cache.lookup(feature: "support", templateVersion: 1, embedding: [1, 1, 0, 0])
        guard case .hit = result else {
            return XCTFail("similarity exactly at threshold must hit (>= semantics), got \(result)")
        }
    }

    func testJustBelowThresholdMissesAndReportsBestSimilarity() async throws {
        let cache = makeCache(threshold: 0.90)
        await admit(cache, vector: Vectors.base)
        let result = await cache.lookup(feature: "support", templateVersion: 1, embedding: Vectors.similar85)
        guard case .miss(let best) = result else {
            return XCTFail("expected miss, got \(result)")
        }
        XCTAssertEqual(try XCTUnwrap(best), 0.85, accuracy: 0.01)
    }

    func testMissOnEmptyCacheReportsNilBest() async {
        let cache = makeCache()
        let result = await cache.lookup(feature: "support", templateVersion: 1, embedding: Vectors.base)
        guard case .miss(let best) = result else {
            return XCTFail("expected miss, got \(result)")
        }
        XCTAssertNil(best)
    }

    func testFeatureIsolation() async {
        let cache = makeCache()
        await admit(cache, feature: "support", vector: Vectors.base)
        let result = await cache.lookup(feature: "search", templateVersion: 1, embedding: Vectors.identical)
        guard case .miss = result else {
            return XCTFail("entries must not leak across features, got \(result)")
        }
    }

    func testTemplateVersionIsolation() async {
        let cache = makeCache()
        await admit(cache, version: 1, vector: Vectors.base)
        let result = await cache.lookup(feature: "support", templateVersion: 2, embedding: Vectors.identical)
        guard case .miss = result else {
            return XCTFail("v1 answers must never serve v2 prompts, got \(result)")
        }
    }

    // MARK: TTL

    func testEntryExpiresExactlyAtTTLBoundary() async {
        let clock = ManualClock()
        let cache = makeCache(ttl: 100, clock: clock)
        await admit(cache, vector: Vectors.base)
        clock.advance(by: 100) // expiry is defined as now >= expiresAt
        let result = await cache.lookup(feature: "support", templateVersion: 1, embedding: Vectors.identical)
        guard case .miss = result else {
            return XCTFail("entry at exact TTL boundary must be expired, got \(result)")
        }
        let metrics = await cache.metricsSnapshot()
        XCTAssertEqual(metrics.expiredRemovals, 1)
        let count = await cache.count
        XCTAssertEqual(count, 0)
    }

    func testEntryLivesJustBeforeTTLBoundary() async {
        let clock = ManualClock()
        let cache = makeCache(ttl: 100, clock: clock)
        await admit(cache, vector: Vectors.base)
        clock.advance(by: 99)
        let result = await cache.lookup(feature: "support", templateVersion: 1, embedding: Vectors.identical)
        guard case .hit = result else {
            return XCTFail("entry inside TTL must hit, got \(result)")
        }
    }

    // MARK: Eviction

    func testLRUEvictionEvictsColdestEntry() async {
        let cache = makeCache(maxEntries: 2)
        await admit(cache, text: "a", vector: axis(0))
        await admit(cache, text: "b", vector: axis(1))
        // Touch A so B becomes the coldest.
        _ = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(0))
        await admit(cache, text: "c", vector: axis(2))

        let metrics = await cache.metricsSnapshot()
        XCTAssertEqual(metrics.capacityEvictions, 1)

        guard case .miss = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(1)) else {
            return XCTFail("B (coldest) should have been evicted")
        }
        guard case .hit = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(0)) else {
            return XCTFail("A (recently touched) should have survived")
        }
        guard case .hit = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(2)) else {
            return XCTFail("C (newest) should be present")
        }
    }

    func testByteBudgetEviction() async {
        // Each fixture entry costs 162 approximate bytes (1 + 1 + 4*8 + 128);
        // a 200-byte budget holds exactly one.
        let cache = makeCache(maxEntries: 10, maxTotalBytes: 200)
        let first = await admit(cache, text: "a", vector: [1, 0, 0, 0])
        XCTAssertEqual(first, .admitted(evictedForCapacity: 0))
        let second = await admit(cache, text: "b", vector: [0, 1, 0, 0])
        XCTAssertEqual(second, .admitted(evictedForCapacity: 1))
        let count = await cache.count
        XCTAssertEqual(count, 1)
    }

    func testOversizedEntryRejected() async {
        let cache = makeCache(maxEntries: 10, maxTotalBytes: 100) // below the 162-byte fixture cost
        let outcome = await admit(cache, vector: [1, 0, 0, 0])
        XCTAssertEqual(outcome, .rejectedOversized)
        let count = await cache.count
        XCTAssertEqual(count, 0)
    }

    func testZeroCapacityCacheRejectsEverythingWithoutCrashing() async {
        let cache = makeCache(maxEntries: 0)
        let outcome = await admit(cache, vector: Vectors.base)
        XCTAssertEqual(outcome, .rejectedZeroCapacity)
        guard case .miss = await cache.lookup(feature: "support", templateVersion: 1, embedding: Vectors.base) else {
            return XCTFail("zero-capacity cache must always miss")
        }
    }

    func testNearDuplicateAdmissionRejected() async {
        let cache = makeCache(threshold: 0.90)
        await admit(cache, vector: Vectors.base)
        let outcome = await admit(cache, text: "near dup", vector: Vectors.similar95)
        guard case .rejectedNearDuplicate(_, let similarity) = outcome else {
            return XCTFail("expected near-duplicate rejection, got \(outcome)")
        }
        XCTAssertEqual(similarity, 0.95, accuracy: 0.01)
        let count = await cache.count
        XCTAssertEqual(count, 1)
    }

    func testEmptyEmbeddingRejected() async {
        let cache = makeCache()
        let outcome = await admit(cache, vector: [])
        XCTAssertEqual(outcome, .rejectedInvalidEmbedding)
    }

    // MARK: Invalidation & trim

    func testInvalidateFeatureRemovesOnlyThatFeature() async {
        let cache = makeCache()
        await admit(cache, feature: "support", vector: axis(0))
        await admit(cache, feature: "search", vector: axis(1))
        let removed = await cache.invalidate(feature: "support")
        XCTAssertEqual(removed, 1)
        guard case .hit = await cache.lookup(feature: "search", templateVersion: 1, embedding: axis(1)) else {
            return XCTFail("other features must survive invalidation")
        }
    }

    func testPruneTemplateVersionsRemovesOnlyOlderVersions() async {
        let cache = makeCache()
        await admit(cache, version: 1, vector: axis(0))
        await admit(cache, version: 2, vector: axis(1))
        let removed = await cache.pruneTemplateVersions(olderThan: 2, feature: "support")
        XCTAssertEqual(removed, 1)
        guard case .hit = await cache.lookup(feature: "support", templateVersion: 2, embedding: axis(1)) else {
            return XCTFail("current-version entry must survive pruning")
        }
    }

    func testTrimKeepsWarmestEntries() async {
        let cache = makeCache()
        await admit(cache, text: "a", vector: axis(0))
        await admit(cache, text: "b", vector: axis(1))
        await admit(cache, text: "c", vector: axis(2))
        // Touch A: recency (cold→warm) becomes B, C, A.
        _ = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(0))

        let removed = await cache.trim(toFraction: 0.34) // keeps floor(3 * 0.34) = 1
        XCTAssertEqual(removed, 2)
        guard case .hit = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(0)) else {
            return XCTFail("warmest entry must survive a trim")
        }
    }

    func testTrimClampsFraction() async {
        let cache = makeCache()
        await admit(cache, text: "a", vector: axis(0))
        await admit(cache, text: "b", vector: axis(1))
        let removedByOverflowFraction = await cache.trim(toFraction: 2.0)  // clamped to 1 → keep all
        XCTAssertEqual(removedByOverflowFraction, 0)
        let removedByNegativeFraction = await cache.trim(toFraction: -1.0) // clamped to 0 → remove all
        XCTAssertEqual(removedByNegativeFraction, 2)
        let count = await cache.count
        XCTAssertEqual(count, 0)
    }

    // MARK: Conservation law

    func testMetricsConservationLaw() async {
        let clock = ManualClock()
        let cache = makeCache(maxEntries: 3, ttl: 100, clock: clock)

        await admit(cache, text: "a", vector: axis(0))
        await admit(cache, text: "b", vector: axis(1))
        await admit(cache, text: "c", vector: axis(2))
        await admit(cache, text: "d", vector: axis(3)) // evicts A (capacity)

        clock.advance(by: 101) // B, C, D now expired
        _ = await cache.lookup(feature: "support", templateVersion: 1, embedding: axis(5)) // sweeps 3 expired

        await admit(cache, text: "e", vector: axis(4))
        await admit(cache, text: "f", vector: axis(5))
        _ = await cache.trim(toFraction: 0.5) // removes 1
        _ = await cache.invalidateAll()       // removes 1

        let metrics = await cache.metricsSnapshot()
        let live = await cache.count
        XCTAssertEqual(metrics.admitted, 6)
        XCTAssertEqual(metrics.capacityEvictions, 1)
        XCTAssertEqual(metrics.expiredRemovals, 3)
        XCTAssertEqual(metrics.trimRemovals, 1)
        XCTAssertEqual(metrics.invalidatedRemovals, 1)
        XCTAssertEqual(live, 0)
        // Every admitted entry accounted for through exactly one exit door.
        XCTAssertEqual(
            metrics.admitted,
            live + metrics.expiredRemovals + metrics.capacityEvictions
                + metrics.trimRemovals + metrics.invalidatedRemovals
        )
    }

    // MARK: Snapshots

    func testSnapshotRoundtripRestoresEntries() async {
        let clock = ManualClock()
        let source = makeCache(ttl: 1_000, clock: clock)
        await admit(source, text: "a", vector: axis(0))
        await admit(source, text: "b", vector: axis(1))
        let snapshot = await source.exportSnapshot()

        let target = makeCache(ttl: 1_000, clock: clock)
        let outcome = await target.restore(from: snapshot)
        XCTAssertEqual(outcome.restored, 2)
        XCTAssertFalse(outcome.discardedSchemaMismatch)
        guard case .hit = await target.lookup(feature: "support", templateVersion: 1, embedding: axis(0)) else {
            return XCTFail("restored entry must be servable")
        }
    }

    func testSnapshotSchemaMismatchDiscardsEverything() async {
        let cache = makeCache()
        let alien = CacheSnapshot(schemaVersion: 999, exportedAt: Date(), entries: [])
        let outcome = await cache.restore(from: alien)
        XCTAssertTrue(outcome.discardedSchemaMismatch)
        XCTAssertEqual(outcome.restored, 0)
    }

    func testRestoreDropsExpiredEntries() async {
        let clock = ManualClock()
        let source = makeCache(ttl: 100, clock: clock)
        await admit(source, vector: axis(0))
        let snapshot = await source.exportSnapshot()

        clock.advance(by: 101)
        let target = makeCache(ttl: 100, clock: clock)
        let outcome = await target.restore(from: snapshot)
        XCTAssertEqual(outcome.restored, 0)
        XCTAssertEqual(outcome.droppedExpired, 1)
    }

    func testRestoreRespectsCapacityKeepingWarmest() async {
        let clock = ManualClock()
        let source = makeCache(maxEntries: 10, clock: clock)
        await admit(source, text: "cold", vector: axis(0))
        clock.advance(by: 10)
        await admit(source, text: "mid", vector: axis(1))
        clock.advance(by: 10)
        await admit(source, text: "warm", vector: axis(2))
        let snapshot = await source.exportSnapshot()

        let target = makeCache(maxEntries: 2, clock: clock)
        let outcome = await target.restore(from: snapshot)
        XCTAssertEqual(outcome.restored, 2)
        XCTAssertEqual(outcome.droppedForCapacity, 1)
        guard case .hit = await target.lookup(feature: "support", templateVersion: 1, embedding: axis(2)) else {
            return XCTFail("warmest snapshot entry must be restored")
        }
        guard case .miss = await target.lookup(feature: "support", templateVersion: 1, embedding: axis(0)) else {
            return XCTFail("coldest snapshot entry must be the one dropped")
        }
    }

    func testSnapshotCodableRoundtrip() async throws {
        let cache = makeCache()
        await admit(cache, text: "a", vector: axis(0))
        let snapshot = await cache.exportSnapshot()
        let data = try JSONEncoder().encode(snapshot)
        let decoded = try JSONDecoder().decode(CacheSnapshot.self, from: data)
        XCTAssertEqual(decoded.schemaVersion, CacheSnapshot.currentSchemaVersion)
        XCTAssertEqual(decoded.entries.map(\.id), snapshot.entries.map(\.id))
    }
}
