import XCTest
@testable import SemanticLLMCacheKit

final class SemanticGatewayTests: XCTestCase {

    private struct Fixture {
        let gateway: SemanticGateway
        let cache: SemanticResponseCache
        let ledger: FeatureTokenLedger
        let generator: GatedGenerator
        let clock: ManualClock
    }

    /// Stub vocabulary: "alpha" and "alpha prime" are ~0.95 similar,
    /// "beta" is orthogonal to both.
    private func makeFixture(
        threshold: Double = 0.90,
        budget: Int = 100_000,
        ttl: TimeInterval? = nil,
        completionAllowance: Int = 256,
        failEmbeddingFor: Set<String> = []
    ) -> Fixture {
        let clock = ManualClock()
        let cache = SemanticResponseCache(
            configuration: SemanticCacheConfiguration(
                maxEntries: 64,
                maxTotalBytes: 1_000_000,
                similarityThreshold: threshold,
                ttl: ttl
            ),
            clock: clock
        )
        let ledger = FeatureTokenLedger(
            configuration: TokenBudgetConfiguration(
                windowDuration: 3_600,
                budgets: [:],
                defaultBudget: budget
            ),
            clock: clock
        )
        let generator = GatedGenerator()
        let embedder = StubEmbedder(
            dimension: 4,
            vectors: [
                "alpha": Vectors.base,
                "alpha prime": Vectors.similar95,
                "beta": Vectors.orthogonal
            ],
            failForTexts: failEmbeddingFor
        )
        let gateway = SemanticGateway(
            cache: cache,
            ledger: ledger,
            embedder: embedder,
            generator: generator,
            estimator: HeuristicTokenEstimator(completionAllowance: completionAllowance)
        )
        return Fixture(gateway: gateway, cache: cache, ledger: ledger, generator: generator, clock: clock)
    }

    private func query(_ text: String, feature: String = "chat", version: Int = 1) -> SemanticQuery {
        SemanticQuery(feature: feature, text: text, templateVersion: version)
    }

    // MARK: Basic pipeline

    func testMissGeneratesChargesAndAdmits() async throws {
        let fixture = makeFixture()
        let response = try await fixture.gateway.respond(to: query("alpha"))
        XCTAssertEqual(response.source, .generated)
        XCTAssertEqual(response.tokensCharged, 40) // 10 prompt + 30 completion

        let spent = await fixture.ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 40, "actual token spend must be settled against the budget")
        let cached = await fixture.cache.count
        XCTAssertEqual(cached, 1, "successful generation must be admitted to the cache")
    }

    func testSemanticallySimilarRequestServedFromCacheForFree() async throws {
        let fixture = makeFixture()
        _ = try await fixture.gateway.respond(to: query("alpha"))
        let second = try await fixture.gateway.respond(to: query("alpha prime"))

        guard case .semanticCache(let similarity) = second.source else {
            return XCTFail("expected semantic cache hit, got \(second.source)")
        }
        XCTAssertEqual(similarity, 0.95, accuracy: 0.01)
        XCTAssertEqual(second.tokensCharged, 0)

        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 1, "the similar request must not re-generate")
        let spent = await fixture.ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 40, "cache hits must not charge the budget")
    }

    func testEmptyQueryThrows() async {
        let fixture = makeFixture()
        do {
            _ = try await fixture.gateway.respond(to: query("   \n "))
            XCTFail("expected emptyQuery")
        } catch let error as GatewayError {
            XCTAssertEqual(error, .emptyQuery)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testEmbeddingFailureWrapped() async {
        let fixture = makeFixture(failEmbeddingFor: ["alpha"])
        do {
            _ = try await fixture.gateway.respond(to: query("alpha"))
            XCTFail("expected embeddingFailed")
        } catch let error as GatewayError {
            guard case .embeddingFailed = error else {
                return XCTFail("expected embeddingFailed, got \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: Coalescing

    func testConcurrentIdenticalRequestsCoalesceIntoOneGeneration() async throws {
        let fixture = makeFixture()
        await fixture.generator.gate()
        let identicalQuery = query("alpha")

        let results = try await withThrowingTaskGroup(of: GatewayResponse.self) { group -> [GatewayResponse] in
            for _ in 0..<5 {
                group.addTask { try await fixture.gateway.respond(to: identicalQuery) }
            }
            // Hold the gate until all four late arrivals have joined the flight.
            let joined = await eventually {
                await fixture.gateway.currentStats().coalesced == 4
            }
            XCTAssertTrue(joined, "expected 4 callers to coalesce onto the in-flight generation")
            await fixture.generator.release()

            var collected: [GatewayResponse] = []
            for try await response in group {
                collected.append(response)
            }
            return collected
        }

        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results.filter { $0.source == .generated }.count, 1)
        XCTAssertEqual(results.filter { $0.source == .coalesced }.count, 4)
        XCTAssertTrue(results.filter { $0.source == .coalesced }.allSatisfy { $0.tokensCharged == 0 })

        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 1, "five identical concurrent requests must produce exactly one generation")
        let spent = await fixture.ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 40, "the flight must be paid for exactly once")
        let cached = await fixture.cache.count
        XCTAssertEqual(cached, 1)
    }

    func testNormalizationUnifiesFlights() async throws {
        let fixture = makeFixture()
        await fixture.generator.gate()
        let shouty = query("Alpha")
        let padded = query("  ALPHA \n")

        let results = try await withThrowingTaskGroup(of: GatewayResponse.self) { group -> [GatewayResponse] in
            group.addTask { try await fixture.gateway.respond(to: shouty) }
            group.addTask { try await fixture.gateway.respond(to: padded) }
            let joined = await eventually {
                await fixture.gateway.currentStats().coalesced == 1
            }
            XCTAssertTrue(joined, "differently-formatted identical texts must share one flight")
            await fixture.generator.release()

            var collected: [GatewayResponse] = []
            for try await response in group {
                collected.append(response)
            }
            return collected
        }

        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(results.filter { $0.source == .generated }.count, 1)
        XCTAssertEqual(results.filter { $0.source == .coalesced }.count, 1)
    }

    // MARK: Failure semantics

    func testFailureReachesAllCallersAndDoesNotPoisonNextRequest() async throws {
        let fixture = makeFixture()
        await fixture.generator.gate()
        await fixture.generator.setShouldFail(true)
        let identicalQuery = query("alpha")

        let outcomes = await withTaskGroup(of: Result<GatewayResponse, any Error>.self) { group -> [Result<GatewayResponse, any Error>] in
            for _ in 0..<3 {
                group.addTask {
                    do {
                        return .success(try await fixture.gateway.respond(to: identicalQuery))
                    } catch {
                        return .failure(error)
                    }
                }
            }
            let joined = await eventually {
                await fixture.gateway.currentStats().coalesced == 2
            }
            XCTAssertTrue(joined)
            await fixture.generator.release()

            var collected: [Result<GatewayResponse, any Error>] = []
            for await outcome in group {
                collected.append(outcome)
            }
            return collected
        }

        XCTAssertEqual(outcomes.count, 3)
        for outcome in outcomes {
            guard case .failure(let error) = outcome, let gatewayError = error as? GatewayError,
                  case .generationFailed = gatewayError else {
                return XCTFail("every caller of a failed flight must receive generationFailed, got \(outcome)")
            }
        }

        // Nothing cached, nothing charged, reservation released.
        let cached = await fixture.cache.count
        XCTAssertEqual(cached, 0)
        let spent = await fixture.ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 0)
        let pending = await fixture.ledger.pendingTokens(feature: "chat")
        XCTAssertEqual(pending, 0, "failed flight must release its reservation")

        // The failed flight must not be joinable by the next request.
        await fixture.generator.setShouldFail(false)
        let recovered = try await fixture.gateway.respond(to: query("alpha"))
        XCTAssertEqual(recovered.source, .generated)
        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 2, "recovery must be a fresh generation, not a poisoned join")
    }

    // MARK: Budget interactions

    func testBudgetDeniedSkipsGeneratorEntirely() async {
        // Estimate for "alpha" = max(1, 5/4) + 256 allowance = 257 > budget 10.
        let fixture = makeFixture(budget: 10)
        do {
            _ = try await fixture.gateway.respond(to: query("alpha"))
            XCTFail("expected budgetExhausted")
        } catch let error as GatewayError {
            guard case .budgetExhausted(let feature, _) = error else {
                return XCTFail("expected budgetExhausted, got \(error)")
            }
            XCTAssertEqual(feature, "chat")
        } catch {
            XCTFail("unexpected error: \(error)")
        }

        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 0, "a denied request must never reach the model")
        let stats = await fixture.gateway.currentStats()
        XCTAssertEqual(stats.budgetDenials, 1)
    }

    func testCacheHitStillServedWhenBudgetExhausted() async throws {
        // Budget covers exactly one generation, then denies.
        let fixture = makeFixture(budget: 300)
        await fixture.generator.setResponse(
            GeneratedResponse(text: "expensive-answer", promptTokens: 100, completionTokens: 190)
        )
        _ = try await fixture.gateway.respond(to: query("alpha")) // spends 290 of 300

        // A *different* question is denied (estimate 256+1 > 10 remaining)…
        do {
            _ = try await fixture.gateway.respond(to: query("beta"))
            XCTFail("expected budgetExhausted for the uncached question")
        } catch let error as GatewayError {
            guard case .budgetExhausted = error else {
                return XCTFail("expected budgetExhausted, got \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }

        // …but the semantically-similar cached question still answers, free.
        let cachedAnswer = try await fixture.gateway.respond(to: query("alpha prime"))
        guard case .semanticCache = cachedAnswer.source else {
            return XCTFail("exhausted budget must degrade to cached answers, not errors; got \(cachedAnswer.source)")
        }
        XCTAssertEqual(cachedAnswer.tokensCharged, 0)
        XCTAssertEqual(cachedAnswer.text, "expensive-answer")
    }

    // MARK: Versioning & TTL

    func testTemplateVersionBumpRegenerates() async throws {
        let fixture = makeFixture()
        _ = try await fixture.gateway.respond(to: query("alpha", version: 1))
        let bumped = try await fixture.gateway.respond(to: query("alpha", version: 2))
        XCTAssertEqual(bumped.source, .generated, "a template bump must never serve v1 answers")
        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 2)
    }

    func testExpiredEntryRegeneratesAndReadmits() async throws {
        let fixture = makeFixture(ttl: 100)
        _ = try await fixture.gateway.respond(to: query("alpha"))
        fixture.clock.advance(by: 101)
        let second = try await fixture.gateway.respond(to: query("alpha"))
        XCTAssertEqual(second.source, .generated, "expired entries must regenerate")
        let calls = await fixture.generator.callCount
        XCTAssertEqual(calls, 2)
        let cached = await fixture.cache.count
        XCTAssertEqual(cached, 1, "the fresh answer must be re-admitted")
    }

    // MARK: Stats

    func testStatsAccounting() async throws {
        let fixture = makeFixture()
        _ = try await fixture.gateway.respond(to: query("alpha"))       // generated
        _ = try await fixture.gateway.respond(to: query("alpha prime")) // cache hit
        _ = try await fixture.gateway.respond(to: query("beta"))        // generated

        let stats = await fixture.gateway.currentStats()
        XCTAssertEqual(stats.generated, 2)
        XCTAssertEqual(stats.cacheHits, 1)
        XCTAssertEqual(stats.coalesced, 0)
        XCTAssertEqual(stats.budgetDenials, 0)
        XCTAssertEqual(stats.generationFailures, 0)
    }
}
