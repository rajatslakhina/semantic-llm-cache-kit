import Foundation

/// The client-side AI gateway: one front door for LLM-backed features that
/// composes the semantic cache, the token-budget ledger and single-flight
/// request coalescing around an injected model client.
///
/// ## Request pipeline (order is load-bearing)
///
/// 1. **Normalize** — empty queries fail fast, trivially-equal texts unify.
/// 2. **Embed** — through the `Embedder` seam.
/// 3. **Cache lookup** — BEFORE the budget check, deliberately: cache hits
///    are free, so a feature that has exhausted its budget can still serve
///    every answer it already paid for. Budget exhaustion degrades a feature
///    to "cached answers only" instead of turning it off. (Rejected
///    alternative: budget-first — simpler to reason about, but it makes an
///    exhausted feature return errors for questions it already answered,
///    which is strictly worse for the user and saves nothing.)
/// 4. **Coalesce or register** — an identical normalized request already in
///    flight is joined, not re-generated. The check-then-register step is
///    synchronous inside the actor (no `await` between check and insert), so
///    two racing identical requests can never both register. The budget
///    authorization runs *inside* the flight task for the same reason — if
///    it ran before registration, the suspension it introduces would reopen
///    exactly the duplicate-registration race the synchronous section closes.
/// 5. **Generate → settle → admit** — actual token spend replaces the
///    reservation; only complete successful responses are admitted.
///
/// ## Failure semantics
/// A failed generation: releases its budget reservation in full, caches
/// nothing, delivers the failure to every caller of that flight (owner and
/// coalesced joiners), and removes the flight registration so the *next*
/// identical request generates fresh. Failures never poison future requests.
///
/// ## Coalescing & cancellation
/// Flights run as unstructured `Task`s on purpose: a flight is shared by all
/// callers that joined it, so one caller's cancellation must not tear down
/// everyone else's request. The trade-off (a cancelled sole caller lets the
/// flight complete and warm the cache anyway) is documented behavior, not an
/// accident — the completed response is paid for and immediately useful.
///
/// ## Stats accounting
/// `budgetDenials` and `generationFailures` count *flights* (one per failed
/// generation attempt), while `coalesced` counts *callers* that joined an
/// existing flight. Cache hit/miss detail lives in the cache's own metrics.
public actor SemanticGateway {
    private let cache: SemanticResponseCache
    private let ledger: FeatureTokenLedger
    private let embedder: any Embedder
    private let generator: any ResponseGenerator
    private let estimator: any TokenEstimating

    private struct InFlightGeneration {
        let id: UUID
        let task: Task<GeneratedResponse, Error>
    }

    /// Key: feature|templateVersion|normalizedText (exact match — semantic
    /// near-duplicates are the cache's job; coalescing is only ever safe for
    /// *identical* requests, where callers are provably asking the same thing).
    private var inFlight: [String: InFlightGeneration] = [:]
    private var stats = GatewayStats()

    public init(
        cache: SemanticResponseCache,
        ledger: FeatureTokenLedger,
        embedder: any Embedder,
        generator: any ResponseGenerator,
        estimator: any TokenEstimating = HeuristicTokenEstimator()
    ) {
        self.cache = cache
        self.ledger = ledger
        self.embedder = embedder
        self.generator = generator
        self.estimator = estimator
    }

    // MARK: - Public API

    public func respond(to query: SemanticQuery) async throws -> GatewayResponse {
        let normalized = TextNormalizer.normalize(query.text)
        guard !normalized.isEmpty else {
            throw GatewayError.emptyQuery
        }

        let embedding: [Double]
        do {
            embedding = try await embedder.embed(normalized)
        } catch {
            throw GatewayError.embeddingFailed(reason: String(describing: error))
        }

        // Step 3: cache first — hits are free even under an exhausted budget.
        let lookup = await cache.lookup(
            feature: query.feature,
            templateVersion: query.templateVersion,
            embedding: embedding
        )
        if case .hit(let entry, let similarity) = lookup {
            stats.cacheHits += 1
            return GatewayResponse(
                text: entry.response.text,
                source: .semanticCache(similarity: similarity),
                tokensCharged: 0
            )
        }

        let key = flightKey(feature: query.feature, templateVersion: query.templateVersion, normalized: normalized)

        // Step 4a: join an existing identical flight.
        if let existing = inFlight[key] {
            stats.coalesced += 1
            do {
                let response = try await existing.task.value
                return GatewayResponse(text: response.text, source: .coalesced, tokensCharged: 0)
            } catch {
                throw Self.mapFlightError(error, feature: query.feature)
            }
        }

        // Step 4b: register a new flight. No `await` between the check above
        // and this insert — the actor guarantees atomicity of this section,
        // which is what makes duplicate registration impossible.
        let flightID = UUID()
        let ledger = self.ledger
        let generator = self.generator
        let feature = query.feature
        let estimate = estimator.estimateTokens(for: normalized)
        let task = Task<GeneratedResponse, Error> {
            let decision = await ledger.authorize(feature: feature, estimatedTokens: estimate)
            guard case .allowed(let reservation, _) = decision else {
                if case .denied(let retryAfter) = decision {
                    throw GatewayError.budgetExhausted(feature: feature, retryAfter: retryAfter)
                }
                // Exhaustive switch over a two-case enum; unreachable.
                throw GatewayError.budgetExhausted(feature: feature, retryAfter: nil)
            }
            do {
                let response = try await generator.generate(prompt: normalized, feature: feature)
                await ledger.settle(reservation, actualTokens: response.totalTokens)
                return response
            } catch {
                await ledger.release(reservation)
                throw error
            }
        }
        inFlight[key] = InFlightGeneration(id: flightID, task: task)

        do {
            let response = try await task.value
            // Admit BEFORE deregistering: while admission is in progress,
            // late identical arrivals still coalesce onto the (already
            // resolved) flight instead of missing the not-yet-admitted cache
            // entry and paying for a duplicate generation.
            _ = await cache.admit(
                feature: query.feature,
                templateVersion: query.templateVersion,
                normalizedText: normalized,
                embedding: embedding,
                response: response
            )
            deregisterFlight(key: key, id: flightID)
            stats.generated += 1
            return GatewayResponse(text: response.text, source: .generated, tokensCharged: response.totalTokens)
        } catch {
            // Deregister immediately on failure so the next identical request
            // starts a fresh flight — a failed flight must never be joinable
            // beyond the callers it already has.
            deregisterFlight(key: key, id: flightID)
            let mapped = Self.mapFlightError(error, feature: query.feature)
            if case .budgetExhausted = mapped {
                stats.budgetDenials += 1
            } else {
                stats.generationFailures += 1
            }
            throw mapped
        }
    }

    public func currentStats() -> GatewayStats { stats }

    // MARK: - Private

    private func flightKey(feature: String, templateVersion: Int, normalized: String) -> String {
        "\(feature)|\(templateVersion)|\(normalized)"
    }

    /// Identity-checked removal: only the flight that registered this key may
    /// remove it. Without the ID check, a slow completion could deregister a
    /// *successor* flight for the same key that legitimately replaced it.
    private func deregisterFlight(key: String, id: UUID) {
        if inFlight[key]?.id == id {
            inFlight[key] = nil
        }
    }

    private static func mapFlightError(_ error: any Error, feature: String) -> GatewayError {
        if let gatewayError = error as? GatewayError {
            return gatewayError
        }
        return .generationFailed(feature: feature, reason: String(describing: error))
    }
}
