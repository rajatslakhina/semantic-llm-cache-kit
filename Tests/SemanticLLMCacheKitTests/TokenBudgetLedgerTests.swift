import XCTest
@testable import SemanticLLMCacheKit

final class TokenBudgetLedgerTests: XCTestCase {

    private func makeLedger(
        budget: Int = 1_000,
        window: TimeInterval = 3_600,
        budgets: [String: Int] = [:],
        clock: ManualClock = ManualClock()
    ) -> FeatureTokenLedger {
        FeatureTokenLedger(
            configuration: TokenBudgetConfiguration(
                windowDuration: window,
                budgets: budgets,
                defaultBudget: budget
            ),
            clock: clock
        )
    }

    private struct UnexpectedDecision: Error {
        let decision: BudgetDecision
    }

    private func expectReservation(from decision: BudgetDecision) throws -> BudgetReservation {
        guard case .allowed(let reservation, _) = decision else {
            XCTFail("expected allowed, got \(decision)")
            throw UnexpectedDecision(decision: decision)
        }
        return reservation
    }

    func testAllowUnderBudgetReportsRemaining() async {
        let ledger = makeLedger(budget: 1_000)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: 300)
        guard case .allowed(_, let remaining) = decision else {
            return XCTFail("expected allowed, got \(decision)")
        }
        XCTAssertEqual(remaining, 700)
    }

    func testDenyWhenSpentPlusEstimateExceedsBudget() async throws {
        let ledger = makeLedger(budget: 1_000)
        let first = await ledger.authorize(feature: "chat", estimatedTokens: 800)
        let reservation = try expectReservation(from: first)
        await ledger.settle(reservation, actualTokens: 800)

        let second = await ledger.authorize(feature: "chat", estimatedTokens: 300)
        guard case .denied = second else {
            return XCTFail("expected denial, got \(second)")
        }
    }

    func testDenyEstimateLargerThanWholeBudgetHasNilRetryAfter() async {
        let ledger = makeLedger(budget: 1_000)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: 2_000)
        guard case .denied(let retryAfter) = decision else {
            return XCTFail("expected denial, got \(decision)")
        }
        XCTAssertNil(retryAfter, "no amount of waiting fits a request larger than the whole budget")
    }

    func testRetryAfterComputedFromOldestSpendRecord() async throws {
        let clock = ManualClock()
        let ledger = makeLedger(budget: 100, window: 3_600, clock: clock)
        let first = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        await ledger.settle(try expectReservation(from: first), actualTokens: 60)

        clock.advance(by: 100)
        let second = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        guard case .denied(let retryAfter) = second else {
            return XCTFail("expected denial, got \(second)")
        }
        XCTAssertEqual(try XCTUnwrap(retryAfter), 3_500, accuracy: 0.001)
    }

    func testWindowSlideFreesBudget() async throws {
        let clock = ManualClock()
        let ledger = makeLedger(budget: 100, window: 3_600, clock: clock)
        let first = await ledger.authorize(feature: "chat", estimatedTokens: 100)
        await ledger.settle(try expectReservation(from: first), actualTokens: 100)

        clock.advance(by: 3_601)
        let second = await ledger.authorize(feature: "chat", estimatedTokens: 100)
        guard case .allowed = second else {
            return XCTFail("aged-out spend must free the budget, got \(second)")
        }
    }

    func testPendingReservationBlocksConcurrentOverspend() async {
        let ledger = makeLedger(budget: 100)
        let first = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        guard case .allowed = first else {
            return XCTFail("first reservation must be allowed")
        }
        let second = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        guard case .denied(let retryAfter) = second else {
            return XCTFail("pending reservation must block a concurrent overspend, got \(second)")
        }
        // Denial caused by a pending reservation, not aged spend — there is
        // no aging-out moment to predict.
        XCTAssertNil(retryAfter)
    }

    func testReleaseRestoresCapacity() async throws {
        let ledger = makeLedger(budget: 100)
        let first = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        let reservation = try expectReservation(from: first)
        await ledger.release(reservation)

        let second = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        guard case .allowed = second else {
            return XCTFail("released reservation must restore capacity, got \(second)")
        }
    }

    func testSettleRecordsActualNotEstimate() async throws {
        let ledger = makeLedger(budget: 100)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        await ledger.settle(try expectReservation(from: decision), actualTokens: 10)
        let spent = await ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 10)
        let remaining = await ledger.remainingBudget(feature: "chat")
        XCTAssertEqual(remaining, 90)
    }

    func testSettleIsIdempotent() async throws {
        let ledger = makeLedger(budget: 100)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        let reservation = try expectReservation(from: decision)
        await ledger.settle(reservation, actualTokens: 10)
        await ledger.settle(reservation, actualTokens: 10)
        let spent = await ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 10, "double settle must not double charge")
    }

    func testReleaseAfterSettleHasNoEffect() async throws {
        let ledger = makeLedger(budget: 100)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: 60)
        let reservation = try expectReservation(from: decision)
        await ledger.settle(reservation, actualTokens: 10)
        await ledger.release(reservation)
        let spent = await ledger.spentTokensInWindow(feature: "chat")
        XCTAssertEqual(spent, 10, "release after settle must not erase the spend record")
    }

    func testPerFeatureIsolation() async throws {
        let ledger = makeLedger(budget: 100)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: 100)
        await ledger.settle(try expectReservation(from: decision), actualTokens: 100)

        let other = await ledger.authorize(feature: "search", estimatedTokens: 100)
        guard case .allowed = other else {
            return XCTFail("features must not share budgets, got \(other)")
        }
    }

    func testZeroBudgetFeatureAlwaysDenied() async {
        let ledger = makeLedger(budget: 1_000, budgets: ["disabled": 0])
        let decision = await ledger.authorize(feature: "disabled", estimatedTokens: 1)
        guard case .denied(let retryAfter) = decision else {
            return XCTFail("zero-budget feature must deny, got \(decision)")
        }
        XCTAssertNil(retryAfter)
    }

    func testNegativeEstimateClampedToZero() async {
        let ledger = makeLedger(budget: 100)
        let decision = await ledger.authorize(feature: "chat", estimatedTokens: -50)
        guard case .allowed(let reservation, let remaining) = decision else {
            return XCTFail("negative estimate must clamp to 0 and be allowed, got \(decision)")
        }
        XCTAssertEqual(reservation.estimatedTokens, 0)
        XCTAssertEqual(remaining, 100)
    }

    func testDefaultBudgetFallbackForUnknownFeature() async {
        let ledger = makeLedger(budget: 555, budgets: ["known": 10])
        let remaining = await ledger.remainingBudget(feature: "unknown")
        XCTAssertEqual(remaining, 555)
    }
}
