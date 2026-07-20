import Foundation

// MARK: - Configuration

public struct TokenBudgetConfiguration: Sendable {
    /// Sliding accounting window. Clamped to > 0 (defaults to 1h on misuse).
    public let windowDuration: TimeInterval
    /// Per-feature budgets (tokens per window). Missing features fall back to
    /// `defaultBudget`. Negative budgets are clamped to 0 (= feature disabled).
    public let budgets: [String: Int]
    public let defaultBudget: Int

    public init(
        windowDuration: TimeInterval = 60 * 60,
        budgets: [String: Int] = [:],
        defaultBudget: Int = 10_000
    ) {
        self.windowDuration = windowDuration > 0 ? windowDuration : 60 * 60
        self.budgets = budgets.mapValues { max(0, $0) }
        self.defaultBudget = max(0, defaultBudget)
    }

    public func budget(for feature: String) -> Int {
        budgets[feature] ?? defaultBudget
    }
}

// MARK: - Decisions

/// A granted right to spend up to `estimatedTokens` for one generation.
/// Must be settled (actual spend) or released (failure) — the ledger holds
/// the estimate against the budget until one of the two happens.
public struct BudgetReservation: Sendable, Equatable {
    public let id: UUID
    public let feature: String
    public let estimatedTokens: Int
}

public enum BudgetDecision: Sendable, Equatable {
    case allowed(BudgetReservation, remainingAfterReservation: Int)
    /// `retryAfter`: seconds until the oldest in-window spend record ages
    /// out (i.e. the earliest moment budget can free up). nil when waiting
    /// cannot help — the estimate exceeds the whole configured budget.
    case denied(retryAfter: TimeInterval?)
}

// MARK: - Ledger actor

/// Per-feature sliding-window token accounting with reserve-then-settle
/// semantics.
///
/// ## Why reserve/settle instead of advisory estimates
/// With advisory accounting (check, then record after the fact), N
/// concurrent distinct requests each pass the check against the same
/// un-decremented balance and collectively overshoot the budget by up to
/// N-1 requests. Reservations close that race: the estimate is held against
/// the budget at decision time, then replaced by the actual spend on success
/// or released on failure. Rejected alternative: a mutex-free advisory
/// model — simpler, but its overshoot is unbounded under bursty concurrency,
/// which defeats the point of a budget.
///
/// ## Failure semantics
/// A failed generation releases its reservation in full — users are never
/// charged for answers they did not receive. (The provider may still have
/// billed partial work upstream; reconciling that belongs to server-side
/// metering, not a client ledger.)
public actor FeatureTokenLedger {
    private struct SpendRecord {
        let date: Date
        let tokens: Int
    }

    private let configuration: TokenBudgetConfiguration
    private let clock: any CacheClock

    private var spendRecords: [String: [SpendRecord]] = [:]
    private var pendingReservations: [String: [UUID: Int]] = [:]

    public init(configuration: TokenBudgetConfiguration, clock: any CacheClock = SystemCacheClock()) {
        self.configuration = configuration
        self.clock = clock
    }

    // MARK: Authorization

    public func authorize(feature: String, estimatedTokens: Int) -> BudgetDecision {
        let estimate = max(0, estimatedTokens)
        let now = clock.now
        prune(feature: feature, asOf: now)

        let budget = configuration.budget(for: feature)
        let spent = spentTokensInWindow(feature: feature)
        let pending = pendingTokens(feature: feature)

        guard spent + pending + estimate <= budget else {
            if estimate > budget {
                // No amount of waiting frees enough budget for this request.
                return .denied(retryAfter: nil)
            }
            let retryAfter = spendRecords[feature]?.first.map { oldest in
                max(0, oldest.date.addingTimeInterval(configuration.windowDuration).timeIntervalSince(now))
            }
            // If denial is caused purely by pending reservations (no spend
            // records yet), there is no aging-out moment to predict.
            return .denied(retryAfter: retryAfter)
        }

        let reservation = BudgetReservation(id: UUID(), feature: feature, estimatedTokens: estimate)
        pendingReservations[feature, default: [:]][reservation.id] = estimate
        let remaining = budget - spent - pending - estimate
        return .allowed(reservation, remainingAfterReservation: remaining)
    }

    /// Converts a reservation into an actual spend record. Idempotent: a
    /// second settle of the same reservation records nothing.
    public func settle(_ reservation: BudgetReservation, actualTokens: Int) {
        guard pendingReservations[reservation.feature]?.removeValue(forKey: reservation.id) != nil else {
            return
        }
        let record = SpendRecord(date: clock.now, tokens: max(0, actualTokens))
        spendRecords[reservation.feature, default: []].append(record)
    }

    /// Releases a reservation without recording spend (generation failed or
    /// was cancelled). Idempotent.
    public func release(_ reservation: BudgetReservation) {
        pendingReservations[reservation.feature]?.removeValue(forKey: reservation.id)
    }

    // MARK: Introspection

    public func spentTokensInWindow(feature: String) -> Int {
        prune(feature: feature, asOf: clock.now)
        return (spendRecords[feature] ?? []).reduce(0) { $0 + $1.tokens }
    }

    public func pendingTokens(feature: String) -> Int {
        (pendingReservations[feature] ?? [:]).values.reduce(0, +)
    }

    public func remainingBudget(feature: String) -> Int {
        let budget = configuration.budget(for: feature)
        return max(0, budget - spentTokensInWindow(feature: feature) - pendingTokens(feature: feature))
    }

    // MARK: Private

    private func prune(feature: String, asOf now: Date) {
        guard var records = spendRecords[feature], !records.isEmpty else { return }
        let cutoff = now.addingTimeInterval(-configuration.windowDuration)
        records.removeAll { $0.date <= cutoff }
        spendRecords[feature] = records
    }
}
