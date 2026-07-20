import Foundation

// MARK: - Generator seam

/// The model seam. Production apps plug their real client here — an Apple
/// Foundation Models session, a cloud LLM API client, or an on-device
/// runtime. The gateway only requires that a *complete* response with token
/// accounting comes back; streaming assembly (buffer-then-return) belongs in
/// the conforming client. That "complete responses only" contract is what
/// lets the cache guarantee it never serves a truncated answer.
public protocol ResponseGenerator: Sendable {
    func generate(prompt: String, feature: String) async throws -> GeneratedResponse
}

// MARK: - Token estimation seam

/// Pre-generation spend estimate used for budget reservations.
///
/// Estimates only need to be *proportionate*, not exact: the ledger settles
/// reservations against the provider's actual token counts afterwards.
public protocol TokenEstimating: Sendable {
    func estimateTokens(for text: String) -> Int
}

/// The classic chars/4 heuristic plus a completion allowance.
///
/// Rejected alternative: bundling a real tokenizer (e.g. a BPE vocab) —
/// hundreds of KB of vocabulary tables to improve an estimate whose only
/// job is to be roughly proportional before settle-time correction.
public struct HeuristicTokenEstimator: TokenEstimating {
    /// Expected completion size charged into the estimate, since budgets
    /// cover the whole round trip, not just the prompt.
    public let completionAllowance: Int

    public init(completionAllowance: Int = 256) {
        self.completionAllowance = max(0, completionAllowance)
    }

    public func estimateTokens(for text: String) -> Int {
        let promptEstimate = max(1, text.utf8.count / 4)
        return promptEstimate + completionAllowance
    }
}

// MARK: - Simulated generator

/// Deterministic, honestly-labeled simulation of an LLM backend, used by the
/// demo app and available for previews/tests. Latency is configurable so
/// coalescing behavior is visible in a UI. This is a stand-in at the
/// `ResponseGenerator` seam — swapping in a real client touches zero other
/// code, which is the point of the seam.
public struct SimulatedAssistantGenerator: ResponseGenerator {
    public let latency: Duration

    public init(latency: Duration = .milliseconds(600)) {
        self.latency = latency
    }

    public func generate(prompt: String, feature: String) async throws -> GeneratedResponse {
        if latency > .zero {
            try await Task.sleep(for: latency)
        }
        let answer = Self.cannedAnswer(prompt: prompt, feature: feature)
        return GeneratedResponse(
            text: answer,
            promptTokens: max(1, prompt.utf8.count / 4),
            completionTokens: max(1, answer.utf8.count / 4)
        )
    }

    private static func cannedAnswer(prompt: String, feature: String) -> String {
        "[\(feature)] Simulated assistant answer to: \"\(prompt)\". "
            + "In production this text comes from your real model client behind the ResponseGenerator seam."
    }
}
