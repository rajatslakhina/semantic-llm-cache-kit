import Foundation
@testable import SemanticLLMCacheKit

// MARK: - Manual clock

/// Deterministic, manually-advanced clock. Lock-protected class (not an
/// actor) because `CacheClock.now` is a synchronous requirement; all lock
/// sections are tiny and never suspend, so this is safe under Swift 6.
final class ManualClock: CacheClock, @unchecked Sendable {
    private let lock = NSLock()
    private var current: Date

    init(start: Date = Date(timeIntervalSince1970: 1_000_000)) {
        self.current = start
    }

    var now: Date {
        lock.lock()
        defer { lock.unlock() }
        return current
    }

    func advance(by interval: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        current = current.addingTimeInterval(interval)
    }
}

// MARK: - Stub embedder

/// Returns scripted vectors keyed by normalized text, so tests control
/// similarity relationships exactly instead of reverse-engineering the
/// hash embedder. Unknown texts get a deterministic orthogonal-ish fallback.
struct StubEmbedder: Embedder {
    let dimension: Int
    let vectors: [String: [Double]]
    let failForTexts: Set<String>

    init(dimension: Int = 4, vectors: [String: [Double]] = [:], failForTexts: Set<String> = []) {
        self.dimension = dimension
        self.vectors = vectors
        self.failForTexts = failForTexts
    }

    func embed(_ text: String) async throws -> [Double] {
        let normalized = TextNormalizer.normalize(text)
        if failForTexts.contains(normalized) {
            throw EmbeddingError.emptyInput
        }
        if let vector = vectors[normalized] {
            return vector
        }
        // Deterministic fallback: unit vector on an axis picked by hash.
        // Bit-masking (not abs) makes the value non-negative without the
        // Int.min overflow trap abs() carries.
        var vector = [Double](repeating: 0, count: dimension)
        let index = (normalized.hashValue & Int.max) % max(1, dimension)
        vector[index] = 1
        return vector
    }
}

// MARK: - Scripted generator

enum TestGenerationError: Error, Equatable {
    case scripted
}

/// Controllable generator: counts calls, can fail on demand, and can be
/// "gated" so a generation stays in flight until the test releases it —
/// which is how coalescing windows are held open deterministically.
actor GatedGenerator: ResponseGenerator {
    private(set) var callCount = 0
    private var gated = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    var shouldFail = false
    var response = GeneratedResponse(text: "generated-answer", promptTokens: 10, completionTokens: 30)

    func gate() { gated = true }

    func release() {
        gated = false
        let resumed = waiters
        waiters = []
        for waiter in resumed {
            waiter.resume()
        }
    }

    func setShouldFail(_ value: Bool) { shouldFail = value }
    func setResponse(_ value: GeneratedResponse) { response = value }

    func generate(prompt: String, feature: String) async throws -> GeneratedResponse {
        callCount += 1
        if gated {
            await withCheckedContinuation { continuation in
                if gated {
                    waiters.append(continuation)
                } else {
                    continuation.resume()
                }
            }
        }
        if shouldFail {
            throw TestGenerationError.scripted
        }
        return response
    }
}

// MARK: - Polling helper

/// Polls an async condition without wall-clock sleeps dominating the suite.
/// Fails (returns false) after `attempts` polls, so a broken condition can
/// never hang the test run.
func eventually(attempts: Int = 2_000, _ condition: () async -> Bool) async -> Bool {
    for _ in 0..<attempts {
        if await condition() { return true }
        await Task.yield()
    }
    return false
}

// MARK: - Vector fixtures

enum Vectors {
    /// Unit vectors with controlled pairwise cosine similarities (dimension 4).
    static let base: [Double] = [1, 0, 0, 0]
    static let identical: [Double] = [1, 0, 0, 0]
    static let orthogonal: [Double] = [0, 1, 0, 0]

    /// cos(base, similar95) ≈ 0.95
    static var similar95: [Double] {
        let c = 0.95
        let s = (1 - c * c).squareRoot()
        return [c, s, 0, 0]
    }

    /// cos(base, similar85) ≈ 0.85
    static var similar85: [Double] {
        let c = 0.85
        let s = (1 - c * c).squareRoot()
        return [c, s, 0, 0]
    }

    /// Exactly the configured threshold when threshold = 0.90.
    static var similar90: [Double] {
        let c = 0.90
        let s = (1 - c * c).squareRoot()
        return [c, s, 0, 0]
    }
}
