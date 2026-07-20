import Foundation

// MARK: - Text normalization

/// Canonicalizes query text before embedding, cache keying and coalescing.
///
/// Normalization is load-bearing: "What's my order status?" and
/// "what's  my ORDER status?" must embed identically and share a
/// single-flight key, otherwise trivially-equal requests double-spend.
public enum TextNormalizer {
    public static func normalize(_ text: String) -> String {
        text
            .lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}

// MARK: - Embedder seam

public enum EmbeddingError: Error, Sendable, Equatable {
    /// Input was empty (or whitespace-only) after normalization.
    case emptyInput
}

/// The embedding seam.
///
/// Production apps plug a real model here (Core ML sentence encoder, the
/// Foundation Models embedding API, or a server-side embedding endpoint).
/// The package ships a deterministic hash-based implementation so the
/// *system* — thresholds, TTL, eviction, budgets, coalescing — is what gets
/// exercised and tested, without smuggling a 100MB model into a library repo.
///
/// An embedding cache (memoizing repeated embed calls) is deliberately out of
/// scope: it belongs as a decorator on this seam, and that decorator pattern
/// is already demonstrated in the companion `on-device-rag-kit` repo.
public protocol Embedder: Sendable {
    var dimension: Int { get }
    func embed(_ text: String) async throws -> [Double]
}

/// Deterministic n-gram hashing embedder (FNV-1a over character trigrams,
/// bucketed into a fixed-dimension vector, L2-normalized).
///
/// This is an honestly-labeled *simulation* of a semantic encoder: texts that
/// share wording overlap strongly (high cosine similarity), unrelated texts
/// do not. It is deterministic across runs and platforms, which is exactly
/// what reproducible tests and demos need. It does NOT capture true semantic
/// similarity between differently-worded paraphrases — a real encoder does,
/// and drops into the same `Embedder` seam without touching any other code.
public struct DeterministicHashEmbedder: Embedder {
    public let dimension: Int
    private let gramLength = 3

    public init(dimension: Int = 64) {
        // A tiny dimension makes hash collisions dominate and similarity
        // meaningless; clamp to a floor rather than trusting the caller.
        self.dimension = max(8, dimension)
    }

    public func embed(_ text: String) async throws -> [Double] {
        let normalized = TextNormalizer.normalize(text)
        guard !normalized.isEmpty else { throw EmbeddingError.emptyInput }

        var vector = [Double](repeating: 0, count: dimension)
        let scalars = Array(normalized.unicodeScalars)

        if scalars.count < gramLength {
            // Short inputs get a single whole-string gram; the alternative
            // (throwing) would make two-character queries unusable.
            let bucket = Int(fnv1a(normalized.utf8) % UInt64(dimension))
            vector[bucket] += 1
        } else {
            // Bounds: i ranges over 0...(count - gramLength), so the slice
            // upper bound i + gramLength never exceeds count.
            for i in 0...(scalars.count - gramLength) {
                let gram = String(String.UnicodeScalarView(scalars[i..<(i + gramLength)]))
                let bucket = Int(fnv1a(gram.utf8) % UInt64(dimension))
                vector[bucket] += 1
            }
        }

        let magnitude = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        // Unreachable in practice (at least one gram was counted above), but
        // a division by zero here would poison every similarity comparison,
        // so it is guarded rather than assumed.
        guard magnitude > 0 else { throw EmbeddingError.emptyInput }
        return vector.map { $0 / magnitude }
    }

    private func fnv1a(_ bytes: String.UTF8View) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in bytes {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return hash
    }
}

// MARK: - Cosine similarity

public enum CosineSimilarity {
    /// Cosine similarity in [-1, 1], or nil when the comparison is undefined
    /// (dimension mismatch, empty vectors, or a zero-magnitude vector).
    ///
    /// Returning nil instead of 0 matters: a dimension mismatch is a
    /// programming error worth surfacing as "no comparison possible", not a
    /// legitimate "totally dissimilar" score that silently degrades hit rate.
    public static func between(_ a: [Double], _ b: [Double]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var dot = 0.0
        var magA = 0.0
        var magB = 0.0
        for i in a.indices {
            dot += a[i] * b[i]
            magA += a[i] * a[i]
            magB += b[i] * b[i]
        }
        guard magA > 0, magB > 0 else { return nil }
        return dot / (magA.squareRoot() * magB.squareRoot())
    }
}
