import Foundation

/// Injected time source.
///
/// Every time-dependent decision in this package (TTL expiry, budget-window
/// pruning, retry-after computation, LRU recency) reads the clock through this
/// seam so tests can drive time deterministically instead of sleeping.
/// A test suite that sleeps is a test suite that flakes.
public protocol CacheClock: Sendable {
    var now: Date { get }
}

/// Production clock backed by the system time.
public struct SystemCacheClock: CacheClock {
    public init() {}
    public var now: Date { Date() }
}
