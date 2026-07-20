import XCTest
@testable import SemanticLLMCacheKit

final class EmbeddingTests: XCTestCase {

    // MARK: TextNormalizer

    func testNormalizerCollapsesCaseAndWhitespace() {
        XCTAssertEqual(TextNormalizer.normalize("  What's   my ORDER\nstatus? "), "what's my order status?")
    }

    func testNormalizerEmptyAndWhitespaceOnly() {
        XCTAssertEqual(TextNormalizer.normalize(""), "")
        XCTAssertEqual(TextNormalizer.normalize("   \n\t "), "")
    }

    // MARK: DeterministicHashEmbedder

    func testEmbeddingIsDeterministic() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 64)
        let a = try await embedder.embed("reset my password please")
        let b = try await embedder.embed("reset my password please")
        XCTAssertEqual(a, b)
    }

    func testEmbeddingIsNormalizationInvariant() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 64)
        let a = try await embedder.embed("Reset my  PASSWORD please")
        let b = try await embedder.embed("reset my password please")
        XCTAssertEqual(a, b)
    }

    func testDistinctTextsProduceDistinctVectors() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 64)
        let a = try await embedder.embed("reset my password")
        let b = try await embedder.embed("track my recent order delivery")
        XCTAssertNotEqual(a, b)
    }

    func testEmbeddingIsUnitLength() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 64)
        let vector = try await embedder.embed("how do i return an item")
        let magnitude = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-9)
    }

    func testOverlappingTextsScoreHigherThanUnrelatedTexts() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 64)
        let base = try await embedder.embed("track my order status today")
        let overlapping = try await embedder.embed("track my order status now")
        let unrelated = try await embedder.embed("cancel subscription billing plan")
        let simOverlap = try XCTUnwrap(CosineSimilarity.between(base, overlapping))
        let simUnrelated = try XCTUnwrap(CosineSimilarity.between(base, unrelated))
        XCTAssertGreaterThan(simOverlap, simUnrelated)
    }

    func testEmptyInputThrows() async {
        let embedder = DeterministicHashEmbedder()
        do {
            _ = try await embedder.embed("   ")
            XCTFail("expected emptyInput")
        } catch let error as EmbeddingError {
            XCTAssertEqual(error, .emptyInput)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testShortInputStillEmbeds() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 16)
        let vector = try await embedder.embed("ok")
        XCTAssertEqual(vector.count, 16)
        let magnitude = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(magnitude, 1.0, accuracy: 1e-9)
    }

    func testDimensionClampedToFloor() {
        XCTAssertEqual(DeterministicHashEmbedder(dimension: 2).dimension, 8)
        XCTAssertEqual(DeterministicHashEmbedder(dimension: -10).dimension, 8)
        XCTAssertEqual(DeterministicHashEmbedder(dimension: 128).dimension, 128)
    }

    // MARK: CosineSimilarity

    func testIdenticalVectorsScoreOne() throws {
        let similarity = try XCTUnwrap(CosineSimilarity.between([1, 2, 3], [1, 2, 3]))
        XCTAssertEqual(similarity, 1.0, accuracy: 1e-12)
    }

    func testOrthogonalVectorsScoreZero() throws {
        let similarity = try XCTUnwrap(CosineSimilarity.between([1, 0], [0, 1]))
        XCTAssertEqual(similarity, 0.0, accuracy: 1e-12)
    }

    func testMismatchedDimensionsReturnNil() {
        XCTAssertNil(CosineSimilarity.between([1, 0, 0], [1, 0]))
    }

    func testEmptyVectorsReturnNil() {
        XCTAssertNil(CosineSimilarity.between([], []))
    }

    func testZeroMagnitudeReturnsNil() {
        XCTAssertNil(CosineSimilarity.between([0, 0], [1, 0]))
    }
}
