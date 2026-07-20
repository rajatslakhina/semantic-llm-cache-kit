// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SemanticLLMCacheKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        // Library product only — the runnable demo lives in a separate repo
        // (semantic-llm-cache-kit-demo-app) that consumes this package as a
        // remote dependency, the same way any external consumer would.
        .library(
            name: "SemanticLLMCacheKit",
            targets: ["SemanticLLMCacheKit"]
        )
    ],
    targets: [
        .target(
            name: "SemanticLLMCacheKit"
        ),
        .testTarget(
            name: "SemanticLLMCacheKitTests",
            dependencies: ["SemanticLLMCacheKit"]
        )
    ]
)
