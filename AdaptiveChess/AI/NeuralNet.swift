import Foundation

/// A multilayer perceptron for chess position evaluation, implemented
/// from scratch in Swift (no Core ML) so the weights can live inside the
/// player's personal profile file.
///
/// Input: 769 floats — 64 squares × 12 one-hot features (6 piece types ×
/// 2 colors), plus side-to-move at index 768 (+1 White / −1 Black).
///
/// Architecture: 769 → 512 → 256 → 128 → 64 → 32 → 1. Hidden layers use
/// ReLU, the output uses tanh, so the evaluation is always in [−1, 1]
/// **from White's perspective** (+1 = White winning, −1 = Black winning).
///
/// ~569k parameters. One forward pass is ~569k multiply-adds — still
/// trivial for an iPhone, even for 1-ply reply checks.
final class NeuralNet: Codable {

    // MARK: - Architecture

    static let inputSize = 769
    static let hidden1Size = 512
    static let hidden2Size = 256
    static let hidden3Size = 128
    static let hidden4Size = 64
    static let hidden5Size = 32

    // w1[512][769], b1[512]
    // w2[256][512], b2[256]
    // w3[128][256], b3[128]
    // w4[64][128],  b4[64]
    // w5[32][64],   b5[32]
    // w6[1][32],    b6[1]
    var w1: [[Float]]; var b1: [Float]
    var w2: [[Float]]; var b2: [Float]
    var w3: [[Float]]; var b3: [Float]
    var w4: [[Float]]; var b4: [Float]
    var w5: [[Float]]; var b5: [Float]
    var w6: [[Float]]; var b6: [Float]

    // MARK: - Init

    /// Random initialization: He init for the ReLU layers, Xavier for the
    /// tanh output layer. A fresh network plays essentially randomly —
    /// which is exactly what "level 1 is beatable" needs.
    init() {
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size
        let nH3 = NeuralNet.hidden3Size
        let nH4 = NeuralNet.hidden4Size
        let nH5 = NeuralNet.hidden5Size

        func rand(fanIn: Int, fanOut: Int, relu: Bool) -> Float {
            let limit: Float = relu
                ? sqrtf(2.0 / Float(fanIn))
                : sqrtf(6.0 / Float(fanIn + fanOut))
            return Float.random(in: -limit...limit)
        }

        w1 = (0..<nH1).map { _ in (0..<nIn).map { _ in rand(fanIn: nIn, fanOut: nH1, relu: true) } }
        b1 = [Float](repeating: 0, count: nH1)
        w2 = (0..<nH2).map { _ in (0..<nH1).map { _ in rand(fanIn: nH1, fanOut: nH2, relu: true) } }
        b2 = [Float](repeating: 0, count: nH2)
        w3 = (0..<nH3).map { _ in (0..<nH2).map { _ in rand(fanIn: nH2, fanOut: nH3, relu: true) } }
        b3 = [Float](repeating: 0, count: nH3)
        w4 = (0..<nH4).map { _ in (0..<nH3).map { _ in rand(fanIn: nH3, fanOut: nH4, relu: true) } }
        b4 = [Float](repeating: 0, count: nH4)
        w5 = (0..<nH5).map { _ in (0..<nH4).map { _ in rand(fanIn: nH4, fanOut: nH5, relu: true) } }
        b5 = [Float](repeating: 0, count: nH5)
        w6 = (0..<1).map { _ in (0..<nH5).map { _ in rand(fanIn: nH5, fanOut: 1, relu: false) } }
        b6 = [Float](repeating: 0, count: 1)
    }

    // MARK: - Forward pass

    /// Evaluate a position. Returns a value in [−1, 1] from White's perspective.
    func forward(_ x: [Float]) -> Float {
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size
        let nH3 = NeuralNet.hidden3Size
        let nH4 = NeuralNet.hidden4Size
        let nH5 = NeuralNet.hidden5Size

        var h1 = [Float](repeating: 0, count: nH1)
        for i in 0..<nH1 {
            var s = b1[i]
            let row = w1[i]
            for j in 0..<nIn { s += row[j] * x[j] }
            h1[i] = s > 0 ? s : 0
        }
        var h2 = [Float](repeating: 0, count: nH2)
        for i in 0..<nH2 {
            var s = b2[i]
            let row = w2[i]
            for j in 0..<nH1 { s += row[j] * h1[j] }
            h2[i] = s > 0 ? s : 0
        }
        var h3 = [Float](repeating: 0, count: nH3)
        for i in 0..<nH3 {
            var s = b3[i]
            let row = w3[i]
            for j in 0..<nH2 { s += row[j] * h2[j] }
            h3[i] = s > 0 ? s : 0
        }
        var h4 = [Float](repeating: 0, count: nH4)
        for i in 0..<nH4 {
            var s = b4[i]
            let row = w4[i]
            for j in 0..<nH3 { s += row[j] * h3[j] }
            h4[i] = s > 0 ? s : 0
        }
        var h5 = [Float](repeating: 0, count: nH5)
        for i in 0..<nH5 {
            var s = b5[i]
            let row = w5[i]
            for j in 0..<nH4 { s += row[j] * h4[j] }
            h5[i] = s > 0 ? s : 0
        }

        var z = b6[0]
        let row = w6[0]
        for j in 0..<nH5 { z += row[j] * h5[j] }
        return tanh(z)
    }

    // MARK: - Training

    /// One SGD step on mean-squared error: L = (target − out)².
    func train(x: [Float], target: Float, lr: Float) {
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size
        let nH3 = NeuralNet.hidden3Size
        let nH4 = NeuralNet.hidden4Size
        let nH5 = NeuralNet.hidden5Size

        // ---- forward pass, caching pre-activations ----
        var z1 = [Float](repeating: 0, count: nH1)
        var h1 = [Float](repeating: 0, count: nH1)
        for i in 0..<nH1 {
            var s = b1[i]
            let row = w1[i]
            for j in 0..<nIn { s += row[j] * x[j] }
            z1[i] = s; h1[i] = s > 0 ? s : 0
        }
        var z2 = [Float](repeating: 0, count: nH2)
        var h2 = [Float](repeating: 0, count: nH2)
        for i in 0..<nH2 {
            var s = b2[i]
            let row = w2[i]
            for j in 0..<nH1 { s += row[j] * h1[j] }
            z2[i] = s; h2[i] = s > 0 ? s : 0
        }
        var z3 = [Float](repeating: 0, count: nH3)
        var h3 = [Float](repeating: 0, count: nH3)
        for i in 0..<nH3 {
            var s = b3[i]
            let row = w3[i]
            for j in 0..<nH2 { s += row[j] * h2[j] }
            z3[i] = s; h3[i] = s > 0 ? s : 0
        }
        var z4 = [Float](repeating: 0, count: nH4)
        var h4 = [Float](repeating: 0, count: nH4)
        for i in 0..<nH4 {
            var s = b4[i]
            let row = w4[i]
            for j in 0..<nH3 { s += row[j] * h3[j] }
            z4[i] = s; h4[i] = s > 0 ? s : 0
        }
        var z5 = [Float](repeating: 0, count: nH5)
        var h5 = [Float](repeating: 0, count: nH5)
        for i in 0..<nH5 {
            var s = b5[i]
            let row = w5[i]
            for j in 0..<nH4 { s += row[j] * h4[j] }
            z5[i] = s; h5[i] = s > 0 ? s : 0
        }
        var z6 = b6[0]
        do {
            let row = w6[0]
            for j in 0..<nH5 { z6 += row[j] * h5[j] }
        }
        let out = tanh(z6)

        // ---- backward pass ----
        var d6 = clip(-2 * (target - out) * (1 - out * out))

        // delta for hidden5 (uses w6 BEFORE update)
        var d5 = [Float](repeating: 0, count: nH5)
        do {
            let row = w6[0]
            for i in 0..<nH5 { d5[i] = clip(d6 * row[i] * (z5[i] > 0 ? 1 : 0)) }
        }
        // apply output-layer update
        do {
            for j in 0..<nH5 { w6[0][j] -= lr * d6 * h5[j] }
            b6[0] -= lr * d6
        }

        // delta for hidden4 (uses w5 BEFORE update)
        var d4 = [Float](repeating: 0, count: nH4)
        for j in 0..<nH4 {
            var s: Float = 0
            for i in 0..<nH5 { s += d5[i] * w5[i][j] }
            d4[j] = clip(s * (z4[j] > 0 ? 1 : 0))
        }
        for i in 0..<nH5 {
            let d = d5[i]
            b5[i] -= lr * d
            for j in 0..<nH4 { w5[i][j] -= lr * d * h4[j] }
        }

        // delta for hidden3 (uses w4 BEFORE update)
        var d3 = [Float](repeating: 0, count: nH3)
        for j in 0..<nH3 {
            var s: Float = 0
            for i in 0..<nH4 { s += d4[i] * w4[i][j] }
            d3[j] = clip(s * (z3[j] > 0 ? 1 : 0))
        }
        for i in 0..<nH4 {
            let d = d4[i]
            b4[i] -= lr * d
            for j in 0..<nH3 { w4[i][j] -= lr * d * h3[j] }
        }

        // delta for hidden2 (uses w3 BEFORE update)
        var d2 = [Float](repeating: 0, count: nH2)
        for j in 0..<nH2 {
            var s: Float = 0
            for i in 0..<nH3 { s += d3[i] * w3[i][j] }
            d2[j] = clip(s * (z2[j] > 0 ? 1 : 0))
        }
        for i in 0..<nH3 {
            let d = d3[i]
            b3[i] -= lr * d
            for j in 0..<nH2 { w3[i][j] -= lr * d * h2[j] }
        }

        // delta for hidden1 (uses w2 BEFORE update)
        var d1 = [Float](repeating: 0, count: nH1)
        for j in 0..<nH1 {
            var s: Float = 0
            for i in 0..<nH2 { s += d2[i] * w2[i][j] }
            d1[j] = clip(s * (z1[j] > 0 ? 1 : 0))
        }
        for i in 0..<nH2 {
            let d = d2[i]
            b2[i] -= lr * d
            for j in 0..<nH1 { w2[i][j] -= lr * d * h1[j] }
        }

        // apply hidden-layer-1 update
        for j in 0..<nH1 {
            let d = d1[j]
            b1[j] -= lr * d
            for k in 0..<nIn { w1[j][k] -= lr * d * x[k] }
        }
    }

    /// Gradient clipping keeps one bad position from blowing up the weights.
    private func clip(_ v: Float) -> Float {
        return min(max(v, -1), 1)
    }

    /// Export weights as flat arrays for profile persistence.
    func exportWeights() -> NetworkWeights {
        return NetworkWeights(
            w1: w1.flatMap { $0 }, b1: b1,
            w2: w2.flatMap { $0 }, b2: b2,
            w3: w3.flatMap { $0 }, b3: b3,
            w4: w4.flatMap { $0 }, b4: b4,
            w5: w5.flatMap { $0 }, b5: b5,
            w6: w6.flatMap { $0 }, b6: b6
        )
    }

    /// Initialize from saved weights. Returns nil if shapes don't match.
    init?(weights: NetworkWeights) {
        guard weights.isValid else { return nil }
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size
        let nH3 = NeuralNet.hidden3Size
        let nH4 = NeuralNet.hidden4Size
        let nH5 = NeuralNet.hidden5Size
        w1 = stride(from: 0, to: weights.w1.count, by: nIn).map {
            Array(weights.w1[$0..<min($0 + nIn, weights.w1.count)])
        }
        b1 = weights.b1
        w2 = stride(from: 0, to: weights.w2.count, by: nH1).map {
            Array(weights.w2[$0..<min($0 + nH1, weights.w2.count)])
        }
        b2 = weights.b2
        w3 = stride(from: 0, to: weights.w3.count, by: nH2).map {
            Array(weights.w3[$0..<min($0 + nH2, weights.w3.count)])
        }
        b3 = weights.b3
        w4 = stride(from: 0, to: weights.w4.count, by: nH3).map {
            Array(weights.w4[$0..<min($0 + nH3, weights.w4.count)])
        }
        b4 = weights.b4
        w5 = stride(from: 0, to: weights.w5.count, by: nH4).map {
            Array(weights.w5[$0..<min($0 + nH4, weights.w5.count)])
        }
        b5 = weights.b5
        w6 = [weights.w6]
        b6 = weights.b6
    }
}
