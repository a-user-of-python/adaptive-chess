import Foundation

/// A small multilayer perceptron for chess position evaluation, implemented
/// from scratch in Swift (no Core ML) so the weights can live inside the
/// player's personal profile file.
///
/// Input: 769 floats — 64 squares × 12 one-hot features (6 piece types ×
/// 2 colors), plus side-to-move at index 768 (+1 White / −1 Black).
///
/// Architecture: 769 → 128 → 32 → 1. Hidden layers use ReLU, the output uses
/// tanh, so the evaluation is always in [−1, 1] **from White's perspective**
/// (+1 = White winning, −1 = Black winning).
///
/// ~103k parameters. One forward pass is ~100k multiply-adds — trivial for
/// an iPhone, even for 1-ply reply checks (a few thousand evals per move).
final class NeuralNet: Codable {

    // MARK: - Architecture

    static let inputSize = 769
    static let hidden1Size = 128
    static let hidden2Size = 32

    // w1[hidden1][input], b1[hidden1]
    // w2[hidden2][hidden1], b2[hidden2]
    // w3[1][hidden2], b3[1]
    var w1: [[Float]]
    var b1: [Float]
    var w2: [[Float]]
    var b2: [Float]
    var w3: [[Float]]
    var b3: [Float]

    // MARK: - Init

    /// Random initialization: He init for the ReLU layers, Xavier for the
    /// tanh output layer. A fresh network plays essentially randomly —
    /// which is exactly what "level 1 is beatable" needs.
    init() {
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size

        func rand(fanIn: Int, fanOut: Int, relu: Bool) -> Float {
            // He: sqrt(2/fanIn) for ReLU. Xavier: sqrt(6/(fanIn+fanOut)) for tanh.
            let limit: Float = relu
                ? sqrtf(2.0 / Float(fanIn))
                : sqrtf(6.0 / Float(fanIn + fanOut))
            return Float.random(in: -limit...limit)
        }

        w1 = (0..<nH1).map { _ in (0..<nIn).map { _ in rand(fanIn: nIn, fanOut: nH1, relu: true) } }
        b1 = [Float](repeating: 0, count: nH1)
        w2 = (0..<nH2).map { _ in (0..<nH1).map { _ in rand(fanIn: nH1, fanOut: nH2, relu: true) } }
        b2 = [Float](repeating: 0, count: nH2)
        w3 = (0..<1).map { _ in (0..<nH2).map { _ in rand(fanIn: nH2, fanOut: 1, relu: false) } }
        b3 = [Float](repeating: 0, count: 1)
    }

    // MARK: - Forward pass

    /// Evaluate a position. Returns a value in [−1, 1] from White's perspective.
    func forward(_ x: [Float]) -> Float {
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size

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

        var z = b3[0]
        let row = w3[0]
        for j in 0..<nH2 { z += row[j] * h2[j] }
        return tanh(z)
    }

    // MARK: - Training

    /// One SGD step on mean-squared error: L = (target − out)².
    ///
    /// - Parameters:
    ///   - x: 769 input features (see ``PositionEncoder``).
    ///   - target: desired output in [−1, 1] from White's perspective.
    ///   - lr: learning rate. Kept small (0.01) so the network adapts
    ///     gradually over many games instead of overfitting one game.
    func train(x: [Float], target: Float, lr: Float) {
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size

        // ---- forward pass, caching pre-activations ----
        var z1 = [Float](repeating: 0, count: nH1)
        var h1 = [Float](repeating: 0, count: nH1)
        for i in 0..<nH1 {
            var s = b1[i]
            let row = w1[i]
            for j in 0..<nIn { s += row[j] * x[j] }
            z1[i] = s
            h1[i] = s > 0 ? s : 0
        }

        var z2 = [Float](repeating: 0, count: nH2)
        var h2 = [Float](repeating: 0, count: nH2)
        for i in 0..<nH2 {
            var s = b2[i]
            let row = w2[i]
            for j in 0..<nH1 { s += row[j] * h1[j] }
            z2[i] = s
            h2[i] = s > 0 ? s : 0
        }

        var z3 = b3[0]
        do {
            let row = w3[0]
            for j in 0..<nH2 { z3 += row[j] * h2[j] }
        }
        let out = tanh(z3)

        // ---- backward pass ----
        // dL/dz3 for L = (target − out)² with out = tanh(z3):
        //   dL/dout = −2(target − out), dout/dz3 = 1 − out²
        var d3 = -2 * (target - out) * (1 - out * out)
        d3 = clip(d3)

        // delta for hidden layer 2 (uses w3 BEFORE it is updated)
        var d2 = [Float](repeating: 0, count: nH2)
        do {
            let row = w3[0]
            for i in 0..<nH2 {
                d2[i] = clip(d3 * row[i] * (z2[i] > 0 ? 1 : 0))
            }
        }

        // apply output-layer update
        do {
            for j in 0..<nH2 { w3[0][j] -= lr * d3 * h2[j] }
            b3[0] -= lr * d3
        }

        // delta for hidden layer 1 (uses w2 BEFORE it is updated)
        var d1 = [Float](repeating: 0, count: nH1)
        for j in 0..<nH1 {
            var s: Float = 0
            for i in 0..<nH2 { s += d2[i] * w2[i][j] }
            d1[j] = clip(s * (z1[j] > 0 ? 1 : 0))
        }

        // apply hidden-layer-2 update
        for i in 0..<nH2 {
            let d = d2[i]
            b2[i] -= lr * d
            for j in 0..<nH1 { w2[i][j] -= lr * d * h1[j] }
        }

        // apply hidden-layer-1 update (x is sparse; zero entries are no-ops)
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
            w1: w1.flatMap { $0 },
            b1: b1,
            w2: w2.flatMap { $0 },
            b2: b2,
            w3: w3.flatMap { $0 },
            b3: b3
        )
    }

    /// Initialize from saved weights. Returns nil if shapes don't match.
    init?(weights: NetworkWeights) {
        guard weights.isValid else { return nil }
        let nIn = NeuralNet.inputSize
        let nH1 = NeuralNet.hidden1Size
        let nH2 = NeuralNet.hidden2Size
        // Unflatten w1: [128][769]
        w1 = stride(from: 0, to: weights.w1.count, by: nIn).map {
            Array(weights.w1[$0..<min($0 + nIn, weights.w1.count)])
        }
        b1 = weights.b1
        // Unflatten w2: [32][128]
        w2 = stride(from: 0, to: weights.w2.count, by: nH1).map {
            Array(weights.w2[$0..<min($0 + nH1, weights.w2.count)])
        }
        b2 = weights.b2
        // Unflatten w3: [1][32]
        w3 = [weights.w3]
        b3 = weights.b3
    }
}
