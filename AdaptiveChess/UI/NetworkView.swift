import SwiftUI

// Visual explorer for the profile's neural network (769 → 128 → 32 → 1).
// Each neuron is a circle; tap one to see its incoming weights.
// The bottom meter shows how much the weights moved in the last game.
struct NetworkView: View {
    @ObservedObject var store: ProfileStore
    @State private var selectedNeuron: NeuronSelection? = nil

    var body: some View {
        ScrollView {
            VStack(spacing: 28) {
                header

                if let weights = store.profile.networkWeights, weights.isValid {
                    outputLayer(weights: weights)
                    hiddenLayer2(weights: weights)
                    hiddenLayer1(weights: weights)
                    inputLayer
                } else {
                    Text("No trained weights yet. Finish a game and the network will appear here.")
                        .font(.title3)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding()
                }

                weightChangeMeter
            }
            .padding()
        }
        .navigationTitle("Neural Network")
        .sheet(item: $selectedNeuron) { selection in
            NeuronDetailView(selection: selection, weights: store.profile.networkWeights)
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(spacing: 6) {
            Text("769 → 128 → 32 → 1")
                .font(.title2.bold())
            Text("Tap any circle to see that neuron's weights.")
                .font(.body)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Layers

    private func outputLayer(weights: NetworkWeights) -> some View {
        VStack(spacing: 10) {
            layerLabel("Output", count: 1, description: "Win probability: -1 (loss) to +1 (win)")
            NeuronCircle(index: 0, value: weights.b3.first ?? 0, diameter: 88) {
                selectedNeuron = NeuronSelection(layer: .output, index: 0)
            }
        }
    }

    private func hiddenLayer2(weights: NetworkWeights) -> some View {
        VStack(spacing: 10) {
            layerLabel("Hidden layer 2", count: 32, description: "Combines patterns into strategy")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 8), spacing: 8) {
                ForEach(0..<32, id: \.self) { i in
                    NeuronCircle(index: i, value: weights.b2[i], diameter: 44) {
                        selectedNeuron = NeuronSelection(layer: .hidden2, index: i)
                    }
                }
            }
        }
    }

    private func hiddenLayer1(weights: NetworkWeights) -> some View {
        VStack(spacing: 10) {
            layerLabel("Hidden layer 1", count: 128, description: "Detects board patterns")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 8), spacing: 6) {
                ForEach(0..<128, id: \.self) { i in
                    NeuronCircle(index: i, value: weights.b1[i], diameter: 36) {
                        selectedNeuron = NeuronSelection(layer: .hidden1, index: i)
                    }
                }
            }
        }
    }

    private var inputLayer: some View {
        VStack(spacing: 10) {
            layerLabel("Input", count: 769, description: "Board squares × piece types + side to move")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 4), count: 12), spacing: 4) {
                ForEach(0..<769, id: \.self) { i in
                    NeuronCircle(index: i, value: 0, diameter: 28, showValue: false) {
                        selectedNeuron = NeuronSelection(layer: .input, index: i)
                    }
                }
            }
        }
    }

    private func layerLabel(_ name: String, count: Int, description: String) -> some View {
        VStack(spacing: 2) {
            Text("\(name)  ·  \(count)")
                .font(.headline)
            Text(description)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Weight change meter

    private var weightChangeMeter: some View {
        VStack(spacing: 10) {
            Text("Learning Since Last Game")
                .font(.headline)
            if let change = store.lastWeightChange {
                VStack(spacing: 6) {
                    Text(String(format: "%.6f", change))
                        .font(.system(.title, design: .monospaced).bold())
                    Text("\(store.lastChangedWeightCount) of 102,656 weights moved")
                        .font(.body)
                        .foregroundColor(.secondary)
                    // Visual bar: log-scaled so tiny changes are still visible
                    GeometryReader { geo in
                        let fraction = min(1.0, max(0.02, log10(max(change, 1e-7)) + 7) / 4))
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color(.systemGray5))
                                .frame(height: 14)
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.accentColor)
                                .frame(width: geo.size.width * CGFloat(fraction), height: 14)
                        }
                    }
                    .frame(height: 14)
                    Text("Bigger bar = more learning from that game. Early games move more; it settles as the AI converges on your style.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
            } else {
                Text("Finish a game and the change will appear here.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
            }
        }
    }
}

// MARK: - Neuron circle

private struct NeuronCircle: View {
    let index: Int
    let value: Float
    let diameter: CGFloat
    var showValue: Bool = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(circleColor)
                VStack(spacing: 0) {
                    Text("\(index)")
                        .font(.system(size: diameter * 0.28, weight: .bold, design: .monospaced))
                        .foregroundColor(.primary)
                    if showValue && diameter >= 36 {
                        Text(String(format: "%.2f", value))
                            .font(.system(size: diameter * 0.20, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }
            }
            .frame(width: diameter, height: diameter)
        }
        .buttonStyle(.plain)
    }

    private var circleColor: Color {
        // Color by bias sign/magnitude: blue = positive, orange = negative
        let v = Double(value)
        if abs(v) < 0.05 { return Color(.systemGray5) }
        return v > 0 ? Color.blue.opacity(min(0.85, 0.25 + abs(v) * 0.5))
                     : Color.orange.opacity(min(0.85, 0.25 + abs(v) * 0.5))
    }
}

// MARK: - Neuron selection + detail

private struct NeuronSelection: Identifiable {
    enum Layer { case input, hidden1, hidden2, output }
    let layer: Layer
    let index: Int
    var id: String { "\(layer)-\(index)" }
}

private struct NeuronDetailView: View {
    let selection: NeuronSelection
    let weights: NetworkWeights?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(title)
                        .font(.title2.bold())
                    Text(explanation)
                        .font(.body)
                        .foregroundColor(.secondary)

                    if let rows = weightRows, !rows.isEmpty {
                        Text("Strongest incoming weights")
                            .font(.headline)
                        ForEach(rows, id: \.index) { row in
                            HStack {
                                Text(describeInput(row.index))
                                    .font(.body.monospaced())
                                Spacer()
                                Text(String(format: "%+.4f", row.weight))
                                    .font(.body.monospaced().bold())
                                    .foregroundColor(row.weight >= 0 ? .blue : .orange)
                            }
                            .padding(.vertical, 4)
                            Divider()
                        }
                        if let bias = biasValue {
                            HStack {
                                Text("bias")
                                    .font(.body.monospaced())
                                Spacer()
                                Text(String(format: "%+.4f", bias))
                                    .font(.body.monospaced().bold())
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Neuron \(selection.index)")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var title: String {
        switch selection.layer {
        case .input: return "Input #\(selection.index)"
        case .hidden1: return "Hidden 1 · Neuron #\(selection.index)"
        case .hidden2: return "Hidden 2 · Neuron #\(selection.index)"
        case .output: return "Output neuron"
        }
    }

    private var explanation: some View {
        Text({
            switch selection.layer {
            case .input:
                return inputDescription(selection.index)
            case .hidden1:
                return "Takes all 769 inputs, weighs each one, and fires when its pattern is on the board."
            case .hidden2:
                return "Combines the 128 pattern detectors into higher-level strategy signals."
            case .output:
                return "The final verdict: positive means good for the side to move, negative means bad. This is the number the AI uses to pick moves."
            }
        }())
    }

    private func inputDescription(_ i: Int) -> String {
        if i == 768 { return "1 when it's White to move, 0 for Black." }
        let pieceNames = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        let side = i < 384 ? "White" : "Black"
        let piece = pieceNames[(i % 384) / 64]
        let sq = i % 64
        let file = String(UnicodeScalar(97 + sq % 8)!)
        let rank = sq / 8 + 1
        return "1 when \(side) \(piece) is on \(file)\(rank), else 0."
    }

    private func describeInput(_ i: Int) -> String {
        switch selection.layer {
        case .hidden1:
            if i == 768 { return "side-to-move" }
            let side = i < 384 ? "W" : "B"
            let pieces = ["P", "N", "B", "R", "Q", "K"]
            let piece = pieces[(i % 384) / 64]
            let sq = i % 64
            let file = String(UnicodeScalar(97 + sq % 8)!)
            let rank = sq / 8 + 1
            return "\(side)\(piece) \(file)\(rank)"
        case .hidden2, .output:
            return "hidden1 #\(i)"
        case .input:
            return ""
        }
    }

    private var weightRows: [(index: Int, weight: Float)]? {
        guard let w = weights, w.isValid else { return nil }
        let raw: [Float]
        switch selection.layer {
        case .hidden1:
            let start = selection.index * 769
            raw = Array(w.w1[start..<(start + 769)])
        case .hidden2:
            let start = selection.index * 128
            raw = Array(w.w2[start..<(start + 128)])
        case .output:
            raw = w.w3
        case .input:
            return nil
        }
        return raw.enumerated()
            .map { (index: $0.offset, weight: $0.element) }
            .sorted { abs($0.weight) > abs($1.weight) }
            .prefix(15)
            .map { $0 }
    }

    private var biasValue: Float? {
        guard let w = weights, w.isValid else { return nil }
        switch selection.layer {
        case .hidden1: return w.b1[selection.index]
        case .hidden2: return w.b2[selection.index]
        case .output: return w.b3.first
        case .input: return nil
        }
    }
}
