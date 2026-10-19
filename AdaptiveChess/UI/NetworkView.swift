import SwiftUI

// Zoomable visual explorer for the profile's neural network (769 → 256 → 128 → 1).
// Neurons are circles connected by lines; pinch to zoom and everything
// (circles, lines, text) scales together. Tap a neuron for its weights.
// The bottom meter shows how much the weights moved in the last game.
struct NetworkView: View {
    @ObservedObject var store: ProfileStore
    @State private var zoom: CGFloat = 0.5
    @State private var selectedNeuron: NeuronSelection? = nil

    var body: some View {
        VStack(spacing: 0) {
            zoomControls
            if let weights = store.profile.networkWeights, weights.isValid {
                ScrollView([.horizontal, .vertical]) {
                    NetworkCanvas(weights: weights, zoom: zoom, onTap: { selectedNeuron = $0 })
                        .scaleEffect(zoom, anchor: .topLeading)
                        .frame(
                            width: NetworkLayout.canvasWidth * zoom,
                            height: NetworkLayout.canvasHeight * zoom,
                            alignment: .topLeading
                        )
                }
                .gesture(
                    MagnifyGesture()
                        .onChanged { value in
                            zoom = clampZoom(baseZoom * value.magnification)
                        }
                        .onEnded { _ in baseZoom = zoom }
                )
            } else {
                Text("No trained weights yet. Finish a game and the network will appear here.")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding()
                Spacer()
            }
            weightChangeMeter
        }
        .navigationTitle("Neural Network")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selectedNeuron) { selection in
            NeuronDetailView(selection: selection, weights: store.profile.networkWeights)
        }
        .onAppear { baseZoom = zoom }
    }

    @State private var baseZoom: CGFloat = 0.5

    private func clampZoom(_ z: CGFloat) -> CGFloat {
        min(3.0, max(0.15, z))
    }

    private var zoomControls: some View {
        HStack {
            Button { zoom = clampZoom(zoom - 0.15); baseZoom = zoom } label: {
                Image(systemName: "minus.magnifyingglass").font(.title3)
            }
            Text("\(Int(zoom * 100))%")
                .font(.body.monospaced())
                .frame(minWidth: 60)
            Button { zoom = clampZoom(zoom + 0.15); baseZoom = zoom } label: {
                Image(systemName: "plus.magnifyingglass").font(.title3)
            }
            Spacer()
            Text("Pinch to zoom · tap a neuron for weights")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Weight change meter

    private var weightChangeMeter: some View {
        VStack(spacing: 8) {
            Text("Learning Since Last Game")
                .font(.headline)
            if let change = store.lastWeightChange {
                VStack(spacing: 6) {
                    Text(String(format: "%.6f", change))
                        .font(.system(.title2, design: .monospaced).bold())
                    Text("\(store.lastChangedWeightCount) of 230,145 weights moved")
                        .font(.callout)
                        .foregroundColor(.secondary)
                    GeometryReader { geo in
                        let fraction = min(1.0, max(0.02, log10(max(change, 1e-7)) + 7) / 4)
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color(.systemGray5))
                                .frame(height: 12)
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.accentColor)
                                .frame(width: geo.size.width * CGFloat(fraction), height: 12)
                        }
                    }
                    .frame(height: 12)
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
            } else {
                Text("Finish a game and the change will appear here.")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
            }
        }
        .padding()
        .background(Color(.systemBackground).opacity(0.95))
    }
}

// MARK: - Layout

private enum NetworkLayout {
    // Column X centers
    static let inputX: CGFloat = 320
    static let hidden1X: CGFloat = 900
    static let hidden2X: CGFloat = 1320
    static let outputX: CGFloat = 1600

    // Grid configs: (columns, cell size)
    static let inputCols = 24
    static let inputCell: CGFloat = 26
    static let hidden1Cols = 16
    static let hidden1Cell: CGFloat = 36
    static let hidden2Cell: CGFloat = 44
    static let hidden2Cell: CGFloat = 44
    static let outputDiameter: CGFloat = 90

    static let topMargin: CGFloat = 80
    static let canvasWidth: CGFloat = 1800
    static let canvasHeight: CGFloat = 1400

    static func position(layer: NeuronSelection.Layer, index: Int) -> CGPoint {
        switch layer {
        case .input:
            let row = index / inputCols
            let col = index % inputCols
            let gridW = CGFloat(inputCols) * inputCell
            let x = inputX - gridW / 2 + CGFloat(col) * inputCell + inputCell / 2
            let y = topMargin + CGFloat(row) * inputCell + inputCell / 2
            return CGPoint(x: x, y: y)
        case .hidden1:
            let row = index / hidden1Cols
            let col = index % hidden1Cols
            let gridW = CGFloat(hidden1Cols) * hidden1Cell
            let x = hidden1X - gridW / 2 + CGFloat(col) * hidden1Cell + hidden1Cell / 2
            let y = topMargin + 120 + CGFloat(row) * hidden1Cell + hidden1Cell / 2
            return CGPoint(x: x, y: y)
        case .hidden2:
            let row = index / hidden2Cols
            let col = index % hidden2Cols
            let gridW = CGFloat(hidden2Cols) * hidden2Cell
            let x = hidden2X - gridW / 2 + CGFloat(col) * hidden2Cell + hidden2Cell / 2
            let y = topMargin + 280 + CGFloat(row) * hidden2Cell + hidden2Cell / 2
            return CGPoint(x: x, y: y)
        case .output:
            return CGPoint(x: outputX, y: topMargin + 480)
        }
    }

    static func radius(layer: NeuronSelection.Layer) -> CGFloat {
        switch layer {
        case .input: return inputCell / 2 - 2
        case .hidden1: return hidden1Cell / 2 - 3
        case .hidden2: return hidden2Cell / 2 - 4
        case .output: return outputDiameter / 2
        }
    }

    /// Hit-test a point against all neurons. Returns the tapped neuron, if any.
    static func hitTest(_ point: CGPoint) -> NeuronSelection? {
        // Check smallest layers first (they're on top visually)
        let checks: [(NeuronSelection.Layer, Int)] = [
            (.output, 1), (.hidden2, 128), (.hidden1, 256), (.input, 769),
        ]
        for (layer, count) in checks {
            let r = radius(layer: layer)
            for i in 0..<count {
                let p = position(layer: layer, index: i)
                let dx = point.x - p.x, dy = point.y - p.y
                if dx * dx + dy * dy <= (r + 6) * (r + 6) {
                    return NeuronSelection(layer: layer, index: i)
                }
            }
        }
        return nil
    }
}

// MARK: - Canvas

private struct NetworkCanvas: View {
    let weights: NetworkWeights
    let zoom: CGFloat
    let onTap: (NeuronSelection) -> Void

    var body: some View {
        Canvas { context, size in
            drawConnections(context: &context)
            drawLayer(context: &context, layer: .input, count: 769, biases: nil)
            drawLayer(context: &context, layer: .hidden1, count: 256, biases: weights.b1)
            drawLayer(context: &context, layer: .hidden2, count: 128, biases: weights.b2)
            drawLayer(context: &context, layer: .output, count: 1, biases: weights.b3)
            drawLabels(context: &context)
        }
        .frame(width: NetworkLayout.canvasWidth, height: NetworkLayout.canvasHeight)
        .onTapGesture { location in
            // location is in scaled space; convert to canvas logical coordinates
            let logical = CGPoint(x: location.x / zoom, y: location.y / zoom)
            if let hit = NetworkLayout.hitTest(logical) {
                onTap(hit)
            }
        }
    }

    private func drawConnections(context: inout GraphicsContext) {
        // Output layer: all 32 connections
        let outP = NetworkLayout.position(layer: .output, index: 0)
        for i in 0..<128 {
            let p = NetworkLayout.position(layer: .hidden2, index: i)
            let w = weights.w3[i]
            var path = Path()
            path.move(to: p)
            path.addLine(to: outP)
            context.stroke(path, with: .color(connectionColor(w).opacity(0.7)), lineWidth: 2)
        }
        // Hidden1 -> Hidden2: all 4096, very faint
        for j in 0..<128 {
            let dst = NetworkLayout.position(layer: .hidden2, index: j)
            for i in 0..<256 {
                let src = NetworkLayout.position(layer: .hidden1, index: i)
                let w = weights.w2[j * 256 + i]
                if abs(w) < 0.02 { continue }
                var path = Path()
                path.move(to: src)
                path.addLine(to: dst)
                context.stroke(path, with: .color(connectionColor(w).opacity(0.12)), lineWidth: 1)
            }
        }
        // Input -> Hidden1: only the strongest ~1500 connections to avoid clutter
        var strongest: [(w: Float, src: Int, dst: Int)] = []
        strongest.reserveCapacity(2000)
        for j in 0..<256 {
            for i in 0..<769 {
                let w = weights.w1[j * 769 + i]
                if abs(w) > 0.15 {
                    strongest.append((w, i, j))
                }
            }
        }
        strongest.sort { abs($0.w) > abs($1.w) }
        for conn in strongest.prefix(1500) {
            let src = NetworkLayout.position(layer: .input, index: conn.src)
            let dst = NetworkLayout.position(layer: .hidden1, index: conn.dst)
            var path = Path()
            path.move(to: src)
            path.addLine(to: dst)
            context.stroke(path, with: .color(connectionColor(conn.w).opacity(0.25)), lineWidth: 1)
        }
    }

    private func connectionColor(_ w: Float) -> Color {
        w >= 0 ? .blue : .orange
    }

    private func drawLayer(context: inout GraphicsContext, layer: NeuronSelection.Layer, count: Int, biases: [Float]?) {
        for i in 0..<count {
            let p = NetworkLayout.position(layer: layer, index: i)
            let r = NetworkLayout.radius(layer: layer)
            let bias = (biases != nil && i < biases!.count) ? biases![i] : 0

            let circle = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
            context.fill(circle, with: .color(neuronColor(bias: bias)))
            context.stroke(circle, with: .color(.primary.opacity(0.3)), lineWidth: 1)

            // Index number inside — scales with zoom because it's drawn in canvas space
            if r >= 10 {
                let fontSize = min(r * 0.7, 16)
                let text = Text("\(i)").font(.system(size: fontSize, weight: .bold, design: .monospaced))
                context.draw(text, at: p, anchor: .center)
            }
        }
    }

    private func neuronColor(bias: Float) -> Color {
        // All neurons fire on every evaluation — color is decorative by layer,
        // not an "active/inactive" indicator. Tap a neuron to see its weights.
        Color.blue.opacity(0.35)
    }

    private func drawLabels(context: inout GraphicsContext) {
        let labels: [(String, CGFloat)] = [
            ("Input · 769", NetworkLayout.inputX),
            ("Hidden 1 · 256", NetworkLayout.hidden1X),
            ("Hidden 2 · 128", NetworkLayout.hidden2X),
            ("Output · 1", NetworkLayout.outputX),
        ]
        for (text, x) in labels {
            let t = Text(text).font(.system(size: 28, weight: .bold))
            context.draw(t, at: CGPoint(x: x, y: 30), anchor: .center)
        }
    }
}

// MARK: - Neuron selection + detail (unchanged)

struct NeuronSelection: Identifiable {
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
                    explanation
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

    private var explanation: Text {
        switch selection.layer {
        case .input:
            return Text(inputDescription(selection.index))
        case .hidden1:
            return Text("Takes all 769 inputs, weighs each one, and fires when its pattern is on the board.")
        case .hidden2:
            return Text("Combines the 128 pattern detectors into higher-level strategy signals.")
        case .output:
            return Text("The final verdict: positive means good for the side to move, negative means bad. This is the number the AI uses to pick moves.")
        }
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
            let start = selection.index * 256
            raw = Array(w.w2[start..<(start + 256)])
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
