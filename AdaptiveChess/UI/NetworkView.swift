import SwiftUI

// Zoomable visual explorer for the profile's neural network (911 → 512 → 256 → 128 → 64 → 32 → 3).
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
            networkThoughts
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

    private var weightChangeMeter: some View {        VStack(spacing: 8) {
            Text("Learning Since Last Game")
                .font(.headline)
            if let change = store.lastWeightChange {
                VStack(spacing: 6) {
                    Text(String(format: "%.6f", change))
                        .font(.system(.title2, design: .monospaced).bold())
                    Text("\(store.lastChangedWeightCount) of 641,603 weights moved")
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

    // MARK: - What the network thinks is new

    /// Plain-language summary from the on-device AI, reset every game.
    /// Also saved to network-thoughts.txt in the app's documents.
    @ViewBuilder
    private var networkThoughts: some View {
        if #available(iOS 26, *) {
            VStack(alignment: .leading, spacing: 8) {
                Text("What Your Network Thinks Is New")
                    .font(.headline)
                if let summary = NetworkNarrator.shared.latestSummary {
                    Text(summary)
                        .font(.body)
                        .foregroundColor(.primary)
                } else {
                    Text("Finish a game and the on-device AI will explain what changed.")
                        .font(.callout)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(12)
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
    }
}

// MARK: - Layout

private enum NetworkLayout {
    // Column X centers — spaced so no two neuron grids overlap.
    // Grid widths: input 624, h1 600, h2 576, h3 480, h4 384, h5 416, out 90.
    // 250pt gaps between adjacent grid edges.
    static let inputX: CGFloat = 412
    static let hidden1X: CGFloat = 1274
    static let hidden2X: CGFloat = 2112
    static let hidden3X: CGFloat = 2890
    static let hidden4X: CGFloat = 3572
    static let hidden5X: CGFloat = 4222
    static let outputX: CGFloat = 4725

    // Grid configs: (columns, cell size)
    static let inputCols = 24
    static let inputCell: CGFloat = 26
    static let hidden1Cols = 20
    static let hidden1Cell: CGFloat = 30
    static let hidden2Cols = 16
    static let hidden2Cell: CGFloat = 36
    static let hidden3Cols = 12
    static let hidden3Cell: CGFloat = 40
    static let hidden4Cols = 8
    static let hidden4Cell: CGFloat = 48
    static let hidden5Cols = 8
    static let hidden5Cell: CGFloat = 52
    static let outputDiameter: CGFloat = 90

    static let topMargin: CGFloat = 80
    static let canvasWidth: CGFloat = 4900
    static let canvasHeight: CGFloat = 1700

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
            let y = topMargin + 160 + CGFloat(row) * hidden2Cell + hidden2Cell / 2
            return CGPoint(x: x, y: y)
        case .hidden3:
            let row = index / hidden3Cols
            let col = index % hidden3Cols
            let gridW = CGFloat(hidden3Cols) * hidden3Cell
            let x = hidden3X - gridW / 2 + CGFloat(col) * hidden3Cell + hidden3Cell / 2
            let y = topMargin + 240 + CGFloat(row) * hidden3Cell + hidden3Cell / 2
            return CGPoint(x: x, y: y)
        case .hidden4:
            let row = index / hidden4Cols
            let col = index % hidden4Cols
            let gridW = CGFloat(hidden4Cols) * hidden4Cell
            let x = hidden4X - gridW / 2 + CGFloat(col) * hidden4Cell + hidden4Cell / 2
            let y = topMargin + 300 + CGFloat(row) * hidden4Cell + hidden4Cell / 2
            return CGPoint(x: x, y: y)
        case .hidden5:
            let row = index / hidden5Cols
            let col = index % hidden5Cols
            let gridW = CGFloat(hidden5Cols) * hidden5Cell
            let x = hidden5X - gridW / 2 + CGFloat(col) * hidden5Cell + hidden5Cell / 2
            let y = topMargin + 360 + CGFloat(row) * hidden5Cell + hidden5Cell / 2
            return CGPoint(x: x, y: y)
        case .output:
            // 3 horizon outputs stacked vertically: now, +1 move, +2 moves.
            let oy = topMargin + 380 + CGFloat(index) * (outputDiameter + 30)
            return CGPoint(x: outputX, y: oy)
        }
    }

    static func radius(layer: NeuronSelection.Layer) -> CGFloat {
        switch layer {
        case .input: return inputCell / 2 - 2
        case .hidden1: return hidden1Cell / 2 - 3
        case .hidden2: return hidden2Cell / 2 - 4
        case .hidden3: return hidden3Cell / 2 - 4
        case .hidden4: return hidden4Cell / 2 - 5
        case .hidden5: return hidden5Cell / 2 - 5
        case .output: return outputDiameter / 2
        }
    }

    /// Hit-test a point against all neurons. Returns the tapped neuron, if any.
    static func hitTest(_ point: CGPoint) -> NeuronSelection? {
        // Check smallest layers first (they're on top visually)
        let checks: [(NeuronSelection.Layer, Int)] = [
            (.output, 3), (.hidden5, 32), (.hidden4, 64), (.hidden3, 128), (.hidden2, 256), (.hidden1, 512), (.input, 911),
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
            drawLayer(context: &context, layer: .input, count: 911, biases: nil)
            drawLayer(context: &context, layer: .hidden1, count: 512, biases: weights.b1)
            drawLayer(context: &context, layer: .hidden2, count: 256, biases: weights.b2)
            drawLayer(context: &context, layer: .hidden3, count: 128, biases: weights.b3)
            drawLayer(context: &context, layer: .hidden4, count: 64, biases: weights.b4)
            drawLayer(context: &context, layer: .hidden5, count: 32, biases: weights.b5)
            drawLayer(context: &context, layer: .output, count: 3, biases: weights.b6)
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
        // Sean's call: draw EVERY connection in the network, not a sample.
        // Lines are batched by sign into one path per color per layer pair,
        // so 568K connections become 12 strokes. Thin + translucent so dense
        // regions read as a cloud, not a solid blob.
        func drawAll(w: [Float], srcLayer: NeuronSelection.Layer, srcCount: Int,
                     dstLayer: NeuronSelection.Layer, dstCount: Int,
                     opacity: Double, lineWidth: CGFloat) {
            // Precompute positions once per layer.
            var srcPts: [CGPoint] = []
            srcPts.reserveCapacity(srcCount)
            for i in 0..<srcCount { srcPts.append(NetworkLayout.position(layer: srcLayer, index: i)) }
            var dstPts: [CGPoint] = []
            dstPts.reserveCapacity(dstCount)
            for j in 0..<dstCount { dstPts.append(NetworkLayout.position(layer: dstLayer, index: j)) }

            var posPath = Path()
            var negPath = Path()
            for j in 0..<dstCount {
                let dst = dstPts[j]
                let base = j * srcCount
                for i in 0..<srcCount {
                    let wt = w[base + i]
                    let src = srcPts[i]
                    if wt >= 0 {
                        posPath.move(to: src); posPath.addLine(to: dst)
                    } else {
                        negPath.move(to: src); negPath.addLine(to: dst)
                    }
                }
            }
            context.stroke(posPath, with: .color(.blue.opacity(opacity)), lineWidth: lineWidth)
            context.stroke(negPath, with: .color(.orange.opacity(opacity)), lineWidth: lineWidth)
        }

        // Hidden5 -> Output: 32 connections
        drawAll(w: weights.w6, srcLayer: .hidden5, srcCount: 32,
                dstLayer: .output, dstCount: 3, opacity: 0.6, lineWidth: 2)
        // Hidden4 -> Hidden5: 2,048
        drawAll(w: weights.w5, srcLayer: .hidden4, srcCount: 64,
                dstLayer: .hidden5, dstCount: 32, opacity: 0.35, lineWidth: 1)
        // Hidden3 -> Hidden4: 8,192
        drawAll(w: weights.w4, srcLayer: .hidden3, srcCount: 128,
                dstLayer: .hidden4, dstCount: 64, opacity: 0.25, lineWidth: 0.8)
        // Hidden2 -> Hidden3: 32,768
        drawAll(w: weights.w3, srcLayer: .hidden2, srcCount: 256,
                dstLayer: .hidden3, dstCount: 128, opacity: 0.18, lineWidth: 0.6)
        // Hidden1 -> Hidden2: 131,072
        drawAll(w: weights.w2, srcLayer: .hidden1, srcCount: 512,
                dstLayer: .hidden2, dstCount: 256, opacity: 0.12, lineWidth: 0.5)
        // Input -> Hidden1: 393,728
        drawAll(w: weights.w1, srcLayer: .input, srcCount: 911,
                dstLayer: .hidden1, dstCount: 512, opacity: 0.08, lineWidth: 0.4)
    }

    private func drawLayer(context: inout GraphicsContext, layer: NeuronSelection.Layer, count: Int, biases: [Float]?) {
        // Sean's call: the text always shows and scales with the zoom, the same
        // way the circles do — it shrinks with them when zooming out instead of
        // vanishing. Font size is set in canvas coordinates, so the canvas
        // scaleEffect(zoom) scales text and circles together automatically.
        for i in 0..<count {
            let p = NetworkLayout.position(layer: layer, index: i)
            let r = NetworkLayout.radius(layer: layer)
            let bias: Float? = (biases != nil && i < biases!.count) ? biases![i] : nil

            let circle = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
            context.fill(circle, with: .color(neuronColor(bias: bias ?? 0)))
            context.stroke(circle, with: .color(.primary.opacity(0.3)), lineWidth: 1)

            // The neuron's own weight (bias) inside the circle.
            // Input neurons have no weights, so they show their index.
            if r >= 10 {
                let fontSize = min(r * 0.55, 14)
                let label: String
                if let b = bias {
                    label = String(format: "%+.2f", b)
                } else {
                    label = "\(i)"
                }
                let text = Text(label).font(.system(size: fontSize, weight: .bold, design: .monospaced))
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
            ("Input · 911", NetworkLayout.inputX),
            ("Hidden 1 · 512", NetworkLayout.hidden1X),
            ("Hidden 2 · 256", NetworkLayout.hidden2X),
            ("Hidden 3 · 128", NetworkLayout.hidden3X),
            ("Hidden 4 · 64", NetworkLayout.hidden4X),
            ("Hidden 5 · 32", NetworkLayout.hidden5X),
            ("Output · 3", NetworkLayout.outputX),
        ]
        for (text, x) in labels {
            let t = Text(text).font(.system(size: 28, weight: .bold))
            context.draw(t, at: CGPoint(x: x, y: 30), anchor: .center)
        }
    }
}

// MARK: - Neuron selection + detail (unchanged)

struct NeuronSelection: Identifiable {
    enum Layer { case input, hidden1, hidden2, hidden3, hidden4, hidden5, output }
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
        case .hidden3: return "Hidden 3 · Neuron #\(selection.index)"
        case .hidden4: return "Hidden 4 · Neuron #\(selection.index)"
        case .hidden5: return "Hidden 5 · Neuron #\(selection.index)"
        case .output:
            switch selection.index {
            case 0: return "Output · Now"
            case 1: return "Output · +1 move"
            case 2: return "Output · +2 moves"
            default: return "Output neuron"
            }
        }
    }

    private var explanation: Text {
        switch selection.layer {
        case .input:
            return Text(inputDescription(selection.index))
        case .hidden1:
            return Text("Takes all 911 inputs, weighs each one, and fires when its pattern is on the board.")
        case .hidden2:
            return Text("Combines the 512 pattern detectors into higher-level strategy signals.")
        case .hidden3:
            return Text("Distills strategy into 128 tactical concepts.")
        case .hidden4:
            return Text("Refines tactics into 64 positional judgments.")
        case .hidden5:
            return Text("Final reasoning layer — 32 high-level evaluations feed the output.")
        case .output:
            switch selection.index {
            case 0:
                return Text("Value of this position right now. Positive = good for White, negative = good for Black.")
            case 1:
                return Text("The network's prediction of the position value 1 full move from now — its internal lookahead, no search tree.")
            case 2:
                return Text("The network's prediction 2 full moves from now. This is how it 'thinks ahead' without ever generating a move.")
            default:
                return Text("Output neuron")
            }
        }
    }

    private func inputDescription(_ i: Int) -> String {
        switch i {
        case 768: return "1 when it's White to move, -1 for Black."
        case 769: return "1 when White can still castle kingside."
        case 770: return "1 when White can still castle queenside."
        case 771: return "1 when Black can still castle kingside."
        case 772: return "1 when Black can still castle queenside."
        case 773..<781:
            let file = String(UnicodeScalar(97 + (i - 773))!)
            return "1 when en passant is available on the \(file)-file."
        case 781: return "Half-moves since the last pawn move or capture, /100 (50-move rule)."
        case 782: return "1 when the side to move is in check."
        case 783..<847:
            let sq = i - 783
            let file = String(UnicodeScalar(97 + sq % 8)!)
            let rank = sq / 8 + 1
            return "1 when \(file)\(rank) is attacked by White."
        case 847..<911:
            let sq = i - 847
            let file = String(UnicodeScalar(97 + sq % 8)!)
            let rank = sq / 8 + 1
            return "1 when \(file)\(rank) is attacked by Black."
        default: break
        }
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
            switch i {
            case 768: return "side-to-move"
            case 769: return "white-can-castle-kingside"
            case 770: return "white-can-castle-queenside"
            case 771: return "black-can-castle-kingside"
            case 772: return "black-can-castle-queenside"
            case 773..<781:
                let file = String(UnicodeScalar(97 + (i - 773))!)
                return "en-passant-on-\(file)-file"
            case 781: return "halfmove-clock-50mr"
            case 782: return "side-to-move-in-check"
            case 783..<847:
                let sq = i - 783
                let file = String(UnicodeScalar(97 + sq % 8)!)
                let rank = sq / 8 + 1
                return "attacked-by-W-\(file)\(rank)"
            case 847..<911:
                let sq = i - 847
                let file = String(UnicodeScalar(97 + sq % 8)!)
                let rank = sq / 8 + 1
                return "attacked-by-B-\(file)\(rank)"
            default: break
            }
            let side = i < 384 ? "W" : "B"
            let pieces = ["P", "N", "B", "R", "Q", "K"]
            let piece = pieces[(i % 384) / 64]
            let sq = i % 64
            let file = String(UnicodeScalar(97 + sq % 8)!)
            let rank = sq / 8 + 1
            return "\(side)\(piece) \(file)\(rank)"
        case .hidden2:
            return "hidden1 #\(i)"
        case .hidden3:
            return "hidden2 #\(i)"
        case .hidden4:
            return "hidden3 #\(i)"
        case .hidden5:
            return "hidden4 #\(i)"
        case .output:
            return "hidden5 #\(i)"
        case .input:
            return ""
        }
    }

    private var weightRows: [(index: Int, weight: Float)]? {
        guard let w = weights, w.isValid else { return nil }
        let raw: [Float]
        switch selection.layer {
        case .hidden1:
            let start = selection.index * 911
            raw = Array(w.w1[start..<(start + 911)])
        case .hidden2:
            let start = selection.index * 512
            raw = Array(w.w2[start..<(start + 512)])
        case .hidden3:
            let start = selection.index * 256
            raw = Array(w.w3[start..<(start + 256)])
        case .hidden4:
            let start = selection.index * 128
            raw = Array(w.w4[start..<(start + 128)])
        case .hidden5:
            let start = selection.index * 64
            raw = Array(w.w5[start..<(start + 64)])
        case .output:
            let start = selection.index * 32
            raw = Array(w.w6[start..<(start + 32)])
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
        case .hidden3: return w.b3[selection.index]
        case .hidden4: return w.b4[selection.index]
        case .hidden5: return w.b5[selection.index]
        case .output:
            guard selection.index < w.b6.count else { return nil }
            return w.b6[selection.index]
        case .input: return nil
        }
    }
}
