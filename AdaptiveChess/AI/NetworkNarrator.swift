import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Uses the on-device AI (Foundation Models, iOS 26+) to explain — in plain
/// language — what the chess network learned from the last game.
///
/// After every game the narrator diffs the old and new weights, picks the
/// most-changed connections, describes them in human terms ("White pawn on e4
/// → pattern detector #42"), and asks the on-device model to summarize what's
/// new. The result is written to `network-thoughts.txt` in the app's documents
/// directory, overwriting the previous game's summary. It is also published
/// via `latestSummary` so the UI can show it.
///
/// On devices without Foundation Models (older iOS, model not downloaded) it
/// falls back to a short template summary — the file still gets written.
@available(iOS 26, *)
final class NetworkNarrator: ObservableObject {
    static let shared = NetworkNarrator()

    /// The most recent summary text, for UI display.
    @Published var latestSummary: String?

    /// File name in the app's documents directory. Reset every game.
    static let fileName = "network-thoughts.txt"

    private init() {
        latestSummary = Self.readFile()
    }

    // MARK: - Entry point

    /// Call after training on a finished game. Runs off the main thread.
    func narrate(oldWeights: NetworkWeights?,
                 newWeights: NetworkWeights,
                 result: GameResult,
                 userColor: PieceColor,
                 moveCount: Int) {
        Task.detached(priority: .utility) {
            let summary = await Self.buildSummary(
                oldWeights: oldWeights,
                newWeights: newWeights,
                result: result,
                userColor: userColor,
                moveCount: moveCount
            )
            let text = Self.formatFile(summary: summary, result: result,
                                       moveCount: moveCount)
            Self.writeFile(text)
            await MainActor.run {
                self.latestSummary = summary
            }
        }
    }

    // MARK: - Summary building

    private static func buildSummary(oldWeights: NetworkWeights?,
                                     newWeights: NetworkWeights,
                                     result: GameResult,
                                     userColor: PieceColor,
                                     moveCount: Int) async -> String {
        let changes = describeChanges(old: oldWeights, new: newWeights)
        let fallback = templateSummary(changes: changes, result: result,
                                       moveCount: moveCount)

        #if canImport(FoundationModels)
        guard isModelAvailable() else { return fallback }
        do {
            let prompt = buildPrompt(changes: changes, result: result,
                                     userColor: userColor, moveCount: moveCount)
            let session = LanguageModelSession()
            let response = try await session.respond(to: prompt)
            let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? fallback : text
        } catch {
            return fallback
        }
        #else
        return fallback
        #endif
    }

    #if canImport(FoundationModels)
    private static func isModelAvailable() -> Bool {
        // SystemLanguageModel is the iOS 26+ availability gate.
        if #available(iOS 26, *) {
            return SystemLanguageModel.default.isAvailable
        }
        return false
    }
    #endif

    // MARK: - Change description

    /// Human-readable descriptions of the most-changed connections.
    private struct WeightChange {
        let description: String
        let oldValue: Float
        let newValue: Float
    }

    private static func describeChanges(old: NetworkWeights?,
                                        new: NetworkWeights,
                                        limit: Int = 12) -> [WeightChange] {
        guard let old = old, old.isValid, new.isValid else { return [] }
        guard old.w1.count == new.w1.count else { return [] }

        // Focus on input → hidden1: the only connections we can name in
        // plain language ("White pawn on e4 → detector #42").
        var deltas: [(index: Int, delta: Float, old: Float, new: Float)] = []
        deltas.reserveCapacity(new.w1.count)
        for i in 0..<new.w1.count {
            let d = new.w1[i] - old.w1[i]
            if abs(d) > 1e-5 {
                deltas.append((i, d, old.w1[i], new.w1[i]))
            }
        }
        deltas.sort { abs($0.delta) > abs($1.delta) }

        return deltas.prefix(limit).map { entry in
            let dstNeuron = entry.index / 783
            let srcIndex = entry.index % 783
            let desc = "\(inputName(srcIndex)) → detector #\(dstNeuron)"
            return WeightChange(description: desc,
                                oldValue: entry.old, newValue: entry.new)
        }
    }

    /// Plain-language name for an input neuron index.
    private static func inputName(_ i: Int) -> String {
        switch i {
        case 768: return "side to move"
        case 769: return "White kingside castling right"
        case 770: return "White queenside castling right"
        case 771: return "Black kingside castling right"
        case 772: return "Black queenside castling right"
        case 773..<781:
            let file = String(UnicodeScalar(97 + (i - 773))!)
            return "en passant on \(file)-file"
        case 781: return "50-move clock"
        case 782: return "check status"
        default: break
        }
        let side = i < 384 ? "White" : "Black"
        let pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        let piece = pieces[(i % 384) / 64]
        let sq = i % 64
        let file = String(UnicodeScalar(97 + sq % 8)!)
        let rank = sq / 8 + 1
        return "\(side) \(piece) on \(file)\(rank)"
    }

    // MARK: - Prompt

    private static func buildPrompt(changes: [WeightChange],
                                    result: GameResult,
                                    userColor: PieceColor,
                                    moveCount: Int) -> String {
        let resultText: String
        switch result {
        case .whiteWins: resultText = "White won"
        case .blackWins: resultText = "Black won"
        case .draw: resultText = "draw"
        }

        var lines = changes.map { c in
            let dir = c.newValue > c.oldValue ? "strengthened" : "weakened"
            return "- \(c.description): \(dir) from \(fmt(c.oldValue)) to \(fmt(c.newValue))"
        }
        if lines.isEmpty {
            lines = ["(no single connection moved much — learning was spread thin)"]
        }

        return """
        You are explaining a chess AI's brain to a teenager who plays chess.
        This AI learns only from their own games. After one game, these are the
        connections in its neural network that changed the most:

        \(lines.joined(separator: "\n"))

        Game: \(resultText) in \(moveCount) moves.

        In 3-4 short sentences, explain what the network seems to have learned \
        from this game. Plain language, no jargon, no bullet points. Start with \
        "This game, your network ". Do not mention numbers with more than 2 decimals.
        """
    }

    private static func fmt(_ v: Float) -> String {
        String(format: "%.2f", v)
    }

    // MARK: - Fallback

    private static func templateSummary(changes: [WeightChange],
                                        result: GameResult,
                                        moveCount: Int) -> String {
        let resultText: String
        switch result {
        case .whiteWins: resultText = "White won"
        case .blackWins: resultText = "Black won"
        case .draw: resultText = "draw"
        }
        if changes.isEmpty {
            return "This game, your network spread tiny adjustments across many connections — nothing stood out. \(resultText) in \(moveCount) moves."
        }
        let top = changes.prefix(3).map { c in
            let dir = c.newValue > c.oldValue ? "now trusts more" : "now trusts less"
            return "\(c.description) (\(dir))"
        }.joined(separator: "; ")
        return "This game, your network shifted what it pays attention to. Biggest movers: \(top). \(resultText) in \(moveCount) moves."
    }

    // MARK: - File

    private static func formatFile(summary: String, result: GameResult,
                                   moveCount: Int) -> String {
        let date = DateFormatter.localizedString(from: Date(), dateStyle: .medium,
                                                 timeStyle: .short)
        let resultText: String
        switch result {
        case .whiteWins: resultText = "White won"
        case .blackWins: resultText = "Black won"
        case .draw: resultText = "Draw"
        }
        return """
        What your network thinks is new
        ================================
        After game on \(date) — \(resultText), \(moveCount) moves.
        (This file resets every game.)

        \(summary)
        """
    }

    private static var fileURL: URL {
        FileManager.default.urls(for: .documentDirectory,
                                  in: .userDomainMask)[0]
            .appendingPathComponent(fileName)
    }

    private static func writeFile(_ text: String) {
        try? text.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    static func readFile() -> String? {
        try? String(contentsOf: fileURL, encoding: .utf8)
    }
}
