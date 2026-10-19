import Foundation

/// On-device neural-network chess player.
///
/// The intelligence lives entirely in ``NeuralNet``: a small MLP trained on
/// the user's own games. There is no minimax and no search tree — the network
/// *is* the player.
///
/// - Game 1: the weights are random, so it plays randomly. Easy to beat.
/// - After ~20 games: it has absorbed the user's openings and patterns.
/// - After ~50+ games: it is genuinely personalized and hard to beat —
///   it avoids the positions the user wins from and steers toward the ones
///   the user loses from.
///
/// Levels change *how* the network's judgment is used, not the judgment:
/// - L1–L3: pick the best-scoring move with decreasing exploration noise
///   (noise makes it feel human and beatable at low levels).
/// - L4–L5: no/little noise plus a 1-ply blunder check — after each candidate
///   move, look at the opponent's best reply (scored by the same network)
///   and pick the move with the best worst-case. This is a tactical sanity
///   filter, not a search; the intelligence stays in the weights.
///
/// Call ``findBestMove(game:profile:)`` off the main thread.
final class ChessBrain {

    /// 1 (beatable) … 5 (strongest). Clamped on set.
    var level: Int {
        didSet { level = min(max(level, 1), 5) }
    }

    /// The learned weights. This object is what gets saved into the
    /// player's personal file — it encodes everything learned about them.
    var network: NeuralNet

    init(level: Int = 3, network: NeuralNet = NeuralNet()) {
        self.level = min(max(level, 1), 5)
        self.network = network
    }

    // MARK: - Move selection

    /// Pick the AI's move for the side to move.
    /// - Returns: the chosen move, or nil when there are no legal moves.
    func findBestMove(game: ChessGame, profile: PlayerProfile) -> Move? {
        let moves = game.legalMoves()
        guard !moves.isEmpty else { return nil }
        if moves.count == 1 { return moves[0] }

        let aiIsWhite = (game.sideToMove == .white)
        let noise = noiseScale * experienceFactor(profile: profile)
        let blunderCheck = level >= 4

        var bestMove = moves[0]
        var bestValue: Float = aiIsWhite ? -.infinity : .infinity

        for m in moves {
            game.makeMove(m)
            let value: Float
            if game.isGameOver {
                value = terminalValue(game)
            } else if blunderCheck {
                value = worstReplyValue(game: game)
            } else {
                value = network.forward(PositionEncoder.encode(game))
            }
            game.undoMove()

            var noisy = value
            if noise > 0 {
                // Sum of two uniforms ≈ triangular noise in [−noise, +noise].
                noisy += (Float.random(in: -1...1) + Float.random(in: -1...1)) * 0.5 * noise
            }

            if aiIsWhite ? (noisy > bestValue) : (noisy < bestValue) {
                bestValue = noisy
                bestMove = m
            }
        }
        return bestMove
    }

    /// 1-ply blunder check: score every opponent reply with the same network
    /// and return the value of the opponent's best reply (worst for us).
    /// This stops the AI hanging pieces at high levels; it is a tactical
    /// filter, not a search — no alpha-beta, no depth.
    private func worstReplyValue(game: ChessGame) -> Float {
        let replies = game.legalMoves()
        guard !replies.isEmpty else { return terminalValue(game) }

        let oppIsWhite = (game.sideToMove == .white)
        var worst: Float = oppIsWhite ? -.infinity : .infinity

        for r in replies {
            game.makeMove(r)
            let v: Float = game.isGameOver
                ? terminalValue(game)
                : network.forward(PositionEncoder.encode(game))
            game.undoMove()
            if oppIsWhite ? (v > worst) : (v < worst) {
                worst = v
            }
        }
        return worst
    }

    /// Game-over value from White's perspective: ±1 for checkmate
    /// (the side to move is the one mated), 0 for any draw.
    private func terminalValue(_ game: ChessGame) -> Float {
        if game.isCheckmate {
            return game.sideToMove == .white ? -1 : 1
        }
        return 0
    }

    // MARK: - Levels

    /// Exploration noise per level. High at L1 (beatable, human-feeling),
    /// zero at L5 (plays the network's true judgment).
    private var noiseScale: Float {
        switch level {
        case 1: return 0.45
        case 2: return 0.25
        case 3: return 0.10
        case 4: return 0.03
        default: return 0.0
        }
    }

    /// Slightly reduce exploration as the network gains experience, so a
    /// well-trained brain at L1 still feels casual but less clownish.
    private func experienceFactor(profile: PlayerProfile) -> Float {
        let g = Float(profile.gamesPlayed)
        return max(0.6, 1.0 - 0.008 * g) // floors at 0.6 after ~50 games
    }

    // MARK: - Learning

    /// Train the network on a finished game.
    ///
    /// Every position in the game — before each move by *either* player —
    /// becomes a training sample, so the network absorbs the user's patterns
    /// directly: lines the user wins with get reinforced as good-for-White,
    /// lines the user loses get reinforced as good-for-Black. The AI (Black)
    /// then avoids giving the user positions it has learned are strong for
    /// them, and steers toward ones it has learned they mishandle.
    ///
    /// Targets use a TD(λ)-style blend: mostly the network's own bootstrapped
    /// estimate of the next position, anchored to the actual game result.
    /// Learning rate is small on purpose — adaptation happens gradually over
    /// many games ("over time you start losing more"), not from one game.
    func trainOnGame(moves: [Move], result: GameResult, userColor: Color) {
        guard !moves.isEmpty else { return }

        // Rebuild every position from the opening position.
        let game = ChessGame()
        var positions: [[Float]] = []
        positions.reserveCapacity(moves.count)
        for m in moves {
            positions.append(PositionEncoder.encode(game))
            game.makeMove(m)
        }

        let finalResult: Float
        switch result {
        case .whiteWins: finalResult = 1
        case .blackWins: finalResult = -1
        case .draw: finalResult = 0
        }

        let lambda: Float = 0.7
        let lr: Float = 0.01
        let epochs = 2

        for _ in 0..<epochs {
            var bootstrap = finalResult
            for x in positions.reversed() {
                let target = lambda * bootstrap + (1 - lambda) * finalResult
                network.train(x: x, target: target, lr: lr)
                bootstrap = network.forward(x)
            }
        }
    }

    // MARK: - Persistence

    /// Save the learned weights (the "personal file") to disk.
    /// ProfileStore should call this after every recorded game.
    func save(to url: URL) throws {
        let data = try JSONEncoder().encode(network)
        try data.write(to: url, options: .atomic)
    }

    /// Load previously learned weights.
    func load(from url: URL) throws {
        network = try JSONDecoder().decode(NeuralNet.self, from: Data(contentsOf: url))
    }

    /// Forget everything learned — fresh random weights.
    /// ProfileStore should call this when the user deletes their profile
    /// (allowed once per 7 days per the spec).
    func resetKnowledge() {
        network = NeuralNet()
    }
}

// MARK: - Position encoding

/// 769-feature board encoding: 64 squares × 12 one-hot features
/// (6 piece types × 2 colors), plus side-to-move at index 768
/// (+1 White to move / −1 Black to move).
enum PositionEncoder {
    static func encode(_ game: ChessGame) -> [Float] {
        var x = [Float](repeating: 0, count: NeuralNet.inputSize)
        for rank in 0..<8 {
            for file in 0..<8 {
                guard let p = game.piece(at: Square(file: file, rank: rank)) else { continue }
                let base = (rank * 8 + file) * 12 + (p.color == .white ? 0 : 6)
                switch p.type {
                case .pawn:   x[base] = 1
                case .knight: x[base + 1] = 1
                case .bishop: x[base + 2] = 1
                case .rook:   x[base + 3] = 1
                case .queen:  x[base + 4] = 1
                case .king:   x[base + 5] = 1
                }
            }
        }
        x[NeuralNet.inputSize - 1] = (game.sideToMove == .white) ? 1 : -1
        return x
    }
}
