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
                // Search inside the net: one forward pass predicts now,
                // +1 move, +2 moves. Weight toward the near future.
                let horizons = network.forwardMulti(PositionEncoder.encode(game))
                value = 0.5 * horizons[0] + 0.3 * horizons[1] + 0.2 * horizons[2]
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
                : {
                    let h = network.forwardMulti(PositionEncoder.encode(game))
                    return 0.5 * h[0] + 0.3 * h[1] + 0.2 * h[2]
                }()
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

    /// Recent encoded positions, for online per-move training.
    /// Kept short — only the last few plies matter for the lookahead check.
    private var recentPositions: [[Float]] = []
    private let maxHistory = 6

    /// Call after every move (by either side) during a live game.
    /// Does one light TD step: checks whether the network's past "future"
    /// predictions match what actually arrived, and corrects them.
    ///
    /// This is the tight feedback loop — the network learns DURING the game,
    /// not just at the end. The learning rate is deliberately small so a
    /// single game can't overwrite what older games taught.
    ///
    /// - Parameter game: the position AFTER the move was made.
    func trainOnMove(game: ChessGame) {
        let x = PositionEncoder.encode(game)
        recentPositions.append(x)
        if recentPositions.count > maxHistory {
            recentPositions.removeFirst()
        }

        let lr: Float = 0.005
        let n = recentPositions.count
        let nowPred = network.forwardMulti(x)

        // 2 plies ago, out[1] predicted "value now". Correct it toward
        // what the network actually sees now.
        if n >= 3 {
            let pastX = recentPositions[n - 3]
            let pastPred = network.forwardMulti(pastX)
            network.train(x: pastX,
                          targets: [pastPred[0], nowPred[0], pastPred[2]],
                          lr: lr)
        }
        // 4 plies ago, out[2] predicted "value now". Same correction.
        if n >= 5 {
            let pastX = recentPositions[n - 5]
            let pastPred = network.forwardMulti(pastX)
            network.train(x: pastX,
                          targets: [pastPred[0], pastPred[1], nowPred[0]],
                          lr: lr)
        }
    }

    /// Call when a new game starts.
    func resetMoveHistory() {
        recentPositions.removeAll()
    }

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
    func trainOnGame(moves: [Move], result: GameResult, userColor: PieceColor) {
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
        let lr: Float = 0.015
        let epochs = 3

        // Three horizons, all from White's perspective:
        //   out[0] = value now, out[1] = value 2 plies out, out[2] = value 4 plies out.
        // The network learns to "see the future" internally — no move tree,
        // no external search. Each position's future targets come from the
        // actual positions reached later in THIS game.
        for _ in 0..<epochs {
            // Bootstrap from the end, per horizon.
            var boot0 = finalResult
            var boot1 = finalResult
            var boot2 = finalResult
            for idx in positions.indices.reversed() {
                let x = positions[idx]
                // Future positions for the lookahead targets (clamped to game end).
                let f1 = positions[min(idx + 2, positions.count - 1)]
                let f2 = positions[min(idx + 4, positions.count - 1)]

                let t0 = lambda * boot0 + (1 - lambda) * finalResult
                // The +1-move target: what the network itself predicted for
                // the position 2 plies out, anchored to the final result.
                let t1 = lambda * boot1 + (1 - lambda) * finalResult
                let t2 = lambda * boot2 + (1 - lambda) * finalResult

                network.train(x: x, targets: [t0, t1, t2], lr: lr)

                let preds = network.forwardMulti(x)
                boot0 = preds[0]
                // Bootstrap the future horizons from the network's own
                // predictions at the future positions — this teaches it
                // temporal consistency: "what I predict for 2 moves out
                // should match what I predict when I get there."
                boot1 = network.forwardMulti(f1)[0]
                boot2 = network.forwardMulti(f2)[0]
            }
        }

        // ---- Blunder-focused extra training ----
        // If the AI keeps falling for the same trap, normal training isn't
        // punishing it enough: the trap position gets diluted among all
        // other positions, and 70% of its target is the network's own
        // (wrong) opinion. So we find blunders explicitly and hammer them.
        //
        // A blunder = the AI's eval dropped sharply after its move
        // (opponent's reply was much better than the AI expected).
        // For blunders we use pure outcome (λ=0, no bootstrapping the
        // wrong opinion), higher lr, and more epochs.
        let aiIsWhite = (userColor == .black)
        let aiSign: Float = aiIsWhite ? 1 : -1  // +1 if AI is White
        let blunderLr: Float = 0.05
        let blunderEpochs = 8
        let blunderThreshold: Float = 0.35

        for idx in positions.indices {
            // Only the AI's moves (even plies = White, odd plies = Black).
            let isAiMove = aiIsWhite ? (idx % 2 == 0) : (idx % 2 == 1)
            guard isAiMove else { continue }
            // Need the opponent's reply to exist.
            guard idx + 2 < positions.count else { continue }

            let before = network.forwardMulti(positions[idx])[0]
            let after = network.forwardMulti(positions[idx + 2])[0]
            // Swing from the AI's perspective (positive = good for AI).
            let swing = (after - before) * aiSign

            if swing < -blunderThreshold {
                // Blunder confirmed. Train all 3 horizons toward the true
                // outcome — no bootstrapping, just "this was bad."
                for _ in 0..<blunderEpochs {
                    network.train(x: positions[idx],
                                  targets: [finalResult, finalResult, finalResult],
                                  lr: blunderLr)
                }
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

/// 911-feature board encoding: 64 squares × 12 one-hot features
/// (6 piece types × 2 colors), plus side-to-move at index 768
/// (+1 White to move / −1 Black to move).
enum PositionEncoder {
    /// Full 911-input encoding — every distinct element of chess state:
    /// 768 piece squares + side to move + 4 castling rights +
    /// 8 en-passant file one-hot + halfmove clock + check flag +
    /// 128 attack map (64 squares × attacked-by-white, 64 × attacked-by-black).
    ///
    /// The attack map is the "why": the network doesn't just see WHERE
    /// pieces are, it sees WHAT THEY THREATEN. A knight on f3 is just a
    /// knight — but a knight on f3 attacking h2 next to an undefended king
    /// is a mating threat. The network can now learn that distinction.
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
        x[768] = (game.sideToMove == .white) ? 1 : -1
        // Castling rights
        x[769] = game.whiteCanCastleKingside ? 1 : 0
        x[770] = game.whiteCanCastleQueenside ? 1 : 0
        x[771] = game.blackCanCastleKingside ? 1 : 0
        x[772] = game.blackCanCastleQueenside ? 1 : 0
        // En passant file (one-hot over 8 files; all zero when none)
        if let epFile = game.enPassantFile, (0..<8).contains(epFile) {
            x[773 + epFile] = 1
        }
        // Halfmove clock, normalized (50-move rule triggers at 100)
        x[781] = min(Float(game.halfmovesSincePawnOrCapture) / 100.0, 1.0)
        // Is the side to move in check?
        x[782] = game.isInCheck(game.sideToMove) ? 1 : 0
        // Attack maps: 64 squares attacked-by-white, 64 attacked-by-black.
        // This is what lets the network learn WHY a piece is dangerous.
        for rank in 0..<8 {
            for file in 0..<8 {
                let sq = Square(file: file, rank: rank)
                let idx = rank * 8 + file
                if game.isSquareAttacked(sq, by: .white) { x[783 + idx] = 1 }
                if game.isSquareAttacked(sq, by: .black) { x[847 + idx] = 1 }
            }
        }
        return x
    }
}
