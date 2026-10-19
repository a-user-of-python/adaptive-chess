import Foundation

/// How a finished game turned out from the user's perspective.
enum PlayerGameOutcome: String, Codable {
    case win
    case loss
    case draw
}

/// Win/loss/draw breakdown for a single opening line.
struct OpeningRecord: Codable {
    var games: Int = 0
    var wins: Int = 0
    var losses: Int = 0
    var draws: Int = 0
}

/// Learned neural-network weights — the machine-readable "personal file on
/// how you play." The AI trains these after each game and they persist in the
/// same JSON as the human-readable stats.
///
/// Architecture 769→128→32→1 (MLP):
/// - w1: 769*128 weights, b1: 128 biases  (input → hidden 1)
/// - w2: 128*32 weights,  b2: 32 biases   (hidden 1 → hidden 2)
/// - w3: 32*1 weights,    b3: 1 bias      (hidden 2 → output)
/// `nil` means no trained network yet — the AI starts from random weights.
struct NetworkWeights: Codable {
    var w1: [Float]
    var b1: [Float]
    var w2: [Float]
    var b2: [Float]
    var w3: [Float]
    var b3: [Float]
    var w4: [Float]
    var b4: [Float]
    var w5: [Float]
    var b5: [Float]
    var w6: [Float]
    var b6: [Float]

    /// Expected element counts for the 769→512→256→128→64→32→1 architecture (~569k params).
    static let w1Count = 769 * 512
    static let b1Count = 512
    static let w2Count = 512 * 256
    static let b2Count = 256
    static let w3Count = 256 * 128
    static let b3Count = 128
    static let w4Count = 128 * 64
    static let b4Count = 64
    static let w5Count = 64 * 32
    static let b5Count = 32
    static let w6Count = 32
    static let b6Count = 1

    /// True when every array matches the architecture. Guards against
    /// loading weights saved by a different network shape.
    var isValid: Bool {
        w1.count == Self.w1Count && b1.count == Self.b1Count &&
        w2.count == Self.w2Count && b2.count == Self.b2Count &&
        w3.count == Self.w3Count && b3.count == Self.b3Count &&
        w4.count == Self.w4Count && b4.count == Self.b4Count &&
        w5.count == Self.w5Count && b5.count == Self.b5Count &&
        w6.count == Self.w6Count && b6.count == Self.b6Count
    }
}

/// One finished game, as remembered by the profile.
///
/// The profile keeps a capped list of these records and derives every aggregate
/// from them, so shrinking the context window can never leave stale numbers
/// behind. Records are append-only from the profile's point of view.
struct GameRecord: Codable {
    let date: Date
    let userColor: PieceColor
    let outcome: PlayerGameOutcome
    let plies: Int                        // half-moves in the whole game
    let pieceMoveCounts: [String: Int]    // user's moves only, keyed by PieceType raw value
    let captures: Int                     // captures made by the user
    let checksGiven: Int                  // checks given by the user
    let castledKingside: Bool
    let castledQueenside: Bool
    let openingKey: String                // first 4 half-moves in UCI, space-separated
}

/// The player's learning profile.
///
/// READ-ONLY by design: every stat is `private(set)`. The only way stats change
/// is through the internal `append(_:window:)` / `reset()` helpers, which are
/// called by `ProfileStore` — never by UI code. There are no public setters.
struct PlayerProfile: Codable {
    private(set) var gamesPlayed: Int = 0
    private(set) var wins: Int = 0
    private(set) var losses: Int = 0
    private(set) var draws: Int = 0
    private(set) var totalMoves: Int = 0              // half-moves across remembered games
    private(set) var pieceMoveCounts: [String: Int] = [:]
    private(set) var captures: Int = 0
    private(set) var checksGiven: Int = 0
    private(set) var castledKingside: Int = 0
    private(set) var castledQueenside: Int = 0
    private(set) var openings: [String: Int] = [:]   // openingKey -> games
    private(set) var openingResults: [String: OpeningRecord] = [:]
    private(set) var avgGameLength: Double = 0        // half-moves per game
    private(set) var lastUpdated: Date = Date()
    private(set) var recentGames: [GameRecord] = []   // capped by the context window

    /// The trained network weights, or `nil` when no network has been trained
    /// yet. `nil` = the AI plays from freshly randomized weights.
    /// Wiped (back to `nil`) by `reset()` / weekly delete.
    private(set) var networkWeights: NetworkWeights? = nil

    static var empty: PlayerProfile { PlayerProfile() }

    /// Appends a game record, enforces the context window, and recomputes
    /// every aggregate from the surviving records. `window <= 0` = unlimited.
    mutating func append(_ record: GameRecord, window: Int) {
        recentGames.append(record)
        enforceWindow(window)
    }

    /// Drops the oldest records beyond `window` (`<= 0` = keep all),
    /// then recomputes aggregates.
    mutating func enforceWindow(_ window: Int) {
        if window > 0 && recentGames.count > window {
            recentGames.removeFirst(recentGames.count - window)
        }
        recompute()
    }

    /// Wipes everything back to a fresh profile, including the network weights.
    mutating func reset() {
        self = PlayerProfile()
    }

    /// Internal: replaces the learned network weights. Called by
    /// `ProfileStore.updateWeights(_:)` after the AI trains — never by UI.
    mutating func setWeights(_ weights: NetworkWeights?) {
        networkWeights = weights
    }

    // MARK: - Private

    /// Rebuilds every aggregate from `recentGames`. This is the single source
    /// of truth for stats — nothing is ever incremented in place.
    private mutating func recompute() {
        gamesPlayed = recentGames.count
        wins = 0
        losses = 0
        draws = 0
        totalMoves = 0
        pieceMoveCounts = [:]
        captures = 0
        checksGiven = 0
        castledKingside = 0
        castledQueenside = 0
        openings = [:]
        openingResults = [:]

        for record in recentGames {
            switch record.outcome {
            case .win:  wins += 1
            case .loss: losses += 1
            case .draw: draws += 1
            }

            totalMoves += record.plies
            for (piece, count) in record.pieceMoveCounts {
                pieceMoveCounts[piece, default: 0] += count
            }
            captures += record.captures
            checksGiven += record.checksGiven
            if record.castledKingside { castledKingside += 1 }
            if record.castledQueenside { castledQueenside += 1 }

            openings[record.openingKey, default: 0] += 1
            var opening = openingResults[record.openingKey] ?? OpeningRecord()
            opening.games += 1
            switch record.outcome {
            case .win:  opening.wins += 1
            case .loss: opening.losses += 1
            case .draw: opening.draws += 1
            }
            openingResults[record.openingKey] = opening
        }

        avgGameLength = gamesPlayed > 0 ? Double(totalMoves) / Double(gamesPlayed) : 0
        lastUpdated = Date()
    }
}
