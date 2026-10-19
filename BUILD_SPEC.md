# Adaptive Chess — Build Spec

## Architecture

```
AdaptiveChess/
├── AdaptiveChessApp.swift      — app entry, tab view (Game / Profile / Settings)
├── Engine/
│   ├── ChessTypes.swift        — Color, PieceType, Piece, Square, Move
│   ├── ChessGame.swift         — game state, legal move gen, make/undo, terminal detection
├── AI/
│   ├── Search.swift            — negamax + alpha-beta + iterative deepening + move ordering
│   ├── Evaluation.swift        — material + piece-square tables + profile exploitation
├── Profile/
│   ├── PlayerProfile.swift     — Codable profile: stats, openings, tendencies
│   ├── ProfileStore.swift      — JSON persistence, context window, weekly delete cooldown
│   └── GameRecorder.swift      — records moves/games into profile
└── UI/
    ├── BoardView.swift         — SwiftUI chess board, tap-to-move, legal move dots
    ├── GameView.swift          — game screen, new game, resign, status
    ├── SettingsView.swift      — AI level (1-5), context window (games remembered)
    └── ProfileView.swift       — READ-ONLY profile viewer, delete button (weekly cooldown)
```

## Core Interface (Engine)

```swift
enum Color: String, Codable { case white, black }
enum PieceType: String, Codable { case pawn, knight, bishop, rook, queen, king }
struct Piece: Equatable, Hashable, Codable { let type: PieceType; let color: Color }
struct Square: Equatable, Hashable, Codable { let file: Int; let rank: Int } // 0-7

struct Move: Equatable, Hashable {
    let from: Square
    let to: Square
    let promotion: PieceType?   // nil unless pawn promotion
    // Engine fills these:
    let isCapture: Bool
    let isCastleKingside: Bool
    let isCastleQueenside: Bool
    let isEnPassant: Bool
}

class ChessGame {
    init() // standard starting position
    init(fen: String) // for testing
    var sideToMove: Color
    func legalMoves() -> [Move]       // ALL legal moves, no illegal ones
    func legalMoves(from: Square) -> [Move]
    func makeMove(_ move: Move)
    func undoMove()
    func isInCheck(_ color: Color) -> Bool
    var isCheckmate: Bool
    var isStalemate: Bool
    var isDrawByFiftyMove: Bool
    var isDrawByRepetition: Bool
    var isDrawByInsufficientMaterial: Bool
    var isGameOver: Bool
    var result: GameResult? // whiteWins, blackWins, draw
    func piece(at: Square) -> Piece?
    var moveHistory: [Move] // for recorder
    func toFEN() -> String
}
```

## AI Interface

```swift
class ChessAI {
    var level: Int // 1-5, maps to depth/time
    func findBestMove(game: ChessGame, profile: PlayerProfile) -> Move?
    // Must run off main thread. Level 1: depth 1, Level 5: depth 4 + quiescence.
}
```

## Profile

```swift
struct PlayerProfile: Codable {
    var gamesPlayed: Int
    var wins: Int; var losses: Int; var draws: Int
    var totalMoves: Int
    var pieceMoveCounts: [String: Int]  // "knight": 42
    var captures: Int
    var checksGiven: Int
    var castledKingside: Int; var castledQueenside: Int
    var openings: [String: Int]  // "e4 e5": 5 — first 4 half-moves as key
    var avgGameLength: Double
    var lastUpdated: Date
}

class ProfileStore: ObservableObject {
    static let shared: ProfileStore
    var profile: PlayerProfile
    var contextWindow: Int // max games remembered (10/25/50/100/0=all)
    func recordGame(moves: [Move], result: GameResult, userColor: Color)
    func canDelete() -> Bool // 7-day cooldown
    func deleteProfile() // only if canDelete()
    func save()
}
```

## Rules

- Engine: NO illegal moves ever. Full chess rules: check, checkmate, stalemate, castling (all conditions), en passant, promotion (all 4 pieces), 50-move rule, threefold repetition, insufficient material.
- AI: plays as Black by default (user is White). Must be beatable at level 1, strong at level 5.
- Profile exploitation: AI adjusts evaluation based on profile. Examples:
  - User castles kingside often → AI weights kingside attack higher
  - User is aggressive (high captures) → AI plays more solid, punishes overextension
  - User favors knights → AI trades knights favorably
  - User plays same opening → AI prepares specific counter-lines
- Profile file: JSON in app Documents, viewable in-app but NEVER editable.
- Delete: allowed once per 7 days. Store lastDeleteDate in UserDefaults.
- UI: simple, high contrast, large touch targets (bad vision). Board with coordinates, legal move dots, last-move highlight.
