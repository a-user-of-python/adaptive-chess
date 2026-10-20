// GameView.swift
// Adaptive Chess — the game screen.
//
// Board + status bar + New Game / Resign buttons. You play White; the
// on-device AI plays Black and moves automatically after you, on a
// background thread, with a "thinking" indicator. Finished games are
// recorded into the learning profile exactly once.

import SwiftUI

// MARK: - View model

/// Owns the live game and the AI turn loop.
/// @MainActor because it drives SwiftUI state and talks to ProfileStore.
@MainActor
final class GameViewModel: ObservableObject {
    @Published var game = ChessGame()
    @Published var selectedSquare: Square?
    @Published var lastMove: Move?
    @Published var aiThinking = false
    @Published var statusText = "White to move"
    @Published var gameOverText: String? = nil

    /// Guards the profile write so a game is recorded exactly once.
    private var recorded = false

    /// Bumps on every new game so a stale background AI move can never
    /// land on the wrong position.
    private var gameID = UUID()

    /// The network for THIS game. Persistent across moves so per-move
    /// training accumulates; saved back to the profile at game end.
    /// Recreated from the profile at every newGame().
    private var gameNetwork: NeuralNet
    /// Brain wrapping gameNetwork. Its move history resets each game.
    private var gameBrain: ChessBrain

    /// ChessGame is a reference type, so in-place mutations (makeMove)
    /// don't trip @Published. Call this after every mutation.
    private func refresh() {
        objectWillChange.send()
    }

    init() {
        // Fresh network from the profile for this game.
        if let w = ProfileStore.shared.currentWeights,
           let net = NeuralNet(weights: w) {
            gameNetwork = net
        } else {
            gameNetwork = NeuralNet()
        }
        gameBrain = ChessBrain(level: AppSettings.shared.aiLevel,
                               network: gameNetwork)
        updateStatus()
    }

    /// Rebuild the per-game brain from the current profile weights.
    /// Called by newGame() so each game starts from the latest learning.
    private func resetBrainForNewGame() {
        if let w = ProfileStore.shared.currentWeights,
           let net = NeuralNet(weights: w) {
            gameNetwork = net
        } else {
            gameNetwork = NeuralNet()
        }
        gameBrain = ChessBrain(level: AppSettings.shared.aiLevel,
                               network: gameNetwork)
        gameBrain.resetMoveHistory()
    }

    // MARK: - User actions

    /// Called by BoardView with the user's chosen legal move.
    func userDidChooseMove(_ move: Move) {
        guard !aiThinking, !game.isGameOver,
              game.sideToMove == .white else { return }
        game.makeMove(move)
        refresh()
        lastMove = move
        selectedSquare = nil
        // Online learning: one light TD step after every move.
        gameBrain.trainOnMove(game: game)
        updateStatus()
        if checkGameOver() { return }
        triggerAIMove()
    }

    func newGame() {
        // Safety net: if the previous game had meaningful play but was never
        // recorded (e.g. the user hit New Game thinking it was over), record
        // it as a draw rather than silently losing it. Needs ≥4 plies so
        // accidental taps don't pollute the profile.
        if !recorded && !game.isGameOver && game.moveHistory.count >= 4 {
            finishGame(result: .draw, message: "Game abandoned.")
        }
        gameID = UUID()
        game = ChessGame()
        selectedSquare = nil
        lastMove = nil
        aiThinking = false
        recorded = false
        gameOverText = nil
        resetBrainForNewGame()
        updateStatus()
    }

    func resign() {
        guard !game.isGameOver, !recorded else { return }
        finishGame(result: .blackWins, message: "You resigned. AI wins.")
    }

    // MARK: - AI turn

    private func triggerAIMove() {
        guard !game.isGameOver, game.sideToMove == .black,
              !recorded else { return }
        aiThinking = true
        updateStatus()

        // Snapshot what the background thread needs. The FEN string keeps
        // the AI's position fully independent of the live game object.
        let id = gameID
        let fen = game.toFEN()
        let level = AppSettings.shared.aiLevel
        let profile = ProfileStore.shared.profile

        let savedWeights = ProfileStore.shared.currentWeights
        DispatchQueue.global(qos: .userInitiated).async {
            // NOTE: ChessBrain is the AI agent's class name (spec said ChessAI).
            let network: NeuralNet
            if let w = savedWeights, let net = NeuralNet(weights: w) {
                network = net
            } else {
                network = NeuralNet() // fresh random weights for new profile
            }
            let brain = ChessBrain(level: level, network: network)
            guard let position = ChessGame(fen: fen) else { return }
            let move = brain.findBestMove(game: position, profile: profile)

            Task { @MainActor [weak self] in
                guard let self = self,
                      self.gameID == id,
                      !self.recorded else { return }
                self.aiThinking = false
                if let move = move {
                    self.game.makeMove(move)
                    self.refresh()
                    self.lastMove = move
                    self.selectedSquare = nil
                    // Online learning: one light TD step after the AI's move too.
                    // Note: the AI moved using a snapshot network; the per-move
                    // training here applies to the persistent game network.
                    self.gameBrain.trainOnMove(game: self.game)
                }
                self.updateStatus()
                _ = self.checkGameOver()
            }
        }
    }

    // MARK: - Status & game over

    private func updateStatus() {
        if let over = gameOverText {
            statusText = over
            return
        }
        if aiThinking {
            statusText = "Black (AI) is thinking…"
        } else if game.sideToMove == .white {
            statusText = game.isInCheck(.white) ? "White to move — Check!" : "White to move"
        } else {
            statusText = game.isInCheck(.black) ? "Black to move — Check!" : "Black (AI) to move"
        }
    }

    /// Detects a finished game, records it to the profile, and reports
    /// whether the game is over.
    @discardableResult
    private func checkGameOver() -> Bool {
        guard game.isGameOver, !recorded else { return game.isGameOver }
        let result = game.result ?? .draw
        let message: String
        switch result {
        case .whiteWins:
            message = "Checkmate! You win!"
        case .blackWins:
            message = "Checkmate. AI wins."
        case .draw:
            if game.isStalemate {
                message = "Stalemate. Draw."
            } else if game.isDrawByFiftyMove {
                message = "Draw by the fifty-move rule."
            } else if game.isDrawByRepetition {
                message = "Draw by threefold repetition."
            } else if game.isDrawByInsufficientMaterial {
                message = "Draw by insufficient material."
            } else {
                message = "Draw."
            }
        }
        finishGame(result: result, message: message)
        return true
    }

    /// Single choke point for ending a game: sets the UI state and writes
    /// the game to the learning profile exactly once.
    private func finishGame(result: GameResult, message: String) {
        recorded = true
        aiThinking = false
        gameOverText = message
        statusText = message
        let moves = game.moveHistory
        // Train the neural net on this game (background thread).
        // Uses the persistent game network, which already accumulated
        // per-move online learning during the game.
        let gameNet = self.gameNetwork
        let savedWeights = ProfileStore.shared.currentWeights
        let level = AppSettings.shared.aiLevel
        DispatchQueue.global(qos: .utility).async {
            let brain = ChessBrain(level: level, network: gameNet)
            brain.trainOnGame(moves: moves, result: result, userColor: .white)
            let newWeights = gameNet.exportWeights()
            Task { @MainActor in
                ProfileStore.shared.updateWeights(newWeights)
            }
            // Narrate what changed, using the on-device AI (iOS 26+).
            // The text file resets every game.
            if #available(iOS 26, *) {
                NetworkNarrator.shared.narrate(oldWeights: savedWeights,
                                              newWeights: newWeights,
                                              result: result,
                                              userColor: .white,
                                              moveCount: moves.count)
            }
        }
        ProfileStore.shared.recordGame(moves: moves,
                                       result: result,
                                       userColor: .white)
    }
}

// MARK: - View

struct GameView: View {
    @StateObject private var viewModel = GameViewModel()
    @State private var showResignConfirm = false
    @State private var showGameOverAlert = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 14) {
                statusCard

                BoardView(
                    game: viewModel.game,
                    selectedSquare: $viewModel.selectedSquare,
                    lastMove: $viewModel.lastMove,
                    interactionEnabled: !viewModel.aiThinking && !viewModel.game.isGameOver,
                    userColor: .white,
                    onMove: { viewModel.userDidChooseMove($0) }
                )
                .padding(.horizontal)

                if viewModel.aiThinking {
                    HStack(spacing: 12) {
                        ProgressView()
                            .scaleEffect(1.4)
                        Text("AI is thinking…")
                            .font(.title3)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                    .accessibilityLabel("AI is thinking")
                }

                Spacer(minLength: 0)

                buttonRow
            }
            .padding(.vertical)
            .navigationTitle("Adaptive Chess")
            .alert("Resign the game?", isPresented: $showResignConfirm) {
                Button("Resign", role: .destructive) {
                    viewModel.resign()
                }
                Button("Keep Playing", role: .cancel) {}
            } message: {
                Text("This counts as a loss and will be recorded in your profile.")
            }
            .onChange(of: viewModel.gameOverText) { _ in
                showGameOverAlert = viewModel.gameOverText != nil
            }
            .alert("Game Over", isPresented: $showGameOverAlert) {
                Button("New Game") { viewModel.newGame() }
                Button("Review Board", role: .cancel) {}
            } message: {
                Text(viewModel.gameOverText ?? "")
            }
        }
    }

    private var statusCard: some View {
        Text(viewModel.statusText)
            .font(.title2)
            .bold()
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(16)
            .padding(.horizontal)
            .accessibilityLabel("Game status: \(viewModel.statusText)")
    }

    private var buttonRow: some View {
        HStack(spacing: 16) {
            Button("New Game") {
                viewModel.newGame()
            }
            .font(.title2)
            .bold()
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.accentColor)
            .foregroundColor(.white)
            .cornerRadius(16)

            Button("Resign") {
                showResignConfirm = true
            }
            .font(.title2)
            .bold()
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color(.secondarySystemBackground))
            .foregroundColor(.red)
            .cornerRadius(16)
            .disabled(viewModel.game.isGameOver)
        }
        .padding(.horizontal)
    }
}
