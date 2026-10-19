import Foundation

/// Converts a finished `ChessGame` into a profile update.
///
/// This is the only entry point the UI uses to feed games into the profile:
/// when a game ends, the UI calls
/// `GameRecorder.recordFinishedGame(game, userColor: .white)` and the game is
/// recorded. Nothing else in the app can mutate the profile.
@MainActor
enum GameRecorder {

    /// Records a finished game into `store` (default: the shared store).
    /// Does nothing unless the game is over and has a result — unfinished or
    /// abandoned games never touch the profile.
    static func recordFinishedGame(
        _ game: ChessGame,
        userColor: PieceColor,
        into store: ProfileStore = .shared
    ) {
        guard game.isGameOver, let result = game.result else { return }
        store.recordGame(moves: game.moveHistory, result: result, userColor: userColor)
    }
}
