import Foundation
import Combine

/// Lightweight descriptor for the profile picker UI.
///
/// The full stats live in each profile's own JSON file; this is just the
/// list. `gamesPlayed` / `lastPlayed` are denormalized here so the picker
/// doesn't have to load every JSON — they are updated by `ProfileStore`
/// on every `recordGame`, the only path that changes them.
struct ProfileInfo: Codable, Equatable, Identifiable {
    var id: String          // UUID string
    var name: String        // user-chosen display name, e.g. "Sean"
    var createdAt: Date
    var gamesPlayed: Int
    var wins: Int = 0
    var losses: Int = 0
    var draws: Int = 0
    var lastPlayed: Date?
}

/// Owns the player's learning profiles: JSON persistence per profile,
/// the context window, profile switching, and per-profile weekly-delete
/// cooldowns.
///
/// Multi-profile design: each person on the device gets their own profile
/// with a COMPLETELY separate AI — own network weights, own stats, own
/// game history, own JSON file (`profile_<id>.json`). Nothing is shared
/// between profiles.
///
/// Thread-safety: the whole store lives on `@MainActor`, which is where all
/// SwiftUI interaction happens. `recordGame` replays at most a few hundred
/// moves — fast enough to stay on the main actor.
///
/// The profile data itself is `private(set)`: stats can only change via
/// `recordGame(_:result:userColor:)` or the delete paths. There are no
/// public setters, so a profile can never be edited — only learned from,
/// renamed, or deleted (once a week per profile).
@MainActor
final class ProfileStore: ObservableObject {

    static let shared = ProfileStore()

    /// All profiles on this device, for the picker UI.
    @Published private(set) var profiles: [ProfileInfo] = []

    /// ID of the active profile. Published so UI can react to switches
    /// (e.g. reload the AI's network weights).
    @Published private(set) var currentProfileId: String = ""

    /// The active profile's data. Read-only from outside; mutated only by
    /// `recordGame` / `updateWeights` / delete / switch.
    @Published private(set) var profile: PlayerProfile = .empty

    /// The active profile's learned network weights, or `nil` when untrained.
    ///
    /// THE AI HOOK: the AI must load these before playing (a fresh
    /// `NeuralNet` when `nil`) and persist trained weights via
    /// `updateWeights(_:)`. Always reflects the CURRENT profile — after
    /// `switchProfile(id:)` this changes, so the AI must re-read it.
    var currentWeights: NetworkWeights? { profile.networkWeights }

    /// Display name of the active profile.
    var currentProfileName: String {
        profiles.first(where: { $0.id == currentProfileId })?.name ?? "Player 1"
    }

    /// Max games remembered: 10 / 25 / 50 / 100, or 0 = unlimited.
    /// Persisted in UserDefaults; shrinking it drops the oldest games
    /// from the CURRENT profile.
    @Published var contextWindow: Int {
        didSet {
            guard Self.validWindows.contains(contextWindow) else {
                contextWindow = 50
                return
            }
            UserDefaults.standard.set(contextWindow, forKey: Self.contextWindowKey)
            profile.enforceWindow(contextWindow)
            save()
        }
    }

    // MARK: - Storage

    private static let registryKey = "AdaptiveChess.profileRegistry"
    private static let currentProfileKey = "AdaptiveChess.currentProfileId"
    private static let contextWindowKey = "AdaptiveChess.contextWindow"
    private static let validWindows = [0, 10, 25, 50, 100]
    private static let defaultWindow = 50
    private static let deleteCooldown: TimeInterval = 7 * 24 * 60 * 60 // 7 days

    /// Legacy single-profile file, migrated on first launch after this update.
    private static let legacyFileName = "player_profile.json"
    private static let legacyDeleteKey = "AdaptiveChess.lastProfileDelete"

    private init() {
        let saved = UserDefaults.standard.object(forKey: Self.contextWindowKey) as? Int
            ?? Self.defaultWindow
        self.contextWindow = Self.validWindows.contains(saved) ? saved : Self.defaultWindow

        // Load the registry; migrate the old single profile on first run.
        var registry = Self.loadRegistry()
        if registry.isEmpty {
            registry = self.migrateLegacyOrFresh()
        }
        self.profiles = registry

        // Resolve the active profile.
        let savedId = UserDefaults.standard.string(forKey: Self.currentProfileKey)
        let resolved: String
        if let savedId, registry.contains(where: { $0.id == savedId }) {
            resolved = savedId
        } else {
            resolved = registry[0].id
            UserDefaults.standard.set(resolved, forKey: Self.currentProfileKey)
        }
        self.currentProfileId = resolved

        var loaded = Self.loadProfile(id: resolved)
        loaded.enforceWindow(self.contextWindow)
        self.profile = loaded
    }

    // MARK: - Profile management

    /// Switches to another profile. Saves the outgoing profile first.
    /// Does nothing for unknown IDs or the already-active profile.
    ///
    /// UI HOOK: after this returns, `profile`, `currentWeights`, and
    /// `currentProfileName` all reflect the new profile. The game screen
    /// should reload the AI's network from `currentWeights` and abandon
    /// any in-progress game (it belonged to the old profile's AI).
    func switchProfile(id: String) {
        guard id != currentProfileId,
              profiles.contains(where: { $0.id == id }) else { return }
        save() // persist outgoing profile
        activateProfile(id: id)
    }

    /// Creates a new empty profile (fresh random network weights on next
    /// load — `nil` until the AI trains), switches to it, and returns its ID.
    /// Empty/blank names become "Player N" (unique).
    func createProfile(name: String) -> String {
        let id = newProfileEntry(name: name)
        switchProfile(id: id) // saves outgoing, activates the new one
        return id
    }

    /// Renames a profile. Only the display name changes — the learning,
    /// weights, and history are untouched. Allowed anytime. Blank names
    /// and unknown IDs are ignored.
    func renameProfile(id: String, name: String) {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty,
              let i = profiles.firstIndex(where: { $0.id == id }) else { return }
        profiles[i].name = uniqueName(clean, excluding: id)
        persistRegistry()
    }

    // MARK: - Recording (the ONLY way stats change)

    /// Records a finished game into the CURRENT profile. Drops the oldest
    /// game when the record list exceeds `contextWindow` (0 = unlimited).
    func recordGame(moves: [Move], result: GameResult, userColor: Color) {
        let record = Self.analyze(moves: moves, result: result, userColor: userColor)
        profile.append(record, window: contextWindow)
        if let i = profiles.firstIndex(where: { $0.id == currentProfileId }) {
            profiles[i].gamesPlayed = profile.gamesPlayed
            profiles[i].wins = profile.wins
            profiles[i].losses = profile.losses
            profiles[i].draws = profile.draws
            profiles[i].lastPlayed = Date()
            persistRegistry()
        }
        save()
    }

    // MARK: - Neural network weights

    /// Replaces the CURRENT profile's learned network weights and persists
    /// them to that profile's JSON. Called by the AI after training on a
    /// finished game. Internal — the profile is never user-editable, and
    /// UI code cannot call this.
    func updateWeights(_ weights: NetworkWeights) {
        profile.setWeights(weights)
        save()
    }

    // MARK: - Weekly delete (per profile)

    /// True when deletion is allowed for the given profile: never deleted
    /// before, or 7+ days since its last delete.
    func canDelete(id: String) -> Bool {
        guard let last = UserDefaults.standard.object(forKey: Self.deleteKey(for: id)) as? Date else {
            return true
        }
        return Date().timeIntervalSince(last) >= Self.deleteCooldown
    }

    /// True when the CURRENT profile may be deleted. (ProfileView hook.)
    func canDelete() -> Bool { canDelete(id: currentProfileId) }

    /// When deletion will next be allowed for the given profile.
    /// `nil` means it is allowed now. Useful for a countdown in the UI.
    func nextDeleteDate(id: String) -> Date? {
        guard let last = UserDefaults.standard.object(forKey: Self.deleteKey(for: id)) as? Date else {
            return nil
        }
        let next = last.addingTimeInterval(Self.deleteCooldown)
        return next > Date() ? next : nil
    }

    /// When the CURRENT profile may next be deleted. (ProfileView hook.)
    var nextDeleteDate: Date? { nextDeleteDate(id: currentProfileId) }

    /// Deletes the CURRENT profile entirely — its JSON file, its registry
    /// entry, its learned weights. Does nothing when the 7-day cooldown
    /// has not elapsed for this profile. (ProfileView hook.)
    ///
    /// After deletion the store switches to another profile, or creates a
    /// fresh "Player 1" when none are left — there is always a current
    /// profile.
    func deleteProfile() {
        deleteProfile(id: currentProfileId)
    }

    /// Deletes the given profile entirely. Per-profile 7-day cooldown.
    /// If it was the current profile, switches to another profile (or a
    /// fresh one when none are left).
    func deleteProfile(id: String) {
        guard canDelete(id: id),
              let index = profiles.firstIndex(where: { $0.id == id }) else { return }
        let wasCurrent = (id == currentProfileId)

        // Remove the JSON file and the registry entry.
        try? FileManager.default.removeItem(at: Self.profileURL(for: id))
        profiles.remove(at: index)
        persistRegistry()

        // Stamp this profile's cooldown.
        UserDefaults.standard.set(Date(), forKey: Self.deleteKey(for: id))

        if wasCurrent {
            if let next = profiles.first {
                // Switch WITHOUT saving — the outgoing profile is gone.
                activateProfile(id: next.id)
            } else {
                // None left: create a fresh profile and activate it directly.
                let freshId = newProfileEntry(name: "Player 1")
                activateProfile(id: freshId)
            }
        }
    }

    // MARK: - Persistence

    /// Writes the CURRENT profile's JSON to the app's Documents directory.
    /// Called automatically after every mutation; also exposed for manual use.
    func save() {
        Self.saveProfile(profile, id: currentProfileId)
    }

    // MARK: - Private

    /// Switches the in-memory state to the given profile WITHOUT saving the
    /// outgoing profile. Use when the outgoing profile was just deleted.
    private func activateProfile(id: String) {
        currentProfileId = id
        UserDefaults.standard.set(id, forKey: Self.currentProfileKey)
        var loaded = Self.loadProfile(id: id)
        loaded.enforceWindow(contextWindow)
        profile = loaded
    }

    /// Appends a new empty profile entry to the registry, writes its empty
    /// JSON, and returns its ID. Does NOT switch to it.
    private func newProfileEntry(name: String) -> String {
        let id = UUID().uuidString
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalName = clean.isEmpty ? uniqueName("Player \(profiles.count + 1)") : uniqueName(clean)
        profiles.append(ProfileInfo(id: id, name: finalName, createdAt: Date(),
                                    gamesPlayed: 0, lastPlayed: nil))
        persistRegistry()
        Self.saveProfile(.empty, id: id)
        return id
    }

    /// Returns `base` unless taken, else "base 2", "base 3", … (case-insensitive).
    private func uniqueName(_ base: String, excluding: String? = nil) -> String {
        let taken = Set(profiles.filter { $0.id != excluding }.map { $0.name.lowercased() })
        var candidate = base
        var n = 2
        while taken.contains(candidate.lowercased()) {
            candidate = "\(base) \(n)"
            n += 1
        }
        return candidate
    }

    private func persistRegistry() {
        Self.persistRegistry(profiles)
    }

    // MARK: Static storage helpers

    private static func documentsDir() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Each profile gets its own JSON file — fully isolated weights, stats,
    /// and history. Nothing is shared between profiles.
    private static func profileURL(for id: String) -> URL {
        documentsDir().appendingPathComponent("profile_\(id).json")
    }

    private static func deleteKey(for id: String) -> String {
        "AdaptiveChess.lastDelete.\(id)"
    }

    private static func saveProfile(_ p: PlayerProfile, id: String) {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(p)
            try data.write(to: profileURL(for: id), options: .atomic)
        } catch {
            // Profile persistence must never crash the app.
            print("[ProfileStore] save failed: \(error)")
        }
    }

    private static func loadProfile(id: String) -> PlayerProfile {
        let url = profileURL(for: id)
        guard let data = try? Data(contentsOf: url) else { return .empty }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let loaded = try decoder.decode(PlayerProfile.self, from: data)
            // Guard against weights saved by a different network shape.
            if let w = loaded.networkWeights, !w.isValid {
                var fixed = loaded
                fixed.setWeights(nil)
                return fixed
            }
            return loaded
        } catch {
            print("[ProfileStore] load failed, starting fresh: \(error)")
            return .empty
        }
    }

    private static func loadRegistry() -> [ProfileInfo] {
        guard let data = UserDefaults.standard.data(forKey: registryKey),
              let list = try? JSONDecoder().decode([ProfileInfo].self, from: data),
              !list.isEmpty else { return [] }
        return list
    }

    private static func persistRegistry(_ list: [ProfileInfo]) {
        if let data = try? JSONEncoder().encode(list) {
            UserDefaults.standard.set(data, forKey: registryKey)
        }
    }

    /// First launch after the multi-profile update: adopt the legacy
    /// single-profile file as "Player 1" (keeping its delete cooldown),
    /// or create a fresh "Player 1" when there is nothing to migrate.
    /// Returns the initial registry.
    private func migrateLegacyOrFresh() -> [ProfileInfo] {
        let id = UUID().uuidString
        let legacyURL = Self.documentsDir().appendingPathComponent(Self.legacyFileName)
        if FileManager.default.fileExists(atPath: legacyURL.path) {
            try? FileManager.default.moveItem(at: legacyURL, to: Self.profileURL(for: id))
            if let last = UserDefaults.standard.object(forKey: Self.legacyDeleteKey) as? Date {
                UserDefaults.standard.set(last, forKey: Self.deleteKey(for: id))
                UserDefaults.standard.removeObject(forKey: Self.legacyDeleteKey)
            }
            let p = Self.loadProfile(id: id)
            let info = ProfileInfo(id: id, name: "Player 1", createdAt: Date(),
                                   gamesPlayed: p.gamesPlayed, wins: p.wins,
                                   losses: p.losses, draws: p.draws,
                                   lastPlayed: p.lastUpdated)
            Self.persistRegistry([info])
            return [info]
        } else {
            let info = ProfileInfo(id: id, name: "Player 1", createdAt: Date(),
                                   gamesPlayed: 0, lastPlayed: nil)
            Self.saveProfile(.empty, id: id)
            Self.persistRegistry([info])
            return [info]
        }
    }

    /// Replays a finished game and builds the per-game record for the user.
    /// Only the user's own moves contribute to piece counts, captures, checks,
    /// and castling; the opening key uses the game's first 4 half-moves.
    private static func analyze(moves: [Move], result: GameResult, userColor: Color) -> GameRecord {
        let game = ChessGame()
        var pieceCounts: [String: Int] = [:]
        var captures = 0
        var checks = 0
        var kingside = false
        var queenside = false

        for (index, move) in moves.enumerated() {
            let mover: Color = (index % 2 == 0) ? .white : .black
            let opponent: Color = (mover == .white) ? .black : .white
            let isUserMove = (mover == userColor)

            if isUserMove {
                if let piece = game.piece(at: move.from) {
                    pieceCounts[piece.type.rawValue, default: 0] += 1
                }
                if move.isCapture || move.isEnPassant { captures += 1 }
                if move.isCastleKingside { kingside = true }
                if move.isCastleQueenside { queenside = true }
            }

            game.makeMove(move)

            if isUserMove && game.isInCheck(opponent) {
                checks += 1
            }
        }

        let outcome: PlayerGameOutcome
        switch result {
        case .whiteWins: outcome = (userColor == .white) ? .win : .loss
        case .blackWins: outcome = (userColor == .black) ? .win : .loss
        case .draw:      outcome = .draw
        }

        let openingKey = moves.prefix(4).map { uci($0) }.joined(separator: " ")

        return GameRecord(
            date: Date(),
            userColor: userColor,
            outcome: outcome,
            plies: moves.count,
            pieceMoveCounts: pieceCounts,
            captures: captures,
            checksGiven: checks,
            castledKingside: kingside,
            castledQueenside: queenside,
            openingKey: openingKey
        )
    }

    private static func uci(_ move: Move) -> String {
        var text = uciSquare(move.from) + uciSquare(move.to)
        if let promotion = move.promotion {
            switch promotion {
            case .knight: text += "n"
            case .bishop: text += "b"
            case .rook:   text += "r"
            case .queen:  text += "q"
            case .king:   text += "k" // unreachable; keeps the switch exhaustive
            case .pawn:   text += "p" // unreachable; keeps the switch exhaustive
            }
        }
        return text
    }

    private static func uciSquare(_ square: Square) -> String {
        let file = Character(UnicodeScalar(97 + square.file)!)
        return "\(file)\(square.rank + 1)"
    }
}
