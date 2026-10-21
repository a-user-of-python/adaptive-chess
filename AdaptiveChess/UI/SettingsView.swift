// SettingsView.swift
// Adaptive Chess — settings screen.
//
// Two controls, both large and simple:
//   1. AI difficulty (1-5), persisted in UserDefaults.
//   2. Context window — how many recent games the profile remembers.
//      This lives on ProfileStore (also persisted); shrinking it drops
//      the oldest games from the AI's memory.

import SwiftUI

// MARK: - App settings

/// AI difficulty level, shared across the app. Persists in UserDefaults.
/// (The context window itself lives on ProfileStore per the build spec.)
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var aiLevel: Int {
        didSet {
            UserDefaults.standard.set(aiLevel, forKey: Self.aiLevelKey)
        }
    }

    /// Which engine plays the AI's moves: "network" (the learning neural net)
    /// or "stockfish" (embedded Stockfish; the network still learns via
    /// distillation). Persists in UserDefaults.
    @Published var aiEngine: String {
        didSet {
            UserDefaults.standard.set(aiEngine, forKey: Self.aiEngineKey)
        }
    }

    var useStockfish: Bool { aiEngine == "stockfish" }

    private static let aiLevelKey = "AdaptiveChess.aiLevel"
    private static let aiEngineKey = "AdaptiveChess.aiEngine"

    private init() {
        let saved = UserDefaults.standard.integer(forKey: Self.aiLevelKey)
        self.aiLevel = (1...5).contains(saved) ? saved : 3
        let savedEngine = UserDefaults.standard.string(forKey: Self.aiEngineKey) ?? "network"
        self.aiEngine = (savedEngine == "stockfish") ? "stockfish" : "network"
    }
}

// MARK: - View

struct SettingsView: View {
    @StateObject private var settings = AppSettings.shared
    @StateObject private var store = ProfileStore.shared

    private static let windowOptions: [(value: Int, label: String)] = [
        (10, "Last 10 games"),
        (25, "Last 25 games"),
        (50, "Last 50 games"),
        (100, "Last 100 games"),
        (0, "All games"),
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    engineSection
                    difficultySection
                    contextWindowSection
                }
                .padding()
            }
            .navigationTitle("Settings")
        }
    }

    // MARK: - AI engine

    private var engineSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AI Engine")
                .font(.title)
                .bold()

            Text("Who plays the computer's moves. Your network keeps learning either way — when Stockfish plays, it teaches your network after every move.")
                .font(.body)
                .foregroundColor(.secondary)

            EngineRowView(
                name: "Your Network",
                description: "The AI that learns your style. Starts weak, gets personal.",
                isSelected: !settings.useStockfish
            ) {
                settings.aiEngine = "network"
            }

            EngineRowView(
                name: "Stockfish",
                description: "The world-champion engine plays. Your network watches and learns from it.",
                isSelected: settings.useStockfish
            ) {
                settings.aiEngine = "stockfish"
            }
        }
    }

    // MARK: - AI difficulty

    private var difficultySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AI Difficulty")
                .font(.title)
                .bold()

            Text("How strong the computer plays. Higher levels think harder and exploit your habits more.")
                .font(.body)
                .foregroundColor(.secondary)

            ForEach(1...5, id: \.self) { level in
                DifficultyRowView(
                    level: level,
                    name: Self.levelName(level),
                    description: Self.levelDescription(level),
                    isSelected: settings.aiLevel == level
                ) {
                    settings.aiLevel = level
                }
            }
        }
    }

    // MARK: - Context window

    private var contextWindowSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Memory")
                .font(.title)
                .bold()

            Text("How many of your recent games the AI remembers when learning your style. Older games are forgotten, so the AI adapts to how you play *now*.")
                .font(.body)
                .foregroundColor(.secondary)

            ForEach(Self.windowOptions, id: \.value) { option in
                WindowOptionRowView(
                    label: option.label,
                    isSelected: store.contextWindow == option.value
                ) {
                    store.contextWindow = option.value
                }
            }
        }
    }

    // MARK: - Level copy

    private static func levelName(_ level: Int) -> String {
        switch level {
        case 1: return "Casual"
        case 2: return "Relaxed"
        case 3: return "Club Player"
        case 4: return "Strong"
        case 5: return "Brutal"
        default: return ""
        }
    }

    private static func levelDescription(_ level: Int) -> String {
        switch level {
        case 1: return "Learning the ropes. Very beatable."
        case 2: return "A friendly game. It makes mistakes."
        case 3: return "A solid challenge. Plays real chess."
        case 4: return "Tough. Punishes sloppy play."
        case 5: return "Relentless. Bring your best — and keep evolving."
        default: return ""
        }
    }
}

// MARK: - Extracted rows (kept separate so the compiler can type-check quickly)

private struct EngineRowView: View {
    let name: String
    let description: String
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(name)
                        .font(.title3)
                        .bold()
                        .foregroundColor(.primary)
                    Text(description)
                        .font(.body)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.accentColor)
                        .accessibilityHidden(true)
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(16)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(name) engine")
    }
}

private struct DifficultyRowView: View {
    let level: Int
    let name: String
    let description: String
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Level \(level) — \(name)")
                        .font(.title3)
                        .bold()
                        .foregroundColor(.primary)
                    Text(description)
                        .font(.body)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.accentColor)
                        .accessibilityHidden(true)
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(16)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("AI level \(level), \(name)")
    }
}

private struct WindowOptionRowView: View {
    let label: String
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 14) {
                Text(label)
                    .font(.title3)
                    .bold()
                    .foregroundColor(.primary)
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.accentColor)
                        .accessibilityHidden(true)
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(16)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Remember \(label)")
    }
}
