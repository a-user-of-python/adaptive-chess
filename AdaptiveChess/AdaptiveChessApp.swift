// AdaptiveChessApp.swift
// Adaptive Chess — app entry point.
//
// Three tabs: Game, Profile, Settings.
// The shared ProfileStore is injected as an environment object so every
// tab observes the same learning profile.
// Stockfish starts on launch; if its NNUE models aren't cached, a
// first-launch download screen appears instead of the tabs.

import SwiftUI

@main
struct AdaptiveChessApp: App {
    @StateObject private var store = ProfileStore.shared
    @ObservedObject private var engine = StockfishEngine.shared
    @State private var downloadComplete = false

    init() {
        // Start Stockfish on launch. If NNUE models aren't cached,
        // the download screen appears (see body).
        StockfishEngine.shared.start()
    }

    var body: some Scene {
        WindowGroup {
            if engine.needsDownload && !downloadComplete {
                ModelDownloadView {
                    downloadComplete = true
                }
            } else {
                TabView {
                    GameView()
                        .tabItem {
                            Label("Game", systemImage: "gamecontroller")
                        }

                    ProfileView()
                        .tabItem {
                            Label("Profile", systemImage: "person.crop.circle")
                        }

                    SettingsView()
                        .tabItem {
                            Label("Settings", systemImage: "gear")
                        }
                }
                .environmentObject(store)
            }
        }
    }
}
