// AdaptiveChessApp.swift
// Adaptive Chess — app entry point.
//
// Three tabs: Game, Profile, Settings.
// The shared ProfileStore is injected as an environment object so every
// tab observes the same learning profile.

import SwiftUI

@main
struct AdaptiveChessApp: App {
    @StateObject private var store = ProfileStore.shared

    var body: some Scene {
        WindowGroup {
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
