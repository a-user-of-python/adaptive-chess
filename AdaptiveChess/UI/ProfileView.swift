// ProfileView.swift
// Adaptive Chess — read-only profile viewer.
//
// Shows everything the AI has learned about you: record, playing style,
// favorite openings, and piece preferences — plus the raw JSON. The
// profile can NEVER be edited here; there are no text fields and no
// setters are reachable. The only mutation allowed is the weekly wipe,
// gated by ProfileStore.canDelete().

import SwiftUI

struct ProfileView: View {
    @StateObject private var store = ProfileStore.shared
    @State private var showDeleteConfirm = false
    @State private var showAddProfile = false
    @State private var newProfileName = ""
    @State private var renameTarget: ProfileInfo? = nil
    @State private var renameText = ""
    @State private var deleteTarget: ProfileInfo? = nil
    @State private var showDeleteProfileConfirm = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    profilesSection
                    if store.profile.gamesPlayed == 0 {
                        emptyState
                    } else {
                        recordSection
                        styleSection
                        openingsSection
                        piecesSection
                    }
                    jsonSection
                    deleteSection
                }
                .padding()
            }
            .navigationTitle("Profiles")
            .alert("Delete this profile?", isPresented: $showDeleteConfirm) {
                Button("Delete Profile", role: .destructive) {
                    store.deleteProfile()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This permanently removes the profile and everything its AI learned. This cannot be undone.")
            }
            .alert("New profile", isPresented: $showAddProfile) {
                TextField("Name", text: $newProfileName)
                Button("Create") {
                    store.createProfile(name: newProfileName)
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Starts with a fresh AI that knows nothing about you — yet.")
            }
            .alert("Rename profile", isPresented: Binding(
                get: { renameTarget != nil },
                set: { if !$0 { renameTarget = nil } }
            )) {
                TextField("Name", text: $renameText)
                Button("Rename") {
                    if let target = renameTarget {
                        store.renameProfile(id: target.id, name: renameText)
                    }
                    renameTarget = nil
                }
                Button("Cancel", role: .cancel) { renameTarget = nil }
            }
            .alert("Delete profile?", isPresented: $showDeleteProfileConfirm, presenting: deleteTarget) { target in
                if store.canDelete(id: target.id) {
                    Button("Delete", role: .destructive) {
                        store.deleteProfile(id: target.id)
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: { target in
                if store.canDelete(id: target.id) {
                    Text("\"\(target.name)\" and everything its AI learned will be permanently erased. This cannot be undone.")
                } else if let next = store.nextDeleteDate(id: target.id) {
                    let days = max(1, Calendar.current.dateComponents([.day], from: Date(), to: next).day ?? 1)
                    Text("Delete is on cooldown for this profile — available in \(days) day\(days == 1 ? "" : "s").")
                } else {
                    Text("Delete is on cooldown for this profile.")
                }
            }
        }
    }

    // MARK: - Profile switcher

    /// Who's playing? Each profile gets its own completely separate AI —
    /// own learning, own weights, own history. Tap to switch, swipe a row
    /// for rename/delete.
    private var profilesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Who's Playing")

            VStack(spacing: 0) {
                ForEach(store.profiles) { info in
                    VStack(spacing: 0) {
                        Button {
                            store.switchProfile(id: info.id)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(info.name)
                                        .font(.title3)
                                        .bold()
                                        .foregroundColor(.primary)
                                    Text(profileRowSubtitle(info))
                                        .font(.body)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                if info.id == store.currentProfileId {
                                    Image(systemName: "checkmark.circle.fill")
                                        .font(.title2)
                                        .foregroundColor(.accentColor)
                                        .accessibilityLabel("Current profile")
                                }
                            }
                            .padding(.vertical, 10)
                        }
                        .swipeActions(edge: .trailing) {
                            Button("Rename") {
                                renameTarget = info
                                renameText = info.name
                            }
                            .tint(.blue)
                            Button("Delete", role: .destructive) {
                                deleteTarget = info
                                showDeleteProfileConfirm = true
                            }
                            .tint(.red)
                        }
                        if info.id != store.profiles.last?.id { Divider() }
                    }
                }
            }
            .padding(.horizontal)
            .background(Color(.secondarySystemBackground))
            .cornerRadius(16)

            Button {
                newProfileName = ""
                showAddProfile = true
            } label: {
                Label("Add Person", systemImage: "plus.circle.fill")
                    .font(.title3)
                    .bold()
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(16)
            }

            Text("Each person gets their own AI. It only learns from your games — never anyone else's.")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    private func profileRowSubtitle(_ info: ProfileInfo) -> String {
        guard info.gamesPlayed > 0 else { return "No games yet" }
        return "\(info.gamesPlayed) games · \(info.wins)–\(info.losses)–\(info.draws)"
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 12) {
            Text("No games yet")
                .font(.title)
                .bold()
            Text("Play a game and the AI will start learning how you play — then use it against you.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
    }

    // MARK: - Record

    private var recordSection: some View {
        let p = store.profile
        return VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Record")
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                statCard(value: "\(p.gamesPlayed)", label: "Games")
                statCard(value: "\(p.wins)–\(p.losses)–\(p.draws)", label: "W / L / D")
                statCard(value: winRateText, label: "Win Rate")
                statCard(value: String(format: "%.1f", p.avgGameLength / 2), label: "Avg Moves/Game")
            }
        }
    }

    private var winRateText: String {
        let p = store.profile
        guard p.gamesPlayed > 0 else { return "—" }
        return String(format: "%.0f%%", 100.0 * Double(p.wins) / Double(p.gamesPlayed))
    }

    private func statCard(value: String, label: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title)
                .bold()
                .minimumScaleFactor(0.6)
                .lineLimit(1)
            Text(label)
                .font(.body)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(16)
    }

    // MARK: - Playing style

    private var styleSection: some View {
        let p = store.profile
        let games = max(p.gamesPlayed, 1)
        let capturesPerGame = Double(p.captures) / Double(games)
        let checksPerGame = Double(p.checksGiven) / Double(games)

        return VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Playing Style")
            VStack(spacing: 0) {
                statRow(label: "Aggression", value: aggressionLabel(capturesPerGame))
                Divider()
                statRow(label: "Captures per game", value: String(format: "%.1f", capturesPerGame))
                Divider()
                statRow(label: "Checks per game", value: String(format: "%.1f", checksPerGame))
                Divider()
                statRow(label: "Castled kingside", value: "\(p.castledKingside)×")
                Divider()
                statRow(label: "Castled queenside", value: "\(p.castledQueenside)×")
            }
            .padding(.horizontal)
            .background(Color(.secondarySystemBackground))
            .cornerRadius(16)

            Text("The AI studies these tendencies and adjusts its play to exploit them.")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    private func aggressionLabel(_ capturesPerGame: Double) -> String {
        switch capturesPerGame {
        case ..<0.5: return "Calm"
        case ..<1.5: return "Balanced"
        case ..<3.0: return "Aggressive"
        default: return "Wild"
        }
    }

    // MARK: - Openings

    private var openingsSection: some View {
        let top = store.profile.openings
            .sorted { $0.value > $1.value }
            .prefix(5)

        return VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Favorite Openings")
            if top.isEmpty {
                Text("No openings recorded yet.")
                    .font(.body)
                    .foregroundColor(.secondary)
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(top.enumerated()), id: \.element.key) { index, entry in
                        VStack(spacing: 0) {
                            HStack {
                                Text(formatOpening(entry.key))
                                    .font(.title3)
                                    .bold()
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.7)
                                Spacer()
                                Text("×\(entry.value)")
                                    .font(.title3)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 10)
                            if index < top.count - 1 { Divider() }
                        }
                    }
                }
                .padding(.horizontal)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(16)
            }
            Text("Your most-played first moves. The AI prepares counters for lines you repeat.")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    /// Turns "e2e4 e7e5 g1f3" into "e4 e5 f3" for readability.
    private func formatOpening(_ key: String) -> String {
        key.split(separator: " ").map { uci in
            let s = String(uci)
            guard s.count >= 4 else { return s }
            // Show just the destination square, like "e4".
            let to = String(s.dropFirst(2).prefix(2))
            let promo = s.count > 4 ? "=\(s.dropFirst(4).uppercased())" : ""
            return to + promo
        }.joined(separator: " ")
    }

    // MARK: - Piece preferences

    private var piecesSection: some View {
        let sorted = store.profile.pieceMoveCounts
            .sorted { $0.value > $1.value }
        let total = max(sorted.map(\.value).reduce(0, +), 1)

        return VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Piece Preferences")
            if sorted.isEmpty {
                Text("No piece data yet.")
                    .font(.body)
                    .foregroundColor(.secondary)
            } else {
                VStack(spacing: 10) {
                    ForEach(sorted, id: \.key) { key, count in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(pieceDisplayName(key))
                                    .font(.title3)
                                    .bold()
                                Spacer()
                                Text("\(count)×")
                                    .font(.title3)
                                    .foregroundColor(.secondary)
                            }
                            GeometryReader { geo in
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(Color.accentColor.opacity(0.75))
                                    .frame(width: geo.size.width * CGFloat(count) / CGFloat(total),
                                           height: 10)
                            }
                            .frame(height: 10)
                            .accessibilityHidden(true)
                        }
                    }
                }
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(16)
            }
            Text("Which pieces you move most. The AI will happily trade away your favorites.")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    private func pieceDisplayName(_ key: String) -> String {
        key.prefix(1).uppercased() + key.dropFirst() + "s"
    }

    // MARK: - Raw JSON

    private var jsonSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Profile Data")
            DisclosureGroup {
                ScrollView([.horizontal, .vertical]) {
                    Text(profileJSON)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                }
                .frame(maxHeight: 320)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
            } label: {
                Text("View raw JSON")
                    .font(.title3)
                    .bold()
            }
            Text("This is the exact file stored on your device. It is read-only — it cannot be edited, only learned from or deleted.")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    private var profileJSON: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(store.profile),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    // MARK: - Weekly delete

    private var deleteSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Delete Profile")

            if store.canDelete() {
                Button(role: .destructive) {
                    showDeleteConfirm = true
                } label: {
                    Text("Delete \"\(store.currentProfileName)\"")
                        .font(.title2)
                        .bold()
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(16)
                }
            } else {
                Text(deleteCountdownText)
                    .font(.title3)
                    .bold()
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(16)
                    .foregroundColor(.secondary)
            }

            Text("Removes this profile and everything its AI learned. Allowed once every 7 days per profile.")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    private var deleteCountdownText: String {
        guard let next = store.nextDeleteDate else {
            return "Delete available now"
        }
        let components = Calendar.current.dateComponents([.day, .hour], from: Date(), to: next)
        if let days = components.day, days >= 1 {
            return "Available in \(days) day\(days == 1 ? "" : "s")"
        } else if let hours = components.hour, hours >= 1 {
            return "Available in about \(hours) hour\(hours == 1 ? "" : "s")"
        } else {
            return "Available soon"
        }
    }

    // MARK: - Helpers

    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.title2)
            .bold()
            .accessibilityAddTraits(.isHeader)
    }

    private func statRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.title3)
            Spacer()
            Text(value)
                .font(.title3)
                .bold()
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 10)
    }
}
