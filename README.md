# AdaptiveChess

An iOS chess app with an on-device AI that learns how **you** play — and uses it against you.

## What it is

- **Full legal chess** — every legal move is playable, no illegal moves, ever. Check, checkmate, stalemate, castling, en passant, promotion, 50-move rule, threefold repetition, insufficient material.
- **On-device neural-network AI** — a from-scratch 769→128→32→1 MLP written in Swift. No Core ML, no cloud, no API keys. It starts from random weights and trains purely on your games, on your device.
- **Adaptive opponent** — as you play, the app builds a personal profile file (openings, piece preferences, aggression, castling habits…). The AI exploits what it learns, so over time you start losing more and have to develop new ways to beat it.
- **Simple UI** — large high-contrast board with coordinates, legal-move dots, last-move highlight. Adjustable computer level and "context window" (how many games the profile remembers).
- **Read-only profile viewer** — you can look at your profile file in-app but never edit it. You may delete it once per week (7-day cooldown), then the AI starts learning you all over again.

## Building

Builds run on GitHub Actions — there is nothing to compile locally.

1. Push to `main`, or open the **Actions** tab and run **"Build unsigned IPA"** via *workflow_dispatch*.
2. When the run finishes, download the **AdaptiveChess-unsigned** artifact (an `.ipa` file).

> **No releases are ever published.** Per the owner's requirement, IPAs are distributed only as Actions workflow artifacts — never as GitHub Releases.

## Installing the IPA

The IPA is **unsigned**, so it won't install directly. Use one of these (free):

- **Sideloadly** (Mac/Windows) — drag the IPA in, sign with your Apple ID, install over USB/Wi-Fi.
- **AltStore** (Mac/Windows) — same idea, installs via its companion app.

Both re-sign the app with a free Apple developer certificate (7-day expiry, just re-install weekly).

## Project layout

```
AdaptiveChess/
├── AdaptiveChessApp.swift      — app entry, tab view (Game / Profile / Settings)
├── Engine/
│   ├── ChessTypes.swift        — Color, PieceType, Piece, Square, Move
│   ├── ChessGame.swift         — game state, legal move gen, make/undo, terminal detection
├── AI/
│   ├── NeuralNet.swift         — the MLP: forward pass, backprop, weight init
│   ├── Search.swift            — move search guided by the network
│   └── Evaluation.swift        — board features + profile exploitation
├── Profile/
│   ├── PlayerProfile.swift     — Codable profile: stats, openings, tendencies (+ network weights)
│   ├── ProfileStore.swift      — JSON persistence, context window, weekly delete cooldown
│   └── GameRecorder.swift      — records moves/games into profile, triggers training
└── UI/
    ├── BoardView.swift         — SwiftUI board, tap-to-move, legal move dots
    ├── GameView.swift          — game screen, new game, resign, status
    ├── SettingsView.swift      — AI level, context window size
    └── ProfileView.swift       — READ-ONLY profile viewer, weekly delete button
```

Xcode project: `AdaptiveChess.xcodeproj` (iOS 17+, Swift, bundle ID `com.user.adaptivechess`).
CI: `.github/workflows/build-ipa.yml` — archive unsigned → zip Payload → upload artifact.
