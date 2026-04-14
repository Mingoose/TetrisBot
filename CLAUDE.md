# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tetris Web App (`tetris-web/`)

### Running

```bash
cd tetris-web
npm install
npm run dev      # Vite dev server with hot reload
npm run build    # TypeScript compile + Vite bundle
npm run preview  # Preview production build
```

Requires a `.env` file (or environment variables) for Supabase auth:
```
VITE_SUPABASE_URL=...
VITE_SUPABASE_ANON_KEY=...
```

### Architecture

Twenty-two TypeScript source files in `src/`, each with a single responsibility:

**Core game engine:**
- **`types.ts`** — All shared interfaces: `GameState`, `Snapshot`, `ActivePiece`, `CellValue`, `GameMode`, `GameVariant`
- **`pieces.ts`** — Piece rotation matrices, SRS wall-kick tables (`WALL_KICKS_JLSTZ`, `WALL_KICKS_I`), piece colors
- **`board.ts`** — Pure functions: `collides`, `hardDropY`, `lockPiece`, `clearLines`, `isGameOver`, `scoreForLines`, `gravityInterval`
- **`bag.ts`** — 7-bag randomizer with `getState()`/`restoreState()` for snapshot-accurate rewind
- **`game.ts`** — Game loop (`processFrame`), gravity, DAS/ARR, lock delay, `lockAndSpawn`, hold, hard drop, sprint completion; `setLockHook()` for versus mode callbacks
- **`input.ts`** — Keyboard event handler; `flushInput()` clears per-frame edge state each loop
- **`rewind.ts`** — `pushHistory`/`rewind`: snapshots taken pre-lock in `lockAndSpawn`; one undo = one piece; max 50 snapshots

**AI engine (Web Worker pipeline):**
- **`engine.ts`** — Public types: `EngineRequest`, `EngineAnalysis`, `EngineLine`, `EngineMove`; `gameStateToEngineRequest()` helper
- **`ai.ts`** — Beam search implementation; `findBestMove`/`findBestMoveHard`/`analyzePositionHard`; `AiDifficulty` levels (easy=greedy, medium=beam W20 D4, hard=beam W32 D5 + advanced eval)
- **`ai.worker.ts`** — Web Worker entry point; dispatches `type:'analyze'` to `analyzePositionHard`, otherwise to bot-move functions
- **`moveQuality.ts`** — `classifyMove()`: matches player placement against engine lines to rate quality (great/good/mistake/blunder) by score delta

**Game modes:**
- **`versus.ts`** — Versus/Watch/Bot-vs-Bot modes; `VersusData`, `BotVsBotData`, `BotBoard`, `CombatState`; garbage math (combo table, B2B); player lock hook; bot move application; shared piece sequence for bot-vs-bot
- **`replay.ts`** — `SprintReplay` and `VersusReplay` types + frame-index helpers for playback; each entry stores a pre-lock snapshot and elapsed time

**UI and rendering:**
- **`editor.ts`** — Board editor mode (creative only); click/drag toggles cells; `CELL_SIZE`/`BOARD_OFFSET_*` constants used by renderer
- **`renderer.ts`** — All Canvas 2D drawing; full redraw each frame; ghost piece, hold, next queue (5 pieces), HUD, overlays, sprint/versus complete screens
- **`engineOverlay.ts`** — Canvas overlay rendered during creative pause; animates the selected engine line on the board + shows a notation panel listing all lines with scores and clear labels
- **`settings.ts`** — `KeyBindings` interface, `Settings` type (keybindings + DAS/ARR + sonicDrop toggle), `keyLabel()` helper
- **`settingsUI.ts`** — Modal overlay for keybinding remapping, DAS/ARR sliders, sonic drop toggle, save/cancel/reset flow
- **`storage.ts`** — `StorageAdapter` interface + `LocalStorageAdapter` (localStorage); `loadSettings()` merges saved with defaults
- **`supabase.ts`** — Supabase client singleton; reads `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` from env
- **`authUI.ts`** — Sign-in/sign-up overlay backed by Supabase auth; auto-skips if a session already exists
- **`main.ts`** — Entry point: creates canvas, wires everything, loads settings, registers menu click handler, starts `requestAnimationFrame` loop

### Key Design Notes

**Bag/rewind consistency:** The module-level `bag` in `game.ts` is always restored from `state.bagState` before consuming — never consumed directly. This keeps rewind deterministic.

**Snapshot fields:** `Snapshot` (the rewind unit) is a strict subset of `GameState` containing only serializable data. Transient timing fields (`lockDelayMs`, `gravityAccumMs`, etc.) are reset on restore, never saved.

**Layout constants:** `CELL_SIZE`, `BOARD_OFFSET_X`, `BOARD_OFFSET_Y` live in `editor.ts` and are imported by `renderer.ts`. Changing these changes both the editor hit-testing and the visual layout simultaneously.

**Game variants:** `'sprint'` races to 40 lines; `'creative'` is free play with board editor + engine analysis on pause; `'versus'` pits player against the bot with garbage exchange; `'watch'` is bot-only; `'botvsbot'` runs two bots head-to-head on a shared piece sequence.

**Web Worker AI:** `ai.worker.ts` runs beam search off the main thread. `main.ts` posts requests via `requestBotMove()` (from `versus.ts`) and receives responses with the chosen move. Engine analysis for creative pause uses a separate `type:'analyze'` message path and returns `EngineAnalysis` with `topN` ranked lines.

**Versus garbage:** `versus.ts` owns all combat math — `computeGarbage()`, `handleLock()`, and `CombatState`. Garbage is exchanged on each piece lock; cancellation is applied before routing the remainder to the opponent.

**Bot-vs-bot piece sync:** Both bots share a single `bvbSeq[]` array grown lazily by one Bag. Each bot tracks its own `pieceIndex` so they draw deterministically from the same sequence regardless of play speed.

**Lock delay:** 500ms, resets on movement, max 15 resets before force-lock.

**Sonic drop:** Optional setting; when enabled, soft-drop teleports piece to bottom without locking.

**Controls (defaults):** `←/→` move (DAS=133ms, ARR=10ms), `↓` soft drop, `Space` hard drop, `↑/X` rotate CW, `Z` rotate CCW, `A` rotate 180°, `C/Shift` hold, `R` rewind, `E` editor mode, `P/Esc` pause. All bindings are remappable via the settings modal.

## Running the Bot

```bash
python tetris-bot/main_app.py
```

Requires ChromeDriver in PATH and these Python packages:
```bash
pip install selenium pillow pyautogui
```

The bot opens Chrome, logs into JStris, and begins autonomous Sprint mode play.

## Bot Architecture

Three modules in `tetris-bot/`:

**`main_app.py`** — Browser automation + computer vision layer
- Uses Selenium to drive Chrome on jstris.jezevec10.com
- Captures screenshots with PyAutoGUI; parses the 10×20 board into a binary matrix (brightness >30 = filled)
- Identifies pieces by sampling pixel RGB values (`classify_by_pixel`)
- Translates engine move commands into keyboard inputs (`execute_move`)
- Game loop: capture → parse → call engine → execute → repeat (0.1s sleep)

**`engine.py`** — AI move solver
- Generates all valid placements (rotations × horizontal positions) via `get_valid_moves()`
- Simulates placement and line clears with `make_move()`
- Scores board states with `evaluate_board()`: +20/line cleared, −2/height unit, −2/bumpiness unit, −20/hole
- `get_best_move()` runs DFS to configurable depth (default=1) and returns the optimal first move

**`coin.py`** — Standalone probability utility, not used by the bot

## Key Hardcoded Values

All screen coordinates in `main_app.py` assume a specific screen resolution/browser position:
- Canvas region: `(140, 270, 440, 480)` px (PyAutoGUI screenshot region)
- Next queue offset: `x=360, y=0`
- Hold box offset: `x=0, y=10`

Piece colors are identified by exact RGB thresholds in `classify_by_pixel`. If the game UI changes, these need updating.

Login credentials are hardcoded in `log_in()` (`ababababab` / `ababababab_test`).
