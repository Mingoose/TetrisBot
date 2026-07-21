# TetrisBot

A full-featured Tetris web app with a beam-search AI, CNN position evaluator, and a Selenium bot that plays [Jstris](https://jstris.jezevec10.com/) autonomously.

![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178C6?logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5.0-646CFF?logo=vite&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=tensorflow&logoColor=white)

<!-- add screenshot here -->

## Features

- **Playable Tetris** with SRS rotation, DAS/ARR, lock delay, sonic drop, hold, and full 7-bag randomizer
- **Four AI difficulty levels** — easy (greedy), medium (beam W20 D4), hard (beam W32 D5 + advanced heuristics), experimental (CNN-guided beam search)
- **Five game modes** — Sprint (40-line race), Creative (board editor + engine analysis), Versus (player vs bot), Watch (bot-only), Bot-vs-Bot
- **CNN position evaluator** — a TensorFlow.js model trained in PyTorch, exported to a flat binary weights file, and run in-browser
- **Web Worker AI** — beam search runs off the main thread so the UI never stutters
- **Engine overlay** — pause in Creative mode to see the AI's top-ranked move sequences animated on the board
- **Move quality rating** — every piece you place is graded (great / good / mistake / blunder) against the engine's best line
- **Rewind** — undo up to 50 pieces; deterministic replay thanks to snapshot-accurate bag state
- **Sprint & Versus replays** — watch any completed game back frame by frame
- **Selenium bot** — Python automation that plays Jstris Sprint using computer vision and keyboard injection

## Tech Stack

| Layer | Tech |
|---|---|
| Web app | TypeScript, Vite, Canvas 2D |
| AI (web) | Custom beam search, TensorFlow.js (CNN) |
| Auth & storage | Supabase |
| CNN training | PyTorch, ONNX |
| Jstris bot | Python, Selenium, PyAutoGUI, Pillow |

## Getting Started

### Web App

**Prerequisites:** Node.js 18+, a free [Supabase](https://supabase.com) project

```bash
cd tetris-web
npm install
cp .env.example .env.local   # fill in your Supabase URL and anon key
npm run dev                  # starts Vite dev server at http://localhost:5173
```

```bash
npm run build    # TypeScript compile + Vite bundle → dist/
npm run preview  # serve the production build locally
```

### Jstris Bot

**Prerequisites:** Python 3.11+, ChromeDriver in your PATH

```bash
pip install selenium pillow pyautogui
python tetris-bot/main_app.py
```

The bot opens Chrome, logs into Jstris, and begins playing Sprint autonomously.

### CNN Training (optional)

```bash
pip install -r training/requirements.txt

python training/generate_data.py   # generate self-play positions → training/data/positions.npz
python training/train.py           # train PyTorch model → training/models/tetris_eval.pt
python training/export_onnx.py     # export weights binary → tetris-web/public/models/tetris_eval_weights.bin
```

## Usage

### Controls (web app)

| Key | Action |
|---|---|
| `← / →` | Move (DAS 133ms, ARR 10ms) |
| `↓` | Soft drop |
| `Space` | Hard drop |
| `↑ / X` | Rotate clockwise |
| `Z` | Rotate counter-clockwise |
| `A` | Rotate 180° |
| `C / Shift` | Hold |
| `R` | Rewind one piece |
| `E` | Toggle board editor (Creative mode) |
| `P / Esc` | Pause |

All bindings are remappable via the in-game Settings modal, which also exposes DAS/ARR sliders and a sonic-drop toggle.

### Game Modes

- **Sprint** — race to clear 40 lines as fast as possible
- **Creative** — free play with a board editor; pause to run the engine and see annotated move lines
- **Versus** — play against the bot with garbage exchange (combo table, back-to-back bonuses)
- **Watch** — observe the bot play solo
- **Bot-vs-Bot** — two bots share a deterministic piece sequence and play head-to-head

## Architecture

```
TetrisBot/
├── tetris-web/          # TypeScript Vite web app
│   └── src/
│       ├── game.ts      # game loop, gravity, DAS/ARR, lock delay
│       ├── ai.ts        # beam search (all difficulty levels)
│       ├── ai.worker.ts # Web Worker wrapper — keeps UI thread free
│       ├── cnnEvaluator.ts  # TF.js CNN loaded from .bin weights
│       ├── versus.ts    # combat math, garbage, bot-vs-bot sync
│       ├── renderer.ts  # Canvas 2D draw loop
│       └── ...          # 18 more single-responsibility modules
├── tetris-bot/          # Selenium Jstris automation
│   ├── main_app.py      # screen capture, board parsing, key injection
│   └── engine.py        # placement generator + board evaluator
└── training/            # CNN training pipeline
    ├── generate_data.py # self-play data generation
    ├── train.py         # PyTorch training
    └── export_onnx.py   # export weights to flat .bin for TF.js
```

The web app's AI runs entirely in a **Web Worker** (`ai.worker.ts`). The main thread posts an `EngineRequest` (board state + piece queue) and receives an `EngineAnalysis` or `EngineMove` back — beam search never blocks rendering.

The CNN evaluator (`cnnEvaluator.ts`) loads a flat `float32` binary blob of BatchNorm-folded Conv weights and runs a 3-conv + 2-dense network on 20×10 binary board tensors. It augments the hard/experimental beam search as a position scorer but is not required for other difficulty levels.

## Configuration

### Web App — `.env.local`

```env
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

### Jstris Bot — hardcoded values in `tetris-bot/main_app.py`

The bot assumes a fixed screen layout. If your browser window is in a different position, update these constants:

```python
# Canvas region (PyAutoGUI screenshot coords)
CANVAS_REGION = (140, 270, 440, 480)

# Login credentials (change before running)
USERNAME = "ababababab"
PASSWORD = "ababababab_test"
```

Piece colors are identified by exact RGB thresholds in `classify_by_pixel` — update these if the Jstris UI changes.

## Contributing

Pull requests are welcome. For significant changes, open an issue first to discuss the approach. The web app's modules are deliberately single-responsibility — keep new logic in the file that owns that concern rather than adding to `main.ts`. The AI difficulty levels live in `ai.ts`; new evaluator strategies should slot in alongside the existing `easy` / `medium` / `hard` / `experimental` branches.
