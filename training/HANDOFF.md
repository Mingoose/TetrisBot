# Handoff: the value-net Tetris bot

Status as of 2026-10-08, written on the M1 Max for the next session (likely on
the user's other laptop). Read the root `CLAUDE.md` first for the codebase; this
file covers the value-network project: where it stands, how to pick it up,
what was tried, and what is left.

## Where things stand

**The best bot is `rank_v2_c16` with a small search over the visible queue,
and it is in the app** as the Experimental difficulty
(`tetris-web/public/models/value_net.{json,bin}`).

| Bot (200 games vs hard mode, benchmark seed 1) | Result | Attack/piece (bot vs hard) |
|---|---|---|
| `rank_v2_c16`, top 3 moves + top 2 replies (`@3,2`, 3 pieces) | **161–39 (80.5%)** | 0.62 vs 0.55 |
| same, one more ply (`@3,2,2`, 4 pieces) | 168–32 (84%) | 0.69 vs 0.56 |

The app uses `findBestMoveTimed` (`src/valueBot.ts`): it deepens `@3` → `@3,2`
→ `@3,2,2` while a 1.2 s budget allows (the app falls back to another bot at
2 s). In Chrome on the M1 Max, 102 of 150 moves reached 4 pieces; median
843 ms, max 1.21 s per move. A slower machine settles at 3 or 2 pieces.

Three self-improvement rounds (fine-tuning on data the net bot generated) all
came out equal or worse, so they were not shipped; see "What was tried".

**Not yet done by anyone:** playing the Experimental difficulty in the real app
UI. The app needs Supabase credentials (`tetris-web/.env`) and a sign-in, which
the M1 Max session did not have, so it was tested through a temporary page using
the same worker and request code. Do this first on a machine with `.env`.

## Picking it up on another machine

Everything needed is in the repo except the candidate files (5 GB, regenerable).

```bash
git pull
cd training
python3 --version                      # needs >= 3.10 (Apple's python3 is 3.9: brew install python@3.12)
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install torch numpy
.venv/bin/python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
.venv/bin/python selfplay_data.py data/runs/*     # summaries of all six runs (see Data below)

cd ../tetris-web && npm install
npm run harness:typecheck && npx tsc --noEmit -p .
```

Regenerate what is not committed, only when needed:
```bash
# Candidate files for train_rank.py (~5 GB total, ~5 min): one per run you train on
for r in teacher-v2 teacher-v2b vs-r2; do npm run candidates -- ../training/data/runs/$r; done
# TypeScript test exports (weights + 1000 check positions) for duels / valuecheck
cd ../training && .venv/bin/python export_value_ts.py models/rank_v2_c16.pt \
    --out models/rank_v2_c16.json data/runs/teacher-v2 data/runs/teacher-v2b
cd ../tetris-web && npm run valuecheck -- ../training/models/rank_v2_c16.json   # expect max error ~2e-6
```

Quick sanity check that everything works (≈5 min on 8 cores):
`npm run duel -- --a net:../training/models/rank_v2_c16.json@3,2 --b hard --games 20`
should win roughly 16 of 20.

## The pipeline (commands)

All TypeScript tools run from `tetris-web/`, Python from `training/`.

| Step | Command | Notes |
|---|---|---|
| Self-play data | `npm run harness -- --out ../training/data/runs/NAME --games N [--net FILE.json@3,2]` | hard bot teacher by default; `--net` = value-net teacher |
| Versus data | `npm run duel -- --a net:FILE.json@3,2 --b hard --games N --seed S --record ../training/data/runs/NAME` | records both sides + side file (ready garbage, search scores) |
| Candidates | `npm run candidates -- RUN_DIR` | every placement of every decision; needed by `train_rank.py` |
| Value-only training | `train_value.py RUNS --channels 16 --out models/X.pt` | the original v2 nets |
| Ranking training | `train_rank.py RUNS --channels 16 [--init CKPT] --out models/X.pt` | see options below |
| Export for TS tools | `export_value_ts.py CKPT --out models/X.json RUNS` | weights + check positions |
| Export for the app | `export_value_ts.py CKPT --app ../tetris-web/public/models` | compact `value_net.{json,bin}` |
| Check TS = PyTorch | `npm run valuecheck -- models/X.json [--deep]` | also teacher agreement and speed |
| 1v1 | `npm run duel -- --a SPEC --b SPEC --games 200 --out FILE.jsonl` | specs: `hard`, `hard:W:D`, `net:FILE.json[@K,K2,…]` |

`train_rank.py` options that matter: `--rank-weight` (teacher-target loss),
`--soft-weight`/`--soft-temp` (targets from the net's own search, for runs made
with `duel --record`), `--anchor-fraction` (share of each epoch from
teacher-target runs), `--lr` (use 3e-4 when fine-tuning), `--patience 0` (no early
stop; best and `.last.pt` are both saved), `--death-penalty`. With `--init` the
starting checkpoint's label mean/std are kept (recomputing them silently rescales
the value against the attack term; that hurt round 1).

**Judging a new net:** always by duels, never by validation loss. Gate it head to
head against `rank_v2_c16@3,2` (200 games; needs ≳55%), then 200 games vs hard
on seed 1 and compare game by game with `data/duels/depth-3-2-vs-hard.jsonl`
(paired sign test). Train on seeds other than 1.

## Data in the repo

`training/data/runs/*/` holds the raw records (`w*.bin`), side files (`s*.bin`,
versus runs only) and `meta.json`. Candidate files (`cand*.bin`,
`candidates.json`) and `models/*.json` are gitignored and regenerable.

| Run | Games | Positions | Teacher / source | Seed, game ids |
|---|---|---|---|---|
| `teacher-v2` | 809 | 216K | hard W32 D5, 5% random drops, 0.2 garbage/piece | 1, ids 0–808 (other Mac) |
| `teacher-v2b` | 2,167 | 581K | same | 1, ids 3000+ |
| `net-r1` | 700 | 186K | `rank_v2_c16@3,2` self-play | 7, ids 0–699 |
| `net-r1b` | 300 | 81K | same | 7, ids 10000+ |
| `net-r1c` | 150 | 38K | same | 7, ids 20000+ |
| `vs-r2` | 800 duels | 275K (both sides) | `rank_v2_c16@3,2` vs hard, real garbage | 11; game id = 2 × duel + board |

Not on the M1 Max (may still be on the other laptop): `teacher-v1` (old
JStris-style rules, pipeline testing only, never mix) and
`obsolete-teacher-v2-srs-oldgarbage` (pre-SRS+/garbage fix, don't use).
Known teacher quirk: the hard bot's search sees the true order of the remaining
bag; records store only the set.

Models (`training/models/*.pt`, ~1 MB each): `value_v2*` (value-only),
`rank_v2_c16` (**best**), `rank025_v2_c16`, `rank_r1_c16`, `rank_r2_c16`,
`surv_c16` (+ `.last.pt` variants). Logs in `data/logs/`, every duel's per-game
results in `data/duels/*.jsonl`.

## What was tried (results history)

**1. Value-only net (v2).** Labels: discounted attack over the next 12 pieces
(γ 0.97), −10 if the bot dies in that window. Val R² 0.313 (32 ch), 0.311
(16 ch), 0.297 (half data); context shuffle drops R² to ~0.21, so it uses the
queue. 1-ply bot vs hard: **0–100**. It only ever saw boards the teacher chose,
so the search over ~70 candidates found boards it misjudged (it rated hole-ridden
boards ~8).

**2. Ranking loss** (`train_rank.py`): softmax over every candidate's
attack + value, target = the teacher's move, plus the value loss. Teacher
agreement 17% → 36% (top 3: 36% → 66%); value R² fell to ~0 (the ranking loss
only constrains differences within a position). 1-ply: 98–2 vs the 1-ply
heuristic, 12–88 vs hard. `--rank-weight 0.25` gave R² 0.18 and played the same.

**3. Search over the visible queue** (`findBestMoveDeep`, 200 games vs hard):
`@3` 98–102, `@5` 115–85, **`@3,2` 161–39**, `@3,2,2` 168–32. The third piece
matters far more than width; the fourth is not significantly better
(paired 30 vs 23, p ≈ 0.41).

**4. Speed.** TS inference (`src/valueNet.ts`): skips rows above the stack
(cached empty-board activations and first-dense-layer sums), sliding-window
convolution over 4 output channels × 4 cells: 2.6 ms → 0.36 ms per board in Node,
identical outputs. The net is ~99% of search time; move generation ~5 µs per
candidate.

**5. Self-improvement rounds — all failed the gate:**

| Round | Data | Targets | vs rank_v2 (gate) | vs hard |
|---|---|---|---|---|
| r1 `rank_r1_c16` | net self-play, 305K | search's pick (hard), new data only, lr 1e-3 | — | 136–64 / 128–72 (worse, p < 0.01) |
| r2 `rank_r2_c16` | versus vs hard, 275K + teacher anchor 50% | soft targets from own search; lr 3e-4 | 100–100 | 150–50 (p ≈ 0.24) |
| r3 `surv_c16` | same as r2 | no search targets, death penalty 30 | 88–111 | 155–45 (p ≈ 0.56) |

Rounds 1–2 made the net more aggressive (attack/piece 0.68–0.73 vs 0.62) without
winning more; round 3 lowered aggression (0.65) and still didn't win more, so
"too aggressive" is not the explanation. Every large gain came from a structural
change (ranking loss, queue search), not from more training on the bot's own
games. On this hardware a round costs 3–5 h, too few rounds for an
AlphaZero-style loop to pay off.

## What is left (in rough order of value)

1. **Play the Experimental difficulty in the app** (needs `.env` + sign-in):
   menu enables once the net loads; check versus, watch and bot-vs-bot modes.
2. **Slower devices:** if the bot should be strong on older laptops/phones, split
   the search's independent branches across a few workers (~2–3× wall-clock on
   the deep plies), or WebAssembly SIMD for the convolution (~2–3× per board,
   bigger job). Not worth it for this machine (4-piece search mostly fits).
3. **Training, if resumed:** the one untested idea is win/loss value labels from
   the versus games (does this side eventually win?); ~30 min training + 1 h
   duels with the existing `vs-r2` data. Expect a similar outcome to r2/r3.
4. Housekeeping: `run_v2.sh` and `train_value.py` still default to the v2
   setup; fine as they are.

## Pitfalls already hit

- **Lid closed = everything pauses.** `caffeinate -i` prevents idle sleep only;
  closing the lid still sleeps the Mac. Long runs survive (they resume), but
  per-piece timings in logs spanning a sleep are meaningless.
- **zsh doesn't word-split `$VAR`** (`R="a b"; cmd $R` passes one argument).
  Write paths out or use `sh`.
- **`pkill -f PATTERN` can match its own shell** when the pattern appears in the
  same command line; it killed a wrapper once. Prefer PIDs.
- **Chrome throttles timers in hidden tabs**; benchmark pages must not yield via
  `setTimeout` (Web Workers are not throttled).
- **Duels under CPU load:** ms/move in duel logs is with all cores busy; measure
  speed separately (valuecheck, or the browser).
- `source .venv/bin/activate` only lasts for that terminal; call
  `.venv/bin/python` directly.

## Project background

**Goal.** A Tetris bot whose network judges a board *together with the upcoming
pieces*, so it needs only a shallow search instead of the hard bot's deep beam
search (width 32, depth 5). The user's framing: "if I create the overhang and I
have a T in the next few pieces then I should be fine — I don't need to compute
the moves in between." Outcome: the net replaces most of the depth; a 2–3 ply
search over the *visible* queue (no guessing) does the rest.

**Rules.** TETR.IO multiplayer: All-Mini+ spins, multiplier combos, B2B level
with surge, SRS+ kicks, garbage cap 8 / 20-frame travel / lands only on
non-clearing locks. All in `tetris-web/src/rules.ts` and `versus.ts`;
`npm run test:rules` verifies against Triangle.js.

**User decisions to respect.**
- Do **not** propose tuning the hard bot's weights with an optimiser (CMA-ES).
- The net should replace search depth, not sit beside a deep search (a small
  search over the visible queue was agreed).
- The hard bot is a benchmark, not a teacher to imitate further: the user wants
  a bot on the frontier, not a copy of hard mode.
- AI style preferences: favour T-spin setups and B2B over single-line clears;
  don't penalise interior wells.
- The user is the only developer; commit straight to `main`, no PRs.
