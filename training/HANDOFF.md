# Handoff: training the v2 value network on the M1 Max

Written 2026-10-05 for a Claude Code session on the user's M1 Max, continuing
work started on their M1 MacBook (8 GB, 4 performance cores). Read the root
`CLAUDE.md` first for the codebase; this file covers where the value-network
project stands and what to do next.

## The task right now

Train three value networks on the self-play data already on this machine, run
the T-slot probe, and send the results back to the other Mac. Concretely:

1. Get the environment and data in place (checklist below).
2. Run `training/run_v2.sh` (it checks the setup first, then trains all three
   models in sequence and shows a macOS notification when done).
3. Report the results (see "What to report").

The user has been hitting setup errors running commands by hand — that is why
this session exists. Do the setup and launching for them.

## Setup checklist (all from `training/`)

**Python environment.** `.venv` may not exist yet.
```bash
python3 --version          # PyTorch needs >= 3.10; Apple's built-in python3 is 3.9
python3 -m venv .venv      # or, if 3.9: brew install python@3.12 && python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch numpy
.venv/bin/python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
```
If MPS is available, training uses the GPU automatically (`--device auto`).
On the other Mac (macOS 13) torch 2.11 reported MPS unavailable.

**Data.** Both runs must be in `training/data/runs/` (gitignored, not in the repo):
- `teacher-v2b/` — generated on this M1 Max (games 3000+, ~2,100 games).
- `teacher-v2/` — generated on the other Mac (games 0–808, 216,293 positions),
  sent over by AirDrop as `~/Downloads/teacher-v2.zip`. If not yet unpacked:
  `unzip ~/Downloads/teacher-v2.zip -d data/runs/`
- Verify: `ls data/runs/*/meta.json` lists both, and
  `.venv/bin/python selfplay_data.py data/runs/teacher-v2 data/runs/teacher-v2b`
  shows ~809 and ~2,100 games with the `spins … full / … mini` summary line.
  Game ids must not overlap (0–808 vs 3000+).

Make sure no self-play run is still writing to `teacher-v2b`
(`pgrep -fl selfplay.mjs` should print nothing; stop with `pkill -f selfplay.mjs`).

**Launch.**
```bash
nohup caffeinate -i ./run_v2.sh > data/logs/run_v2.out 2>&1 &
tail -f data/logs/run_v2.out            # step start/done lines
tail -f data/logs/train-v2.log          # per-epoch progress of the current model
```
Overrides: `DEVICE=cpu ./run_v2.sh` if MPS errors; `THREADS=8` default;
`EXTRA="--epochs 1"` for a quick smoke test (used to verify the script).

Expected time (estimates, not measured on this machine): ~780K training samples.
CPU with 8 threads ≈ 2 min/epoch for the 32-channel model, 12–18 epochs with
early stopping → whole script ~1–1¼ h. With MPS maybe 25–40 min. Multiply the
first epoch's `(Ns)` by ~15 for a real estimate.

## Pitfalls already hit

- **zsh doesn't word-split `$VAR`.** `R="a b"; cmd $R` passes one argument in
  zsh (macOS default shell). Write paths out or use a `sh` script (run_v2.sh does).
- **"meta.json does not exist"** = a run folder path is wrong (usually the above).
- **"no such file models/value_v2.pt"** = training failed earlier; read
  `data/logs/train-v2.log`, not the probe error.
- `source .venv/bin/activate` only lasts for that terminal; run_v2.sh calls
  `.venv/bin/python` directly so activation doesn't matter.

## What the three models test

| Model | Command difference | Question it answers |
|---|---|---|
| `value_v2` | — (32 channels, all data) | The main model |
| `value_v2_half` | `--train-fraction 0.5` | Would more data help? Same validation games as `value_v2` |
| `value_v2_c16` | `--channels 16` | Is a ~4× cheaper model (for in-browser inference) nearly as good? |

## What to report

From each `data/logs/train-v2*.log`: best epoch, final `val loss`, `R²`, and
the context-shuffled R² (last 3 lines). From `probe-v2*.txt`: the table.
Then zip results for the other Mac:
```bash
zip -r ~/Downloads/v2-results.zip data/logs models/value_v2*.pt
```

How to read them:
- **Full vs half:** if full-data R² is clearly higher than half-data, the model
  is data-limited → generating more self-play data is worth it. If similar,
  the bottleneck is labels or architecture.
- **32 vs 16 channels:** if R² is within ~0.01, prefer 16 channels for the
  browser.
- **Context shuffle:** R² dropping noticeably when the piece context is
  shuffled means the net uses the queue. The previous model (old rules, 190K
  samples) went 0.189 → 0.120.
- **T-slot probe:** the slot board should beat the flat board most when a T is
  close (next / hold) and least when no T comes until the next bag. Previous
  model: +0.93 (T in hold) … +0.60 (no T until next bag) — right direction, too
  timid.

## 1v1 results (2026-10-05, M1 Max)

v2 trained (val R² 0.313 full / 0.297 half / 0.311 c16). Exported with
`export_value_ts.py`, ported to TS (`src/valueNet.ts`, `src/valueBot.ts`);
`npm run valuecheck` matches PyTorch to 2e-6 with identical boards/contexts on
1,000 validation positions. Duels (`npm run duel`, 2 pps both sides):

| A vs B | Result | Attack/piece A vs B |
|---|---|---|
| net c16 1-ply vs hard | 0–100 | 0.22 vs 0.39 |
| net 32ch 1-ply vs hard | 3–47 | 0.23 vs 0.38 |
| hard heuristic 1-ply (hard:1:1) vs hard | 1–49 | 0.17 vs 0.39 |
| net c16 1-ply vs hard:1:1 | 15–35 | 0.23 vs 0.15 |

The net bot attacks more than the 1-ply heuristic but tops out sooner. Watching
it play solo: it builds boards full of holes and rates them highly (~8). It
only ever trained on afterstates the teacher chose, so 1-ply argmax over ~70
candidates finds its blind spots. It picks the teacher's move 17.5% of the time
(the 1-ply heuristic: 29.3%). Fix candidates: train on the other candidates too
(a ranking loss over each position's placements, target = teacher's move), and/or
label states the net bot itself visits.

### Ranking loss (step 1 of the fix)

`npm run candidates -- RUN_DIR` (tetris-web/) writes every placement of every
recorded decision (~69 each, 55M total for v2+v2b, ~3 GB, `cand*.bin` in the
run dir). `train_rank.py` adds a softmax loss over attack + value with the
teacher's move as target (`--rank-weight 1`, `--temp 1`), starting from
`value_v2_c16.pt`: 12 epochs × 200K decisions, ~2¼ min each on MPS.
Log: `data/logs/train-rank-c16.log`; model `models/rank_v2_c16.pt`.

- Teacher agreement 17.0% → 35.7% (top 3: 36% → 66%). Value R² fell to ~−0.05:
  the output now only ranks, it no longer predicts attack on its own.
- vs hard:1:1 (1-ply heuristic): **98–2** (old net 15–35).
- vs hard: **12–88** (old net 0–100); games 154 pieces (was 32), attack/piece 0.39 vs 0.50.

### Selective deepening (top K re-searched one piece deeper)

`findBestMoveDeep` in `valueBot.ts`; duel spec `net:FILE.json@K`. The top K
1-ply moves are re-scored as attack + best (attack + value) with the next,
visible piece. Same seeds as the 1-ply duels:

| Net, search | vs hard | Attack/piece (net vs hard) | Game length | ms/move |
|---|---|---|---|---|
| rank_v2_c16, 1-ply | 12–88 | 0.39 vs 0.50 | 154 | ~210 |
| rank_v2_c16, K=3 | **51–49** | 0.54 vs 0.54 | 224 | ~850 |
| rank025_v2_c16, K=3 | **54–46** | 0.59 vs 0.53 | 115 | ~870 |

`rank025` = `--rank-weight 0.25` (val R² 0.18, teacher agreement 34.1%). The two
K=3 results are within noise of each other (paired games: 22 won only by
rank 1.0, 25 only by 0.25); the 0.25 net attacks more and ends games faster.
ms/move is Node with 8–9 duel workers sharing the CPU.

### Faster net and deeper search (2026-10-06)

`valueNet.ts` now skips rows above the stack (their activations equal an
empty board's, cached per layer, with precomputed first-dense-layer sums) and
register-blocks the convolutions: 2.6 ms → 0.45 ms per board (c16), still
within 2.5e-6 of PyTorch. Duel spec widths are per ply: `@3,2` = top 3 moves,
then the top 2 replies of each scored with a third piece. rank_v2_c16 vs hard,
200 games, same seeds (data/duels/depth-*-vs-hard.jsonl):

| Search | vs hard | Attack/piece (net vs hard) | Length | ms/move |
|---|---|---|---|---|
| `@3` | 98–102 (49%) | 0.542 vs 0.546 | 214 | 172 |
| `@5` | 115–85 (57.5%) | 0.571 vs 0.548 | 187 | 252 |
| `@3,2` | **161–39 (80.5%)** | 0.622 vs 0.549 | 170 | 419 |

Paired games: `@3,2` beats `@3` (84 vs 21 games won only by one, p < 0.0001)
and `@5` (68 vs 22, p < 0.0001); `@5` vs `@3` is 57 vs 40 (p ≈ 0.10).
The third ply matters more than a wider second ply.

## Project background

**Goal.** A Tetris bot whose network judges a board *together with the upcoming
pieces*, so it can search shallowly (1 ply, ~70 evaluations per move) instead
of the hard bot's deep beam search (width 32, depth 5). The user's framing:
"if I create the overhang and I have a T in the next few pieces then I should
be fine — I don't need to compute the moves in between."

**Plan.**
1. ✅ Node self-play harness running the real `ai.ts` (`tetris-web/harness/`).
2. ✅ Teacher data from the hard bot under TETR.IO rules (`teacher-v2`, `teacher-v2b`).
3. ⏳ Train the value net (this step). Labels: discounted attack over the next
   12 pieces, −10 if the bot dies in that window (`training/value_data.py`).
4. Build the experimental difficulty as 1-ply search scored by the net, in the
   browser (hand-written TS inference or a smaller model), within the 2 s
   bot-move timeout.
5. Play it against hard mode (bot-vs-bot); later, self-improvement rounds where
   the net bot generates its own data.

**Why the original CNN failed** (it is still in `cnnEvaluator.ts`): inference was
numerically correct but TF.js CPU took ~13 ms/board (20–40 s per move); its
labels were 93% identical; it never saw the queue or hold.

**Rules.** The game targets TETR.IO multiplayer: All-Mini+ spins, multiplier
combos, B2B level with surge, SRS+ kicks, garbage cap 8 / 20-frame travel /
lands only on non-clearing locks. All in `tetris-web/src/rules.ts` and
`versus.ts`; `npm run test:rules` verifies against Triangle.js.

**Data provenance.**
- `teacher-v2`, `teacher-v2b`: current rules — use these.
- `teacher-v1`: old JStris-style rules — pipeline testing only, never mix.
- `obsolete-teacher-v2-srs-oldgarbage`: pre-SRS+/garbage fix — don't use.
- Known teacher quirk: the bot's search sees the true order of the remaining
  bag, slightly more than a player knows; recorded data stores only the set.

**User decisions to respect.**
- Do **not** propose tuning the hard bot's weights with an optimiser (CMA-ES);
  the user decided to use the current hard bot as the teacher.
- The CNN should replace search depth, not sit beside a deep search.
- AI style preferences: favour T-spin setups and B2B over single-line clears;
  don't penalise interior wells.
- The user is the only developer; commit straight to `main`, no PRs.

**Throughput reference.** Hard teacher: ~190 ms/piece per core on the M1;
data generation ~90K positions/hour on 4 cores there.
