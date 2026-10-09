// Bot scored by the value net (valueNet.ts).
//
// Every reachable placement of the active piece, and of the hold option, is
// scored as its exact attack plus the net's value of the afterstate given the
// pieces still to come. findBestMoveDeep then re-searches the best few with the
// next (visible) piece: the net replaces most of the depth, not all of it.

import type { CellValue, PieceType } from './types';
import type { BotBoard, CombatState } from './versus';
import { BOARD_COLS, BOARD_ROWS, spawnPosition } from './board';
import { resolveClear, RULES } from './rules';
import { bmCollides, bmLockAndClear, cellBoardToBm, findReachablePlacements } from './ai';
import { ALL_PIECE_TYPES } from './pieces';
import { NO_PIECE, ValueNet, encodeContext, pieceIndex } from './valueNet';

export interface ValueSearchState {
  board: CellValue[][];
  active: PieceType;
  hold: PieceType | null;
  queue: PieceType[];       // next pieces after the active one (5 visible)
  bagMask: number;          // pieces left in the current bag (bit i = ALL_PIECE_TYPES[i]); 0x7f if fresh
  combo: number;            // before this move (-1 = none)
  b2b: number;              // before this move (-1 = none)
  // Garbage that lands if this move clears nothing: hole column per row, at most
  // RULES.garbageCap rows, only chunks that have finished travelling.
  landingCols: number[];
  // All queued garbage rows, oldest first (landingCols is a prefix). Used to look
  // past this move; defaults to landingCols.
  pendingCols?: number[];
}

export interface BotMove { rotationIndex: number; x: number; y: number; useHold: boolean }

export interface Candidate {
  move: BotMove;
  lines: number;
  attack: number;
  after: Uint16Array;   // board after line clears, before garbage lands
  dies: boolean;        // tops out after this move (including landing garbage)
  // Context inputs for the net (see valueNet.ts encodeContext).
  next: number;
  queue4: number[];
  hold: number;
  combo: number;        // after this move
  b2b: number;          // after this move
  landing: number;      // garbage rows that land on it
}

export interface ValueCandidate extends Candidate {
  value: number;        // net's predicted future attack
  score: number;
}

// Below any real score, so a move that tops out is only played when all do.
const DEATH_SCORE = -1000;
const FULL_ROW = (1 << BOARD_COLS) - 1;

const NO_MOVE: BotMove = { rotationIndex: 0, x: 0, y: 0, useHold: false };

/**
 * The search state for a versus.ts bot at time `now`. bagMask: the pieces left
 * in the current bag (0x7f when the next draw starts a new bag).
 */
export function searchStateFor(bot: BotBoard, combat: CombatState, bagMask: number, now: number): ValueSearchState {
  // Queued garbage rows, and those that would land on a non-clearing lock now
  // (finished travelling, at most the cap; versus.ts tankGarbage).
  const pendingCols: number[] = [];
  let ready = 0;
  for (const chunk of combat.incoming) {
    for (let i = 0; i < chunk.amount; i++) pendingCols.push(chunk.column);
    if (chunk.readyAt <= now && ready === pendingCols.length - chunk.amount) ready = pendingCols.length;
  }
  return {
    board: bot.board, active: bot.active.type, hold: bot.hold, queue: bot.nextQueue,
    bagMask, combo: combat.combo, b2b: combat.b2b,
    landingCols: pendingCols.slice(0, Math.min(ready, RULES.garbageCap)), pendingCols,
  };
}

/** Every placement of the active piece and of the hold option, unscored. */
export function listCandidates(state: ValueSearchState): Candidate[] {
  return candidatesOn(cellBoardToBm(state.board), state);
}

// The 4 queue pieces after `next` as context indices, NO_PIECE where unseen.
function queue4Of(pieces: PieceType[]): number[] {
  const q = pieces.slice(0, 4).map(pieceIndex);
  while (q.length < 4) q.push(NO_PIECE);
  return q;
}

function candidatesOn(board: Uint16Array, state: Omit<ValueSearchState, 'board'>): Candidate[] {
  const q = state.queue;
  // [piece played, uses hold, next active, the 4 queue pieces after it, hold after]
  const options: [PieceType, boolean, PieceType, number[], PieceType | null][] = [
    [state.active, false, q[0], queue4Of(q.slice(1)), state.hold],
  ];
  if (state.hold !== null) {
    options.push([state.hold, true, q[0], queue4Of(q.slice(1)), state.active]);
  } else if (q.length >= 2) {
    // Holding into an empty hold plays q[0] and draws two pieces; the second is unseen.
    options.push([q[0], true, q[1], queue4Of(q.slice(2, 5)), state.active]);
  }

  const out: Candidate[] = [];
  for (const [type, useHold, next, queue4, hold] of options) {
    for (const p of findReachablePlacements(board, type)) {
      const piece = { type, rotationIndex: p.rotationIndex, x: p.x, y: p.y };
      const { bm: after, linesCleared } = bmLockAndClear(board, piece);
      let perfectClear = linesCleared > 0;
      for (let r = 0; perfectClear && r < BOARD_ROWS; r++) if (after[r]) perfectClear = false;
      const clear = resolveClear(state.combo, state.b2b, linesCleared, p.spin, perfectClear);
      const landing = linesCleared > 0 ? 0 : Math.min(state.landingCols.length, RULES.garbageCap);
      out.push({
        move: { rotationIndex: p.rotationIndex, x: p.x, y: p.y, useHold },
        lines: linesCleared, attack: clear.attack, after,
        dies: topsOut(after, state.landingCols.slice(0, landing), next),
        next: pieceIndex(next), queue4, hold: pieceIndex(hold),
        combo: clear.combo, b2b: clear.b2b, landing,
      });
    }
  }
  return out;
}

export function candidateContext(c: Candidate, bagMask: number): Float32Array {
  return encodeContext(c.next, c.queue4, c.hold, bagMask, c.combo, c.b2b, c.landing);
}

/** Score every placement, best first. */
export function scoreCandidates(state: ValueSearchState, net: ValueNet): (ValueCandidate & { ctx: Float32Array })[] {
  return scoreOn(listCandidates(state), state.bagMask, net);
}

function scoreOn(cands: Candidate[], bagMask: number, net: ValueNet): (ValueCandidate & { ctx: Float32Array })[] {
  return cands.map(c => {
    const ctx = candidateContext(c, bagMask);
    const value = net.evaluate(c.after, ctx);
    return { ...c, ctx, value, score: (c.dies ? DEATH_SCORE : 0) + c.attack + value };
  }).sort((a, b) => b.score - a.score);
}

export function findBestMoveValue(state: ValueSearchState, net: ValueNet): BotMove {
  return scoreCandidates(state, net)[0]?.move ?? NO_MOVE;
}

/**
 * Re-search the best placements deeper with the visible queue. widths[d] is how
 * many of the best moves at ply d+1 (by attack + value) are expanded with the
 * next piece: [3] re-scores the top 3 as attack + best follow-up; [3, 2] also
 * expands the top 2 follow-ups of each with a third piece. [] is 1-ply. A line
 * scores the attack along it plus the value of the board it ends on. The pieces
 * involved are all visible (5 in the queue), so the search never guesses.
 */
export function findBestMoveDeep(state: ValueSearchState, net: ValueNet, widths: number[]): BotMove {
  return searchDeep(state, net, widths).move;
}

/**
 * Time-aware search for the app (iterative deepening): search with the first
 * width of `maxWidths`, then add one ply at a time while the time budget allows.
 * A deeper pass is only started if its estimated cost fits, and is abandoned at
 * the deadline, keeping the last completed result. Passes share scored
 * positions, so each one pays only for its new ply; a slow machine ends up with
 * a shallower search instead of missing the move deadline.
 */
export function findBestMoveTimed(
  state: ValueSearchState, net: ValueNet, maxWidths: number[], budgetMs: number,
): { move: BotMove; widths: number[]; ms: number } {
  const t0 = performance.now();
  const memo: Memo = new Map();
  let depth = Math.min(1, maxWidths.length);
  let widths = maxWidths.slice(0, depth);
  const first: SearchCtx = { memo, deadline: Infinity, scored: 0 };
  let result = searchDeep(state, net, widths, first);
  // Time per scored position (one candidate list run through the net), measured.
  const perList = (performance.now() - t0) / Math.max(1, first.scored ?? 0);
  while (depth < maxWidths.length) {
    // The next pass only scores the new leaves: one list per line at the new depth.
    const newLists = maxWidths.slice(0, depth + 1).reduce((a, b) => a * b, 1);
    if (performance.now() - t0 + perList * newLists > budgetMs) break;
    const deeper = maxWidths.slice(0, depth + 1);
    try {
      result = searchDeep(state, net, deeper, { memo, deadline: t0 + budgetMs, scored: 0 });
    } catch (e) {
      if (e !== TIMEOUT) throw e;
      break;
    }
    widths = deeper;
    depth++;
  }
  return { move: result.move, widths, ms: performance.now() - t0 };
}

/** The root moves the search expanded, with their deep scores (attack along the line + leaf value). */
export interface SearchResult {
  move: BotMove;
  expanded: { move: BotMove; score: number }[];
}

type Scored = (ValueCandidate & { ctx: Float32Array })[];
type Memo = Map<string, Scored>;
interface SearchCtx { memo?: Memo; deadline: number; scored?: number }
const TIMEOUT = Symbol('search timeout');
const moveKey = (m: BotMove) => `${m.rotationIndex},${m.x},${m.y},${m.useHold ? 1 : 0}`;

export function searchDeep(state: ValueSearchState, net: ValueNet, widths: number[], sc: SearchCtx = { deadline: Infinity }): SearchResult {
  let roots = sc.memo?.get('');
  if (!roots) {
    roots = scoreOn(candidatesOn(cellBoardToBm(state.board), state), state.bagMask, net);
    sc.memo?.set('', roots);
    if (sc.scored !== undefined) sc.scored++;
  }
  const live = roots.filter(c => !c.dies).slice(0, widths[0] ?? 0);
  if (live.length <= 1) return { move: roots[0]?.move ?? NO_MOVE, expanded: [] };
  const expanded = live.map(c => ({
    move: c.move, score: c.attack + lineValue(state, c, net, widths, 1, sc, moveKey(c.move)),
  }));
  let best = expanded[0];
  for (const e of expanded) if (e.score > best.score) best = e;
  return { move: best.move, expanded };
}

// Best score reachable from the position `c` leaves: attack + value one ply on,
// or deeper while widths[depth] says to keep expanding. `path` names the line
// for the memo.
function lineValue(
  state: SearchFields, c: Candidate, net: ValueNet, widths: number[], depth: number, sc: SearchCtx, path: string,
): number {
  if (performance.now() > sc.deadline) throw TIMEOUT;
  const child = positionAfter(state, c);
  let scored = sc.memo?.get(path);
  if (!scored) {
    scored = scoreOn(candidatesOn(child.board, child.state), child.state.bagMask, net);
    sc.memo?.set(path, scored);
    if (sc.scored !== undefined) sc.scored++;
  }
  if (!scored.length) return DEATH_SCORE;
  const live = depth < widths.length ? scored.filter(x => !x.dies).slice(0, widths[depth]) : [];
  if (!live.length) return scored[0].score;
  let best = -Infinity;
  for (const x of live) {
    best = Math.max(best, x.attack + lineValue(child.state, x, net, widths, depth + 1, sc, `${path}|${moveKey(x.move)}`));
  }
  return best;
}

type SearchFields = Omit<ValueSearchState, 'board'>;

// The decision after playing `c`: its board with any landed garbage, the next
// piece, and what is left of the garbage queue.
function positionAfter(state: SearchFields, c: Candidate): { board: Uint16Array; state: SearchFields } {
  // A clear cancels queued rows oldest first; otherwise the ready rows land.
  // Whatever is left is assumed to have arrived by the next move.
  const pending = state.pendingCols ?? state.landingCols;
  let board = c.after;
  let rest: number[];
  if (c.lines > 0) rest = pending.slice(Math.min(c.attack, pending.length));
  else {
    board = withGarbage(c.after, state.landingCols.slice(0, c.landing));
    rest = pending.slice(c.landing);
  }
  const piece = (i: number) => ALL_PIECE_TYPES[i];
  return {
    board,
    state: {
      active: piece(c.next),
      hold: c.hold === NO_PIECE ? null : piece(c.hold),
      queue: c.queue4.filter(i => i !== NO_PIECE).map(piece),
      bagMask: state.bagMask, // the draw after this move is unseen, as in training
      combo: c.combo, b2b: c.b2b,
      landingCols: rest.slice(0, RULES.garbageCap), pendingCols: rest,
    },
  };
}

// `after` with garbage rows pushed in from the bottom (versus.ts tankGarbage
// order: the first row to land ends up highest). Rows pushed off the top are lost.
function withGarbage(after: Uint16Array, cols: number[]): Uint16Array {
  const n = cols.length;
  if (!n) return after;
  const b = new Uint16Array(BOARD_ROWS);
  for (let r = 0; r < BOARD_ROWS - n; r++) b[r] = after[r + n];
  for (let i = 0; i < n; i++) b[BOARD_ROWS - n + i] = FULL_ROW & ~(1 << cols[i]);
  return b;
}

// The bot's own, stricter top-out model, which the net was trained with: a
// block in the top two visible rows after garbage lands, or the next piece
// blocked at the top of the visible field. The game itself only ends on a
// TETR.IO block out or lock out (board.ts), above the visible field.
function topsOut(after: Uint16Array, landingCols: number[], next: PieceType): boolean {
  const n = landingCols.length;
  for (let r = 0; r < n; r++) if (after[r]) return true; // pushed off the top
  const b = withGarbage(after, landingCols);
  return b[0] !== 0 || b[1] !== 0 || bmCollides(b, { ...spawnPosition(next), y: 0 }, 0, 0);
}
