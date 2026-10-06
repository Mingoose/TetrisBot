// Bot scored by the value net (valueNet.ts).
//
// Every reachable placement of the active piece, and of the hold option, is
// scored as its exact attack plus the net's value of the afterstate given the
// pieces still to come. findBestMoveDeep then re-searches the best few with the
// next (visible) piece: the net replaces most of the depth, not all of it.

import type { CellValue, PieceType } from './types';
import { BOARD_COLS, BOARD_ROWS } from './board';
import { spawnPiece } from './game';
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
 * Take the `topK` best placements by 1-ply score and re-score each as its attack
 * plus the best follow-up with the next piece (attack + net value). The next
 * piece is always visible, so this second ply needs no guessing.
 */
export function findBestMoveDeep(state: ValueSearchState, net: ValueNet, topK: number): BotMove {
  const roots = scoreCandidates(state, net);
  const live = roots.filter(c => !c.dies).slice(0, topK);
  if (topK <= 1 || live.length <= 1) return roots[0]?.move ?? NO_MOVE;
  let best = live[0];
  let bestScore = -Infinity;
  for (const c of live) {
    const s = c.attack + bestFollowUp(state, c, net);
    if (s > bestScore) { bestScore = s; best = c; }
  }
  return best.move;
}

// Best attack + value over the placements after `c`, from the position it leaves.
function bestFollowUp(state: ValueSearchState, c: Candidate, net: ValueNet): number {
  // Garbage after this move: a clear cancels queued rows oldest first; otherwise
  // the ready rows land. Whatever is left is assumed to have arrived by the next move.
  const pending = state.pendingCols ?? state.landingCols;
  let board = c.after;
  let rest: number[];
  if (c.lines > 0) rest = pending.slice(Math.min(c.attack, pending.length));
  else {
    board = withGarbage(c.after, state.landingCols.slice(0, c.landing));
    rest = pending.slice(c.landing);
  }
  const piece = (i: number) => ALL_PIECE_TYPES[i];
  const children = candidatesOn(board, {
    active: piece(c.next),
    hold: c.hold === NO_PIECE ? null : piece(c.hold),
    queue: c.queue4.filter(i => i !== NO_PIECE).map(piece),
    bagMask: state.bagMask, // the draw after this move is unseen, as in training
    combo: c.combo, b2b: c.b2b,
    landingCols: rest.slice(0, RULES.garbageCap), pendingCols: rest,
  });
  const scored = scoreOn(children, state.bagMask, net);
  return scored.length ? scored[0].score : DEATH_SCORE;
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

// Game over after landing garbage under `after`: a block in the top two rows
// (board.ts isGameOver) or the next piece can't spawn.
function topsOut(after: Uint16Array, landingCols: number[], next: PieceType): boolean {
  const n = landingCols.length;
  for (let r = 0; r < n; r++) if (after[r]) return true; // pushed off the top
  const b = withGarbage(after, landingCols);
  return b[0] !== 0 || b[1] !== 0 || bmCollides(b, spawnPiece(next), 0, 0);
}
