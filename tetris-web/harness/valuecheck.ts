// Check the TypeScript value net and 1-ply bot against Python, then measure how
// often the bot picks the teacher's move.
//
// Uses the check positions written by training/export_value_ts.py: validation
// positions with the teacher's move and Python's sample for it.
//
// Usage (from tetris-web/):
//   npm run valuecheck -- ../training/models/value_v2_c16.json [--deep]
//
// --deep also measures agreement with the top K re-searched one piece deeper
// (findBestMoveDeep), on the first DEEP_CHECKS positions; it is ~5× slower per position.

import { readFileSync } from 'node:fs';

import type { PieceType } from '../src/types';
import { ALL_PIECE_TYPES } from '../src/pieces';
import { BOARD_COLS, BOARD_ROWS } from '../src/board';
import { ValueNet, ValueNetFile } from '../src/valueNet';
import { findBestMoveDeep, scoreCandidates } from '../src/valueBot';
import { findBestMoveHard } from '../src/ai';

interface Check {
  board: number[]; active: number; hold: number; queue: number[]; bag_mask: number;
  combo: number; b2b: number; move: [number, number, number, number];
  lines: number; attack: number; garbage_in: number;
  after: number[]; ctx: number[]; value: number;
}

const piece = (i: number): PieceType | null => (i === 7 ? null : ALL_PIECE_TYPES[i]);

function cellBoard(bits: number[]) {
  return Array.from({ length: BOARD_ROWS }, (_, r) =>
    Array.from({ length: BOARD_COLS }, (_, c) => ((bits[r] >> c) & 1 ? 'X' : 0) as 0 | 'X'));
}

const file = JSON.parse(readFileSync(process.argv[2], 'utf8')) as ValueNetFile & { checks: Check[] };
const net = new ValueNet(file);
const checks = file.checks;

let maxNetErr = 0, ctxMismatch = 0, boardMismatch = 0, attackMismatch = 0, notFound = 0;
let top1 = 0, top3 = 0, rankSum = 0, hard1Top1 = 0;
let netMs = 0, evals = 0;
const DEEP_K = process.argv.includes('--deep') ? [3, 5] : [];
const DEEP_CHECKS = 200;
let deepN = 0;
const deepTop1 = DEEP_K.map(() => 0);

for (const ch of checks) {
  // 1. Net alone, on Python's own inputs.
  const v = net.evaluate(Uint16Array.from(ch.after), Float32Array.from(ch.ctx));
  maxNetErr = Math.max(maxNetErr, Math.abs(v - ch.value));

  // 2. The bot's search, from the decision the teacher faced. Which garbage was
  // ready isn't recorded; use what landed after the teacher's move.
  const state = {
    board: cellBoard(ch.board), active: piece(ch.active)!, hold: piece(ch.hold),
    queue: ch.queue.map(i => piece(i)!), bagMask: ch.bag_mask, combo: ch.combo, b2b: ch.b2b,
    landingCols: new Array(ch.lines > 0 ? 0 : ch.garbage_in).fill(0),
  };
  const t0 = performance.now();
  const cands = scoreCandidates(state, net);
  netMs += performance.now() - t0;
  evals += cands.length;

  const [rot, x, y, hold] = ch.move;
  const rank = cands.findIndex(c => c.move.rotationIndex === rot && c.move.x === x && c.move.y === y
    && c.move.useHold === (hold === 1));
  if (rank < 0) { notFound++; continue; }
  const t = cands[rank];
  if (t.after.some((b, r) => b !== ch.after[r])) boardMismatch++;
  if (t.ctx.some((c, i) => Math.abs(c - ch.ctx[i]) > 1e-6)) ctxMismatch++;
  if (t.attack !== ch.attack) attackMismatch++;
  if (DEEP_K.length && deepN < DEEP_CHECKS) {
    deepN++;
    DEEP_K.forEach((k, i) => {
      const m = findBestMoveDeep(state, net, [k]);
      if (m.rotationIndex === rot && m.x === x && m.y === y && m.useHold === (hold === 1)) deepTop1[i]++;
    });
  }
  if (rank === 0) top1++;
  if (rank < 3) top3++;
  rankSum += rank + 1;

  // Reference: the hard heuristic with no lookahead (depth 1).
  const bot = {
    board: state.board, active: { type: state.active, rotationIndex: 0, x: 0, y: 0 },
    nextQueue: state.queue, hold: state.hold, holdUsed: false, bagState: [] as PieceType[],
    pieceIndex: -1, lines: 0, dead: false,
  };
  const h = findBestMoveHard(bot, ch.lines > 0 ? 0 : ch.garbage_in, 1, 1, ch.combo, ch.b2b);
  if (h.rotationIndex === rot && h.x === x && h.y === y && h.useHold === (hold === 1)) hard1Top1++;
}

const n = checks.length;
const found = n - notFound;
const pct = (k: number, d = found) => `${(100 * k / d).toFixed(1)}%`;
console.log(`${n} validation positions, ${net.channels}-channel net`);
console.log(`net output vs PyTorch: max abs error ${maxNetErr.toExponential(2)}`);
console.log(`teacher move not among candidates: ${notFound}`);
console.log(`teacher move encoded differently from Python: board ${boardMismatch}, context ${ctxMismatch}, attack ${attackMismatch}`);
console.log(`picks the teacher's move: net 1-ply ${pct(top1)} (top 3: ${pct(top3)}, mean rank ${(rankSum / found).toFixed(1)}), ` +
  `hard heuristic 1-ply ${pct(hard1Top1)}`);
if (DEEP_K.length) console.log(`with the top K re-searched one piece deeper (first ${deepN} positions): ` +
  DEEP_K.map((k, i) => `K=${k} ${pct(deepTop1[i], deepN)}`).join(', '));
console.log(`speed: ${(netMs / n).toFixed(1)} ms/move, ${(1000 * netMs / evals).toFixed(0)} µs/board, ${(evals / n).toFixed(0)} candidates/move`);
if (maxNetErr > 1e-3 || boardMismatch || ctxMismatch || attackMismatch) process.exitCode = 1;
