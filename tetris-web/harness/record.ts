// Fixed-width binary record for one decision point in a self-play game.
//
// One record = the position the bot saw before moving, the move it made, and
// what that move produced. Python loads these with numpy using RECORD_FIELDS
// (also written to each run's meta.json), so the two must stay in sync.
//
// All multi-byte fields are little-endian. Pieces are encoded as indices into
// PIECE_ORDER; 7 means "none".

import type { PieceType } from '../src/types';
import { ALL_PIECE_TYPES } from '../src/pieces';

export const PIECE_ORDER: readonly PieceType[] = ALL_PIECE_TYPES; // I O T S Z J L
export const NO_PIECE = 7;

export const TERMINAL_NONE = 0;
export const TERMINAL_DIED = 1;      // bot topped out after this move
export const TERMINAL_TRUNCATED = 2; // game hit the piece cap after this move

// [name, numpy dtype, count]
export const RECORD_FIELDS: [string, string, number][] = [
  ['game_id',     '<u4', 1],
  ['ply',         '<u2', 1],
  ['board',       '<u2', 20], // row 0 = top; bit c set = column c filled
  ['active',      'u1',  1],
  ['hold',        'u1',  1],
  ['queue',       'u1',  5],
  ['bag_mask',    'u1',  1],  // bit i set = PIECE_ORDER[i] still in current bag (the set, not the order)
  ['combo',       'i1',  1],  // -1 = no streak
  ['b2b',         'i1',  1],  // B2B level: -1 = none, 0 = first qualifying clear
  ['incoming',    'u1',  1],  // garbage queued against the bot before this move (landed or not)
  ['move_rot',    'u1',  1],
  ['move_x',      'i1',  1],
  ['move_y',      'i1',  1],
  ['move_hold',   'u1',  1],
  ['move_random', 'u1',  1],  // 1 = exploration move, not the teacher's choice
  ['lines',       'u1',  1],  // lines cleared by this move
  ['attack',      'u1',  1],  // gross garbage generated (before cancelling incoming)
  ['spin',        'u1',  1],  // 0 none, 1 mini, 2 full (rules.ts, All-Mini+)
  ['surge',       'u1',  1],  // part of attack released by breaking a charged B2B
  ['perfect_clear','u1', 1],
  ['garbage_in',  'u1',  1],  // garbage rows that landed on the board after this move
  ['terminal',    'u1',  1],
];

const SIZES: Record<string, number> = { '<u4': 4, '<u2': 2, 'u1': 1, 'i1': 1 };
export const RECORD_SIZE = RECORD_FIELDS.reduce((s, [, t, n]) => s + SIZES[t] * n, 0);

export interface PositionRecord {
  gameId: number;
  ply: number;
  board: Uint16Array;
  active: number;
  hold: number;
  queue: number[];
  bagMask: number;
  combo: number;
  b2b: number;
  incoming: number;
  moveRot: number;
  moveX: number;
  moveY: number;
  moveHold: boolean;
  moveRandom: boolean;
  lines: number;
  attack: number;
  spin: number;
  surge: number;
  perfectClear: boolean;
  garbageIn: number;
  terminal: number;
}

const clampU8 = (v: number) => Math.max(0, Math.min(255, v));

export function writeRecord(buf: Buffer, offset: number, r: PositionRecord): number {
  let o = offset;
  o = buf.writeUInt32LE(r.gameId, o);
  o = buf.writeUInt16LE(r.ply, o);
  for (let i = 0; i < 20; i++) o = buf.writeUInt16LE(r.board[i], o);
  o = buf.writeUInt8(r.active, o);
  o = buf.writeUInt8(r.hold, o);
  for (let i = 0; i < 5; i++) o = buf.writeUInt8(r.queue[i] ?? NO_PIECE, o);
  o = buf.writeUInt8(r.bagMask, o);
  o = buf.writeInt8(Math.max(-128, Math.min(127, r.combo)), o);
  o = buf.writeInt8(Math.max(-128, Math.min(127, r.b2b)), o);
  o = buf.writeUInt8(clampU8(r.incoming), o);
  o = buf.writeUInt8(r.moveRot, o);
  o = buf.writeInt8(r.moveX, o);
  o = buf.writeInt8(r.moveY, o);
  o = buf.writeUInt8(r.moveHold ? 1 : 0, o);
  o = buf.writeUInt8(r.moveRandom ? 1 : 0, o);
  o = buf.writeUInt8(clampU8(r.lines), o);
  o = buf.writeUInt8(clampU8(r.attack), o);
  o = buf.writeUInt8(r.spin, o);
  o = buf.writeUInt8(clampU8(r.surge), o);
  o = buf.writeUInt8(r.perfectClear ? 1 : 0, o);
  o = buf.writeUInt8(clampU8(r.garbageIn), o);
  o = buf.writeUInt8(r.terminal, o);
  return o;
}

// Side file written next to the records by duel.ts --record (s<N>.bin): what a
// record alone doesn't hold. `ready` is the garbage rows that would land on a
// non-clearing move (at most the cap); for a value-net move, n > 0 and the
// moves its search expanded with their deep scores (valueBot.ts searchDeep).
export const SEARCH_K = 3;
export const SEARCH_FIELDS: [string, string, number][] = [
  ['game_id',   '<u4', 1],
  ['ply',       '<u2', 1],
  ['ready',     'u1',  1],
  ['n',         'u1',  1],
  ['move_rot',  'u1',  SEARCH_K],
  ['move_x',    'i1',  SEARCH_K],
  ['move_y',    'i1',  SEARCH_K],
  ['move_hold', 'u1',  SEARCH_K],
  ['score',     '<f4', SEARCH_K],
];
export const SEARCH_SIZE = 8 + SEARCH_K * 8;

export interface SearchRecord {
  gameId: number;
  ply: number;
  ready: number;
  expanded: { move: { rotationIndex: number; x: number; y: number; useHold: boolean }; score: number }[];
}

export function writeSearchRecord(buf: Buffer, offset: number, r: SearchRecord): number {
  let o = offset;
  const e = r.expanded.slice(0, SEARCH_K);
  o = buf.writeUInt32LE(r.gameId, o);
  o = buf.writeUInt16LE(r.ply, o);
  o = buf.writeUInt8(clampU8(r.ready), o);
  o = buf.writeUInt8(e.length, o);
  for (let i = 0; i < SEARCH_K; i++) o = buf.writeUInt8(e[i]?.move.rotationIndex ?? 0, o);
  for (let i = 0; i < SEARCH_K; i++) o = buf.writeInt8(e[i]?.move.x ?? 0, o);
  for (let i = 0; i < SEARCH_K; i++) o = buf.writeInt8(e[i]?.move.y ?? 0, o);
  for (let i = 0; i < SEARCH_K; i++) o = buf.writeUInt8(e[i]?.move.useHold ? 1 : 0, o);
  for (let i = 0; i < SEARCH_K; i++) o = buf.writeFloatLE(e[i]?.score ?? 0, o);
  return o;
}
