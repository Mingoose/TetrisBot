import { PieceType, RotationMatrix } from './types';

// All 7 pieces with 4 rotation states (ported from engine.py, expanded to full 4 rotations).
// Each rotation is a 2D array where 1 = filled cell.
// All rotation matrices use the full SRS bounding box so that piece positions
// match the Tetris Guideline exactly (important for wall kicks and column offsets).
// State order: 0 (spawn) → R (CW) → 2 (180°) → L (CCW)
export const ROTATIONS: Record<PieceType, RotationMatrix[]> = {
  // I uses a 4×4 bounding box. CW lands at col 2, CCW at col 1 — intentionally asymmetric.
  I: [
    [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], // 0
    [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], // R
    [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], // 2
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], // L
  ],
  O: [
    [[1, 1], [1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1], [1, 1]],
  ],
  T: [
    [[0, 1, 0], [1, 1, 1], [0, 0, 0]], // 0
    [[0, 1, 0], [0, 1, 1], [0, 1, 0]], // R
    [[0, 0, 0], [1, 1, 1], [0, 1, 0]], // 2
    [[0, 1, 0], [1, 1, 0], [0, 1, 0]], // L
  ],
  // S and Z each have 4 distinct states — states 0/2 share the same shape but
  // occupy different rows of the 3×3 bounding box, which affects wall kicks.
  S: [
    [[0, 1, 1], [1, 1, 0], [0, 0, 0]], // 0
    [[0, 1, 0], [0, 1, 1], [0, 0, 1]], // R
    [[0, 0, 0], [0, 1, 1], [1, 1, 0]], // 2
    [[1, 0, 0], [1, 1, 0], [0, 1, 0]], // L
  ],
  Z: [
    [[1, 1, 0], [0, 1, 1], [0, 0, 0]], // 0
    [[0, 0, 1], [0, 1, 1], [0, 1, 0]], // R
    [[0, 0, 0], [1, 1, 0], [0, 1, 1]], // 2
    [[0, 1, 0], [1, 1, 0], [1, 0, 0]], // L
  ],
  J: [
    [[1, 0, 0], [1, 1, 1], [0, 0, 0]], // 0
    [[0, 1, 1], [0, 1, 0], [0, 1, 0]], // R
    [[0, 0, 0], [1, 1, 1], [0, 0, 1]], // 2
    [[0, 1, 0], [0, 1, 0], [1, 1, 0]], // L
  ],
  L: [
    [[0, 0, 1], [1, 1, 1], [0, 0, 0]], // 0
    [[0, 1, 0], [0, 1, 0], [0, 1, 1]], // R
    [[0, 0, 0], [1, 1, 1], [1, 0, 0]], // 2
    [[1, 1, 0], [0, 1, 0], [0, 1, 0]], // L
  ],
};

// Standard Tetris guideline colors
export const PIECE_COLORS: Record<PieceType | 'X', string> = {
  I: '#00f0f0',
  O: '#f0f000',
  T: '#a000f0',
  S: '#00f000',
  Z: '#f00000',
  J: '#0000f0',
  L: '#f0a000',
  X: '#888888', // editor-placed cell
};

// SRS+ wall kicks (TETR.IO), keyed by "<from><to>" rotation state.
// dx: positive = right. dy: positive = DOWN (board Y increases downward).
// Each list excludes the unkicked test, which getKicks prepends. Source:
// Triangle.js kick data ("SRS+"), github.com/halp1/triangle, which uses the
// same sign convention. J/L/S/T/Z 90° kicks are plain SRS; I kicks and all
// 180° kicks differ from SRS.
type KickTable = Record<string, ReadonlyArray<readonly [number, number]>>;

const KICKS_JLSTZ: KickTable = {
  '01': [[-1, 0], [-1, -1], [0, 2], [-1, 2]],
  '10': [[1, 0], [1, 1], [0, -2], [1, -2]],
  '12': [[1, 0], [1, 1], [0, -2], [1, -2]],
  '21': [[-1, 0], [-1, -1], [0, 2], [-1, 2]],
  '23': [[1, 0], [1, -1], [0, 2], [1, 2]],
  '32': [[-1, 0], [-1, 1], [0, -2], [-1, -2]],
  '30': [[-1, 0], [-1, 1], [0, -2], [-1, -2]],
  '03': [[1, 0], [1, -1], [0, 2], [1, 2]],
  '02': [[0, -1], [1, -1], [-1, -1], [1, 0], [-1, 0]],
  '13': [[1, 0], [1, -2], [1, -1], [0, -2], [0, -1]],
  '20': [[0, 1], [-1, 1], [1, 1], [-1, 0], [1, 0]],
  '31': [[-1, 0], [-1, -2], [-1, -1], [0, -2], [0, -1]],
};

const KICKS_I: KickTable = {
  '01': [[1, 0], [-2, 0], [-2, 1], [1, -2]],
  '10': [[-1, 0], [2, 0], [-1, 2], [2, -1]],
  '12': [[-1, 0], [2, 0], [-1, -2], [2, 1]],
  '21': [[-2, 0], [1, 0], [-2, -1], [1, 2]],
  '23': [[2, 0], [-1, 0], [2, -1], [-1, 2]],
  '32': [[1, 0], [-2, 0], [1, -2], [-2, 1]],
  '30': [[1, 0], [-2, 0], [1, 2], [-2, -1]],
  '03': [[-1, 0], [2, 0], [2, 1], [-1, -2]],
  '02': [[0, -1]],
  '13': [[1, 0]],
  '20': [[0, 1]],
  '31': [[-1, 0]],
};

const NO_KICK: ReadonlyArray<readonly [number, number]> = [[0, 0]];

// Precomputed [table][from * 4 + to] with the unkicked test first; 180° and 90°
// share one index space. Built once so the bot's search does a plain array lookup.
const KICK_LISTS = [KICKS_JLSTZ, KICKS_I].map(table => {
  const lists: ReadonlyArray<readonly [number, number]>[] = new Array(16).fill(NO_KICK);
  for (const key of Object.keys(table)) lists[+key[0] * 4 + +key[1]] = [[0, 0], ...table[key]];
  return lists;
});

// Kick tests for rotating `type` from one state to another, unkicked test first.
// Index 4 of a J/L/S/T/Z 90° list is the "TST / fin" kick (see rules.isTstKick).
export function getKicks(type: PieceType, from: number, to: number): ReadonlyArray<readonly [number, number]> {
  if (type === 'O') return NO_KICK;
  return KICK_LISTS[type === 'I' ? 1 : 0][(from & 3) * 4 + (to & 3)];
}

export function getRotation(type: PieceType, index: number): RotationMatrix {
  const rots = ROTATIONS[type];
  return rots[((index % rots.length) + rots.length) % rots.length];
}

export const ALL_PIECE_TYPES: PieceType[] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
