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

// SRS wall-kick offsets: [fromRotation][kickIndex] = [dx, dy]
// Used when a rotation is blocked — try each kick offset in order.
// dx: positive = right. dy: positive = DOWN (board Y increases downward).
// Source: Tetris Guideline (Y-up) with dy signs negated for board coordinates.
export const WALL_KICKS_JLSTZ: Array<Array<[number, number]>> = [
  // 0 → R
  [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  // R → 2
  [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  // 2 → L
  [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  // L → 0
  [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
];

// Counter-clockwise = next rotation index - 1, use reversed kicks
export const WALL_KICKS_I: Array<Array<[number, number]>> = [
  // 0 → 1
  [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  // 1 → 2
  [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  // 2 → 3
  [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  // 3 → 0
  [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
];

export function getRotation(type: PieceType, index: number): RotationMatrix {
  const rots = ROTATIONS[type];
  return rots[((index % rots.length) + rots.length) % rots.length];
}

export function getWallKicks(type: PieceType, fromIndex: number): Array<[number, number]> {
  const normalized = ((fromIndex % 4) + 4) % 4;
  return type === 'I' ? WALL_KICKS_I[normalized] : WALL_KICKS_JLSTZ[normalized];
}

export const ALL_PIECE_TYPES: PieceType[] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
