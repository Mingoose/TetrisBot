// Queue-aware value net (training/value_net.py) evaluated in plain TypeScript.
//
// Scores a post-placement board given the pieces to come: predicted discounted
// attack over the next ~12 pieces (a death inside that window counts -10).
// Weights come from training/export_value_ts.py; the context layout must match
// training/value_data.py encode_context.
//
// Layout: 3× Conv3×3 (padding 1) + ReLU → Conv1×1 to `squeeze` channels + ReLU →
// flatten; context → Linear(64) + ReLU; concat → Linear(hidden) → Linear(hidden) → 1.

import type { PieceType } from './types';
import { ALL_PIECE_TYPES } from './pieces';
import { BOARD_COLS, BOARD_ROWS } from './board';

export const NO_PIECE = 7;
export const CONTEXT_SIZE = 7 + 4 * 8 + 8 + 7 + 3;

export interface ValueNetFile {
  config: { channels: number; squeeze: number; hidden: number; context_size: number };
  label_mean: number;
  label_std: number;
  weights: Record<string, { shape: number[]; data: number[] }>;
}

export const pieceIndex = (p: PieceType | null): number => (p ? ALL_PIECE_TYPES.indexOf(p) : NO_PIECE);

/**
 * Context vector for the position after a placement (see value_data.py).
 * queue4: the 4 pieces after `next`, NO_PIECE where unseen. bagMask: bit i set =
 * ALL_PIECE_TYPES[i] left in the current bag (0x7f for a fresh bag). combo and
 * b2b are the levels after the lock; garbageLanding is the rows that land on it.
 */
export function encodeContext(
  next: number, queue4: number[], hold: number, bagMask: number,
  combo: number, b2b: number, garbageLanding: number,
): Float32Array {
  const ctx = new Float32Array(CONTEXT_SIZE);
  ctx[next] = 1;
  for (let i = 0; i < 4; i++) ctx[7 + i * 8 + queue4[i]] = 1;
  ctx[39 + hold] = 1;
  for (let p = 0; p < 7; p++) ctx[47 + p] = (bagMask >> p) & 1;
  ctx[54] = (combo + 1) / 10;
  ctx[55] = (Math.min(Math.max(b2b, -1), 19) + 1) / 10;
  ctx[56] = garbageLanding / 10;
  return ctx;
}

// Feature maps are stored padded to (BOARD_ROWS + 2) × (BOARD_COLS + 2) so a 3×3
// convolution is a shifted multiply-add over one contiguous run of cells.
const PW = BOARD_COLS + 2;
const PH = BOARD_ROWS + 2;
const PLANE = PW * PH;
// The run covers interior cell (1, 1) through (BOARD_ROWS, BOARD_COLS); the pad
// columns inside it get written too and are re-zeroed after each layer.
const RUN_START = PW + 1;
const RUN_LEN = (BOARD_ROWS - 1) * PW + BOARD_COLS;
const CELLS = BOARD_ROWS * BOARD_COLS;

export class ValueNet {
  readonly channels: number;
  private readonly squeeze: number;
  private readonly hidden: number;
  private readonly mean: number;
  private readonly std: number;
  private readonly conv: { w: Float32Array; b: Float32Array; cin: number }[];
  private readonly sqW: Float32Array; private readonly sqB: Float32Array;
  private readonly ctxW: Float32Array; private readonly ctxB: Float32Array;
  private readonly h0W: Float32Array; private readonly h0B: Float32Array;
  private readonly h1W: Float32Array; private readonly h1B: Float32Array;
  private readonly h2W: Float32Array; private readonly h2B: number;
  // Scratch buffers, reused across calls.
  private readonly bufA: Float32Array;
  private readonly bufB: Float32Array;
  private readonly feat: Float32Array;
  private readonly hid0: Float32Array;
  private readonly hid1: Float32Array;

  constructor(file: ValueNetFile) {
    const { channels, squeeze, hidden, context_size } = file.config;
    if (context_size !== CONTEXT_SIZE) throw new Error(`context size ${context_size}, expected ${CONTEXT_SIZE}`);
    this.channels = channels;
    this.squeeze = squeeze;
    this.hidden = hidden;
    this.mean = file.label_mean;
    this.std = file.label_std;
    const t = (name: string, shape: number[]): Float32Array => {
      const w = file.weights[name];
      if (!w || w.shape.join() !== shape.join()) throw new Error(`weight ${name}: expected shape [${shape}], got [${w?.shape}]`);
      return Float32Array.from(w.data);
    };
    this.conv = [0, 2, 4].map((layer, i) => {
      const cin = i === 0 ? 1 : channels;
      return { w: t(`board.${layer}.weight`, [channels, cin, 3, 3]), b: t(`board.${layer}.bias`, [channels]), cin };
    });
    this.sqW = t('board.6.weight', [squeeze, channels, 1, 1]);
    this.sqB = t('board.6.bias', [squeeze]);
    this.ctxW = t('context.0.weight', [64, CONTEXT_SIZE]);
    this.ctxB = t('context.0.bias', [64]);
    this.h0W = t('head.0.weight', [hidden, squeeze * CELLS + 64]);
    this.h0B = t('head.0.bias', [hidden]);
    this.h1W = t('head.2.weight', [hidden, hidden]);
    this.h1B = t('head.2.bias', [hidden]);
    this.h2W = t('head.4.weight', [1, hidden]);
    this.h2B = t('head.4.bias', [1])[0];

    this.bufA = new Float32Array(channels * PLANE);
    this.bufB = new Float32Array(channels * PLANE);
    this.feat = new Float32Array(squeeze * CELLS + 64);
    this.hid0 = new Float32Array(hidden);
    this.hid1 = new Float32Array(hidden);
  }

  /** Predicted future attack for a board (row bitmasks, row 0 = top) and context. */
  evaluate(board: Uint16Array, ctx: Float32Array): number {
    // Input plane.
    let src = this.bufA;
    src.fill(0, 0, PLANE);
    for (let r = 0; r < BOARD_ROWS; r++) {
      const bits = board[r];
      if (!bits) continue;
      for (let c = 0; c < BOARD_COLS; c++) if ((bits >> c) & 1) src[(r + 1) * PW + c + 1] = 1;
    }
    let dst = this.bufB;

    for (const { w, b, cin } of this.conv) {
      this.conv3x3(src, dst, w, b, cin);
      const tmp = src; src = dst; dst = tmp;
    }

    // 1×1 squeeze + ReLU, flattened channel-major like torch's Flatten.
    const C = this.channels;
    const feat = this.feat;
    for (let s = 0; s < this.squeeze; s++) {
      const out = s * CELLS;
      feat.fill(this.sqB[s], out, out + CELLS);
      for (let ci = 0; ci < C; ci++) {
        const wv = this.sqW[s * C + ci];
        const base = ci * PLANE;
        for (let r = 0; r < BOARD_ROWS; r++) {
          const ir = base + (r + 1) * PW + 1;
          const or = out + r * BOARD_COLS;
          for (let c = 0; c < BOARD_COLS; c++) feat[or + c] += wv * src[ir + c];
        }
      }
      for (let i = out; i < out + CELLS; i++) if (feat[i] < 0) feat[i] = 0;
    }

    // Context path.
    const off = this.squeeze * CELLS;
    for (let j = 0; j < 64; j++) {
      let s = this.ctxB[j];
      const row = j * CONTEXT_SIZE;
      for (let i = 0; i < CONTEXT_SIZE; i++) s += this.ctxW[row + i] * ctx[i];
      feat[off + j] = s > 0 ? s : 0;
    }

    dense(this.h0W, this.h0B, feat, this.hid0, true);
    dense(this.h1W, this.h1B, this.hid0, this.hid1, true);
    let out = this.h2B;
    for (let i = 0; i < this.hidden; i++) out += this.h2W[i] * this.hid1[i];
    return out * this.std + this.mean;
  }

  // 3×3 convolution, padding 1, + ReLU. src holds `cin` padded planes; dst gets
  // `channels` padded planes with zeroed borders.
  private conv3x3(src: Float32Array, dst: Float32Array, w: Float32Array, b: Float32Array, cin: number): void {
    for (let co = 0; co < this.channels; co++) {
      const ob = co * PLANE;
      dst.fill(0, ob, ob + PLANE);
      dst.fill(b[co], ob + RUN_START, ob + RUN_START + RUN_LEN);
      for (let ci = 0; ci < cin; ci++) {
        const ib = ci * PLANE;
        const wb = (co * cin + ci) * 9;
        for (let k = 0; k < 9; k++) {
          const wv = w[wb + k];
          if (wv === 0) continue;
          const shift = ((k / 3) | 0) * PW + (k % 3) - PW - 1;
          const o0 = ob + RUN_START;
          const i0 = ib + RUN_START + shift;
          for (let i = 0; i < RUN_LEN; i++) dst[o0 + i] += wv * src[i0 + i];
        }
      }
      // ReLU, and re-zero the pad columns the run wrote into.
      for (let r = 1; r <= BOARD_ROWS; r++) {
        const rb = ob + r * PW;
        dst[rb] = 0;
        dst[rb + PW - 1] = 0;
        for (let c = 1; c <= BOARD_COLS; c++) if (dst[rb + c] < 0) dst[rb + c] = 0;
      }
    }
  }
}

function dense(w: Float32Array, b: Float32Array, x: Float32Array, y: Float32Array, relu: boolean): void {
  const n = x.length;
  for (let j = 0; j < y.length; j++) {
    let s = b[j];
    const row = j * n;
    for (let i = 0; i < n; i++) s += w[row + i] * x[i];
    y[j] = relu && s < 0 ? 0 : s;
  }
}
