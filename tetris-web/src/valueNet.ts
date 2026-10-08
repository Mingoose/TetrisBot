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
  weights: Record<string, { shape: number[]; data: ArrayLike<number> }>;
}

/** The compact app format from export_value_ts.py --app: a manifest plus one float32 blob. */
export interface ValueNetManifest {
  config: ValueNetFile['config'];
  label_mean: number;
  label_std: number;
  tensors: { name: string; shape: number[]; offset: number }[];
}

export function valueNetFromBinary(manifest: ValueNetManifest, blob: ArrayBuffer): ValueNetFile {
  const all = new Float32Array(blob);
  const weights: ValueNetFile['weights'] = {};
  for (const t of manifest.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (t.offset + size > all.length) throw new Error(`value net blob too short for ${t.name}`);
    weights[t.name] = { shape: t.shape, data: all.subarray(t.offset, t.offset + size) };
  }
  return { config: manifest.config, label_mean: manifest.label_mean, label_std: manifest.label_std, weights };
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
// convolution reads fixed offsets around each cell without bounds checks.
const PW = BOARD_COLS + 2;
const PH = BOARD_ROWS + 2;
const PLANE = PW * PH;
const CELLS = BOARD_ROWS * BOARD_COLS;
// Offset of each 3×3 tap from the centre cell, row-major like torch's kernels.
const TAPS = [-PW - 1, -PW, -PW + 1, -1, 0, 1, PW - 1, PW, PW + 1];
// One past the last interior cell, (BOARD_ROWS, BOARD_COLS) in padded coordinates.
const RUN_END = BOARD_ROWS * PW + BOARD_COLS + 1;

// Speed-up: the empty part of the board above the stack. A cell whose receptive
// field holds no block has the same activations as on an empty board: after conv
// layer L that is every row more than L rows above the highest block. Those rows
// are copied from an empty-board pass instead of computed, and the first dense
// layer starts from a precomputed sum over them.

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
  // Empty-board activations: conv layer outputs (padded) and squeeze features.
  private readonly emptyConv: Float32Array[];
  private readonly emptyFeat: Float32Array;
  // h0Base[s]: first dense layer's bias plus the empty-board rows above row s.
  private readonly h0Base: Float32Array[];
  // Scratch buffers, reused across calls.
  private readonly bufA: Float32Array;
  private readonly bufB: Float32Array;
  private readonly feat: Float32Array;
  private readonly hid0: Float32Array;
  private readonly hid1: Float32Array;

  constructor(file: ValueNetFile) {
    const { channels, squeeze, hidden, context_size } = file.config;
    if (context_size !== CONTEXT_SIZE) throw new Error(`context size ${context_size}, expected ${CONTEXT_SIZE}`);
    if (channels % 4) throw new Error(`channels must be a multiple of 4, got ${channels}`);
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

    // Empty-board pass, computing every row.
    this.emptyConv = [];
    let src = this.bufA, dst = this.bufB;
    src.fill(0);
    for (const { w, b, cin } of this.conv) {
      dst.fill(0);
      this.conv3x3(src, dst, w, b, cin, 0);
      this.emptyConv.push(dst.slice());
      const tmp = src; src = dst; dst = tmp;
    }
    this.squeezeRows(src, 0);
    this.emptyFeat = this.feat.slice(0, squeeze * CELLS);
    const n = squeeze * CELLS + 64;
    this.h0Base = [];
    for (let s = 0; s <= BOARD_ROWS; s++) {
      const base = Float32Array.from(this.h0B);
      for (let j = 0; j < hidden; j++) {
        let acc = 0;
        for (let ch = 0; ch < squeeze; ch++) {
          for (let i = ch * CELLS; i < ch * CELLS + s * BOARD_COLS; i++) acc += this.h0W[j * n + i] * this.emptyFeat[i];
        }
        base[j] += acc;
      }
      this.h0Base.push(base);
    }
  }

  /** Predicted future attack for a board (row bitmasks, row 0 = top) and context. */
  evaluate(board: Uint16Array, ctx: Float32Array): number {
    let top = 0;
    while (top < BOARD_ROWS && !board[top]) top++;

    // Input plane.
    let src = this.bufA;
    src.fill(0, 0, PLANE);
    for (let r = top; r < BOARD_ROWS; r++) {
      const bits = board[r];
      if (!bits) continue;
      for (let c = 0; c < BOARD_COLS; c++) if ((bits >> c) & 1) src[(r + 1) * PW + c + 1] = 1;
    }
    let dst = this.bufB;

    for (let l = 0; l < this.conv.length; l++) {
      const { w, b, cin } = this.conv[l];
      dst.set(this.emptyConv[l]);
      this.conv3x3(src, dst, w, b, cin, Math.max(0, top - l - 1));
      const tmp = src; src = dst; dst = tmp;
    }

    const start = Math.max(0, top - this.conv.length);
    const feat = this.feat;
    feat.set(this.emptyFeat);
    this.squeezeRows(src, start);

    // Context path.
    const off = this.squeeze * CELLS;
    for (let j = 0; j < 64; j++) {
      let s = this.ctxB[j];
      const row = j * CONTEXT_SIZE;
      for (let i = 0; i < CONTEXT_SIZE; i++) s += this.ctxW[row + i] * ctx[i];
      feat[off + j] = s > 0 ? s : 0;
    }

    // First dense layer: the empty rows above `start` are already in h0Base.
    const n = off + 64;
    const base = this.h0Base[start];
    const W = this.h0W;
    for (let j = 0; j < this.hidden; j++) {
      let acc = base[j];
      const row = j * n;
      for (let ch = 0; ch < this.squeeze; ch++) {
        const end = row + (ch + 1) * CELLS;
        for (let i = row + ch * CELLS + start * BOARD_COLS, f = ch * CELLS + start * BOARD_COLS; i < end; i++, f++) {
          acc += W[i] * feat[f];
        }
      }
      for (let i = 0; i < 64; i++) acc += W[row + off + i] * feat[off + i];
      this.hid0[j] = acc > 0 ? acc : 0;
    }
    dense(this.h1W, this.h1B, this.hid0, this.hid1, true);
    let out = this.h2B;
    for (let i = 0; i < this.hidden; i++) out += this.h2W[i] * this.hid1[i];
    return out * this.std + this.mean;
  }

  // 1×1 squeeze + ReLU into feat (channel-major like torch's Flatten), rows from startRow.
  private squeezeRows(src: Float32Array, startRow: number): void {
    const C = this.channels;
    for (let s = 0; s < this.squeeze; s++) {
      const wb = s * C;
      for (let r = startRow; r < BOARD_ROWS; r++) {
        const ir = (r + 1) * PW + 1;
        const or = s * CELLS + r * BOARD_COLS;
        for (let c = 0; c < BOARD_COLS; c++) {
          let acc = this.sqB[s];
          for (let ci = 0; ci < C; ci++) acc += this.sqW[wb + ci] * src[ci * PLANE + ir + c];
          this.feat[or + c] = acc > 0 ? acc : 0;
        }
      }
    }
  }

  // 3×3 convolution, padding 1, + ReLU, for board rows startRow and below. src
  // holds `cin` padded planes; dst gets `channels` padded planes. Rows above
  // startRow and the outer border are left as they are; the pad columns inside
  // the computed rows are re-zeroed. Four output channels × four cells at a time
  // with a sliding window, so each loaded input feeds 12 multiply-adds.
  private conv3x3(src: Float32Array, dst: Float32Array, w: Float32Array, b: Float32Array, cin: number, startRow: number): void {
    const C = this.channels;
    const p0 = (startRow + 1) * PW + 1;
    const K = cin * 9; // weights per output channel
    for (let co = 0; co < C; co += 4) {
      const o0 = co * PLANE, o1 = o0 + PLANE, o2 = o1 + PLANE, o3 = o2 + PLANE;
      const b0 = b[co], b1 = b[co + 1], b2 = b[co + 2], b3 = b[co + 3];
      let p = p0;
      for (; p + 4 <= RUN_END; p += 4) {
        let a0 = b0, a1 = b0, a2 = b0, a3 = b0;
        let c0 = b1, c1 = b1, c2 = b1, c3 = b1;
        let d0 = b2, d1 = b2, d2 = b2, d3 = b2;
        let e0 = b3, e1 = b3, e2 = b3, e3 = b3;
        let wk = co * K;
        for (let ci = 0; ci < cin; ci++) {
          // Per kernel row, the 6 inputs x0..x5 cover the three horizontal taps of four cells.
          let q = ci * PLANE + p - PW - 1;
          for (let ky = 0; ky < 3; ky++, q += PW, wk += 3) {
            const x0 = src[q], x1 = src[q + 1], x2 = src[q + 2], x3 = src[q + 3], x4 = src[q + 4], x5 = src[q + 5];
            let u0 = w[wk], u1 = w[wk + 1], u2 = w[wk + 2];
            a0 += u0 * x0 + u1 * x1 + u2 * x2; a1 += u0 * x1 + u1 * x2 + u2 * x3;
            a2 += u0 * x2 + u1 * x3 + u2 * x4; a3 += u0 * x3 + u1 * x4 + u2 * x5;
            u0 = w[wk + K]; u1 = w[wk + K + 1]; u2 = w[wk + K + 2];
            c0 += u0 * x0 + u1 * x1 + u2 * x2; c1 += u0 * x1 + u1 * x2 + u2 * x3;
            c2 += u0 * x2 + u1 * x3 + u2 * x4; c3 += u0 * x3 + u1 * x4 + u2 * x5;
            u0 = w[wk + 2 * K]; u1 = w[wk + 2 * K + 1]; u2 = w[wk + 2 * K + 2];
            d0 += u0 * x0 + u1 * x1 + u2 * x2; d1 += u0 * x1 + u1 * x2 + u2 * x3;
            d2 += u0 * x2 + u1 * x3 + u2 * x4; d3 += u0 * x3 + u1 * x4 + u2 * x5;
            u0 = w[wk + 3 * K]; u1 = w[wk + 3 * K + 1]; u2 = w[wk + 3 * K + 2];
            e0 += u0 * x0 + u1 * x1 + u2 * x2; e1 += u0 * x1 + u1 * x2 + u2 * x3;
            e2 += u0 * x2 + u1 * x3 + u2 * x4; e3 += u0 * x3 + u1 * x4 + u2 * x5;
          }
        }
        dst[o0 + p] = a0 > 0 ? a0 : 0; dst[o0 + p + 1] = a1 > 0 ? a1 : 0; dst[o0 + p + 2] = a2 > 0 ? a2 : 0; dst[o0 + p + 3] = a3 > 0 ? a3 : 0;
        dst[o1 + p] = c0 > 0 ? c0 : 0; dst[o1 + p + 1] = c1 > 0 ? c1 : 0; dst[o1 + p + 2] = c2 > 0 ? c2 : 0; dst[o1 + p + 3] = c3 > 0 ? c3 : 0;
        dst[o2 + p] = d0 > 0 ? d0 : 0; dst[o2 + p + 1] = d1 > 0 ? d1 : 0; dst[o2 + p + 2] = d2 > 0 ? d2 : 0; dst[o2 + p + 3] = d3 > 0 ? d3 : 0;
        dst[o3 + p] = e0 > 0 ? e0 : 0; dst[o3 + p + 1] = e1 > 0 ? e1 : 0; dst[o3 + p + 2] = e2 > 0 ? e2 : 0; dst[o3 + p + 3] = e3 > 0 ? e3 : 0;
      }
      for (; p < RUN_END; p++) {
        for (let j = 0; j < 4; j++) {
          let acc = b[co + j];
          const wb = (co + j) * K;
          for (let ci = 0; ci < cin; ci++) {
            const ib = ci * PLANE + p;
            for (let k = 0; k < 9; k++) acc += w[wb + ci * 9 + k] * src[ib + TAPS[k]];
          }
          dst[o0 + j * PLANE + p] = acc > 0 ? acc : 0;
        }
      }
      // The computed run wrote into the pad columns between rows; zero them.
      for (let r = startRow + 1; r <= BOARD_ROWS; r++) {
        for (const o of [o0, o1, o2, o3]) { dst[o + r * PW] = 0; dst[o + r * PW + PW - 1] = 0; }
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
