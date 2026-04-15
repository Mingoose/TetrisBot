/**
 * CNN-based position evaluator for the "experimental" difficulty.
 *
 * Uses TensorFlow.js (CPU backend) to run inference on a small CNN trained to
 * score Tetris board afterstates. Weights are loaded from a binary blob that
 * was exported from PyTorch with BatchNorm folded into the Conv layers.
 *
 * Model architecture (BN folded):
 *   Conv2D(1→32, 3×3, same) + ReLU
 *   Conv2D(32→64, 3×3, same) + ReLU
 *   Conv2D(64→32, 3×3, same) + ReLU
 *   GlobalAveragePooling2D
 *   Dense(32→64) + ReLU
 *   Dense(64→1)
 *
 * Input:  [N, 20, 10, 1] float32 — binary board occupancy (NHWC)
 * Output: [N, 1] float32 — scalar score per board (higher = better)
 */

import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import * as tfl from '@tensorflow/tfjs-layers';
import type { CellValue } from './types';

const BOARD_ROWS = 20;
const BOARD_COLS = 10;

// Weight layout in tetris_eval_weights.bin (float32, little-endian):
const WEIGHT_SPECS: { shape: number[] }[] = [
  { shape: [3, 3, 1,  32] }, // conv1 kernel  [H,W,in,out]
  { shape: [32] },            // conv1 bias
  { shape: [3, 3, 32, 64] }, // conv2 kernel
  { shape: [64] },            // conv2 bias
  { shape: [3, 3, 64, 32] }, // conv3 kernel
  { shape: [32] },            // conv3 bias
  { shape: [32, 64] },        // dense1 kernel [in,out]
  { shape: [64] },            // dense1 bias
  { shape: [64,  1] },        // dense2 kernel
  { shape: [1] },             // dense2 bias
];

let model: tfl.LayersModel | null = null;

function buildModel(): tfl.Sequential {
  return tfl.sequential({
    layers: [
      tfl.layers.conv2d({
        inputShape: [BOARD_ROWS, BOARD_COLS, 1],
        filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', useBias: true,
      }),
      tfl.layers.conv2d({
        filters: 64, kernelSize: 3, padding: 'same', activation: 'relu', useBias: true,
      }),
      tfl.layers.conv2d({
        filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', useBias: true,
      }),
      tfl.layers.globalAveragePooling2d({}),
      tfl.layers.dense({ units: 64, activation: 'relu' }),
      tfl.layers.dense({ units: 1 }),
    ],
  });
}

export async function loadCnnModel(): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();

  const resp = await fetch('/models/tetris_eval_weights.bin');
  if (!resp.ok) throw new Error(`Failed to fetch weights: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const all = new Float32Array(buf);

  const seq = buildModel();

  // Slice the flat buffer into per-layer tensors
  let offset = 0;
  const tensors: tf.Tensor[] = [];
  for (const { shape } of WEIGHT_SPECS) {
    const count = shape.reduce((a, b) => a * b, 1);
    tensors.push(tf.tensor(all.slice(offset, offset + count), shape));
    offset += count;
  }

  // layers: [0]=conv1, [1]=conv2, [2]=conv3, [3]=globalAvgPool, [4]=dense1, [5]=dense2
  seq.layers[0].setWeights([tensors[0], tensors[1]]);
  seq.layers[1].setWeights([tensors[2], tensors[3]]);
  seq.layers[2].setWeights([tensors[4], tensors[5]]);
  seq.layers[4].setWeights([tensors[6], tensors[7]]);
  seq.layers[5].setWeights([tensors[8], tensors[9]]);

  // Warm up with a dummy pass to JIT-compile kernels
  const dummy = tf.zeros([1, BOARD_ROWS, BOARD_COLS, 1]);
  (seq.predict(dummy) as tf.Tensor).dispose();
  dummy.dispose();

  model = seq;
}

export function isCnnReady(): boolean {
  return model !== null;
}

/**
 * Score N board afterstates in one forward pass.
 * Returns a Float32Array of length N (higher = better position).
 */
export async function evaluateBoardsBatch(boards: CellValue[][][]): Promise<Float32Array> {
  if (!model) throw new Error('CNN model not loaded');
  const N = boards.length;

  // Build NHWC input tensor
  const input = new Float32Array(N * BOARD_ROWS * BOARD_COLS);
  for (let i = 0; i < N; i++) {
    const base = i * BOARD_ROWS * BOARD_COLS;
    for (let r = 0; r < BOARD_ROWS; r++) {
      for (let c = 0; c < BOARD_COLS; c++) {
        input[base + r * BOARD_COLS + c] = boards[i][r][c] !== 0 ? 1.0 : 0.0;
      }
    }
  }

  const inputTensor = tf.tensor4d(input, [N, BOARD_ROWS, BOARD_COLS, 1]);
  const outputTensor = model.predict(inputTensor) as tf.Tensor;
  const scores = outputTensor.dataSync() as Float32Array;
  inputTensor.dispose();
  outputTensor.dispose();

  return scores.slice();
}
