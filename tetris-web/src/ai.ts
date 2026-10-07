import { ActivePiece, CellValue, PieceType } from './types';
import type { EngineMove, EngineRequest, EngineAnalysis } from './engine';
import { BOARD_COLS, BOARD_ROWS } from './board';
import { getRotation, getKicks } from './pieces';
import { Bag } from './bag';
import type { BotBoard } from './versus';
import {
  SpinKind, SPIN_NONE, SPIN_FULL, T_CORNERS, T_FRONT_CORNERS, classifySpin, isTstKick, resolveClear,
} from './rules';

const BEAM_WIDTH = 20;
const SEARCH_DEPTH = 4;

export type AiDifficulty = 'easy' | 'medium' | 'hard' | 'experimental';
// valueNet: the value-net bot (valueBot.ts) with these search widths instead of the beam search.
export const AI_DIFFICULTY_PARAMS: Record<AiDifficulty, { beamWidth: number; searchDepth: number; advancedEval: boolean; valueNet?: number[]; label: string; subtitle: string }> = {
  easy:         { beamWidth: 1,  searchDepth: 1, advancedEval: false,                  label: 'EASY',         subtitle: 'greedy one-piece' },
  medium:       { beamWidth: 20, searchDepth: 4, advancedEval: false,                  label: 'MEDIUM',       subtitle: 'beam search' },
  hard:         { beamWidth: 32, searchDepth: 5, advancedEval: true,                   label: 'HARD',         subtitle: 'beam search+' },
  experimental: { beamWidth: 0,  searchDepth: 0, advancedEval: false, valueNet: [3, 2], label: 'EXPERIMENTAL', subtitle: 'value net' },
};

// Module-level bag used only for beam search simulation.
// Always restored from a node's bagState before use.
const searchBag = new Bag();

// Well column persistence — survives between findBestMoveHard calls in the Web Worker.
// Undefined until the first call; reset to undefined when a new game starts (empty board).
let savedWellCol: number | undefined;
const WELL_HYSTERESIS = 2; // min rows lower a new column must be before we switch wells

// ---- BFS-based move generator ----

interface PlacementResult {
  rotationIndex: number;
  x: number;
  y: number;
  spin: SpinKind; // best spin available by rotating into this position as the last action
}

// Pre-allocated BFS buffers — eliminates per-call Set<string> and Map<string,…>
// allocation, and the O(n) queue.shift().
//
// State space: 4 rotations × (BOARD_ROWS + Y_OFF) rows × (BOARD_COLS + 2*X_OFF) cols
// SRS kicks can shift a piece ±2 in x and ±2 in y relative to spawn, so we add
// generous offsets on each axis.
const BFS_X_OFF  = 4;
const BFS_Y_OFF  = 4;
const BFS_X_SIZE = BOARD_COLS + BFS_X_OFF * 2; // 18
const BFS_Y_SIZE = BOARD_ROWS + BFS_Y_OFF;     // 24
const BFS_SLOT   = BFS_X_SIZE * BFS_Y_SIZE;    // 432 per rotation
// Uint32 generation counters: stamp[idx] === gen means "visited this call".
// No fill() needed between calls — just increment the generation.
const bfsVisit  = new Uint32Array(4 * BFS_SLOT);
const bfsGround = new Uint32Array(4 * BFS_SLOT);
let bfsGen = 0;
// Pre-allocated queue buffer: each entry encodes (rot, x, y) as one Uint16.
// Encoding: (rot << 10) | ((y + BFS_Y_OFF) << 5) | (x + BFS_X_OFF)
// Max reachable states = 4 × BFS_SLOT = 1728; buffer is exactly that size.
const bfsQueue = new Uint16Array(4 * BFS_SLOT);
// Best spin reachable by rotating into each state this call (valid where bfsSpinGen === gen).
const bfsSpin    = new Uint8Array(4 * BFS_SLOT);
const bfsSpinGen = new Uint32Array(4 * BFS_SLOT);

function bfsIdx(rot: number, x: number, y: number): number {
  return rot * BFS_SLOT + (y + BFS_Y_OFF) * BFS_X_SIZE + (x + BFS_X_OFF);
}

// ---- Bitmask board representation ----
//
// A board is a Uint16Array of BOARD_ROWS entries.  Bit c of row r = column c filled.
// Full row mask = (1 << BOARD_COLS) - 1 = 0x3FF.
// Benefits over CellValue[][]:
//   • Board copy: Uint16Array.slice() = 40-byte memcpy vs pointer-chasing 200-cell copy
//   • Line-clear detection: row === BM_FULL_ROW (one comparison)
//   • Row-transition count: O(1) per row with popcount + XOR trick
//   • cache locality: 40 contiguous bytes vs 20 heap pointers + 200 cells

const BM_FULL_ROW = (1 << BOARD_COLS) - 1; // 0x3FF

// O(1) popcount for values 0–1023 (10 bits).
const POPCOUNT_LUT = new Uint8Array(1024);
for (let i = 1; i < 1024; i++) POPCOUNT_LUT[i] = POPCOUNT_LUT[i >> 1] + (i & 1);

function popcount10(x: number): number {
  return POPCOUNT_LUT[x & 0x3FF];
}

// Per-(type, rotation): row bitmasks and dimensions.
const ALL_PIECE_TYPES: PieceType[] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
interface PieceBm { masks: number[]; width: number; height: number; }
const PIECE_BM: Record<PieceType, PieceBm[]> = {} as Record<PieceType, PieceBm[]>;
for (const type of ALL_PIECE_TYPES) {
  PIECE_BM[type] = [];
  for (let rot = 0; rot < 4; rot++) {
    const matrix = getRotation(type, rot);
    const h = matrix.length;
    const w = matrix[0].length;
    const masks: number[] = new Array(h).fill(0);
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        if (matrix[r][c]) masks[r] |= (1 << c);
      }
    }
    PIECE_BM[type].push({ masks, width: w, height: h });
  }
}

// Convert a CellValue[][] board to a bitmask board.  Called once at search entry.
function cellBoardToBm(board: CellValue[][]): Uint16Array {
  const bm = new Uint16Array(BOARD_ROWS);
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      if (board[r][c] !== 0) bm[r] |= (1 << c);
    }
  }
  return bm;
}

function bmCollides(bm: Uint16Array, piece: ActivePiece, dx: number, dy: number): boolean {
  const nx = piece.x + dx;
  const ny = piece.y + dy;
  const { masks, height } = PIECE_BM[piece.type][piece.rotationIndex];
  for (let r = 0; r < height; r++) {
    if (!masks[r]) continue;
    const br = ny + r;
    if (br >= BOARD_ROWS) return true;
    // Walls are checked even above the visible board (br < 0): a kick can lift a
    // piece above row 0, and it must not slide past the walls from there.
    let shifted: number;
    if (nx >= 0) {
      shifted = masks[r] << nx;
    } else {
      const nabs = -nx;
      // Any filled bits that would fall off the left edge = wall collision.
      if (masks[r] & ((1 << nabs) - 1)) return true;
      shifted = masks[r] >> nabs;
    }
    if (shifted & ~BM_FULL_ROW) return true; // bits beyond right wall
    if (br >= 0 && (shifted & bm[br])) return true; // board cell occupied
  }
  return false;
}

function bmHardDropY(bm: Uint16Array, piece: ActivePiece): number {
  let dy = 0;
  while (!bmCollides(bm, piece, 0, dy + 1)) dy++;
  return piece.y + dy;
}

function bmLockAndClear(bm: Uint16Array, piece: ActivePiece): { bm: Uint16Array; linesCleared: number } {
  const result = bm.slice();
  const { masks, height } = PIECE_BM[piece.type][piece.rotationIndex];
  for (let r = 0; r < height; r++) {
    if (!masks[r]) continue;
    const br = piece.y + r;
    if (br < 0 || br >= BOARD_ROWS) continue;
    result[br] |= piece.x >= 0 ? (masks[r] << piece.x) : (masks[r] >> (-piece.x));
  }
  // Compact: remove full rows (scan bottom-up, copy surviving rows down).
  let linesCleared = 0;
  let writeRow = BOARD_ROWS - 1;
  const compact = new Uint16Array(BOARD_ROWS);
  for (let r = BOARD_ROWS - 1; r >= 0; r--) {
    if (result[r] !== BM_FULL_ROW) {
      compact[writeRow--] = result[r];
    } else {
      linesCleared++;
    }
  }
  return { bm: compact, linesCleared };
}

// Index of the SRS kick test the last successful bmAttemptRotation used.
let bmLastKickIndex = 0;

function bmAttemptRotation(bm: Uint16Array, piece: ActivePiece, delta: number): ActivePiece | null {
  const newIndex = ((piece.rotationIndex + delta) % 4 + 4) % 4;
  const kickList = getKicks(piece.type, piece.rotationIndex, newIndex);
  for (let k = 0; k < kickList.length; k++) {
    const [kdx, kdy] = kickList[k];
    const candidate: ActivePiece = { ...piece, rotationIndex: newIndex, x: piece.x + kdx, y: piece.y + kdy };
    if (!bmCollides(bm, candidate, 0, 0)) { bmLastKickIndex = k; return candidate; }
  }
  return null;
}

function bmCountTSpinCorners(bm: Uint16Array, piece: ActivePiece): number {
  // Same four corners as board.ts countTSpinCorners (top-left of 3×3 bounding box)
  let count = 0;
  for (const [r, c] of [
    [piece.y, piece.x], [piece.y, piece.x + 2],
    [piece.y + 2, piece.x], [piece.y + 2, piece.x + 2],
  ] as [number, number][]) {
    if (r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS || (bm[r] & (1 << c)) !== 0) count++;
  }
  return count;
}

function bmCornerFilled(bm: Uint16Array, r: number, c: number): boolean {
  return r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS || (bm[r] & (1 << c)) !== 0;
}

// Spin kind for a grounded piece that just rotated into place (see rules.ts).
function bmSpinAfterRotation(bm: Uint16Array, piece: ActivePiece, tstKick: boolean): SpinKind {
  let corners = 0;
  let front = 0;
  if (piece.type === 'T') {
    const [f0, f1] = T_FRONT_CORNERS[piece.rotationIndex];
    for (let i = 0; i < 4; i++) {
      if (bmCornerFilled(bm, piece.y + T_CORNERS[i][0], piece.x + T_CORNERS[i][1])) {
        corners++;
        if (i === f0 || i === f1) front++;
      }
    }
  }
  const immobile = bmCollides(bm, piece, -1, 0) && bmCollides(bm, piece, 1, 0)
    && bmCollides(bm, piece, 0, -1) && bmCollides(bm, piece, 0, 1);
  return classifySpin(piece.type, true, corners, front, immobile, tstKick);
}

// BFS over all reachable (rotationIndex, x, y) positions, starting from the spawn
// position (rotation 0, centered, y=0). All 6 actions are explored: move left/right,
// drop one row, and rotate CW/CCW/180° with SRS wall kicks. This correctly discovers
// placements only reachable by rotating under a roof — the key for T-spins.
//
// The queue stores states as Uint16 integers (no per-entry object allocation).
// A single mutable ActivePiece `cur` is reused for collision/rotation checks.
function findReachablePlacements(bm: Uint16Array, pieceType: PieceType): PlacementResult[] {
  const spawnWidth = getRotation(pieceType, 0)[0].length;
  const startX = Math.floor((BOARD_COLS - spawnWidth) / 2);

  const cur: ActivePiece = { type: pieceType, rotationIndex: 0, x: startX, y: 0 };

  if (bmCollides(bm, cur, 0, 0)) return [];

  const gen = ++bfsGen;
  let qHead = 0;
  let qTail = 0;
  const grounded: PlacementResult[] = [];
  const groundedIdx: number[] = [];

  bfsVisit[bfsIdx(0, startX, 0)] = gen;
  bfsQueue[qTail++] = (0 << 10) | ((0 + BFS_Y_OFF) << 5) | (startX + BFS_X_OFF);

  while (qHead < qTail) {
    const enc = bfsQueue[qHead++];
    cur.rotationIndex = enc >> 10;
    cur.y = ((enc >> 5) & 31) - BFS_Y_OFF;
    cur.x = (enc & 31) - BFS_X_OFF;

    if (bmCollides(bm, cur, 0, 1)) {
      const gk = bfsIdx(cur.rotationIndex, cur.x, cur.y);
      if (bfsGround[gk] !== gen) {
        bfsGround[gk] = gen;
        grounded.push({ rotationIndex: cur.rotationIndex, x: cur.x, y: cur.y, spin: SPIN_NONE });
        groundedIdx.push(gk);
      }
    }

    if (!bmCollides(bm, cur, -1, 0)) {
      const nk = bfsIdx(cur.rotationIndex, cur.x - 1, cur.y);
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        bfsQueue[qTail++] = (cur.rotationIndex << 10) | ((cur.y + BFS_Y_OFF) << 5) | (cur.x - 1 + BFS_X_OFF);
      }
    }
    if (!bmCollides(bm, cur, 1, 0)) {
      const nk = bfsIdx(cur.rotationIndex, cur.x + 1, cur.y);
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        bfsQueue[qTail++] = (cur.rotationIndex << 10) | ((cur.y + BFS_Y_OFF) << 5) | (cur.x + 1 + BFS_X_OFF);
      }
    }
    if (!bmCollides(bm, cur, 0, 1)) {
      const nk = bfsIdx(cur.rotationIndex, cur.x, cur.y + 1);
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        bfsQueue[qTail++] = (cur.rotationIndex << 10) | ((cur.y + 1 + BFS_Y_OFF) << 5) | (cur.x + BFS_X_OFF);
      }
    }
    for (const delta of [1, -1, 2] as const) {
      const next = bmAttemptRotation(bm, cur, delta);
      if (!next) continue;
      // Repeated upward kicks on a tall stack can lift a piece above what the
      // queue encoding holds; a negative y field would corrupt the rotation bits.
      if (next.y < -BFS_Y_OFF) continue;
      const nk = bfsIdx(next.rotationIndex, next.x, next.y);
      // Locking right after this rotation may score a spin. Record the best one per
      // state, even if the state was already reached another way.
      if (bmCollides(bm, next, 0, 1)) {
        const spin = bmSpinAfterRotation(bm, next, isTstKick(pieceType, delta, bmLastKickIndex, next.rotationIndex));
        if (bfsSpinGen[nk] !== gen) { bfsSpinGen[nk] = gen; bfsSpin[nk] = spin; }
        else if (spin > bfsSpin[nk]) bfsSpin[nk] = spin;
      }
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        bfsQueue[qTail++] = (next.rotationIndex << 10) | ((next.y + BFS_Y_OFF) << 5) | (next.x + BFS_X_OFF);
      }
    }
  }

  // Rotations into a grounded state can be found after the state itself was
  // dequeued, so spins are filled in once the search is complete.
  for (let i = 0; i < grounded.length; i++) {
    const gk = groundedIdx[i];
    if (bfsSpinGen[gk] === gen) grounded[i].spin = bfsSpin[gk] as SpinKind;
  }
  return grounded;
}

// Spin a bot earns for a placement: the best available by rotating into it last.
// Bots send only a final position, so this credits them with the best legal
// finesse, as a player could. Unreachable positions score no spin.
export function placementSpin(board: CellValue[][], piece: ActivePiece): SpinKind {
  const bm = cellBoardToBm(board);
  for (const p of findReachablePlacements(bm, piece.type)) {
    if (p.rotationIndex === piece.rotationIndex && p.x === piece.x && p.y === piece.y) return p.spin;
  }
  return SPIN_NONE;
}

// Move generation and bitmask board helpers for other searchers (valueBot.ts).
export { cellBoardToBm, bmCollides, bmLockAndClear, findReachablePlacements };
export type { PlacementResult };

// ---- Evaluation weights ----
// All tunable scoring constants in one place.

const W = {
  // Height — two-tier danger so penalty escalates as the board fills up
  aggHeightBase:    4,      // rows — no height penalty below this average column height
  aggHeight:        -0.03,  // per aggregate-height unit above aggHeightBase
  dangerThreshold:  8,      // first danger tier: mild slope starts here
  dangerSlope:      4.0,    // penalty per row above dangerThreshold (up to dangerThreshold2)
  dangerThreshold2: 16,     // second danger tier: steep slope starts here
  dangerSlope2:     12.0,   // penalty per row above dangerThreshold2

  // Emergency downstacking — kicks in as height approaches the ceiling.
  // heightFactor = (maxHeight − emergencyHeight) / (BOARD_ROWS − emergencyHeight), clamped 0→1.
  // Used to smoothly override the normal "avoid singles" bias when survival matters more.
  emergencyHeight:      8,  // where the emergency scale starts (0 at or below, 1 at BOARD_ROWS)
  emergencyLineClear:   8.0,// bonus per line cleared × heightFactor — turns singles into rewards when tall
  emergencyHoleScale:   1.5,// holes × (1 + heightFactor×scale) — buried holes hurt more when tall
  emergencyWellScale:   1.5,// well depth bonus × (1 + heightFactor×scale) — keep the lane open

  // Board shape
  holes:            -0.80,  // per covered empty cell
  bumpiness:        -0.28,  // per unit of adjacent height-diff (non-well, non-cliff)
  cliffThreshold:   3,      // diffs above this get an extra steep penalty (non-well only)
  cliffPenalty:     -1.00,  // extra penalty per unit of diff exceeding cliffThreshold
  wellDepthMin:     3,      // well column must be at least this many rows below both neighbours

  // Dellacherie features
  rowTransitions:   -0.25,
  colTransitions:   -0.28,
  cumulativeWells:  -0.30,  // triangular-number penalty for wells NOT at targetWellCol
  wellDepthBonus:   0.50,   // reward per row of depth of the well at targetWellCol (capped at 8)
  landingHeight:    -0.06,  // penalty per row above bottom where piece landed

  // Line clears — base shape incentive (emergency bonus layered on top when tall)
  lineClear:        [0, -0.8, -0.1, 0.4, 1.8] as number[],

  // T-spin
  tspinClearBonus:  1.5,
  wastedTPenalty:   -2.5,
  tslotBonus:       2.0,
  tetrisReadiness:  0.80, // reward per ready Tetris row (0→4 scale, 4 = four fully-open rows)

  // Back-to-back state
  b2bBonus:         0.50,  // flat reward for having B2B active after this placement
  b2bLevel:         0.0,   // extra reward per B2B level (surge charges from level 4)
  b2bLevelCap:      10,    // levels beyond this add nothing

  // Combo awareness
  comboBonus:       0.35,  // reward per combo level (combo+1 when active, 0 when broken)

  // Combo potential
  comboPotential:   0.20,  // reward per nearly-full row (scaled by fill fraction above threshold)

  // Perfect clear
  perfectClear:     30.0,

  // Garbage pressure
  garbageValue:     2.0,
  garbageClearBonus: 0.4,
  garbageUrgency:   -0.3,
};

// Override hard-mode weights, for tuning experiments (the self-play harness's
// --weights). Unknown keys are rejected so a typo can't silently do nothing.
export function setWeights(overrides: Record<string, number | number[]>): void {
  for (const [k, v] of Object.entries(overrides)) {
    if (!(k in W)) throw new Error(`Unknown weight: ${k}`);
    (W as Record<string, unknown>)[k] = v;
  }
}


function initHeightArray(): number[] {
  return new Array<number>(BOARD_COLS).fill(0);
}

// Choose a well column that the AI will commit to for the entire search.
// If prevWellCol is supplied (from savedWellCol), sticks with it unless another
// column is at least WELL_HYSTERESIS rows lower — preventing the well from
// drifting every move as heights equalize. Edge columns (9, then 0) win ties
// on a fresh pick so an empty board defaults to the right edge.
function pickTargetWellCol(bm: Uint16Array, prevWellCol?: number): number {
  const heights = computeColumnHeightsBm(bm);
  const minH = Math.min(...heights);
  if (prevWellCol !== undefined && heights[prevWellCol] <= minH + WELL_HYSTERESIS) {
    return prevWellCol;
  }
  if (heights[BOARD_COLS - 1] === minH) return BOARD_COLS - 1;
  if (heights[0] === minH) return 0;
  return heights.indexOf(minH);
}

// ---- Board evaluation (Hard AI) ----

// Score TSD readiness for each valid T-spin slot on the board.
// For each T-piece rotation and straight-drop position where the 3-corner rule
// is satisfied, measure what fraction of the non-T-piece cells in those rows are
// already filled. A fully-ready slot (all surrounding cells filled) scores 1.0;
// an empty slot scores near 0. This creates a gradient that rewards both creating
// the slot geometry AND filling the rows needed to actually fire the T-spin.
// Returns total readiness across all slots, capped at 2.
function scoreTSpinReadiness(bm: Uint16Array): number {
  let total = 0;

  for (let rotIdx = 0; rotIdx < 4; rotIdx++) {
    const { masks, width: pieceW, height: pieceH } = PIECE_BM['T'][rotIdx];

    for (let x = 0; x <= BOARD_COLS - pieceW; x++) {
      const piece: ActivePiece = { type: 'T', rotationIndex: rotIdx, x, y: 0 };
      if (bmCollides(bm, piece, 0, 0)) continue;

      const y = bmHardDropY(bm, piece);
      const landed: ActivePiece = { ...piece, y };

      if (bmCountTSpinCorners(bm, landed) < 3) continue;

      let nonPieceCells = 0;
      let filledCells = 0;
      for (let pr = 0; pr < pieceH; pr++) {
        const row = y + pr;
        if (row < 0 || row >= BOARD_ROWS) continue;
        const pieceMask = (masks[pr] << x) & BM_FULL_ROW;
        const pieceCount = popcount10(pieceMask);
        nonPieceCells += BOARD_COLS - pieceCount;
        filledCells += popcount10(bm[row] & ~pieceMask);
      }

      if (nonPieceCells === 0) continue;
      total += filledCells / nonPieceCells;
    }
  }

  return Math.min(total, 2);
}

function scoreTetrisReadiness(bm: Uint16Array, targetWellCol: number): number {
  const wellBit = 1 << targetWellCol;
  let total = 0;
  let rowsChecked = 0;

  for (let r = BOARD_ROWS - 1; r >= 0 && rowsChecked < 4; r--) {
    if (bm[r] & wellBit) break; // well column blocked — contiguous streak ends here
    total += popcount10(bm[r]) / (BOARD_COLS - 1);
    rowsChecked++;
  }

  return total;
}

// Scan each column top-down for the first filled cell; O(BOARD_COLS × BOARD_ROWS).
// Used at search roots and after line clears where heights can't be updated incrementally.
function computeColumnHeightsBm(bm: Uint16Array): number[] {
  const heights = new Array<number>(BOARD_COLS).fill(0);
  for (let c = 0; c < BOARD_COLS; c++) {
    const colBit = 1 << c;
    for (let r = 0; r < BOARD_ROWS; r++) {
      if (bm[r] & colBit) { heights[c] = BOARD_ROWS - r; break; }
    }
  }
  return heights;
}

// Advance a fixed-length piece queue by one: drop the front, append newPiece.
// Avoids the two-allocation slice()+spread pattern used previously.
function queueAdvance(q: PieceType[], newPiece: PieceType): PieceType[] {
  const n = q.length;
  const next = new Array<PieceType>(n);
  for (let i = 0; i < n - 1; i++) next[i] = q[i + 1];
  next[n - 1] = newPiece;
  return next;
}

// Advance a queue by two: drop the front two, append newPiece1 then newPiece2.
function queueAdvance2(q: PieceType[], newPiece1: PieceType, newPiece2: PieceType): PieceType[] {
  const n = q.length;
  const next = new Array<PieceType>(n);
  for (let i = 0; i < n - 2; i++) next[i] = q[i + 2];
  next[n - 2] = newPiece1;
  next[n - 1] = newPiece2;
  return next;
}

// ---- Medium AI (Easy + Medium difficulty) ----
// Original beam search — cumulative scoring, dynamic well column, no combo/b2b tracking.

const W_MEDIUM = {
  aggHeightBase:    4,
  aggHeight:        -0.03,
  dangerThreshold:  18,
  dangerSlope:      4.0,
  holes:            -1.00,
  bumpiness:        -0.40,
  wellBumpiness:    -0.05,
  cliffThreshold:   3,
  cliffPenalty:     -1.20,
  lineClear:        [0, -1.2, 0.5, 1.8, 5.0],
  tspinClearBonus:  3.5,
  wastedTPenalty:   -2.5,
  tslotBonus:       2.0,
  garbageClearBonus: 0.4,
  garbageUrgency:   -0.3,
};

function evaluateBoardMedium(
  bm: Uint16Array,
  linesCleared: number,
  spin: SpinKind = SPIN_NONE,
  pendingGarbage: number = 0,
  placedType?: PieceType,
): number {
  const heights = initHeightArray();
  let holes = 0;

  for (let c = 0; c < BOARD_COLS; c++) {
    const colBit = 1 << c;
    let foundTop = false;
    for (let r = 0; r < BOARD_ROWS; r++) {
      if (bm[r] & colBit) {
        if (!foundTop) { heights[c] = BOARD_ROWS - r; foundTop = true; }
      } else if (foundTop) {
        holes++;
      }
    }
  }

  const aggHeight = heights.reduce((a, b) => a + b, 0);
  const maxHeight = Math.max(...heights);

  const wellCol = heights.indexOf(Math.min(...heights));
  let bumpiness = 0;
  let wellBumpiness = 0;
  let cliffPenalty = 0;
  for (let c = 0; c < BOARD_COLS - 1; c++) {
    const diff = Math.abs(heights[c] - heights[c + 1]);
    if (c === wellCol || c + 1 === wellCol) {
      wellBumpiness += diff;
    } else {
      bumpiness += diff;
      if (diff > W_MEDIUM.cliffThreshold) cliffPenalty += diff - W_MEDIUM.cliffThreshold;
    }
  }
  if (wellCol > 0 && wellCol < BOARD_COLS - 1) {
    const crossDiff = Math.abs(heights[wellCol - 1] - heights[wellCol + 1]);
    bumpiness += crossDiff;
    if (crossDiff > W_MEDIUM.cliffThreshold) cliffPenalty += crossDiff - W_MEDIUM.cliffThreshold;
  }

  const lineClearScore = W_MEDIUM.lineClear[Math.min(linesCleared, 4)];
  const dangerPenalty  = Math.max(0, maxHeight - W_MEDIUM.dangerThreshold) * W_MEDIUM.dangerSlope;
  const penalisedHeight = Math.max(0, aggHeight - W_MEDIUM.aggHeightBase * BOARD_COLS);

  let score = W_MEDIUM.aggHeight     * penalisedHeight
            + lineClearScore
            + W_MEDIUM.holes         * holes
            + W_MEDIUM.bumpiness     * bumpiness
            + W_MEDIUM.wellBumpiness * wellBumpiness
            + W_MEDIUM.cliffPenalty  * cliffPenalty
            - dangerPenalty;

  if (placedType === 'T' && spin === SPIN_FULL && linesCleared > 0) score += W_MEDIUM.tspinClearBonus * linesCleared;
  if (placedType === 'T' && spin === SPIN_NONE) score += W_MEDIUM.wastedTPenalty;
  score += scoreTSpinReadiness(bm) * W_MEDIUM.tslotBonus;

  if (pendingGarbage > 0) {
    score += W_MEDIUM.garbageClearBonus * linesCleared * pendingGarbage;
    score += W_MEDIUM.garbageUrgency    * pendingGarbage;
  }

  return score;
}

interface BeamNodeMedium {
  board: Uint16Array;
  activeType: PieceType;
  nextQueue: PieceType[];
  hold: PieceType | null;
  holdUsed: boolean;
  bagState: PieceType[];
  score: number;
  firstMove: { rotationIndex: number; x: number; y: number; useHold: boolean } | null;
}

function expandBeamNodeMedium(node: BeamNodeMedium, pendingGarbage: number): BeamNodeMedium[] {
  const successors: BeamNodeMedium[] = [];

  type PlayOption = {
    pieceType: PieceType;
    useHold: boolean;
    nextActiveType: PieceType;
    nextQueue: PieceType[];
    nextHold: PieceType | null;
    nextBagState: PieceType[];
  };

  const options: PlayOption[] = [];

  {
    searchBag.restoreState(node.bagState);
    const newPiece = searchBag.next();
    options.push({
      pieceType: node.activeType,
      useHold: false,
      nextActiveType: node.nextQueue[0],
      nextQueue: queueAdvance(node.nextQueue, newPiece),
      nextHold: node.hold,
      nextBagState: searchBag.getState(),
    });
  }

  if (!node.holdUsed) {
    if (node.hold !== null) {
      searchBag.restoreState(node.bagState);
      const newPiece = searchBag.next();
      options.push({
        pieceType: node.hold,
        useHold: true,
        nextActiveType: node.nextQueue[0],
        nextQueue: queueAdvance(node.nextQueue, newPiece),
        nextHold: node.activeType,
        nextBagState: searchBag.getState(),
      });
    } else if (node.nextQueue.length >= 2) {
      searchBag.restoreState(node.bagState);
      const newPiece1 = searchBag.next();
      const newPiece2 = searchBag.next();
      options.push({
        pieceType: node.nextQueue[0],
        useHold: true,
        nextActiveType: node.nextQueue[1],
        nextQueue: queueAdvance2(node.nextQueue, newPiece1, newPiece2),
        nextHold: node.activeType,
        nextBagState: searchBag.getState(),
      });
    }
  }

  for (const opt of options) {
    const placements = findReachablePlacements(node.board, opt.pieceType);

    for (const placement of placements) {
      const piece: ActivePiece = {
        type: opt.pieceType,
        rotationIndex: placement.rotationIndex,
        x: placement.x,
        y: placement.y,
      };
      const { bm: clearedBoard, linesCleared } = bmLockAndClear(node.board, piece);
      const moveScore = evaluateBoardMedium(
        clearedBoard, linesCleared, placement.spin, pendingGarbage, opt.pieceType,
      );

      const firstMove = node.firstMove ?? {
        rotationIndex: placement.rotationIndex,
        x: placement.x,
        y: placement.y,
        useHold: opt.useHold,
      };

      successors.push({
        board: clearedBoard,
        activeType: opt.nextActiveType,
        nextQueue: opt.nextQueue,
        hold: opt.nextHold,
        holdUsed: false,
        bagState: opt.nextBagState,
        score: node.score + moveScore,
        firstMove,
      });
    }
  }

  return successors;
}

// ---- Partial sort (min-heap topK) ----
// Returns up to k elements with the highest value (best first), in O(n log k).
// Replaces sort+slice throughout the beam search — avoids sorting all n successors
// when only the top beamWidth (≤32) are needed.

function heapSiftDown<T>(heap: T[], i: number, valueFn: (t: T) => number): void {
  const n = heap.length;
  while (true) {
    let min = i;
    const l = 2 * i + 1, r = 2 * i + 2;
    if (l < n && valueFn(heap[l]) < valueFn(heap[min])) min = l;
    if (r < n && valueFn(heap[r]) < valueFn(heap[min])) min = r;
    if (min === i) break;
    const tmp = heap[i]; heap[i] = heap[min]; heap[min] = tmp;
    i = min;
  }
}

function topKDescending<T>(arr: T[], k: number, valueFn: (t: T) => number): T[] {
  if (arr.length <= k) return arr.sort((a, b) => valueFn(b) - valueFn(a));
  // Build a min-heap of the first k elements (heap[0] = worst kept so far).
  const heap = arr.slice(0, k);
  for (let i = Math.floor(k / 2) - 1; i >= 0; i--) heapSiftDown(heap, i, valueFn);
  // Scan the rest: replace heap root when a better element is found.
  for (let i = k; i < arr.length; i++) {
    if (valueFn(arr[i]) > valueFn(heap[0])) {
      heap[0] = arr[i];
      heapSiftDown(heap, 0, valueFn);
    }
  }
  return heap.sort((a, b) => valueFn(b) - valueFn(a));
}

export function findBestMove(
  bot: BotBoard,
  pendingGarbage: number = 0,
  beamWidth: number = BEAM_WIDTH,
  searchDepth: number = SEARCH_DEPTH,
  _combo: number = -1,       // unused by medium eval — included for API consistency with findBestMoveHard
  _b2b: number = -1,
): { rotationIndex: number; x: number; y: number; useHold: boolean } {
  const rootBm = cellBoardToBm(bot.board);
  let beam: BeamNodeMedium[] = [{
    board: rootBm,
    activeType: bot.active.type,
    nextQueue: [...bot.nextQueue],
    hold: bot.hold,
    holdUsed: bot.holdUsed,
    bagState: bot.bagState,
    score: 0,
    firstMove: null,
  }];

  for (let d = 0; d < searchDepth; d++) {
    const nextBeam: BeamNodeMedium[] = [];
    for (const node of beam) {
      nextBeam.push(...expandBeamNodeMedium(node, pendingGarbage));
    }
    if (nextBeam.length === 0) break;
    beam = topKDescending(nextBeam, beamWidth, n => n.score);
  }

  return beam[0]?.firstMove ?? { rotationIndex: 0, x: 0, y: 0, useHold: false };
}

// ---- Engine analysis ----
// Same beam search as findBestMoveHard, but:
//   • movePath tracking is enabled (root initialized with movePath: [])
//   • Returns top N distinct first-move lines instead of a single best move
//   • Intended for human-facing analysis (creative pause, post-match review, etc.)

export function analyzePositionHard(request: EngineRequest): EngineAnalysis {
  const t0 = performance.now();
  const { board, activeType, nextQueue, hold, holdUsed, bagState,
          combo, b2b, pendingGarbage, beamWidth, searchDepth, topN } = request;

  const rootBm = cellBoardToBm(board);
  const targetWellCol = pickTargetWellCol(rootBm);

  let beam: BeamNodeHard[] = [{
    board: rootBm,
    activeType,
    nextQueue: [...nextQueue],
    hold,
    holdUsed,
    bagState: [...bagState],
    score: 0,
    garbageSent: 0,
    combo: combo ?? -1,
    b2b: b2b ?? -1,
    targetWellCol,
    heights: computeColumnHeightsBm(rootBm),
    firstMove: null,
    movePath: [],  // empty array enables path tracking in expandBeamNodeHardHard
  }];

  for (let d = 0; d < searchDepth; d++) {
    const nextBeam: BeamNodeHard[] = [];
    for (const node of beam) {
      nextBeam.push(...expandBeamNodeHardHard(node, pendingGarbage));
    }
    if (nextBeam.length === 0) break;
    beam = topKDescending(nextBeam, beamWidth, n => n.score + n.garbageSent * W.garbageValue);
  }

  // Deduplicate by first move — keep the best-scoring terminal node per unique first placement.
  const seen = new Map<string, BeamNodeHard>();
  for (const node of beam) {
    if (!node.firstMove || !node.movePath) continue;
    const key = `${node.firstMove.rotationIndex},${node.firstMove.x},${node.firstMove.y},${node.firstMove.useHold ? 1 : 0}`;
    const cur = seen.get(key);
    const rank = node.score + node.garbageSent * W.garbageValue;
    if (!cur || rank > (cur.score + cur.garbageSent * W.garbageValue)) {
      seen.set(key, node);
    }
  }

  const lines: EngineAnalysis['lines'] = [...seen.values()]
    .sort((a, b) =>
      (b.score + b.garbageSent * W.garbageValue) -
      (a.score + a.garbageSent * W.garbageValue),
    )
    .slice(0, topN)
    .map(node => ({
      moves: node.movePath!,
      score: node.score + node.garbageSent * W.garbageValue,
      garbageSent: node.garbageSent,
    }));

  return { lines, durationMs: performance.now() - t0 };
}
// ---- Hard AI — advanced beam search ----
// Stable well column, combo/b2b tracking, perfect-clear detection, terminal-quality ranking.

// Evaluate a board state after placing a piece. Higher = better.
// targetWellCol: the column the AI commits to as its Tetris/T-spin well (fixed per search).
// landingHeight: distance from bottom where the piece landed (BOARD_ROWS - piece.y).
export function evaluateBoard(
  bm: Uint16Array,
  linesCleared: number,
  spin: SpinKind = SPIN_NONE,
  pendingGarbage: number = 0,
  placedType?: PieceType,
  targetWellCol: number = BOARD_COLS - 1,
  landingHeight: number = 0,
  tInLookahead: boolean = true,
  preHeights?: number[],
  combo: number = -1,
  b2b: number = -1,
): number {
  // Perfect clear: board is completely empty after this placement.
  let boardEmpty = true;
  for (let r = 0; r < BOARD_ROWS; r++) { if (bm[r]) { boardEmpty = false; break; } }
  if (boardEmpty) return W.perfectClear;

  // If pre-computed heights are supplied, reuse them directly — no copy needed because
  // the scan below never overwrites a column that already has a known height.
  const heights = preHeights ?? initHeightArray();
  let holes = 0;
  let colTransitions = 0;

  for (let c = 0; c < BOARD_COLS; c++) {
    const colBit = 1 << c;
    // When heights are pre-computed, start the inner scan at the first filled row of
    // this column rather than row 0, skipping the empty space above the stack.
    // For an empty column (height 0), startRow == BOARD_ROWS so the loop doesn't run.
    const startRow = preHeights
      ? (heights[c] > 0 ? BOARD_ROWS - heights[c] : BOARD_ROWS)
      : 0;
    let foundTop = preHeights !== undefined && heights[c] > 0;
    let prevFilled = false; // above startRow is always empty

    for (let r = startRow; r < BOARD_ROWS; r++) {
      const filled = (bm[r] & colBit) !== 0;
      if (filled !== prevFilled) colTransitions++;

      if (filled) {
        if (!foundTop) { heights[c] = BOARD_ROWS - r; foundTop = true; }
        prevFilled = true;
      } else {
        if (foundTop) holes++; // covered empty cell = hole
        prevFilled = false;
      }
    }
    if (!prevFilled) colTransitions++; // transition to floor (floor is always filled)
  }

  const aggHeight = heights.reduce((a, b) => a + b, 0);
  const maxHeight = Math.max(...heights);
  const stackTop = maxHeight > 0 ? BOARD_ROWS - maxHeight : BOARD_ROWS;

  // Row transitions using bitmask: O(1) per row.
  // For each row: count left-wall↔col0, adjacent cell pairs (via XOR), and col9↔right-wall.
  let rowTransitions = 0;
  for (let r = stackTop; r < BOARD_ROWS; r++) {
    const row = bm[r];
    rowTransitions += (1 - (row & 1))                        // left wall (filled) → col 0
                    + popcount10((row ^ (row >> 1)) & 0x1FF)  // inner transitions (pairs 0-1 … 8-9)
                    + (1 - ((row >> (BOARD_COLS - 1)) & 1));  // col 9 → right wall (filled)
  }

  // Cumulative wells (Dellacherie): for each column that is lower than both neighbours,
  // sum the triangular number d*(d+1)/2.  The target well column is REWARDED for depth
  // instead of penalised — the AI should actively build and maintain one deep Tetris well.
  let cumulativeWells = 0;
  let targetWellDepth = 0;
  for (let c = 0; c < BOARD_COLS; c++) {
    const leftH  = c > 0             ? heights[c - 1] : BOARD_ROWS + 4; // wall treated as very tall
    const rightH = c < BOARD_COLS - 1 ? heights[c + 1] : BOARD_ROWS + 4;
    const depth = Math.min(leftH, rightH) - heights[c];
    if (depth > 0) {
      if (c === targetWellCol) {
        targetWellDepth = depth;
        // No penalty for the target well — rewarded separately below.
      } else {
        cumulativeWells += depth * (depth + 1) / 2;
      }
    }
  }

  // Bumpiness: adjacent column height differences, excluding the well column when it is genuinely deep.
  const wellLeft  = targetWellCol > 0             ? heights[targetWellCol - 1] - heights[targetWellCol] : Infinity;
  const wellRight = targetWellCol < BOARD_COLS - 1 ? heights[targetWellCol + 1] - heights[targetWellCol] : Infinity;
  const wellIsDeep = Math.min(wellLeft, wellRight) >= W.wellDepthMin;

  let bumpiness = 0;
  let cliffPenalty = 0;
  for (let c = 0; c < BOARD_COLS - 1; c++) {
    const diff = Math.abs(heights[c] - heights[c + 1]);
    const adjacentToWell = wellIsDeep && (c === targetWellCol || c + 1 === targetWellCol);
    if (!adjacentToWell) {
      bumpiness += diff;
      if (diff > W.cliffThreshold) cliffPenalty += diff - W.cliffThreshold;
    }
  }
  // Penalise asymmetry across the well (flanking columns are never directly adjacent in the loop above).
  if (wellIsDeep && targetWellCol > 0 && targetWellCol < BOARD_COLS - 1) {
    const crossDiff = Math.abs(heights[targetWellCol - 1] - heights[targetWellCol + 1]);
    bumpiness += crossDiff;
    if (crossDiff > W.cliffThreshold) cliffPenalty += crossDiff - W.cliffThreshold;
  }

  // Two-tier danger penalty: mild slope from dangerThreshold, steep above dangerThreshold2.
  const danger1 = Math.max(0, Math.min(maxHeight, W.dangerThreshold2) - W.dangerThreshold);
  const danger2 = Math.max(0, maxHeight - W.dangerThreshold2);
  const dangerPenalty = danger1 * W.dangerSlope + danger2 * W.dangerSlope2;

  // heightFactor (0→1): how close to the ceiling the board is.
  // Used to smoothly override normal strategy in favour of survival.
  const heightFactor = Math.max(0, maxHeight - W.emergencyHeight) / (BOARD_ROWS - W.emergencyHeight);

  // Base line-clear score + emergency bonus that turns singles into rewards when tall.
  const baseClearScore = W.lineClear[Math.min(linesCleared, 4)];
  const emergencyBonus = linesCleared > 0 ? heightFactor * W.emergencyLineClear * linesCleared : 0;
  const lineClearScore = baseClearScore + emergencyBonus;

  // Holes hurt more when the stack is tall (harder to dig out, more likely to top out).
  const holePenalty    = W.holes * (1 + heightFactor * W.emergencyHoleScale);
  // Well depth becomes more valuable when tall (Tetris clears are the fastest escape route).
  const wellDepthScore = W.wellDepthBonus * Math.min(targetWellDepth, 8) * (1 + heightFactor * W.emergencyWellScale);

  const penalisedHeight = Math.max(0, aggHeight - W.aggHeightBase * BOARD_COLS);

  let score = W.aggHeight       * penalisedHeight
            + lineClearScore
            + holePenalty       * holes
            + W.bumpiness       * bumpiness
            + W.cliffPenalty    * cliffPenalty
            + W.rowTransitions  * rowTransitions
            + W.colTransitions  * colTransitions
            + W.cumulativeWells * cumulativeWells
            + wellDepthScore
            + W.landingHeight   * landingHeight
            - dangerPenalty;

  // Full T-spins get the shape bonus; minis are paid through attack and B2B instead.
  if (placedType === 'T' && spin === SPIN_FULL && linesCleared > 0) score += W.tspinClearBonus * linesCleared;
  if (placedType === 'T' && spin === SPIN_NONE) score += W.wastedTPenalty;
  // Only run the expensive T-slot scan when a T-piece is actually in the lookahead.
  // T appears once per 7-bag, so this skips the call ~85% of evaluations.
  if (tInLookahead) score += scoreTSpinReadiness(bm) * W.tslotBonus;
  score += scoreTetrisReadiness(bm, targetWellCol) * W.tetrisReadiness;

  if (pendingGarbage > 0) {
    score += W.garbageClearBonus * linesCleared * pendingGarbage;
    score += W.garbageUrgency    * pendingGarbage;
  }

  // Back-to-back: flat bonus for having B2B active after this placement.
  // Preserves the incentive to keep firing spins/quads rather than breaking the streak.
  if (b2b >= 0) score += W.b2bBonus + W.b2bLevel * Math.min(b2b, W.b2bLevelCap);

  // Combo streak: reward for having an active streak after this placement.
  // combo >= 0 means at least one consecutive clear; scale linearly so longer streaks
  // are more valuable (a 5-combo is worth ~2.1 extra, close to a Tetris).
  if (combo >= 0) score += W.comboBonus * (combo + 1);

  // Combo potential: reward rows that are nearly full (8 or 9 filled cells).
  // Full rows excluded — they clear immediately and are already captured by lineClearScore.
  // Scale: 8 filled → +0.33, 9 filled → +0.67 (per row × comboPotential weight).
  let comboReadyScore = 0;
  for (let r = stackTop; r < BOARD_ROWS; r++) {
    const filled = popcount10(bm[r]);
    if (filled >= 8 && filled < BOARD_COLS) {
      comboReadyScore += (filled - 7) / 3;
    }
  }
  score += W.comboPotential * comboReadyScore;

  return score;
}

// ---- Beam search ----

interface BeamNodeHard {
  board: Uint16Array;
  activeType: PieceType;
  nextQueue: PieceType[];   // NEXT_QUEUE_SIZE pieces after the active piece
  hold: PieceType | null;
  holdUsed: boolean;
  bagState: PieceType[];
  score: number;         // current board shape quality — used for beam pruning at each step
  garbageSent: number;   // total garbage lines sent along this search path
  combo: number;         // current combo index (-1 = no streak)
  b2b: number;           // B2B level (-1 = none, 0 = first qualifying clear)
  targetWellCol: number; // fixed well column for the entire search (set once at root)
  heights: number[];     // column heights — maintained incrementally to skip empty rows in evaluateBoard
  firstMove: { rotationIndex: number; x: number; y: number; useHold: boolean } | null;
  // null = normal bot mode (no path tracking); [] = engine analysis mode (full path tracked)
  movePath: EngineMove[] | null;
}

// Expand one beam node into all possible successor nodes (one piece placed).
// Models hold mechanics exactly as game.ts / versus.ts do.
function expandBeamNodeHardHard(node: BeamNodeHard, pendingGarbage: number): BeamNodeHard[] {
  const successors: BeamNodeHard[] = [];

  type PlayOption = {
    pieceType: PieceType;
    useHold: boolean;
    nextActiveType: PieceType;
    nextQueue: PieceType[];
    nextHold: PieceType | null;
    nextBagState: PieceType[];
  };

  const options: PlayOption[] = [];

  // Option 1: play active piece (no hold)
  {
    searchBag.restoreState(node.bagState);
    const newPiece = searchBag.next();
    options.push({
      pieceType: node.activeType,
      useHold: false,
      nextActiveType: node.nextQueue[0],
      nextQueue: queueAdvance(node.nextQueue, newPiece),
      nextHold: node.hold,
      nextBagState: searchBag.getState(),
    });
  }

  if (!node.holdUsed) {
    if (node.hold !== null) {
      // Option 2a: swap with existing hold piece (draws 1 bag piece to refill queue)
      searchBag.restoreState(node.bagState);
      const newPiece = searchBag.next();
      options.push({
        pieceType: node.hold,
        useHold: true,
        nextActiveType: node.nextQueue[0],
        nextQueue: queueAdvance(node.nextQueue, newPiece),
        nextHold: node.activeType,
        nextBagState: searchBag.getState(),
      });
    } else if (node.nextQueue.length >= 2) {
      // Option 2b: hold is empty — play nextQueue[0] instead, draws 2 bag pieces
      searchBag.restoreState(node.bagState);
      const newPiece1 = searchBag.next();
      const newPiece2 = searchBag.next();
      options.push({
        pieceType: node.nextQueue[0],
        useHold: true,
        nextActiveType: node.nextQueue[1],
        nextQueue: queueAdvance2(node.nextQueue, newPiece1, newPiece2),
        nextHold: node.activeType,
        nextBagState: searchBag.getState(),
      });
    }
  }

  for (const opt of options) {
    const placements = findReachablePlacements(node.board, opt.pieceType);
    // Only run the expensive T-slot scan when T is visible in the upcoming queue.
    const tInLookahead = opt.nextActiveType === 'T'
      || opt.nextHold === 'T'
      || opt.nextQueue.some(p => p === 'T');

    for (const placement of placements) {
      const piece: ActivePiece = {
        type: opt.pieceType,
        rotationIndex: placement.rotationIndex,
        x: placement.x,
        y: placement.y,
      };
      const { bm: clearedBoard, linesCleared } = bmLockAndClear(node.board, piece);
      // Advance combo/B2B and compute attack under the shared rules.
      let isPerfectClear = linesCleared > 0;
      for (let r = 0; isPerfectClear && r < BOARD_ROWS; r++) { if (clearedBoard[r]) isPerfectClear = false; }
      const clear = resolveClear(node.combo, node.b2b, linesCleared, placement.spin, isPerfectClear);
      const nextCombo = clear.combo;
      const nextB2b = clear.b2b;
      const garbageOut = clear.attack;

      // Update heights incrementally: if no lines cleared, only the columns the placed
      // piece touches can change. On a line clear, rows shift so a full rescan is needed.
      let nextHeights: number[];
      if (linesCleared === 0) {
        nextHeights = node.heights.slice();
        const rotation = getRotation(opt.pieceType, placement.rotationIndex);
        for (let pc = 0; pc < rotation[0].length; pc++) {
          const c = placement.x + pc;
          if (c < 0 || c >= BOARD_COLS) continue;
          for (let pr = 0; pr < rotation.length; pr++) {
            if (rotation[pr][pc]) {
              const h = BOARD_ROWS - (placement.y + pr);
              if (h > nextHeights[c]) nextHeights[c] = h;
              break; // only the topmost cell in this column matters
            }
          }
        }
      } else {
        nextHeights = computeColumnHeightsBm(clearedBoard);
      }

      // Landing height = rows from bottom where piece's topmost row sits.
      // Higher value = piece landed near top = dangerous; penalised via W.landingHeight.
      const landingHeight = BOARD_ROWS - piece.y;
      const boardScore = evaluateBoard(
        clearedBoard, linesCleared, placement.spin, pendingGarbage,
        opt.pieceType, node.targetWellCol, landingHeight, tInLookahead, nextHeights, nextCombo, nextB2b,
      );

      const firstMove = node.firstMove ?? {
        rotationIndex: placement.rotationIndex,
        x: placement.x,
        y: placement.y,
        useHold: opt.useHold,
      };

      // Track full move sequence when in engine analysis mode (movePath non-null on root).
      const movePath = node.movePath !== null
        ? [...node.movePath, {
            pieceType: opt.pieceType,
            rotationIndex: placement.rotationIndex,
            x: placement.x,
            y: placement.y,
            useHold: opt.useHold,
            linesCleared,
            spin: placement.spin,
            isPerfectClear,
          } satisfies EngineMove]
        : null;

      successors.push({
        board: clearedBoard,
        activeType: opt.nextActiveType,
        nextQueue: opt.nextQueue,
        hold: opt.nextHold,
        holdUsed: false,
        bagState: opt.nextBagState,
        score: boardScore,
        garbageSent: node.garbageSent + garbageOut,
        combo: nextCombo,
        b2b: nextB2b,
        targetWellCol: node.targetWellCol,
        heights: nextHeights,
        firstMove,
        movePath,
      });
    }
  }

  return successors;
}

// Find the best first move via beam search over SEARCH_DEPTH pieces.
// The beam search naturally discovers T-spin setups through lookahead —
// pieces placed now to build a T-slot will be rewarded when the T-piece fires.
export function findBestMoveHard(
  bot: BotBoard,
  pendingGarbage: number = 0,
  beamWidth: number = BEAM_WIDTH,
  searchDepth: number = SEARCH_DEPTH,
  combo: number = -1,
  b2b: number = -1,
): { rotationIndex: number; x: number; y: number; useHold: boolean } {
  const rootBm = cellBoardToBm(bot.board);
  const rootHeights = computeColumnHeightsBm(rootBm);

  // Reset well persistence on a new game (empty board), then pick with hysteresis.
  if (Math.max(...rootHeights) === 0) savedWellCol = undefined;
  const targetWellCol = pickTargetWellCol(rootBm, savedWellCol);
  savedWellCol = targetWellCol;

  let beam: BeamNodeHard[] = [{
    board: rootBm,
    activeType: bot.active.type,
    nextQueue: [...bot.nextQueue],
    hold: bot.hold,
    holdUsed: bot.holdUsed,
    bagState: bot.bagState,
    score: 0,
    garbageSent: 0,
    combo,
    b2b,
    targetWellCol,
    heights: rootHeights,
    firstMove: null,
    movePath: null,
  }];

  for (let d = 0; d < searchDepth; d++) {
    const nextBeam: BeamNodeHard[] = [];
    for (const node of beam) {
      nextBeam.push(...expandBeamNodeHardHard(node, pendingGarbage));
    }
    if (nextBeam.length === 0) break;
    // Rank by current board shape quality + weighted total garbage sent.
    // Using the node's own board score (not cumulative) means a poor placement is
    // immediately visible rather than being masked by a strong earlier move.
    beam = topKDescending(nextBeam, beamWidth, n => n.score + n.garbageSent * W.garbageValue);
  }

  return beam[0]?.firstMove ?? { rotationIndex: 0, x: 0, y: 0, useHold: false };
}
