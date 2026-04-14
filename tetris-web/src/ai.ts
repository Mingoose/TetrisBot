import { ActivePiece, CellValue, PieceType } from './types';
import type { EngineMove, EngineRequest, EngineAnalysis } from './engine';
import {
  collides, lockAndClear, hardDropY, attemptRotation,
  BOARD_COLS, BOARD_ROWS, countTSpinCorners,
} from './board';
import { getRotation } from './pieces';
import { Bag } from './bag';
import type { BotBoard } from './versus';
import { computeGarbage } from './versus';

const BEAM_WIDTH = 20;
const SEARCH_DEPTH = 4;

export type AiDifficulty = 'easy' | 'medium' | 'hard';
export const AI_DIFFICULTY_PARAMS: Record<AiDifficulty, { beamWidth: number; searchDepth: number; advancedEval: boolean; label: string; subtitle: string }> = {
  easy:   { beamWidth: 1,  searchDepth: 1, advancedEval: false, label: 'EASY',   subtitle: 'greedy one-piece' },
  medium: { beamWidth: 20, searchDepth: 4, advancedEval: false, label: 'MEDIUM', subtitle: 'beam search' },
  hard:   { beamWidth: 32, searchDepth: 5, advancedEval: true,  label: 'HARD',   subtitle: 'beam search+' },
};

// Module-level bag used only for beam search simulation.
// Always restored from a node's bagState before use.
const searchBag = new Bag();

// ---- BFS-based move generator ----

interface PlacementResult {
  rotationIndex: number;
  x: number;
  y: number;
  isTSpin: boolean;
}

// Determine whether a grounded T-piece placement is a T-spin.
// Conditions: 3+ corners filled AND the piece couldn't arrive by a straight drop
// (meaning it had to navigate under a roof via rotation).
function isTSpinPlacement(board: CellValue[][], piece: ActivePiece): boolean {
  if (piece.type !== 'T') return false;
  if (countTSpinCorners(board, piece) < 3) return false;
  // If the piece can be placed by a direct drop at this (rotation, x), it's not a T-spin.
  const atTop: ActivePiece = { ...piece, y: 0 };
  if (!collides(board, atTop, 0, 0) && hardDropY(board, atTop) === piece.y) return false;
  return true;
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

function bfsIdx(rot: number, x: number, y: number): number {
  return rot * BFS_SLOT + (y + BFS_Y_OFF) * BFS_X_SIZE + (x + BFS_X_OFF);
}

// BFS over all reachable (rotationIndex, x, y) positions, starting from the spawn
// position (rotation 0, centered, y=0). All 6 actions are explored: move left/right,
// drop one row, and rotate CW/CCW/180° with SRS wall kicks. This correctly discovers
// placements only reachable by rotating under a roof — the key for T-spins.
function findReachablePlacements(board: CellValue[][], pieceType: PieceType): PlacementResult[] {
  const spawnWidth = getRotation(pieceType, 0)[0].length;
  const startX = Math.floor((BOARD_COLS - spawnWidth) / 2);
  const startPiece: ActivePiece = { type: pieceType, rotationIndex: 0, x: startX, y: 0 };

  if (collides(board, startPiece, 0, 0)) return [];

  const gen = ++bfsGen;
  const queue: ActivePiece[] = [startPiece];
  let qHead = 0;
  const grounded: ActivePiece[] = [];

  bfsVisit[bfsIdx(0, startX, 0)] = gen;

  while (qHead < queue.length) {
    const piece = queue[qHead++];

    // Grounded check — record unique resting positions
    if (collides(board, piece, 0, 1)) {
      const gk = bfsIdx(piece.rotationIndex, piece.x, piece.y);
      if (bfsGround[gk] !== gen) { bfsGround[gk] = gen; grounded.push(piece); }
    }

    // Inline all six moves to avoid per-iteration array allocation
    // Left
    if (!collides(board, piece, -1, 0)) {
      const nx = piece.x - 1;
      const nk = bfsIdx(piece.rotationIndex, nx, piece.y);
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        queue.push({ type: pieceType, rotationIndex: piece.rotationIndex, x: nx, y: piece.y });
      }
    }
    // Right
    if (!collides(board, piece, 1, 0)) {
      const nx = piece.x + 1;
      const nk = bfsIdx(piece.rotationIndex, nx, piece.y);
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        queue.push({ type: pieceType, rotationIndex: piece.rotationIndex, x: nx, y: piece.y });
      }
    }
    // Down
    if (!collides(board, piece, 0, 1)) {
      const ny = piece.y + 1;
      const nk = bfsIdx(piece.rotationIndex, piece.x, ny);
      if (bfsVisit[nk] !== gen) {
        bfsVisit[nk] = gen;
        queue.push({ type: pieceType, rotationIndex: piece.rotationIndex, x: piece.x, y: ny });
      }
    }
    // Rotations CW, CCW, 180°
    for (const delta of [1, -1, 2] as const) {
      const next = attemptRotation(board, piece, delta);
      if (!next) continue;
      const nk = bfsIdx(next.rotationIndex, next.x, next.y);
      if (bfsVisit[nk] !== gen) { bfsVisit[nk] = gen; queue.push(next); }
    }
  }

  return grounded.map(piece => ({
    rotationIndex: piece.rotationIndex,
    x: piece.x,
    y: piece.y,
    isTSpin: isTSpinPlacement(board, piece),
  }));
}

// ---- Evaluation weights ----
// All tunable scoring constants in one place.

const W = {
  // Height — two-tier danger so penalty escalates as the board fills up
  aggHeightBase:    4,      // rows — no height penalty below this average column height
  aggHeight:        -0.03,  // per aggregate-height unit above aggHeightBase
  dangerThreshold:  10,     // first danger tier: mild slope starts here
  dangerSlope:      3.0,    // penalty per row above dangerThreshold (up to dangerThreshold2)
  dangerThreshold2: 16,     // second danger tier: steep slope starts here
  dangerSlope2:     12.0,   // penalty per row above dangerThreshold2

  // Emergency downstacking — kicks in as height approaches the ceiling.
  // heightFactor = (maxHeight − emergencyHeight) / (BOARD_ROWS − emergencyHeight), clamped 0→1.
  // Used to smoothly override the normal "avoid singles" bias when survival matters more.
  emergencyHeight:      12, // where the emergency scale starts (0 at or below, 1 at BOARD_ROWS)
  emergencyLineClear:   6.0,// bonus per line cleared × heightFactor — turns singles into rewards when tall
  emergencyHoleScale:   1.5,// holes × (1 + heightFactor×scale) — buried holes hurt more when tall
  emergencyWellScale:   1.5,// well depth bonus × (1 + heightFactor×scale) — keep the lane open

  // Board shape
  holes:            -0.80,  // per covered empty cell
  overhangs:        -0.35,  // per filled cell directly above an empty cell
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
  lineClear:        [0, -0.8, -0.2, 0.3, 0.0] as number[],

  // T-spin
  tspinClearBonus:  1.5,
  wastedTPenalty:   -2.5,
  tslotBonus:       2.0,

  // Perfect clear
  perfectClear:     30.0,

  // Garbage pressure
  garbageValue:     2.0,
  garbageClearBonus: 0.4,
  garbageUrgency:   -0.3,
};


function initHeightArray(): number[] {
  return new Array<number>(BOARD_COLS).fill(0);
}

// Choose a well column that the AI will commit to for the entire search.
// Prefers the right edge (col 9), but switches to left (col 0) if it is already
// meaningfully lower — indicating the game has already developed that way.
function pickTargetWellCol(board: CellValue[][]): number {
  const heights = initHeightArray();
  for (let c = 0; c < BOARD_COLS; c++) {
    for (let r = 0; r < BOARD_ROWS; r++) {
      if (board[r][c] !== 0) { heights[c] = BOARD_ROWS - r; break; }
    }
  }
  if (heights[0] < heights[BOARD_COLS - 1] - 2) return 0;
  return BOARD_COLS - 1;
}

// ---- Board evaluation (Hard AI) ----

// Score TSD readiness for each valid T-spin slot on the board.
// For each T-piece rotation and straight-drop position where the 3-corner rule
// is satisfied, measure what fraction of the non-T-piece cells in those rows are
// already filled. A fully-ready slot (all surrounding cells filled) scores 1.0;
// an empty slot scores near 0. This creates a gradient that rewards both creating
// the slot geometry AND filling the rows needed to actually fire the T-spin.
// Returns total readiness across all slots, capped at 2.
function scoreTSpinReadiness(board: CellValue[][]): number {
  let total = 0;

  for (let rotIdx = 0; rotIdx < 4; rotIdx++) {
    const rotation = getRotation('T', rotIdx);
    const pieceW = rotation[0].length;
    const pieceH = rotation.length;

    for (let x = 0; x <= BOARD_COLS - pieceW; x++) {
      const piece: ActivePiece = { type: 'T', rotationIndex: rotIdx, x, y: 0 };
      if (collides(board, piece, 0, 0)) continue;

      const y = hardDropY(board, piece);
      const landed: ActivePiece = { ...piece, y };

      if (countTSpinCorners(board, landed) < 3) continue;

      // For every row the T-piece occupies, count filled vs empty non-piece cells.
      // Readiness = filled / (filled + empty) across all those rows.
      let nonPieceCells = 0;
      let filledCells = 0;
      for (let pr = 0; pr < pieceH; pr++) {
        const row = y + pr;
        if (row < 0 || row >= BOARD_ROWS) continue;
        for (let c = 0; c < BOARD_COLS; c++) {
          const pc = c - x;
          if (pc >= 0 && pc < pieceW && rotation[pr][pc]) continue; // piece cell
          nonPieceCells++;
          if (board[row][c] !== 0) filledCells++;
        }
      }

      if (nonPieceCells === 0) continue;
      total += filledCells / nonPieceCells;
    }
  }

  return Math.min(total, 2);
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
  board: CellValue[][],
  linesCleared: number,
  isTSpin: boolean = false,
  pendingGarbage: number = 0,
  placedType?: PieceType,
): number {
  const heights = initHeightArray();
  let holes = 0;

  for (let c = 0; c < BOARD_COLS; c++) {
    let foundTop = false;
    for (let r = 0; r < BOARD_ROWS; r++) {
      if (board[r][c] !== 0) {
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

  if (isTSpin && linesCleared > 0) score += W_MEDIUM.tspinClearBonus * linesCleared;
  if (placedType === 'T' && !isTSpin) score += W_MEDIUM.wastedTPenalty;
  score += scoreTSpinReadiness(board) * W_MEDIUM.tslotBonus;

  if (pendingGarbage > 0) {
    score += W_MEDIUM.garbageClearBonus * linesCleared * pendingGarbage;
    score += W_MEDIUM.garbageUrgency    * pendingGarbage;
  }

  return score;
}

interface BeamNodeMedium {
  board: CellValue[][];
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
      nextQueue: [...node.nextQueue.slice(1), newPiece],
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
        nextQueue: [...node.nextQueue.slice(1), newPiece],
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
        nextQueue: [...node.nextQueue.slice(2), newPiece1, newPiece2],
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
      const { board: clearedBoard, linesCleared } = lockAndClear(node.board, piece);
      const moveScore = evaluateBoardMedium(
        clearedBoard, linesCleared, placement.isTSpin, pendingGarbage, opt.pieceType,
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

export function findBestMove(
  bot: BotBoard,
  pendingGarbage: number = 0,
  beamWidth: number = BEAM_WIDTH,
  searchDepth: number = SEARCH_DEPTH,
  _combo: number = -1,       // unused by medium eval — included for API consistency with findBestMoveHard
  _b2bActive: boolean = false,
): { rotationIndex: number; x: number; y: number; useHold: boolean } {
  let beam: BeamNodeMedium[] = [{
    board: bot.board,
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
    nextBeam.sort((a, b) => b.score - a.score);
    beam = nextBeam.slice(0, beamWidth);
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
          combo, b2bActive, pendingGarbage, beamWidth, searchDepth, topN } = request;

  const targetWellCol = pickTargetWellCol(board);

  let beam: BeamNodeHard[] = [{
    board,
    activeType,
    nextQueue: [...nextQueue],
    hold,
    holdUsed,
    bagState: [...bagState],
    score: 0,
    garbageSent: 0,
    combo: combo ?? -1,
    b2bActive: b2bActive ?? false,
    targetWellCol,
    firstMove: null,
    movePath: [],  // empty array enables path tracking in expandBeamNodeHardHard
  }];

  for (let d = 0; d < searchDepth; d++) {
    const nextBeam: BeamNodeHard[] = [];
    for (const node of beam) {
      nextBeam.push(...expandBeamNodeHardHard(node, pendingGarbage));
    }
    if (nextBeam.length === 0) break;
    nextBeam.sort((a, b) =>
      (b.score + b.garbageSent * W.garbageValue) -
      (a.score + a.garbageSent * W.garbageValue),
    );
    beam = nextBeam.slice(0, beamWidth);
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
  board: CellValue[][],
  linesCleared: number,
  isTSpin: boolean = false,
  pendingGarbage: number = 0,
  placedType?: PieceType,
  targetWellCol: number = BOARD_COLS - 1,
  landingHeight: number = 0,
): number {
  // Perfect clear: board is completely empty after this placement.
  if (board.every(row => row.every(c => c === 0))) return W.perfectClear;

  const heights = initHeightArray();
  let holes = 0;
  let overhangs = 0;  // filled cells directly above an empty cell (Cold Clear's "overhang cells")
  let colTransitions = 0;

  for (let c = 0; c < BOARD_COLS; c++) {
    let foundTop = false;
    let prevFilled = false; // tracks column-transition state; starts false (empty above board)

    for (let r = 0; r < BOARD_ROWS; r++) {
      const filled = board[r][c] !== 0;
      if (filled !== prevFilled) colTransitions++;

      if (filled) {
        if (!foundTop) { heights[c] = BOARD_ROWS - r; foundTop = true; }
        // Overhang: this filled cell has an empty cell directly below it.
        if (r + 1 < BOARD_ROWS && board[r + 1][c] === 0) overhangs++;
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

  // Row transitions: count filled↔empty boundaries scanning each row, treating walls as filled.
  // Only count rows within the occupied portion of the board to avoid constant offset from empty rows.
  let rowTransitions = 0;
  const stackTop = maxHeight > 0 ? BOARD_ROWS - maxHeight : BOARD_ROWS;
  for (let r = stackTop; r < BOARD_ROWS; r++) {
    let prevFilled = true; // left wall
    for (let c = 0; c < BOARD_COLS; c++) {
      const filled = board[r][c] !== 0;
      if (filled !== prevFilled) rowTransitions++;
      prevFilled = filled;
    }
    if (!prevFilled) rowTransitions++; // right wall
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
            + W.overhangs       * overhangs
            + W.bumpiness       * bumpiness
            + W.cliffPenalty    * cliffPenalty
            + W.rowTransitions  * rowTransitions
            + W.colTransitions  * colTransitions
            + W.cumulativeWells * cumulativeWells
            + wellDepthScore
            + W.landingHeight   * landingHeight
            - dangerPenalty;

  if (isTSpin && linesCleared > 0) score += W.tspinClearBonus * linesCleared;
  if (placedType === 'T' && !isTSpin) score += W.wastedTPenalty;
  score += scoreTSpinReadiness(board) * W.tslotBonus;

  if (pendingGarbage > 0) {
    score += W.garbageClearBonus * linesCleared * pendingGarbage;
    score += W.garbageUrgency    * pendingGarbage;
  }

  return score;
}

// ---- Beam search ----

interface BeamNodeHard {
  board: CellValue[][];
  activeType: PieceType;
  nextQueue: PieceType[];   // NEXT_QUEUE_SIZE pieces after the active piece
  hold: PieceType | null;
  holdUsed: boolean;
  bagState: PieceType[];
  score: number;         // current board shape quality — used for beam pruning at each step
  garbageSent: number;   // total garbage lines sent along this search path
  combo: number;         // current combo index (-1 = no streak)
  b2bActive: boolean;    // back-to-back T-spin/Tetris qualifier active
  targetWellCol: number; // fixed well column for the entire search (set once at root)
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
      nextQueue: [...node.nextQueue.slice(1), newPiece],
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
        nextQueue: [...node.nextQueue.slice(1), newPiece],
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
        nextQueue: [...node.nextQueue.slice(2), newPiece1, newPiece2],
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
      const { board: clearedBoard, linesCleared } = lockAndClear(node.board, piece);
      // Advance combo/b2b state for this placement.
      let nextCombo: number;
      let nextB2b: boolean;
      if (linesCleared > 0) {
        nextCombo = node.combo < 0 ? 0 : node.combo + 1;
        nextB2b = placement.isTSpin || linesCleared === 4;
      } else {
        nextCombo = -1;
        nextB2b = node.b2bActive; // b2b only resets on a non-qualifying clear, not on 0 lines
      }

      // Compute garbage using the pre-placement b2b and the post-increment combo index.
      const isPerfectClear = clearedBoard.every(row => row.every(c => c === 0));
      const garbageOut = isPerfectClear
        ? 10
        : computeGarbage(linesCleared, placement.isTSpin, node.b2bActive, nextCombo);

      // Landing height = rows from bottom where piece's topmost row sits.
      // Higher value = piece landed near top = dangerous; penalised via W.landingHeight.
      const landingHeight = BOARD_ROWS - piece.y;
      const boardScore = evaluateBoard(
        clearedBoard, linesCleared, placement.isTSpin, pendingGarbage,
        opt.pieceType, node.targetWellCol, landingHeight,
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
            isTSpin: placement.isTSpin,
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
        b2bActive: nextB2b,
        targetWellCol: node.targetWellCol,
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
  b2bActive: boolean = false,
): { rotationIndex: number; x: number; y: number; useHold: boolean } {
  // Determine a stable well column from the current board state and commit to it for the
  // entire search.  This prevents the well from jumping between columns mid-game.
  const targetWellCol = pickTargetWellCol(bot.board);

  let beam: BeamNodeHard[] = [{
    board: bot.board,
    activeType: bot.active.type,
    nextQueue: [...bot.nextQueue],
    hold: bot.hold,
    holdUsed: bot.holdUsed,
    bagState: bot.bagState,
    score: 0,
    garbageSent: 0,
    combo,
    b2bActive,
    targetWellCol,
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
    nextBeam.sort((a, b) =>
      (b.score + b.garbageSent * W.garbageValue) -
      (a.score + a.garbageSent * W.garbageValue),
    );
    beam = nextBeam.slice(0, beamWidth);
  }

  return beam[0]?.firstMove ?? { rotationIndex: 0, x: 0, y: 0, useHold: false };
}
