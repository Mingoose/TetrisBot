import { ActivePiece, CellValue, PieceType } from './types';
import {
  collides, lockPiece, clearLines, hardDropY,
  BOARD_COLS, BOARD_ROWS, countTSpinCorners,
} from './board';
import { getRotation, getWallKicks } from './pieces';
import { Bag } from './bag';
import type { BotBoard } from './versus';

const BEAM_WIDTH = 20;
const SEARCH_DEPTH = 4;

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

// Apply a rotation (delta = +1 CW, -1 CCW, +2 180°) with SRS wall kicks,
// matching the same logic as game.ts's tryRotate.
function applyRotation(board: CellValue[][], piece: ActivePiece, delta: number): ActivePiece | null {
  const newIndex = ((piece.rotationIndex + delta) % 4 + 4) % 4;
  // CW / 180°: use kicks from the 'from' state. CCW: use negated kicks from the 'to' state.
  const kickIndex = delta > 0 ? piece.rotationIndex : newIndex;
  const kicks = getWallKicks(piece.type, kickIndex);
  const kickList: Array<[number, number]> = delta < 0
    ? kicks.map(([dx, dy]) => [-dx, -dy] as [number, number])
    : kicks;

  for (const [kdx, kdy] of kickList) {
    const candidate: ActivePiece = {
      ...piece,
      rotationIndex: newIndex,
      x: piece.x + kdx,
      y: piece.y + kdy,
    };
    if (!collides(board, candidate, 0, 0)) return candidate;
  }
  return null;
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

// BFS over all reachable (rotationIndex, x, y) positions, starting from the spawn
// position (rotation 0, centered, y=0). All 6 actions are explored: move left/right,
// drop one row, and rotate CW/CCW/180° with SRS wall kicks. This correctly discovers
// placements only reachable by rotating under a roof — the key for T-spins.
function findReachablePlacements(board: CellValue[][], pieceType: PieceType): PlacementResult[] {
  const spawnWidth = getRotation(pieceType, 0)[0].length;
  const start: ActivePiece = {
    type: pieceType,
    rotationIndex: 0,
    x: Math.floor((BOARD_COLS - spawnWidth) / 2),
    y: 0,
  };

  if (collides(board, start, 0, 0)) return [];

  const visited = new Set<string>();
  const queue: ActivePiece[] = [start];
  const groundedMap = new Map<string, ActivePiece>();

  const key = (p: ActivePiece) => `${p.rotationIndex},${p.x},${p.y}`;
  visited.add(key(start));

  while (queue.length > 0) {
    const piece = queue.shift()!;

    if (collides(board, piece, 0, 1)) {
      const gk = key(piece);
      if (!groundedMap.has(gk)) groundedMap.set(gk, piece);
    }

    const nexts: (ActivePiece | null)[] = [
      collides(board, piece, -1, 0) ? null : { ...piece, x: piece.x - 1 },
      collides(board, piece, 1, 0)  ? null : { ...piece, x: piece.x + 1 },
      collides(board, piece, 0, 1)  ? null : { ...piece, y: piece.y + 1 },
      applyRotation(board, piece, 1),
      applyRotation(board, piece, -1),
      applyRotation(board, piece, 2),
    ];

    for (const next of nexts) {
      if (!next) continue;
      const nk = key(next);
      if (!visited.has(nk)) {
        visited.add(nk);
        queue.push(next);
      }
    }
  }

  return Array.from(groundedMap.values()).map(piece => ({
    rotationIndex: piece.rotationIndex,
    x: piece.x,
    y: piece.y,
    isTSpin: isTSpinPlacement(board, piece),
  }));
}

// ---- Evaluation weights ----
// All tunable scoring constants in one place.

const W = {
  // Height
  aggHeightBase:    4,      // rows — no height penalty below this average column height
  aggHeight:        -0.03,  // per aggregate-height unit above aggHeightBase
  dangerThreshold:  18,     // rows — steep penalty kicks in above this
  dangerSlope:      4.0,    // penalty per row above dangerThreshold

  // Board shape
  holes:            -1.00,  // per covered empty cell
  bumpiness:        -0.40,  // per unit of adjacent height-diff not involving the well column
  wellBumpiness:    -0.05,  // per unit of height-diff adjacent to the well column (lenient)
  cliffThreshold:   3,      // diffs above this get an extra steep penalty (non-well only)
  cliffPenalty:     -1.20,  // extra penalty per unit of diff exceeding cliffThreshold

  // Line clears (indexed by lines cleared: 0–4)
  lineClear:        [0, -1.2, 0.5, 1.8, 5.0],

  // T-spin
  tspinClearBonus:  3.5,    // multiplied by lines cleared (on top of lineClear reward)
  wastedTPenalty:   -2.5,   // T placed without achieving a T-spin
  tslotBonus:       2.0,    // scales 0→1 per TSD slot by row-fill readiness (max 2 slots)

  // Garbage pressure
  garbageClearBonus: 0.4,   // extra reward per line cleared × pending garbage
  garbageUrgency:   -0.3,   // penalty per unit of pending incoming garbage
};

// ---- Board evaluation ----

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

// Evaluate a board state after placing a piece. Higher = better.
export function evaluateBoard(
  board: CellValue[][],
  linesCleared: number,
  isTSpin: boolean = false,
  pendingGarbage: number = 0,
  placedType?: PieceType,
): number {
  const heights = new Array<number>(BOARD_COLS).fill(0);
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

  // Split bumpiness into well-adjacent vs rest so the well column can be
  // intentionally low (for Tetris/T-spin) without masking unevenness elsewhere.
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
      if (diff > W.cliffThreshold) cliffPenalty += diff - W.cliffThreshold;
    }
  }
  // Penalise asymmetry across the well: the columns on either side of the well
  // are never directly compared above, so one side can be far taller than the other.
  if (wellCol > 0 && wellCol < BOARD_COLS - 1) {
    const crossDiff = Math.abs(heights[wellCol - 1] - heights[wellCol + 1]);
    bumpiness += crossDiff;
    if (crossDiff > W.cliffThreshold) cliffPenalty += crossDiff - W.cliffThreshold;
  }

  const lineClearScore = W.lineClear[Math.min(linesCleared, 4)];
  const dangerPenalty  = Math.max(0, maxHeight - W.dangerThreshold) * W.dangerSlope;

  const penalisedHeight = Math.max(0, aggHeight - W.aggHeightBase * BOARD_COLS);
  let score = W.aggHeight      * penalisedHeight
            + lineClearScore
            + W.holes          * holes
            + W.bumpiness      * bumpiness
            + W.wellBumpiness  * wellBumpiness
            + W.cliffPenalty   * cliffPenalty
            - dangerPenalty;

  if (isTSpin && linesCleared > 0) {
    score += W.tspinClearBonus * linesCleared;
  }

  if (placedType === 'T' && !isTSpin) {
    score += W.wastedTPenalty;
  }

  score += scoreTSpinReadiness(board) * W.tslotBonus;

  if (pendingGarbage > 0) {
    score += W.garbageClearBonus * linesCleared * pendingGarbage;
    score += W.garbageUrgency    * pendingGarbage;
  }

  return score;
}

// ---- Beam search ----

interface BeamNode {
  board: CellValue[][];
  activeType: PieceType;
  nextQueue: PieceType[];   // NEXT_QUEUE_SIZE pieces after the active piece
  hold: PieceType | null;
  holdUsed: boolean;
  bagState: PieceType[];
  score: number;
  firstMove: { rotationIndex: number; x: number; y: number; useHold: boolean } | null;
}

// Expand one beam node into all possible successor nodes (one piece placed).
// Models hold mechanics exactly as game.ts / versus.ts do.
function expandBeamNode(node: BeamNode, pendingGarbage: number): BeamNode[] {
  const successors: BeamNode[] = [];

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
      const locked = lockPiece(node.board, piece);
      const { board: clearedBoard, linesCleared } = clearLines(locked);
      const moveScore = evaluateBoard(clearedBoard, linesCleared, placement.isTSpin, pendingGarbage, opt.pieceType);

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

// Find the best first move via beam search over SEARCH_DEPTH pieces.
// The beam search naturally discovers T-spin setups through lookahead —
// pieces placed now to build a T-slot will be rewarded when the T-piece fires.
export function findBestMove(
  bot: BotBoard,
  pendingGarbage: number = 0,
): { rotationIndex: number; x: number; y: number; useHold: boolean } {
  let beam: BeamNode[] = [{
    board: bot.board,
    activeType: bot.active.type,
    nextQueue: [...bot.nextQueue],
    hold: bot.hold,
    holdUsed: bot.holdUsed,
    bagState: bot.bagState,
    score: 0,
    firstMove: null,
  }];

  for (let d = 0; d < SEARCH_DEPTH; d++) {
    const nextBeam: BeamNode[] = [];
    for (const node of beam) {
      nextBeam.push(...expandBeamNode(node, pendingGarbage));
    }
    if (nextBeam.length === 0) break;
    nextBeam.sort((a, b) => b.score - a.score);
    beam = nextBeam.slice(0, BEAM_WIDTH);
  }

  return beam[0]?.firstMove ?? { rotationIndex: 0, x: 0, y: 0, useHold: false };
}
