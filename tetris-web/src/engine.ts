import type { CellValue, PieceType, GameState } from './types';

/** A single piece placement in a recommended continuation. */
export interface EngineMove {
  pieceType: PieceType;
  rotationIndex: number;
  x: number;
  y: number;
  useHold: boolean;
  linesCleared: number;
  isTSpin: boolean;
  isPerfectClear: boolean;
}

/**
 * One fully-evaluated continuation line.
 * `moves[0]` is the immediate recommended placement; subsequent moves
 * show what the engine would play after that.
 */
export interface EngineLine {
  moves: EngineMove[];
  score: number;        // composite terminal score (board shape + garbage value)
  garbageSent: number;  // total garbage generated along this path
}

/** Full result from analyzePositionHard. Lines are sorted best-first. */
export interface EngineAnalysis {
  lines: EngineLine[];
  durationMs: number;
}

/**
 * Input descriptor for an engine analysis request.
 * Intentionally decoupled from GameState / BotBoard so it can be built
 * from any source: creative pause, post-match review, hint mode, etc.
 */
export interface EngineRequest {
  board: CellValue[][];
  activeType: PieceType;
  nextQueue: PieceType[];
  hold: PieceType | null;
  holdUsed: boolean;
  bagState: PieceType[];
  combo: number;
  b2bActive: boolean;
  pendingGarbage: number;
  // Search parameters
  beamWidth: number;
  searchDepth: number;
  topN: number;  // number of distinct first-move alternatives to return
}

/**
 * Build an EngineRequest from a live GameState.
 *
 * `combo` and `b2bActive` are not stored on GameState (they live on BotBoard
 * in versus mode only), so they default conservatively to -1 / false.
 * Pass them explicitly when you have access to those values.
 */
export function gameStateToEngineRequest(
  state: GameState,
  params: {
    beamWidth?: number;
    searchDepth?: number;
    topN?: number;
    pendingGarbage?: number;
    combo?: number;
    b2bActive?: boolean;
  } = {},
): EngineRequest {
  return {
    board: state.board,
    activeType: state.active.type,
    nextQueue: [...state.nextQueue],
    hold: state.hold,
    holdUsed: state.holdUsed,
    bagState: [...state.bagState],
    combo:          params.combo          ?? -1,
    b2bActive:      params.b2bActive      ?? false,
    pendingGarbage: params.pendingGarbage ?? 0,
    beamWidth:      params.beamWidth      ?? 48,
    searchDepth:    params.searchDepth    ?? 6,
    topN:           params.topN           ?? 5,
  };
}
