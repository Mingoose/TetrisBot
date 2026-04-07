import type { Snapshot, CellValue, ActivePiece, PieceType } from './types';

/** One piece lock event recorded during a sprint game. */
export interface ReplayEntry {
  /** Full game snapshot taken immediately before this piece locked.
   *  snapshot.board  = board before this piece
   *  snapshot.active = piece at its final landing position
   *  snapshot.hold / nextQueue / lines / score = state at that moment */
  snapshot: Snapshot;
  /** Milliseconds elapsed from sprint start when this piece locked. */
  elapsedMs: number;
}

/** A complete sprint replay. Designed to be serialisable for future persistence. */
export interface SprintReplay {
  entries: ReplayEntry[];
  finalElapsedMs: number;
  /** Board state after the last line clear (may be partially cleared or empty). */
  finalBoard: CellValue[][];
}

/**
 * Given a replay and the current playback time, return the index of the entry
 * whose piece is currently "in play" (has not yet locked).
 * Returns -1 when all pieces have locked (replay is done).
 */
export function getReplayFrameIndex(replay: SprintReplay, elapsedMs: number): number {
  return replay.entries.findIndex(e => e.elapsedMs > elapsedMs);
}

// ---- Versus replay ----

/** Minimal bot board snapshot stored in each versus replay entry. */
export interface BotSnapshot {
  board: CellValue[][];
  active: ActivePiece;
  nextQueue: PieceType[];
  hold: PieceType | null;
  holdUsed: boolean;
  lines: number;
  dead: boolean;
}

/**
 * One lock event (player or bot) recorded during a versus game.
 * playerSnapshot is the player's full game snapshot at the moment of the lock.
 * botSnapshot is the bot's board state at the same moment.
 */
export interface VersusReplayEntry {
  elapsedMs: number;
  playerSnapshot: Snapshot;
  botSnapshot: BotSnapshot;
}

/** A complete versus replay. */
export interface VersusReplay {
  entries: VersusReplayEntry[];
  finalElapsedMs: number;
  winner: 'player' | 'bot';
  /** Player board at game end (may be topped out). */
  finalPlayerBoard: CellValue[][];
  /** Bot board at game end (may be topped out). */
  finalBotBoard: CellValue[][];
}

/**
 * Given a versus replay and the current playback time, return the index of the
 * entry currently "in play". Returns -1 when all entries have elapsed (replay done).
 */
export function getVersusReplayFrameIndex(replay: VersusReplay, elapsedMs: number): number {
  return replay.entries.findIndex(e => e.elapsedMs > elapsedMs);
}
