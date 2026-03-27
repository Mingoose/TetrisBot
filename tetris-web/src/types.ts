export type PieceType = 'I' | 'O' | 'T' | 'S' | 'Z' | 'J' | 'L';

// 0 = empty, PieceType = locked piece color, 'X' = editor-placed cell
export type CellValue = 0 | PieceType | 'X';

export type RotationMatrix = ReadonlyArray<ReadonlyArray<0 | 1>>;

export interface ActivePiece {
  type: PieceType;
  rotationIndex: number;
  x: number; // col of bounding-box left edge
  y: number; // row of bounding-box top edge
}

// Fully serializable — the unit of rewind history
export interface Snapshot {
  board: CellValue[][];    // [row][col], row 0 = top
  score: number;
  lines: number;
  level: number;
  nextQueue: PieceType[];  // 5-piece visible lookahead
  hold: PieceType | null;
  holdUsed: boolean;
  active: ActivePiece;
  bagState: PieceType[];   // remaining pieces in current 7-bag
}

export type GameVariant = 'sprint' | 'creative' | 'versus' | 'watch' | 'botvsbot';
export type GameMode = 'menu' | 'countdown' | 'playing' | 'paused' | 'editor' | 'gameover';

export interface GameState extends Snapshot {
  mode: GameMode;
  variant: GameVariant;
  history: Snapshot[]; // rewind stack (max 50 entries)
  // transient — never snapshotted
  lockDelayMs: number;
  lockResetCount: number;
  gravityAccumMs: number;
  lastFrameTime: number;
  rafHandle: number;
  countdownMs: number;          // ms remaining in pre-game countdown (sprint/versus)
  sprintStartTime: number;      // rAF timestamp when sprint became 'playing'; 0 = not started
  sprintElapsedMs: number;      // frozen at sprint-complete for display
  lastActionRotation: boolean;  // true if last player action was a rotation; for T-spin detection
}
