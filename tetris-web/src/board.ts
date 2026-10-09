import { ActivePiece, CellValue, PieceType } from './types';
import { getRotation, getKicks } from './pieces';
import { SpinKind, T_CORNERS, T_FRONT_CORNERS, classifySpin } from './rules';

export const BOARD_COLS = 10;
export const BOARD_ROWS = 20;   // visible rows
// Hidden rows above the visible field (TETR.IO: 20). Game boards are
// BUFFER_ROWS + BOARD_ROWS rows, but piece coordinates stay in the visible
// frame: row 0 is the top visible row and the buffer is rows -1 .. -BUFFER_ROWS.
// Board functions take either a full board or a 20-row visible one (the AI,
// harness and value net work on the visible rows) and index it as
// board[row + bufferOf(board)].
export const BUFFER_ROWS = 20;

export function emptyBoard(rows = BUFFER_ROWS + BOARD_ROWS): CellValue[][] {
  return Array.from({ length: rows }, () => new Array(BOARD_COLS).fill(0) as CellValue[]);
}

// Number of buffer rows stored above the visible field (0 for a visible-only board).
export function bufferOf(board: CellValue[][]): number {
  return board.length - BOARD_ROWS;
}

// The 20 visible rows, for the AI, the value net and uploaded AIs.
export function visibleRows(board: CellValue[][]): CellValue[][] {
  return board.slice(bufferOf(board));
}

// Check if placing `active` at (active.x + dx, active.y + dy) collides with
// board walls or any filled cell. Rows above the stored board count as empty.
export function collides(
  board: CellValue[][],
  active: ActivePiece,
  dx: number,
  dy: number,
): boolean {
  const rotation = getRotation(active.type, active.rotationIndex);
  const newX = active.x + dx;
  const newY = active.y + dy;
  const off = bufferOf(board);
  for (let r = 0; r < rotation.length; r++) {
    for (let c = 0; c < rotation[r].length; c++) {
      if (!rotation[r][c]) continue;
      const col = newX + c;
      const row = newY + r;
      if (col < 0 || col >= BOARD_COLS) return true;
      if (row >= BOARD_ROWS) return true;
      if (row >= -off && board[row + off][col] !== 0) return true;
    }
  }
  return false;
}

// Return the Y position where the piece would land if hard-dropped.
export function hardDropY(board: CellValue[][], active: ActivePiece): number {
  let dy = 0;
  while (!collides(board, active, 0, dy + 1)) dy++;
  return active.y + dy;
}

// Stamp the active piece into a copy of the board.
export function lockPiece(board: CellValue[][], active: ActivePiece): CellValue[][] {
  const rotation = getRotation(active.type, active.rotationIndex);
  const newBoard = board.map(row => [...row]);
  const off = bufferOf(board);
  for (let r = 0; r < rotation.length; r++) {
    for (let c = 0; c < rotation[r].length; c++) {
      if (!rotation[r][c]) continue;
      const row = active.y + r;
      const col = active.x + c;
      if (row >= -off && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
        newBoard[row + off][col] = active.type as PieceType;
      }
    }
  }
  return newBoard;
}

// Remove completed lines and return the new board + count of lines cleared.
export function clearLines(board: CellValue[][]): { board: CellValue[][]; linesCleared: number } {
  const remaining = board.filter(row => row.some(cell => cell === 0));
  const linesCleared = board.length - remaining.length;
  const newRows: CellValue[][] = Array.from({ length: linesCleared }, () =>
    new Array(BOARD_COLS).fill(0) as CellValue[],
  );
  return { board: [...newRows, ...remaining], linesCleared };
}

// ---- Top-out (TETR.IO, as in Triangle.js) ----

// Where a new piece appears: centred, its spawn-state bounding box starting
// three rows above the field, so its lowest cells sit two rows above row 0.
export const SPAWN_Y = -3;

export function spawnPosition(type: PieceType): ActivePiece {
  const width = getRotation(type, 0)[0].length;
  return { type, rotationIndex: 0, x: Math.floor((BOARD_COLS - width) / 2), y: SPAWN_Y };
}

// Block out: a new piece can't appear where it spawns. Clutch: right after a
// line clear the piece may instead appear higher, at the first free row above
// its spawn. Returns the piece to play, or null when the player has topped out.
export function spawnOrBlockOut(board: CellValue[][], type: PieceType, afterClear: boolean): ActivePiece | null {
  const piece = spawnPosition(type);
  if (!collides(board, piece, 0, 0)) return piece;
  if (!afterClear) return null;
  for (let y = piece.y - 1; y >= -bufferOf(board); y--) {
    if (!collides(board, piece, 0, y - piece.y)) return { ...piece, y };
  }
  return null;
}

// Lock out: a piece that locks entirely above the visible field without
// clearing a line tops the player out.
export function isLockOut(piece: ActivePiece, linesCleared: number): boolean {
  if (linesCleared > 0) return false;
  const rotation = getRotation(piece.type, piece.rotationIndex);
  for (let r = 0; r < rotation.length; r++) {
    if (rotation[r].some(c => c !== 0) && piece.y + r >= 0) return false;
  }
  return true;
}

// Scoring per Tetris guideline
export function scoreForLines(lines: number, level: number): number {
  const base = [0, 100, 300, 500, 800];
  return (base[Math.min(lines, 4)] ?? 0) * level;
}

export function gravityInterval(level: number): number {
  return Math.max(50, 800 - (level - 1) * 50);
}

// Inject `lines` garbage rows at the bottom with a hole at gapCol.
// The top `lines` rows are pushed off the top of the board.
export function addGarbageLines(board: CellValue[][], lines: number, gapCol: number): CellValue[][] {
  if (lines <= 0) return board;
  const shifted = board.slice(lines);
  const makeRow = (): CellValue[] => {
    const row = new Array<CellValue>(BOARD_COLS).fill('X');
    row[gapCol] = 0;
    return row;
  };
  return [...shifted, ...Array.from({ length: lines }, makeRow)];
}

// Apply a rotation (delta = +1 CW, -1 CCW, +2 180°) with SRS+ wall kicks.
// Returns the rotated piece if any kick succeeds, or null if all kicks are blocked.
export function attemptRotation(board: CellValue[][], piece: ActivePiece, delta: number): ActivePiece | null {
  return attemptRotationWithKick(board, piece, delta)?.piece ?? null;
}

// Same as attemptRotation, also reporting which kick test succeeded (0 = no kick).
export function attemptRotationWithKick(
  board: CellValue[][],
  piece: ActivePiece,
  delta: number,
): { piece: ActivePiece; kickIndex: number } | null {
  const newIndex = ((piece.rotationIndex + delta) % 4 + 4) % 4;
  const kickList = getKicks(piece.type, piece.rotationIndex, newIndex);
  for (let k = 0; k < kickList.length; k++) {
    const [kdx, kdy] = kickList[k];
    const candidate: ActivePiece = { ...piece, rotationIndex: newIndex, x: piece.x + kdx, y: piece.y + kdy };
    if (!collides(board, candidate, 0, 0)) return { piece: candidate, kickIndex: k };
  }
  return null;
}

// Spin kind for a piece about to lock (see rules.ts). Corners outside the
// board count as filled.
export function detectSpin(
  board: CellValue[][],
  piece: ActivePiece,
  rotatedLast: boolean,
  tstKick: boolean,
): SpinKind {
  if (!rotatedLast) return 0;
  let corners = 0;
  let front = 0;
  const off = bufferOf(board);
  if (piece.type === 'T') {
    const [f0, f1] = T_FRONT_CORNERS[piece.rotationIndex];
    for (let i = 0; i < 4; i++) {
      const r = piece.y + T_CORNERS[i][0];
      const c = piece.x + T_CORNERS[i][1];
      if (r < -off || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS || board[r + off][c] !== 0) {
        corners++;
        if (i === f0 || i === f1) front++;
      }
    }
  }
  const immobile = collides(board, piece, -1, 0) && collides(board, piece, 1, 0)
    && collides(board, piece, 0, -1) && collides(board, piece, 0, 1);
  return classifySpin(piece.type, rotatedLast, corners, front, immobile, tstKick);
}
