import { ActivePiece, CellValue, PieceType } from './types';
import { getRotation, getWallKicks } from './pieces';

export const BOARD_COLS = 10;
export const BOARD_ROWS = 20;

export function emptyBoard(): CellValue[][] {
  return Array.from({ length: BOARD_ROWS }, () => new Array(BOARD_COLS).fill(0) as CellValue[]);
}

// Check if placing `active` at (active.x + dx, active.y + dy) collides with
// board walls or any filled cell.
export function collides(
  board: CellValue[][],
  active: ActivePiece,
  dx: number,
  dy: number,
): boolean {
  const rotation = getRotation(active.type, active.rotationIndex);
  const newX = active.x + dx;
  const newY = active.y + dy;
  for (let r = 0; r < rotation.length; r++) {
    for (let c = 0; c < rotation[r].length; c++) {
      if (!rotation[r][c]) continue;
      const col = newX + c;
      const row = newY + r;
      if (col < 0 || col >= BOARD_COLS) return true;
      if (row >= BOARD_ROWS) return true;
      if (row >= 0 && board[row][col] !== 0) return true;
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
  for (let r = 0; r < rotation.length; r++) {
    for (let c = 0; c < rotation[r].length; c++) {
      if (!rotation[r][c]) continue;
      const row = active.y + r;
      const col = active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
        newBoard[row][col] = active.type as PieceType;
      }
    }
  }
  return newBoard;
}

// Remove completed lines and return the new board + count of lines cleared.
export function clearLines(board: CellValue[][]): { board: CellValue[][]; linesCleared: number } {
  const remaining = board.filter(row => row.some(cell => cell === 0));
  const linesCleared = BOARD_ROWS - remaining.length;
  const newRows: CellValue[][] = Array.from({ length: linesCleared }, () =>
    new Array(BOARD_COLS).fill(0) as CellValue[],
  );
  return { board: [...newRows, ...remaining], linesCleared };
}

// Combined lock + clear optimised for the AI beam search.
// Only deep-copies the 2–4 rows the piece touches instead of all 20,
// and only checks those rows for full-line detection.
export function lockAndClear(
  board: CellValue[][],
  active: ActivePiece,
): { board: CellValue[][]; linesCleared: number } {
  const rotation = getRotation(active.type, active.rotationIndex);

  // Shallow-copy the outer array; deep-copy only rows the piece writes to.
  const newBoard: CellValue[][] = board.slice();
  for (let pr = 0; pr < rotation.length; pr++) {
    const row = active.y + pr;
    if (row < 0 || row >= BOARD_ROWS) continue;
    if (!rotation[pr].some(c => c !== 0)) continue;
    newBoard[row] = board[row].slice();
    for (let pc = 0; pc < rotation[pr].length; pc++) {
      if (!rotation[pr][pc]) continue;
      const col = active.x + pc;
      if (col >= 0 && col < BOARD_COLS) newBoard[row][col] = active.type as PieceType;
    }
  }

  // Find full lines (only touched rows can become full).
  let fullMask = 0; // bitmask of rows 0–19 that are full (up to 4 can be set)
  let linesCleared = 0;
  for (let pr = 0; pr < rotation.length; pr++) {
    const row = active.y + pr;
    if (row < 0 || row >= BOARD_ROWS) continue;
    let full = true;
    for (let c = 0; c < BOARD_COLS; c++) { if (newBoard[row][c] === 0) { full = false; break; } }
    if (full) { fullMask |= (1 << row); linesCleared++; }
  }

  if (linesCleared === 0) return { board: newBoard, linesCleared: 0 };

  // Rebuild board: linesCleared empty rows on top, then surviving rows in order.
  const result: CellValue[][] = new Array(BOARD_ROWS);
  for (let i = 0; i < linesCleared; i++) result[i] = new Array<CellValue>(BOARD_COLS).fill(0);
  let wi = linesCleared;
  for (let r = 0; r < BOARD_ROWS; r++) {
    if (!(fullMask & (1 << r))) result[wi++] = newBoard[r];
  }
  return { board: result, linesCleared };
}

// Game over if any cell in the top two rows is filled after a lock.
export function isGameOver(board: CellValue[][]): boolean {
  return board[0].some(c => c !== 0) || board[1].some(c => c !== 0);
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

// Apply a rotation (delta = +1 CW, -1 CCW, +2 180°) with SRS wall kicks.
// Returns the rotated piece if any kick succeeds, or null if all kicks are blocked.
export function attemptRotation(board: CellValue[][], piece: ActivePiece, delta: number): ActivePiece | null {
  const newIndex = ((piece.rotationIndex + delta) % 4 + 4) % 4;
  // CW / 180°: use kicks from the 'from' state. CCW: use negated kicks from the 'to' state.
  const kickIndex = delta > 0 ? piece.rotationIndex : newIndex;
  const kicks = getWallKicks(piece.type, kickIndex);
  const kickList: Array<[number, number]> = delta < 0
    ? kicks.map(([dx, dy]) => [-dx, -dy] as [number, number])
    : kicks;
  for (const [kdx, kdy] of kickList) {
    const candidate: ActivePiece = { ...piece, rotationIndex: newIndex, x: piece.x + kdx, y: piece.y + kdy };
    if (!collides(board, candidate, 0, 0)) return candidate;
  }
  return null;
}

// Count how many of the 4 corners of a T piece's 3×3 bounding box are occupied
// (either by board cells or out-of-bounds). Used for 3-corner T-spin detection.
export function countTSpinCorners(board: CellValue[][], piece: ActivePiece): number {
  const corners: [number, number][] = [
    [piece.y,     piece.x    ],
    [piece.y,     piece.x + 2],
    [piece.y + 2, piece.x    ],
    [piece.y + 2, piece.x + 2],
  ];
  return corners.filter(([r, c]) =>
    r < 0 || r >= BOARD_ROWS || c < 0 || c >= BOARD_COLS || board[r][c] !== 0,
  ).length;
}
