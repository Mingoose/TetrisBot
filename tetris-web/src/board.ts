import { ActivePiece, CellValue, PieceType } from './types';
import { getRotation } from './pieces';

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
