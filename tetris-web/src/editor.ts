import { GameState, CellValue } from './types';
import { pushHistory } from './rewind';
import { BOARD_COLS, BOARD_ROWS } from './board';

// Must match LOCK_DELAY_MS in game.ts (not imported to avoid circular dependency).
const LOCK_DELAY_MS = 500;

export const CELL_SIZE = 30;
export const BOARD_OFFSET_X = 120; // pixels from canvas left to board left edge
export const BOARD_OFFSET_Y = 20;  // pixels from canvas top to board top edge

let paintMode: 'on' | 'off' | null = null;
let isMouseDown = false;

export function setupEditor(canvas: HTMLCanvasElement, state: GameState): void {
  canvas.addEventListener('mousedown', (e) => {
    if (state.mode !== 'editor') return;
    isMouseDown = true;
    const { row, col } = pixelToCell(canvas, e);
    if (!inBounds(row, col)) return;
    paintMode = state.board[row][col] === 0 ? 'on' : 'off';
    toggleCell(state, row, col);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (state.mode !== 'editor' || !isMouseDown || paintMode === null) return;
    const { row, col } = pixelToCell(canvas, e);
    if (!inBounds(row, col)) return;
    const isEmpty = state.board[row][col] === 0;
    if (paintMode === 'on' && isEmpty) toggleCell(state, row, col);
    if (paintMode === 'off' && !isEmpty) toggleCell(state, row, col);
  });

  canvas.addEventListener('mouseup', () => {
    isMouseDown = false;
    paintMode = null;
  });

  canvas.addEventListener('mouseleave', () => {
    isMouseDown = false;
    paintMode = null;
  });
}

function pixelToCell(canvas: HTMLCanvasElement, e: MouseEvent): { row: number; col: number } {
  const rect = canvas.getBoundingClientRect();
  const pixelX = e.clientX - rect.left;
  const pixelY = e.clientY - rect.top;
  const col = Math.floor((pixelX - BOARD_OFFSET_X) / CELL_SIZE);
  const row = Math.floor((pixelY - BOARD_OFFSET_Y) / CELL_SIZE);
  return { row, col };
}

function inBounds(row: number, col: number): boolean {
  return row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS;
}

function toggleCell(state: GameState, row: number, col: number): void {
  const current = state.board[row][col];
  (state.board[row][col] as CellValue) = current === 0 ? 'X' : 0;
}

// Call when entering editor mode from playing
export function enterEditor(state: GameState): void {
  state.mode = 'editor';
}

// Call when leaving editor mode back to playing
export function exitEditor(state: GameState, spawnFn: (s: GameState) => void): void {
  pushHistory(state); // allow rewinding past the edit session
  spawnFn(state);
  state.mode = 'playing';
  state.lockDelayMs = LOCK_DELAY_MS;
  state.lockResetCount = 0;
  state.gravityAccumMs = 0;
}
