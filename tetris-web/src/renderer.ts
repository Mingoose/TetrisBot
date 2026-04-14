import { GameState, CellValue, PieceType, ActivePiece } from './types';
import type { SprintReplay, VersusReplay, VersusReplayEntry } from './replay';
import { getReplayFrameIndex, getVersusReplayFrameIndex } from './replay';
import type { ClassificationResult } from './moveQuality';
import { qualityColor, qualityLabel } from './moveQuality';
import { PIECE_COLORS, getRotation } from './pieces';
import { hardDropY, BOARD_COLS, BOARD_ROWS } from './board';
import { CELL_SIZE, BOARD_OFFSET_X, BOARD_OFFSET_Y } from './editor';
import { VersusData, BotBoard, BotVsBotData } from './versus';
import { MAX_HISTORY } from './rewind';
import { KeyBindings, DEFAULT_KEYBINDINGS, keyLabel } from './settings';

const GRID_COLOR = '#1e1e3a';
const BG_COLOR = '#0d0d1a';
const PANEL_BG = '#12122a';
const TEXT_COLOR = '#cccccc';
const LABEL_COLOR = '#666688';

// Derived layout constants
export const BOARD_W = BOARD_COLS * CELL_SIZE;
export const BOARD_H = BOARD_ROWS * CELL_SIZE;
const HOLD_X = 10;
const HOLD_Y = BOARD_OFFSET_Y + 30;
const NEXT_X = BOARD_OFFSET_X + BOARD_W + 10;
const NEXT_Y = BOARD_OFFSET_Y;
export const CANVAS_W = BOARD_OFFSET_X + BOARD_W + 120;
export const CANVAS_H = BOARD_H + BOARD_OFFSET_Y * 2;

// Versus layout: bot board drawn to the right of the existing solo layout
const BOT_CELL_SIZE = 20;
const BOT_BOARD_X = CANVAS_W + 20;
const BOT_BOARD_Y = BOARD_OFFSET_Y;
const BOT_BOARD_W = BOARD_COLS * BOT_CELL_SIZE;
const BOT_BOARD_H = BOARD_ROWS * BOT_CELL_SIZE;
export const VERSUS_CANVAS_W = BOT_BOARD_X + BOT_BOARD_W + 20;

// Menu button rects — exported so main.ts can hit-test clicks
export interface ButtonRect { x: number; y: number; w: number; h: number; }
const BTN_W = 160;
const BTN_H = 56;
const BTN_GAP = 20;
const MENU_CX = VERSUS_CANVAS_W / 2;
const MENU_BY = CANVAS_H / 2 + 20;
// Row 1: three non-admin buttons centred
const MENU_ROW1_START = MENU_CX - (3 * BTN_W + 2 * BTN_GAP) / 2;
export const MENU_SPRINT_BTN:   ButtonRect = { x: MENU_ROW1_START,                         y: MENU_BY, w: BTN_W, h: BTN_H };
export const MENU_CREATIVE_BTN: ButtonRect = { x: MENU_ROW1_START + BTN_W + BTN_GAP,       y: MENU_BY, w: BTN_W, h: BTN_H };
export const MENU_VERSUS_BTN:   ButtonRect = { x: MENU_ROW1_START + 2 * (BTN_W + BTN_GAP), y: MENU_BY, w: BTN_W, h: BTN_H };
// Row 2: difficulty selection (shown instead of row 1 when picking versus difficulty)
const MENU_ROW2_Y = MENU_BY + BTN_H + BTN_GAP;
const MENU_ROW2_START = MENU_CX - (3 * BTN_W + 2 * BTN_GAP) / 2;
export const MENU_DIFF_EASY_BTN:   ButtonRect = { x: MENU_ROW1_START,                         y: MENU_BY, w: BTN_W, h: BTN_H };
export const MENU_DIFF_MEDIUM_BTN: ButtonRect = { x: MENU_ROW1_START + BTN_W + BTN_GAP,       y: MENU_BY, w: BTN_W, h: BTN_H };
export const MENU_DIFF_HARD_BTN:   ButtonRect = { x: MENU_ROW1_START + 2 * (BTN_W + BTN_GAP), y: MENU_BY, w: BTN_W, h: BTN_H };
// Row 3: three admin-only buttons centred
export const MENU_WATCH_BTN:  ButtonRect = { x: MENU_ROW2_START,                         y: MENU_ROW2_Y, w: BTN_W, h: BTN_H };
export const MENU_BVB_BTN:    ButtonRect = { x: MENU_ROW2_START + BTN_W + BTN_GAP,       y: MENU_ROW2_Y, w: BTN_W, h: BTN_H };
export const MENU_UPLOAD_BTN: ButtonRect = { x: MENU_ROW2_START + 2 * (BTN_W + BTN_GAP), y: MENU_ROW2_Y, w: BTN_W, h: BTN_H };

// Game review: "SEE BEST MOVE / HIDE BEST MOVE" button — placed below classification badge
// Classification badge starts at HOLD_Y+100+104=254, delta at +18=272; button at +36=290
export const GAME_REVIEW_BTN: ButtonRect = { x: HOLD_X, y: HOLD_Y + 240, w: 110, h: 28 };

// Bot-vs-bot layout: two boards side by side at BOT_CELL_SIZE
const BVB_CELL = BOT_CELL_SIZE;
const BVB_W = BOARD_COLS * BVB_CELL;
const BVB_H = BOARD_ROWS * BVB_CELL;
const BVB_SEP = 30;
const BVB_MARGIN = Math.floor((VERSUS_CANVAS_W - 2 * BVB_W - BVB_SEP) / 2);
const BVB_L_X = BVB_MARGIN;
const BVB_R_X = BVB_MARGIN + BVB_W + BVB_SEP;
const BVB_Y = BOT_BOARD_Y;

function cellColor(cell: CellValue): string {
  if (cell === 0) return GRID_COLOR;
  return PIECE_COLORS[cell as PieceType | 'X'];
}

function drawCell(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  color: string,
  alpha = 1,
): void {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
  // Highlight top-left edge
  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.fillRect(x + 1, y + 1, size - 2, 3);
  ctx.fillRect(x + 1, y + 1, 3, size - 2);
  ctx.globalAlpha = 1;
}

function drawPiecePreview(
  ctx: CanvasRenderingContext2D,
  type: PieceType,
  centerX: number,
  centerY: number,
  size: number,
): void {
  const rotation = getRotation(type, 0);
  const rows = rotation.length;
  const cols = rotation[0].length;
  const startX = centerX - (cols * size) / 2;
  const startY = centerY - (rows * size) / 2;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (rotation[r][c]) {
        drawCell(ctx, startX + c * size, startY + r * size, size, PIECE_COLORS[type]);
      }
    }
  }
}

function drawPanel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
): void {
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(text, x, y);
}

export function draw(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  versusData?: VersusData | null,
  customAiName: string | null = null,
  botVsBotData?: BotVsBotData | null,
  isAdmin = false,
  customAiError: string | null = null,
  customAiWarning: string | null = null,
  difficultyPending = false,
  bot1Name = 'BOT',
  bot2Name = 'BOT',
  keybindings: KeyBindings = DEFAULT_KEYBINDINGS,
): void {
  // Background — fill the full (possibly wider) canvas
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  if (state.mode === 'menu') {
    drawMenu(ctx, customAiName, isAdmin, difficultyPending);
    return;
  }

  if (state.variant === 'botvsbot' && botVsBotData) {
    drawBvbBoard(ctx, botVsBotData.bot1, BVB_L_X, BVB_Y, 'BOT 1', botVsBotData.bot1Combat, bot1Name);
    drawBvbBoard(ctx, botVsBotData.bot2, BVB_R_X, BVB_Y, 'BOT 2', botVsBotData.bot2Combat, bot2Name);
    drawGarbageBar(ctx, botVsBotData.bot1Combat.pendingGarbage, BVB_L_X - 8, BVB_Y, BVB_H, BVB_CELL);
    drawGarbageBar(ctx, botVsBotData.bot2Combat.pendingGarbage, BVB_R_X + BVB_W + 3, BVB_Y, BVB_H, BVB_CELL);
    // VS divider
    const vsx = Math.floor(VERSUS_CANVAS_W / 2);
    ctx.fillStyle = '#333355';
    ctx.fillRect(vsx - 1, BVB_Y, 2, BVB_H);
    ctx.fillStyle = '#555577';
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('VS', vsx, BVB_Y + BVB_H / 2);
    // Hint bar
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('R: rematch   Esc: menu', VERSUS_CANVAS_W / 2, BVB_Y + BVB_H + 14);
    if (customAiError) drawAiError(ctx, customAiError);
    else if (customAiWarning) drawAiWarning(ctx, customAiWarning);
    if (state.mode === 'gameover') drawBvbResult(ctx, botVsBotData.winner);
    return;
  }

  if (state.variant === 'watch' && versusData) {
    drawWatchBoard(ctx, versusData.bot);
    drawHoldBox(ctx, versusData.bot.hold, versusData.bot.holdUsed);
    drawNextQueue(ctx, versusData.bot.nextQueue);
    drawWatchHUD(ctx, versusData.bot, bot1Name);
    if (customAiError) drawAiError(ctx, customAiError);
    else if (customAiWarning) drawAiWarning(ctx, customAiWarning);
    if (versusData.bot.dead) drawOverlay(ctx, 'TOPPED OUT', '#ff6666');
    return;
  }

  drawBoard(ctx, state);
  drawHoldBox(ctx, state.hold, state.holdUsed);
  drawNextQueue(ctx, state.nextQueue);
  drawHUD(ctx, state, versusData ?? null, keybindings);

  if (versusData) {
    drawBotSection(ctx, versusData, bot1Name);
    drawGarbageBar(ctx, versusData.playerCombat.pendingGarbage,
      BOARD_OFFSET_X - 9, BOARD_OFFSET_Y, BOARD_H, CELL_SIZE);
    drawGarbageBar(ctx, versusData.botCombat.pendingGarbage,
      BOT_BOARD_X + BOT_BOARD_W + 3, BOT_BOARD_Y, BOT_BOARD_H, BOT_CELL_SIZE);
  }

  if (state.mode === 'countdown') drawCountdown(ctx, state);
  if (state.mode === 'paused') {
    const pauseSubtext = `${keyLabel(keybindings.pause)}: resume   Esc: menu`;
    const engineHint = state.variant === 'creative'
      ? `${keyLabel(keybindings.engineAnalysis)}: engine analysis`
      : undefined;
    drawOverlay(ctx, 'PAUSED', '#aaaaff', pauseSubtext, engineHint);
  }
  if (state.mode === 'gameover') {
    if (state.variant === 'sprint') drawSprintComplete(ctx, state);
    else if (state.variant === 'versus') drawVersusResult(ctx, versusData?.winner ?? null);
    else drawOverlay(ctx, 'GAME OVER', '#ff6666');
  }
  if (state.mode === 'editor') drawEditorBanner(ctx);
}

// ---- Shared board-drawing primitives ----

function drawGrid(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  cols: number, rows: number,
  cellSize: number,
): void {
  ctx.strokeStyle = '#16163a';
  ctx.lineWidth = 0.5;
  for (let c = 0; c <= cols; c++) {
    const x = bx + c * cellSize;
    ctx.beginPath(); ctx.moveTo(x, by); ctx.lineTo(x, by + rows * cellSize); ctx.stroke();
  }
  for (let r = 0; r <= rows; r++) {
    const y = by + r * cellSize;
    ctx.beginPath(); ctx.moveTo(bx, y); ctx.lineTo(bx + cols * cellSize, y); ctx.stroke();
  }
}

function drawLockedCells(
  ctx: CanvasRenderingContext2D,
  board: CellValue[][],
  bx: number, by: number,
  cellSize: number,
): void {
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const cell = board[r][c];
      if (cell !== 0) drawCell(ctx, bx + c * cellSize, by + r * cellSize, cellSize, cellColor(cell));
    }
  }
}

// Draws ghost (alpha 0.3) then active piece for any board at arbitrary offset/cell size.
function drawActivePiece(
  ctx: CanvasRenderingContext2D,
  active: ActivePiece,
  board: CellValue[][],
  bx: number, by: number,
  cellSize: number,
): void {
  const color = PIECE_COLORS[active.type];
  const rot = getRotation(active.type, active.rotationIndex);
  const ghostY = hardDropY(board, active);
  for (let r = 0; r < rot.length; r++) {
    for (let c = 0; c < rot[r].length; c++) {
      if (!rot[r][c]) continue;
      const row = ghostY + r; const col = active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
        drawCell(ctx, bx + col * cellSize, by + row * cellSize, cellSize, color, 0.3);
    }
  }
  for (let r = 0; r < rot.length; r++) {
    for (let c = 0; c < rot[r].length; c++) {
      if (!rot[r][c]) continue;
      const row = active.y + r; const col = active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
        drawCell(ctx, bx + col * cellSize, by + row * cellSize, cellSize, color);
    }
  }
}

function drawBoardBorder(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
  ctx.strokeStyle = '#3a3a6a';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);
}

// ---- Per-mode board draw functions ----

function drawBoard(ctx: CanvasRenderingContext2D, state: GameState): void {
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, state.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);
  if (state.mode !== 'editor')
    drawActivePiece(ctx, state.active, state.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
}

function drawHoldBox(ctx: CanvasRenderingContext2D, hold: PieceType | null, holdUsed: boolean): void {
  const w = 100, h = 80;
  drawPanel(ctx, HOLD_X, HOLD_Y, w, h);
  drawLabel(ctx, 'HOLD', HOLD_X + w / 2, HOLD_Y - 4);
  if (hold) {
    ctx.globalAlpha = holdUsed ? 0.4 : 1;
    drawPiecePreview(ctx, hold, HOLD_X + w / 2, HOLD_Y + h / 2, 20);
    ctx.globalAlpha = 1;
  }
}

function drawNextQueue(ctx: CanvasRenderingContext2D, queue: PieceType[]): void {
  const w = 100, itemH = 70;
  drawLabel(ctx, 'NEXT', NEXT_X + w / 2, NEXT_Y - 4);
  const count = Math.min(queue.length, 5);
  for (let i = 0; i < count; i++) {
    const panelY = NEXT_Y + i * (itemH + 4);
    drawPanel(ctx, NEXT_X, panelY, w, itemH);
    drawPiecePreview(ctx, queue[i], NEXT_X + w / 2, panelY + itemH / 2, 16);
  }
}

function drawHUDEntry(ctx: CanvasRenderingContext2D, label: string, value: string, x: number, y: number): void {
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(label, x, y);
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = 'bold 20px monospace';
  ctx.fillText(value, x, y + 22);
}

function drawHints(ctx: CanvasRenderingContext2D, hints: string[]): void {
  const hintY = CANVAS_H - 80;
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  for (let i = 0; i < hints.length; i++) {
    ctx.fillText(hints[i], HOLD_X, hintY + i * 14);
  }
}

function formatSprintTime(ms: number): string {
  const totalCs = Math.floor(ms / 10);
  const cs = totalCs % 100;
  const totalS = Math.floor(totalCs / 100);
  const s = totalS % 60;
  const m = Math.floor(totalS / 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

function drawHUD(ctx: CanvasRenderingContext2D, state: GameState, versusData: VersusData | null, kb: KeyBindings): void {
  const x = HOLD_X;
  const startY = HOLD_Y + 100;

  if (state.variant === 'sprint') {
    const elapsedMs = state.mode === 'gameover'
      ? state.sprintElapsedMs
      : (state.sprintStartTime > 0 ? state.lastFrameTime - state.sprintStartTime : 0);
    drawHUDEntry(ctx, 'TIME', formatSprintTime(elapsedMs), x, startY);
    drawHUDEntry(ctx, 'LEFT', String(Math.max(0, 40 - state.lines)), x, startY + 52);
    drawHints(ctx, [
      `${keyLabel(kb.rewind)}: restart`,
      `${keyLabel(kb.pause)}: pause`,
      'Esc: menu',
    ]);
  } else if (state.variant === 'versus') {
    drawHUDEntry(ctx, 'LINES', String(state.lines), x, startY);
    const combo = versusData ? versusData.playerCombat.combo : -1;
    if (combo >= 2) {
      ctx.fillStyle = '#ffcc44';
      ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${combo + 1}× combo`, x, startY + 52);
    }
    drawHints(ctx, [
      `${keyLabel(kb.rewind)}: restart`,
      `${keyLabel(kb.pause)}: pause`,
      'Esc: menu',
    ]);
  } else {
    drawHUDEntry(ctx, 'LINES', String(state.lines), x, startY);
    const rewindCount = state.history.length;
    ctx.fillStyle = rewindCount > 0 ? LABEL_COLOR : '#333355';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`rewind  ${rewindCount}/${MAX_HISTORY}`, x, startY + 52);
    drawHints(ctx, [
      `${keyLabel(kb.rewind)}: rewind`,
      `${keyLabel(kb.editor)}: editor`,
      `${keyLabel(kb.pause)}: pause`,
      'Esc: menu',
    ]);
  }
}

function drawOverlay(ctx: CanvasRenderingContext2D, text: string, color: string, subtext = 'R: restart   Esc: menu', extraHint?: string): void {
  const cx = BOARD_OFFSET_X + BOARD_W / 2;
  const cy = BOARD_OFFSET_Y + BOARD_H / 2;
  ctx.fillStyle = 'rgba(0,0,0,0.65)';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  ctx.fillStyle = color;
  ctx.font = 'bold 28px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(text, cx, cy - 10);
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '13px monospace';
  ctx.fillText(subtext, cx, cy + 20);
  if (extraHint) {
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '11px monospace';
    ctx.fillText(extraHint, cx, cy + 40);
  }
}

function drawSprintComplete(ctx: CanvasRenderingContext2D, state: GameState): void {
  const cx = BOARD_OFFSET_X + BOARD_W / 2;
  const cy = BOARD_OFFSET_Y + BOARD_H / 2;
  ctx.fillStyle = 'rgba(0,0,0,0.65)';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  ctx.fillStyle = '#66ffaa';
  ctx.font = 'bold 22px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('SPRINT COMPLETE', cx, cy - 18);
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = 'bold 26px monospace';
  ctx.fillText(formatSprintTime(state.sprintElapsedMs), cx, cy + 14);
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '13px monospace';
  ctx.fillText('R: play again   Esc: menu', cx, cy + 38);
  ctx.fillStyle = '#5555aa';
  ctx.font = '11px monospace';
  ctx.fillText('W: replay   G: game review   E: engine analysis', cx, cy + 56);
}

function drawMenuButton(
  ctx: CanvasRenderingContext2D,
  r: ButtonRect,
  label: string,
  color: string,
  subtitle: string,
): void {
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(r.x, r.y, r.w, r.h);
  ctx.strokeStyle = color + '88';
  ctx.lineWidth = 1;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  ctx.fillStyle = color;
  ctx.font = 'bold 18px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, r.x + r.w / 2, r.y + r.h / 2 - 4);
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '10px monospace';
  ctx.fillText(subtitle, r.x + r.w / 2, r.y + r.h / 2 + 14);
}

function drawBvbBoard(
  ctx: CanvasRenderingContext2D,
  bot: BotBoard,
  bx: number,
  by: number,
  label: string,
  combat: import('./versus').CombatState,
  aiName: string | null = null,
): void {
  ctx.fillStyle = '#888899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, bx + BVB_W / 2, by - 6);

  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(bx, by, BVB_W, BVB_H);
  drawGrid(ctx, bx, by, BOARD_COLS, BOARD_ROWS, BVB_CELL);
  drawLockedCells(ctx, bot.board, bx, by, BVB_CELL);
  if (!bot.dead) drawActivePiece(ctx, bot.active, bot.board, bx, by, BVB_CELL);
  drawBoardBorder(ctx, bx, by, BVB_W, BVB_H);

  const sy = by + BVB_H + 14;
  ctx.fillStyle = '#666688';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`LINES  ${bot.lines}`, bx + BVB_W / 2, sy);
  if (combat.combo >= 2) {
    ctx.fillStyle = '#ffcc44';
    ctx.font = 'bold 11px monospace';
    ctx.fillText(`${combat.combo + 1}× combo`, bx + BVB_W / 2, sy + 16);
  }
  if (aiName) {
    ctx.fillStyle = '#667788';
    ctx.font = '10px monospace';
    const short = aiName.length > 20 ? aiName.substring(0, 18) + '…' : aiName;
    ctx.fillText(short, bx + BVB_W / 2, sy + 30);
  }
}

function drawBvbResult(ctx: CanvasRenderingContext2D, winner: 'bot1' | 'bot2' | 'draw' | null): void {
  const cx = VERSUS_CANVAS_W / 2;
  const cy = BVB_Y + BVB_H / 2;
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.fillRect(0, BVB_Y, VERSUS_CANVAS_W, BVB_H);
  if (winner === 'draw' || winner === null) {
    ctx.fillStyle = '#aaaaff';
    ctx.font = 'bold 30px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('DRAW', cx, cy - 10);
  } else {
    const label = winner === 'bot1' ? 'BOT 1 WINS' : 'BOT 2 WINS';
    ctx.fillStyle = '#66ffaa';
    ctx.font = 'bold 30px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label, cx, cy - 10);
  }
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '13px monospace';
  ctx.fillText('R: rematch   Esc: menu', cx, cy + 20);
}

function drawMenu(ctx: CanvasRenderingContext2D, customAiName: string | null, isAdmin: boolean, difficultyPending: boolean): void {
  ctx.fillStyle = '#aaaaff';
  ctx.font = 'bold 52px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('TETRIS', MENU_CX, CANVAS_H / 2 - 50);

  if (difficultyPending) {
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '13px monospace';
    ctx.fillText('choose a difficulty', MENU_CX, CANVAS_H / 2 - 14);
    drawMenuButton(ctx, MENU_DIFF_EASY_BTN,   'EASY',   '#66ffaa', 'greedy one-piece');
    drawMenuButton(ctx, MENU_DIFF_MEDIUM_BTN, 'MEDIUM', '#ffcc44', 'beam search');
    drawMenuButton(ctx, MENU_DIFF_HARD_BTN,   'HARD',   '#ff6666', 'beam search+');
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '11px monospace';
    ctx.fillText('Esc to go back', MENU_CX, MENU_BY + BTN_H + 16);
    return;
  }

  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '13px monospace';
  ctx.fillText('choose a mode', MENU_CX, CANVAS_H / 2 - 14);
  // Row 1 — available to all users
  drawMenuButton(ctx, MENU_SPRINT_BTN,   'SPRINT',   '#66ffaa', 'race to 40 lines');
  drawMenuButton(ctx, MENU_CREATIVE_BTN, 'CREATIVE', '#aaaaff', 'free play + editor');
  drawMenuButton(ctx, MENU_VERSUS_BTN,   'VERSUS',   '#ffaa44', 'pick a difficulty');
  // Row 2 — admin only
  if (isAdmin) {
    drawMenuButton(ctx, MENU_WATCH_BTN,  'WATCH',      '#44ccff', 'watch AI play solo');
    drawMenuButton(ctx, MENU_BVB_BTN,    'BOT VS BOT', '#ff8844', 'two AIs compete');
    const aiSubtitle = customAiName
      ? (customAiName.length > 20 ? customAiName.substring(0, 18) + '…' : customAiName)
      : 'upload & select AI';
    drawMenuButton(ctx, MENU_UPLOAD_BTN, 'AI MANAGER', '#ff88cc', aiSubtitle);
  }
}

function drawGarbageBar(
  ctx: CanvasRenderingContext2D,
  pending: number,
  barX: number,
  boardY: number,
  boardH: number,
  cellSize: number,
): void {
  if (pending <= 0) return;
  const lines = Math.min(pending, BOARD_ROWS);
  const barH = lines * cellSize;
  ctx.fillStyle = '#dd2222';
  ctx.fillRect(barX, boardY + boardH - barH, 5, barH);
}

function drawBotSection(ctx: CanvasRenderingContext2D, data: VersusData, botName = 'BOT'): void {
  const bot = data.bot;
  const shortName = botName.length > 18 ? botName.substring(0, 16) + '…' : botName;
  ctx.fillStyle = '#888899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(shortName, BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y - 6);

  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);
  drawGrid(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOARD_COLS, BOARD_ROWS, BOT_CELL_SIZE);
  drawLockedCells(ctx, bot.board, BOT_BOARD_X, BOT_BOARD_Y, BOT_CELL_SIZE);
  if (!bot.dead) drawActivePiece(ctx, bot.active, bot.board, BOT_BOARD_X, BOT_BOARD_Y, BOT_CELL_SIZE);
  drawBoardBorder(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);

  const statsY = BOT_BOARD_Y + BOT_BOARD_H + 14;
  ctx.fillStyle = '#666688';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`LINES  ${bot.lines}`, BOT_BOARD_X + BOT_BOARD_W / 2, statsY);
  const combo = data.botCombat.combo;
  if (combo >= 2) {
    ctx.fillStyle = '#ffcc44';
    ctx.font = 'bold 11px monospace';
    ctx.fillText(`${combo + 1}× combo`, BOT_BOARD_X + BOT_BOARD_W / 2, statsY + 16);
  }
}

function drawWatchBoard(ctx: CanvasRenderingContext2D, bot: BotBoard): void {
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, bot.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);
  if (!bot.dead) drawActivePiece(ctx, bot.active, bot.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
}

function drawAiBanner(
  ctx: CanvasRenderingContext2D,
  message: string,
  bgColor: string,
  borderColor: string,
  textColor: string,
): void {
  const padding = 10;
  const maxWidth = VERSUS_CANVAS_W - padding * 2;
  ctx.font = '11px monospace';
  const words = message.split(' ');
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const test = current ? `${current} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth - 16 && current) {
      lines.push(current);
      current = word;
    } else {
      current = test;
    }
  }
  if (current) lines.push(current);

  const lineH = 16;
  const bannerH = lines.length * lineH + padding * 2;
  const bannerY = CANVAS_H - bannerH - 4;

  ctx.fillStyle = bgColor;
  ctx.fillRect(0, bannerY, VERSUS_CANVAS_W, bannerH);
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.strokeRect(0, bannerY, VERSUS_CANVAS_W, bannerH);

  ctx.fillStyle = textColor;
  ctx.textAlign = 'left';
  lines.forEach((line, i) => {
    ctx.fillText(line, padding, bannerY + padding + lineH * i + 11);
  });
}

function drawAiWarning(ctx: CanvasRenderingContext2D, message: string): void {
  drawAiBanner(ctx, message, 'rgba(60,40,0,0.92)', '#886600', '#ffcc44');
}

function drawAiError(ctx: CanvasRenderingContext2D, message: string): void {
  drawAiBanner(ctx, message, 'rgba(60,0,0,0.92)', '#882222', '#ff8888');
}

function drawWatchHUD(ctx: CanvasRenderingContext2D, bot: BotBoard, botName = 'BOT'): void {
  const x = HOLD_X;
  const startY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'LINES', String(bot.lines), x, startY);
  // Show bot name above the board
  const shortName = botName.length > 18 ? botName.substring(0, 16) + '…' : botName;
  ctx.fillStyle = '#888899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(shortName, BOARD_OFFSET_X + BOARD_W / 2, BOARD_OFFSET_Y - 6);
  drawHints(ctx, ['R: new game', 'Esc: menu']);
}

function drawVersusResult(ctx: CanvasRenderingContext2D, winner: 'player' | 'bot' | null): void {
  // Overlay on player board
  const cx = BOARD_OFFSET_X + BOARD_W / 2;
  const cy = BOARD_OFFSET_Y + BOARD_H / 2;
  ctx.fillStyle = 'rgba(0,0,0,0.65)';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  if (winner === 'player') {
    ctx.fillStyle = '#66ffaa';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('YOU WIN!', cx, cy - 10);
  } else {
    ctx.fillStyle = '#ff6666';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('GAME OVER', cx, cy - 10);
  }
  ctx.fillStyle = TEXT_COLOR;
  ctx.font = '13px monospace';
  ctx.fillText('R: rematch   Esc: menu', cx, cy + 20);
  ctx.fillStyle = '#5555aa';
  ctx.font = '11px monospace';
  ctx.fillText('W: replay   G: game review   E: engine analysis', cx, cy + 40);

  // Overlay on bot board if bot won (to show player what happened)
  if (winner === 'bot') {
    const bcx = BOT_BOARD_X + BOT_BOARD_W / 2;
    const bcy = BOT_BOARD_Y + BOT_BOARD_H / 2;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);
    ctx.fillStyle = '#66ffaa';
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('BOT WINS', bcx, bcy);
  }
}

function drawCountdown(ctx: CanvasRenderingContext2D, state: GameState): void {
  const cx = BOARD_OFFSET_X + BOARD_W / 2;
  const cy = BOARD_OFFSET_Y + BOARD_H / 2;
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  const n = Math.ceil(state.countdownMs / 1000);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 80px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(n), cx, cy);
  ctx.textBaseline = 'alphabetic';
}

function drawEditorBanner(ctx: CanvasRenderingContext2D): void {
  ctx.fillStyle = 'rgba(0,200,100,0.12)';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  ctx.fillStyle = '#00cc66';
  ctx.font = 'bold 13px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('EDITOR MODE — click to draw', BOARD_OFFSET_X + BOARD_W / 2, BOARD_OFFSET_Y + 18);
  ctx.fillStyle = LABEL_COLOR;
  ctx.font = '11px monospace';
  ctx.fillText('E: resume game', BOARD_OFFSET_X + BOARD_W / 2, BOARD_OFFSET_Y + 36);
}

// ---- Replay screen ----

/**
 * Draw a full replay frame for a sprint replay.
 * Covers the full canvas so it can be used as the sole draw call for a frame.
 */
export function drawReplayScreen(
  ctx: CanvasRenderingContext2D,
  replay: SprintReplay,
  replayElapsedMs: number,
  paused: boolean,
): void {
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  const frameIdx = getReplayFrameIndex(replay, replayElapsedMs);
  const isDone   = frameIdx === -1;
  const snap     = isDone ? null : replay.entries[frameIdx].snapshot;
  const lastSnap = replay.entries[replay.entries.length - 1]?.snapshot;

  const board      = isDone ? replay.finalBoard : snap!.board;
  const active     = isDone ? null             : snap!.active;
  const hold       = isDone ? (lastSnap?.hold ?? null)       : snap!.hold;
  const holdUsed   = isDone ? false                          : snap!.holdUsed;
  const nextQueue  = isDone ? (lastSnap?.nextQueue ?? [])    : snap!.nextQueue;
  const lines      = isDone ? 40                             : snap!.lines;

  // Board background + grid
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);

  // Locked cells
  drawLockedCells(ctx, board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);

  // Active piece at landing position (replay ghost)
  if (active) {
    const color = PIECE_COLORS[active.type];
    const rot   = getRotation(active.type, active.rotationIndex);
    for (let r = 0; r < rot.length; r++) {
      for (let c = 0; c < rot[r].length; c++) {
        if (!rot[r][c]) continue;
        const row = active.y + r;
        const col = active.x + c;
        if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
          drawCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, CELL_SIZE, color);
        }
      }
    }
  }

  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  // Side panels
  drawHoldBox(ctx, hold, holdUsed);
  drawNextQueue(ctx, nextQueue);

  // HUD
  const x      = HOLD_X;
  const startY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'REPLAY', formatSprintTime(isDone ? replay.finalElapsedMs : replayElapsedMs), x, startY);
  drawHUDEntry(ctx, 'LINES',  String(lines), x, startY + 52);
  drawHints(ctx, paused
    ? ['Space: resume', 'R: restart', 'Esc: back']
    : ['Space: pause',  'R: restart', 'Esc: back'],
  );

  // Overlays
  if (paused && !isDone) {
    drawOverlay(ctx, 'PAUSED', '#aaaaff', 'Space: resume   Esc: back');
  } else if (isDone) {
    // Sprint complete overlay — same style as the live sprint complete screen
    const cx = BOARD_OFFSET_X + BOARD_W / 2;
    const cy = BOARD_OFFSET_Y + BOARD_H / 2;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
    ctx.fillStyle = '#66ffaa';
    ctx.font = 'bold 22px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('SPRINT COMPLETE', cx, cy - 18);
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = 'bold 26px monospace';
    ctx.fillText(formatSprintTime(replay.finalElapsedMs), cx, cy + 14);
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '13px monospace';
    ctx.fillText('R: watch again   Esc: back', cx, cy + 38);
    ctx.fillStyle = '#5555aa';
    ctx.font = '11px monospace';
    ctx.fillText('G: game review   E: engine analysis', cx, cy + 56);
  }
}

// ---- Versus replay screen ----

/**
 * Draw a full replay frame for a versus replay, covering the full canvas.
 */
export function drawVersusReplayScreen(
  ctx: CanvasRenderingContext2D,
  replay: VersusReplay,
  replayElapsedMs: number,
  paused: boolean,
): void {
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  const frameIdx = getVersusReplayFrameIndex(replay, replayElapsedMs);
  const isDone    = frameIdx === -1;
  const entry     = isDone ? null : replay.entries[frameIdx];
  const lastEntry = replay.entries[replay.entries.length - 1];

  // ---- Player board ----
  const playerBoard   = isDone ? replay.finalPlayerBoard : entry!.playerSnapshot.board;
  const playerActive  = isDone ? null                    : entry!.playerSnapshot.active;
  const playerHold    = isDone ? (lastEntry?.playerSnapshot.hold    ?? null) : entry!.playerSnapshot.hold;
  const playerHoldUsed = isDone ? false                              : entry!.playerSnapshot.holdUsed;
  const playerNext    = isDone ? (lastEntry?.playerSnapshot.nextQueue ?? []) : entry!.playerSnapshot.nextQueue;
  const playerLines   = isDone ? (lastEntry?.playerSnapshot.lines   ?? 0)   : entry!.playerSnapshot.lines;

  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, playerBoard, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);

  // Draw active piece at its stored (landing) position — no ghost, same as sprint replay
  if (playerActive) {
    const color = PIECE_COLORS[playerActive.type];
    const rot   = getRotation(playerActive.type, playerActive.rotationIndex);
    for (let r = 0; r < rot.length; r++) {
      for (let c = 0; c < rot[r].length; c++) {
        if (!rot[r][c]) continue;
        const row = playerActive.y + r;
        const col = playerActive.x + c;
        if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
          drawCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, CELL_SIZE, color);
        }
      }
    }
  }
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  drawHoldBox(ctx, playerHold, playerHoldUsed);
  drawNextQueue(ctx, playerNext);

  // ---- Bot board ----
  const botBoard  = isDone ? replay.finalBotBoard                    : entry!.botSnapshot.board;
  const botActive = isDone ? null                                     : entry!.botSnapshot.active;
  const botLines  = isDone ? (lastEntry?.botSnapshot.lines ?? 0)     : entry!.botSnapshot.lines;
  const botDead   = isDone ? true                                     : entry!.botSnapshot.dead;

  ctx.fillStyle = '#888899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('BOT', BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y - 6);

  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);
  drawGrid(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOARD_COLS, BOARD_ROWS, BOT_CELL_SIZE);
  drawLockedCells(ctx, botBoard, BOT_BOARD_X, BOT_BOARD_Y, BOT_CELL_SIZE);

  // Draw bot's active piece at stored position (no ghost)
  if (botActive && !botDead) {
    const color = PIECE_COLORS[botActive.type];
    const rot   = getRotation(botActive.type, botActive.rotationIndex);
    for (let r = 0; r < rot.length; r++) {
      for (let c = 0; c < rot[r].length; c++) {
        if (!rot[r][c]) continue;
        const row = botActive.y + r;
        const col = botActive.x + c;
        if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
          drawCell(ctx, BOT_BOARD_X + col * BOT_CELL_SIZE, BOT_BOARD_Y + row * BOT_CELL_SIZE, BOT_CELL_SIZE, color);
        }
      }
    }
  }
  drawBoardBorder(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);

  ctx.fillStyle = '#666688';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`LINES  ${botLines}`, BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y + BOT_BOARD_H + 14);

  // ---- HUD ----
  const hudX = HOLD_X;
  const hudY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'REPLAY', formatSprintTime(isDone ? replay.finalElapsedMs : replayElapsedMs), hudX, hudY);
  drawHUDEntry(ctx, 'LINES',  String(playerLines), hudX, hudY + 52);
  drawHints(ctx, paused
    ? ['Space: resume', 'R: restart', 'Esc: back']
    : ['Space: pause',  'R: restart', 'Esc: back'],
  );

  // ---- Overlays ----
  if (paused && !isDone) {
    drawOverlay(ctx, 'PAUSED', '#aaaaff', 'Space: resume   Esc: back');
  } else if (isDone) {
    const cx = BOARD_OFFSET_X + BOARD_W / 2;
    const cy = BOARD_OFFSET_Y + BOARD_H / 2;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
    if (replay.winner === 'player') {
      ctx.fillStyle = '#66ffaa';
      ctx.font = 'bold 28px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('YOU WIN!', cx, cy - 10);
    } else {
      ctx.fillStyle = '#ff6666';
      ctx.font = 'bold 28px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('GAME OVER', cx, cy - 10);
    }
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = '13px monospace';
    ctx.fillText('R: watch again   Esc: back', cx, cy + 20);
    ctx.fillStyle = '#5555aa';
    ctx.font = '11px monospace';
    ctx.fillText('G: game review   E: engine analysis', cx, cy + 38);

    if (replay.winner === 'bot') {
      const bcx = BOT_BOARD_X + BOT_BOARD_W / 2;
      const bcy = BOT_BOARD_Y + BOT_BOARD_H / 2;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);
      ctx.fillStyle = '#66ffaa';
      ctx.font = 'bold 20px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('BOT WINS', bcx, bcy);
    }
  }
}

// ---- Game review screens ----
// These draw the board/panels from a replay snapshot. The caller (main.ts)
// draws the engine overlay on top to show analysis for that position.

/**
 * Draw a sprint review frame. Shows the board state just before move `moveIdx`
 * was played. The engine overlay should be drawn on top by the caller.
 */
export function drawReviewScreen(
  ctx: CanvasRenderingContext2D,
  replay: SprintReplay,
  moveIdx: number,
): void {
  const entry = replay.entries[Math.min(moveIdx, replay.entries.length - 1)];
  const snap  = entry.snapshot;

  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  // Board — engine overlay will replace this once analysis arrives
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, snap.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);

  // Show the actual piece played at its landing position
  const color = PIECE_COLORS[snap.active.type];
  const rot   = getRotation(snap.active.type, snap.active.rotationIndex);
  for (let r = 0; r < rot.length; r++) {
    for (let c = 0; c < rot[r].length; c++) {
      if (!rot[r][c]) continue;
      const row = snap.active.y + r;
      const col = snap.active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
        drawCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, CELL_SIZE, color);
    }
  }
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  drawHoldBox(ctx, snap.hold, snap.holdUsed);
  drawNextQueue(ctx, snap.nextQueue);

  const x = HOLD_X;
  const startY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'REVIEW', `${moveIdx + 1}/${replay.entries.length}`, x, startY);
  drawHUDEntry(ctx, 'LINES', String(snap.lines), x, startY + 52);
  drawHints(ctx, ['←→: step moves', '↑↓: cycle lines', 'Esc: back']);
}

/**
 * Draw a versus review frame. Shows the player's board state just before move
 * `moveIdx`. The engine overlay should be drawn on top by the caller.
 */
export function drawVersusReviewScreen(
  ctx: CanvasRenderingContext2D,
  replay: VersusReplay,
  moveIdx: number,
): void {
  const entry     = replay.entries[Math.min(moveIdx, replay.entries.length - 1)];
  const playerSnap = entry.playerSnapshot;
  const botSnap    = entry.botSnapshot;

  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  // ---- Player board ----
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, playerSnap.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);

  // Show the actual piece played at its landing position
  const pColor = PIECE_COLORS[playerSnap.active.type];
  const pRot   = getRotation(playerSnap.active.type, playerSnap.active.rotationIndex);
  for (let r = 0; r < pRot.length; r++) {
    for (let c = 0; c < pRot[r].length; c++) {
      if (!pRot[r][c]) continue;
      const row = playerSnap.active.y + r;
      const col = playerSnap.active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
        drawCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, CELL_SIZE, pColor);
    }
  }
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  drawHoldBox(ctx, playerSnap.hold, playerSnap.holdUsed);
  drawNextQueue(ctx, playerSnap.nextQueue);

  // ---- Bot board ----
  ctx.fillStyle = '#888899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('BOT', BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y - 6);

  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);
  drawGrid(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOARD_COLS, BOARD_ROWS, BOT_CELL_SIZE);
  drawLockedCells(ctx, botSnap.board, BOT_BOARD_X, BOT_BOARD_Y, BOT_CELL_SIZE);

  if (!botSnap.dead) {
    const bColor = PIECE_COLORS[botSnap.active.type];
    const bRot   = getRotation(botSnap.active.type, botSnap.active.rotationIndex);
    for (let r = 0; r < bRot.length; r++) {
      for (let c = 0; c < bRot[r].length; c++) {
        if (!bRot[r][c]) continue;
        const row = botSnap.active.y + r;
        const col = botSnap.active.x + c;
        if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
          drawCell(ctx, BOT_BOARD_X + col * BOT_CELL_SIZE, BOT_BOARD_Y + row * BOT_CELL_SIZE, BOT_CELL_SIZE, bColor);
      }
    }
  }
  drawBoardBorder(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);

  ctx.fillStyle = '#666688';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`LINES  ${botSnap.lines}`, BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y + BOT_BOARD_H + 14);

  // ---- HUD ----
  const x = HOLD_X;
  const startY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'REVIEW', `${moveIdx + 1}/${replay.entries.length}`, x, startY);
  drawHUDEntry(ctx, 'LINES', String(playerSnap.lines), x, startY + 52);
  drawHints(ctx, ['←→: step moves', '↑↓: cycle lines', 'Esc: back']);
}

// ---- Game review screens (classification mode) ----

function drawClassificationHUD(
  ctx: CanvasRenderingContext2D,
  x: number,
  startY: number,
  classification: ClassificationResult | null,
  showBestMove: boolean,
): void {
  const badgeY = startY + 104;

  if (!classification) {
    ctx.fillStyle = '#444466';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('analyzing...', x, badgeY);
  } else {
    // Quality label
    ctx.fillStyle = qualityColor(classification.quality);
    ctx.font = 'bold 16px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(qualityLabel(classification.quality).toUpperCase(), x, badgeY);

    // Score delta
    ctx.font = '10px monospace';
    if (classification.delta !== null) {
      ctx.fillStyle = '#666688';
      ctx.fillText(`\u0394 ${classification.delta.toFixed(1)}`, x, badgeY + 18);
    } else {
      ctx.fillStyle = '#664444';
      ctx.fillText('not in top 30', x, badgeY + 18);
    }
  }

  // SEE BEST MOVE / HIDE BEST MOVE button
  const btn = GAME_REVIEW_BTN;
  ctx.fillStyle = showBestMove ? '#1a1a3a' : '#13132a';
  ctx.fillRect(btn.x, btn.y, btn.w, btn.h);
  ctx.strokeStyle = showBestMove ? '#5555aa' : '#3a3a66';
  ctx.lineWidth = 1;
  ctx.strokeRect(btn.x + 0.5, btn.y + 0.5, btn.w - 1, btn.h - 1);
  ctx.fillStyle = showBestMove ? '#8888cc' : '#5555aa';
  ctx.font = 'bold 10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(showBestMove ? 'HIDE BEST' : 'SEE BEST MOVE', btn.x + btn.w / 2, btn.y + btn.h / 2 + 4);
}

/**
 * Draw a sprint game review frame — board from snapshot, classification HUD,
 * and the SEE BEST MOVE button. Caller draws engine overlay on top when active.
 */
export function drawGameReviewScreen(
  ctx: CanvasRenderingContext2D,
  replay: SprintReplay,
  moveIdx: number,
  classification: ClassificationResult | null,
  showBestMove: boolean,
  timestamp: number,
): void {
  const entry = replay.entries[Math.min(moveIdx, replay.entries.length - 1)];
  const snap  = entry.snapshot;

  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  // Board — engine overlay replaces this when showBestMove is true
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, snap.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);

  // Pulse the placed piece to make it clearly identifiable
  const pieceAlpha = 0.75 + 0.25 * Math.sin(timestamp / 200);
  const color = PIECE_COLORS[snap.active.type];
  const rot   = getRotation(snap.active.type, snap.active.rotationIndex);
  for (let r = 0; r < rot.length; r++) {
    for (let c = 0; c < rot[r].length; c++) {
      if (!rot[r][c]) continue;
      const row = snap.active.y + r;
      const col = snap.active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
        drawCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, CELL_SIZE, color, pieceAlpha);
    }
  }
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  drawHoldBox(ctx, snap.hold, snap.holdUsed);
  drawNextQueue(ctx, snap.nextQueue);

  const x = HOLD_X;
  const startY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'REVIEW', `${moveIdx + 1}/${replay.entries.length}`, x, startY);
  drawHUDEntry(ctx, 'LINES', String(snap.lines), x, startY + 52);
  drawClassificationHUD(ctx, x, startY, classification, showBestMove);

  const hints = ['←→: step moves', showBestMove ? 'B: hide best move' : 'B: see best move', 'Esc: back'];
  if (showBestMove) hints.splice(2, 0, '↑↓: cycle lines');
  drawHints(ctx, hints);
}

/**
 * Draw a versus game review frame — player board + bot board, classification HUD,
 * and the SEE BEST MOVE button. Caller draws engine overlay on top when active.
 * `entries` should be the player-lock-only filtered subset of the replay.
 */
export function drawVersusGameReviewScreen(
  ctx: CanvasRenderingContext2D,
  entries: VersusReplayEntry[],
  moveIdx: number,
  classification: ClassificationResult | null,
  showBestMove: boolean,
  timestamp: number,
): void {
  const entry      = entries[Math.min(moveIdx, entries.length - 1)];
  const playerSnap = entry.playerSnapshot;
  const botSnap    = entry.botSnapshot;

  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, VERSUS_CANVAS_W, CANVAS_H);

  // ---- Player board ----
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);
  drawGrid(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_COLS, BOARD_ROWS, CELL_SIZE);
  drawLockedCells(ctx, playerSnap.board, BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE);

  // Pulse the placed piece to make it clearly identifiable
  const pieceAlpha = 0.75 + 0.25 * Math.sin(timestamp / 200);
  const pColor = PIECE_COLORS[playerSnap.active.type];
  const pRot   = getRotation(playerSnap.active.type, playerSnap.active.rotationIndex);
  for (let r = 0; r < pRot.length; r++) {
    for (let c = 0; c < pRot[r].length; c++) {
      if (!pRot[r][c]) continue;
      const row = playerSnap.active.y + r;
      const col = playerSnap.active.x + c;
      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
        drawCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, CELL_SIZE, pColor, pieceAlpha);
    }
  }
  drawBoardBorder(ctx, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  drawHoldBox(ctx, playerSnap.hold, playerSnap.holdUsed);
  drawNextQueue(ctx, playerSnap.nextQueue);

  // ---- Bot board ----
  ctx.fillStyle = '#888899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('BOT', BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y - 6);

  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);
  drawGrid(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOARD_COLS, BOARD_ROWS, BOT_CELL_SIZE);
  drawLockedCells(ctx, botSnap.board, BOT_BOARD_X, BOT_BOARD_Y, BOT_CELL_SIZE);

  if (!botSnap.dead) {
    const bColor = PIECE_COLORS[botSnap.active.type];
    const bRot   = getRotation(botSnap.active.type, botSnap.active.rotationIndex);
    for (let r = 0; r < bRot.length; r++) {
      for (let c = 0; c < bRot[r].length; c++) {
        if (!bRot[r][c]) continue;
        const row = botSnap.active.y + r;
        const col = botSnap.active.x + c;
        if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS)
          drawCell(ctx, BOT_BOARD_X + col * BOT_CELL_SIZE, BOT_BOARD_Y + row * BOT_CELL_SIZE, BOT_CELL_SIZE, bColor);
      }
    }
  }
  drawBoardBorder(ctx, BOT_BOARD_X, BOT_BOARD_Y, BOT_BOARD_W, BOT_BOARD_H);

  ctx.fillStyle = '#666688';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`LINES  ${botSnap.lines}`, BOT_BOARD_X + BOT_BOARD_W / 2, BOT_BOARD_Y + BOT_BOARD_H + 14);

  // ---- HUD ----
  const x = HOLD_X;
  const startY = HOLD_Y + 100;
  drawHUDEntry(ctx, 'REVIEW', `${moveIdx + 1}/${entries.length}`, x, startY);
  drawHUDEntry(ctx, 'LINES', String(playerSnap.lines), x, startY + 52);
  drawClassificationHUD(ctx, x, startY, classification, showBestMove);

  const hints = ['←→: step moves', 'Esc: back'];
  if (showBestMove) hints.splice(1, 0, '↑↓: cycle lines');
  drawHints(ctx, hints);
}
