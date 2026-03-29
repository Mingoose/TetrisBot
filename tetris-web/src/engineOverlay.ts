import type { CellValue } from './types';
import type { EngineAnalysis } from './engine';
import { PIECE_COLORS, getRotation } from './pieces';
import { CELL_SIZE, BOARD_OFFSET_X, BOARD_OFFSET_Y } from './editor';
import { BOARD_COLS, BOARD_ROWS } from './board';

// Layout — derived from the same constants as renderer.ts
const BOARD_W  = BOARD_COLS * CELL_SIZE;   // 300
const BOARD_H  = BOARD_ROWS * CELL_SIZE;   // 600
const NEXT_X   = BOARD_OFFSET_X + BOARD_W + 10;  // 430
const CANVAS_H = BOARD_H + BOARD_OFFSET_Y * 2;   // 640

// Rankings panel sits in the right rail, below the NEXT queue.
// NEXT queue: 5 panels × (70 + 4)px = 370px, starting at NEXT_Y = 20 → bottom ≈ 390.
const PANEL_X = NEXT_X;
const PANEL_Y = 393;
const PANEL_W = 100;
const PANEL_H = CANVAS_H - PANEL_Y - 6;  // ~241px
const ROW_H   = 38;

/**
 * Render the engine analysis overlay on top of an already-drawn game frame.
 *
 * Draws:
 *  1. Ghost pieces for the selected line's moves on the board (decreasing opacity).
 *  2. A rankings panel in the right rail listing top lines with scores.
 *  3. A status/hint banner at the bottom of the board area.
 *
 * @param ctx          - The canvas rendering context (already has the game drawn on it).
 * @param analysis     - Analysis result, or null if still computing.
 * @param selectedLine - Index into analysis.lines for the currently highlighted line.
 * @param board        - Current board state (used only to validate ghost row bounds).
 */
export function drawEngineOverlay(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
  selectedLine: number,
  _board: CellValue[][],
): void {
  drawGhosts(ctx, analysis, selectedLine);
  drawPanel(ctx, analysis, selectedLine);
  drawBanner(ctx, analysis);
}

// ---- Ghost pieces ----

function drawGhosts(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
  selectedLine: number,
): void {
  if (!analysis || analysis.lines.length === 0) return;
  const line = analysis.lines[Math.min(selectedLine, analysis.lines.length - 1)];
  if (!line) return;

  // Opacity schedule: first move solid, second translucent, rest faint.
  const ALPHAS = [0.92, 0.48, 0.24, 0.14, 0.10];

  for (let i = 0; i < line.moves.length; i++) {
    const move  = line.moves[i];
    const alpha = ALPHAS[Math.min(i, ALPHAS.length - 1)];
    const rot   = getRotation(move.pieceType, move.rotationIndex);
    const color = PIECE_COLORS[move.pieceType];

    for (let r = 0; r < rot.length; r++) {
      for (let c = 0; c < rot[r].length; c++) {
        if (!rot[r][c]) continue;
        const row = move.y + r;
        const col = move.x + c;
        if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) continue;

        const px = BOARD_OFFSET_X + col * CELL_SIZE;
        const py = BOARD_OFFSET_Y + row * CELL_SIZE;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2);

        // Matching edge-highlight style from renderer.ts drawCell
        ctx.fillStyle = 'rgba(255,255,255,0.18)';
        ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, 3);
        ctx.fillRect(px + 1, py + 1, 3, CELL_SIZE - 2);
      }
    }
  }
  ctx.globalAlpha = 1;
}

// ---- Rankings panel ----

function drawPanel(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
  selectedLine: number,
): void {
  // Background
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(PANEL_X, PANEL_Y, PANEL_W, PANEL_H);
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.strokeRect(PANEL_X + 0.5, PANEL_Y + 0.5, PANEL_W - 1, PANEL_H - 1);

  // Header
  ctx.fillStyle = '#333366';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('ENGINE LINES', PANEL_X + PANEL_W / 2, PANEL_Y + 13);

  if (!analysis) {
    ctx.fillStyle = '#555577';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('searching...', PANEL_X + PANEL_W / 2, PANEL_Y + PANEL_H / 2 - 4);
    return;
  }

  if (analysis.lines.length === 0) {
    ctx.fillStyle = '#555566';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no moves', PANEL_X + PANEL_W / 2, PANEL_Y + PANEL_H / 2 - 4);
    return;
  }

  const listY = PANEL_Y + 18;

  for (let i = 0; i < analysis.lines.length; i++) {
    const line     = analysis.lines[i];
    const ry       = listY + i * ROW_H;
    const selected = i === selectedLine;

    // Row highlight for selected line
    if (selected) {
      ctx.fillStyle = '#191940';
      ctx.fillRect(PANEL_X + 2, ry, PANEL_W - 4, ROW_H - 2);
      ctx.strokeStyle = '#3a3a88';
      ctx.lineWidth = 1;
      ctx.strokeRect(PANEL_X + 2.5, ry + 0.5, PANEL_W - 5, ROW_H - 3);
    }

    // Rank number
    ctx.fillStyle = selected ? '#aaaaff' : '#444466';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`#${i + 1}`, PANEL_X + 5, ry + 13);

    // Hold indicator
    if (line.moves[0]?.useHold) {
      ctx.fillStyle = selected ? '#9999ee' : '#444466';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText('H', PANEL_X + PANEL_W - 5, ry + 13);
    }

    // Score
    ctx.fillStyle = selected ? '#bbbbcc' : '#444455';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(line.score.toFixed(1), PANEL_X + 5, ry + 26);

    // Garbage sent (only when > 0)
    if (line.garbageSent > 0) {
      ctx.fillStyle = selected ? '#ffaaaa' : '#664444';
      ctx.textAlign = 'right';
      ctx.fillText(`+${line.garbageSent}`, PANEL_X + PANEL_W - 5, ry + 26);
    }
  }

  // Footer
  ctx.fillStyle = '#2a2a44';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('↑↓ cycle lines', PANEL_X + PANEL_W / 2, PANEL_Y + PANEL_H - 5);
}

// ---- Status banner ----

function drawBanner(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
): void {
  const cx      = BOARD_OFFSET_X + BOARD_W / 2;
  const bannerY = BOARD_OFFSET_Y + BOARD_H - 28;

  ctx.globalAlpha = 1;
  ctx.fillStyle = 'rgba(8, 8, 22, 0.88)';
  ctx.fillRect(BOARD_OFFSET_X, bannerY, BOARD_W, 26);

  ctx.fillStyle = analysis ? '#5555aa' : '#444466';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';

  if (analysis) {
    const ms = analysis.durationMs.toFixed(0);
    ctx.fillText(`ENGINE  (F: exit)  ${ms}ms`, cx, bannerY + 17);
  } else {
    ctx.fillText('ENGINE — ANALYZING...', cx, bannerY + 17);
  }
}
