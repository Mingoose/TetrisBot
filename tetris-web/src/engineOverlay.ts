import type { CellValue } from './types';
import type { EngineAnalysis, EngineMove } from './engine';
import { PIECE_COLORS, getRotation } from './pieces';
import { CELL_SIZE, BOARD_OFFSET_X, BOARD_OFFSET_Y } from './editor';
import { BOARD_COLS, BOARD_ROWS } from './board';
import { BOARD_W, BOARD_H, CANVAS_W, CANVAS_H, VERSUS_CANVAS_W } from './renderer';

// Notation panel in the unused right region of the canvas
const PANEL_X = CANVAS_W + 8;
const PANEL_Y = BOARD_OFFSET_Y;
const PANEL_W = VERSUS_CANVAS_W - PANEL_X - 8;
const PANEL_H = CANVAS_H - BOARD_OFFSET_Y * 2;

const COMPACT_ROW_H    = 22;
const MOVE_ROW_H       = 18;
const SELECTED_HEADER_H = 26;

const ROT_LABELS = ['0', 'R', '2', 'L'];

function clearLabel(m: EngineMove): string {
  if (m.isPerfectClear) return 'PC';
  if (m.isTSpin) {
    if (m.linesCleared === 0) return 'TS';
    if (m.linesCleared === 1) return 'TSS';
    if (m.linesCleared === 2) return 'TSD';
    if (m.linesCleared === 3) return 'TST';
  }
  if (m.linesCleared === 1) return 'Single';
  if (m.linesCleared === 2) return 'Double';
  if (m.linesCleared === 3) return 'Triple';
  if (m.linesCleared === 4) return 'Tetris';
  return '';
}

/**
 * Render the engine analysis overlay on top of an already-drawn game frame.
 *
 * Draws:
 *  1. An animated board that cycles through each move in the selected line.
 *  2. A notation panel in the right canvas region listing all lines with placement details.
 *
 * @param ctx         - The canvas rendering context.
 * @param analysis    - Analysis result, or null if still computing.
 * @param selectedLine - Index into analysis.lines for the currently highlighted line.
 * @param boardStates - Pre-computed board states: boardStates[i] = board before move i.
 * @param animFrame   - Index of the move currently being animated.
 * @param timestamp   - rAF timestamp, used for the piece pulse animation.
 */
export function drawEngineOverlay(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
  selectedLine: number,
  boardStates: CellValue[][][],
  animFrame: number,
  timestamp: number,
): void {
  drawAnimatedBoard(ctx, analysis, selectedLine, boardStates, animFrame, timestamp);
  drawPanel(ctx, analysis, selectedLine, animFrame);
}

// ---- Shared cell drawing ----

function drawOverlayCell(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  alpha: number,
  color: string,
): void {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2);
  ctx.fillStyle = 'rgba(255,255,255,0.18)';
  ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, 3);
  ctx.fillRect(px + 1, py + 1, 3, CELL_SIZE - 2);
  ctx.globalAlpha = 1;
}

// ---- Animated board ----

function drawAnimatedBoard(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
  selectedLine: number,
  boardStates: CellValue[][][],
  animFrame: number,
  timestamp: number,
): void {
  if (!analysis || analysis.lines.length === 0 || boardStates.length === 0) return;

  const clampedSel = Math.min(selectedLine, analysis.lines.length - 1);
  const line = analysis.lines[clampedSel];
  if (!line || line.moves.length === 0) return;

  const clampedFrame = Math.min(animFrame, line.moves.length - 1);
  const boardAtFrame = boardStates[clampedFrame] ?? boardStates[0];

  // 1. Cover the paused game board with a fresh background
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#0a0a1e';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  // 2. Grid lines (matches renderer.ts drawGrid)
  ctx.strokeStyle = '#16163a';
  ctx.lineWidth = 0.5;
  for (let c = 0; c <= BOARD_COLS; c++) {
    const x = BOARD_OFFSET_X + c * CELL_SIZE;
    ctx.beginPath(); ctx.moveTo(x, BOARD_OFFSET_Y); ctx.lineTo(x, BOARD_OFFSET_Y + BOARD_H); ctx.stroke();
  }
  for (let r = 0; r <= BOARD_ROWS; r++) {
    const y = BOARD_OFFSET_Y + r * CELL_SIZE;
    ctx.beginPath(); ctx.moveTo(BOARD_OFFSET_X, y); ctx.lineTo(BOARD_OFFSET_X + BOARD_W, y); ctx.stroke();
  }

  // 3. Locked cells from the board state at this frame
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const cell = boardAtFrame[r][c];
      if (!cell) continue;
      const color = PIECE_COLORS[cell as keyof typeof PIECE_COLORS];
      drawOverlayCell(ctx, BOARD_OFFSET_X + c * CELL_SIZE, BOARD_OFFSET_Y + r * CELL_SIZE, 1, color);
    }
  }

  // 4. Current move's piece — pulsing alpha
  const move = line.moves[clampedFrame];
  const rot   = getRotation(move.pieceType, move.rotationIndex);
  const color = PIECE_COLORS[move.pieceType];
  const pulse = 0.75 + 0.25 * Math.sin(timestamp / 200);

  for (let r = 0; r < rot.length; r++) {
    for (let c = 0; c < rot[r].length; c++) {
      if (!rot[r][c]) continue;
      const row = move.y + r;
      const col = move.x + c;
      if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) continue;
      drawOverlayCell(ctx, BOARD_OFFSET_X + col * CELL_SIZE, BOARD_OFFSET_Y + row * CELL_SIZE, pulse, color);
    }
  }

  // 5. Board border (matches renderer.ts drawBoardBorder)
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#3a3a6a';
  ctx.lineWidth = 1;
  ctx.strokeRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, BOARD_H);

  // 6. Move counter at the top of the board
  ctx.fillStyle = 'rgba(8,8,15,0.80)';
  ctx.fillRect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_W, 20);
  ctx.fillStyle = '#6666aa';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(
    `move ${clampedFrame + 1} / ${line.moves.length}`,
    BOARD_OFFSET_X + BOARD_W / 2,
    BOARD_OFFSET_Y + 14,
  );
}

// ---- Notation panel ----

function drawPanel(
  ctx: CanvasRenderingContext2D,
  analysis: EngineAnalysis | null,
  selectedLine: number,
  animFrame: number,
): void {
  // Background
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#08080f';
  ctx.fillRect(PANEL_X, PANEL_Y, PANEL_W, PANEL_H);
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.strokeRect(PANEL_X + 0.5, PANEL_Y + 0.5, PANEL_W - 1, PANEL_H - 1);

  // Header: title left, timing right
  ctx.fillStyle = '#4444aa';
  ctx.font = 'bold 11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('ENGINE LINES', PANEL_X + 6, PANEL_Y + 14);

  if (!analysis) {
    ctx.fillStyle = '#555566';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('…', PANEL_X + PANEL_W - 6, PANEL_Y + 14);
  } else {
    ctx.fillStyle = '#3a3a77';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${analysis.durationMs.toFixed(0)}ms`, PANEL_X + PANEL_W - 6, PANEL_Y + 14);
  }

  // Divider
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PANEL_X + 4, PANEL_Y + 19);
  ctx.lineTo(PANEL_X + PANEL_W - 4, PANEL_Y + 19);
  ctx.stroke();

  if (!analysis) {
    ctx.fillStyle = '#555577';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('searching...', PANEL_X + PANEL_W / 2, PANEL_Y + PANEL_H / 2);
    return;
  }

  if (analysis.lines.length === 0) {
    ctx.fillStyle = '#555566';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('no moves', PANEL_X + PANEL_W / 2, PANEL_Y + PANEL_H / 2);
    return;
  }

  let cursor = PANEL_Y + 24;
  const clampedSel = Math.min(selectedLine, analysis.lines.length - 1);

  for (let i = 0; i < analysis.lines.length; i++) {
    const line     = analysis.lines[i];
    const selected = i === clampedSel;

    if (selected) {
      const expandedH = SELECTED_HEADER_H + line.moves.length * MOVE_ROW_H + 4;

      // Highlight background
      ctx.fillStyle = '#13133a';
      ctx.fillRect(PANEL_X + 2, cursor, PANEL_W - 4, expandedH);
      ctx.strokeStyle = '#3a3a88';
      ctx.lineWidth = 1;
      ctx.strokeRect(PANEL_X + 2.5, cursor + 0.5, PANEL_W - 5, expandedH - 1);

      // Rank + score + garbage header
      ctx.fillStyle = '#aaaaff';
      ctx.font = 'bold 13px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`#${i + 1}`, PANEL_X + 6, cursor + 16);

      ctx.fillStyle = '#888899';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(line.score.toFixed(1), PANEL_X + 28, cursor + 16);

      if (line.garbageSent > 0) {
        ctx.fillStyle = '#ffaaaa';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`+${line.garbageSent}↑`, PANEL_X + PANEL_W - 6, cursor + 16);
      }

      // Move notation rows
      const moveStartY = cursor + SELECTED_HEADER_H;
      for (let m = 0; m < line.moves.length; m++) {
        const mv           = line.moves[m];
        const my           = moveStartY + m * MOVE_ROW_H;
        const pieceColor   = PIECE_COLORS[mv.pieceType];
        const cl           = clearLabel(mv);
        const isActiveMove = m === Math.min(animFrame, line.moves.length - 1);

        // Play arrow for the currently animating move
        if (isActiveMove) {
          ctx.fillStyle = '#aaaaff';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'left';
          ctx.fillText('▶', PANEL_X + 2, my + 13);
        }

        // Step number
        ctx.fillStyle = isActiveMove ? '#aaaaff' : '#555577';
        ctx.font = isActiveMove ? 'bold 10px monospace' : '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${m + 1}.`, PANEL_X + 12, my + 13);

        // Hold indicator
        if (mv.useHold) {
          ctx.fillStyle = '#888866';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'left';
          ctx.fillText('H', PANEL_X + 28, my + 13);
        }

        // Piece type (coloured)
        ctx.fillStyle = pieceColor;
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(mv.pieceType, PANEL_X + 40, my + 13);

        // Rotation
        ctx.fillStyle = '#7777aa';
        ctx.font = '11px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(ROT_LABELS[mv.rotationIndex] ?? '?', PANEL_X + 56, my + 13);

        // Coordinates
        ctx.fillStyle = '#666677';
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`(${mv.x},${mv.y})`, PANEL_X + 70, my + 13);

        // Clear label
        if (cl) {
          const clColor = mv.isPerfectClear ? '#ffdd44'
            : mv.isTSpin                   ? '#44ffcc'
            : mv.linesCleared === 4        ? '#ff8844'
            : '#99cc99';
          ctx.fillStyle = clColor;
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'right';
          ctx.fillText(cl, PANEL_X + PANEL_W - 6, my + 13);
        }
      }

      cursor += expandedH + 4;
    } else {
      // Compact non-selected row
      ctx.fillStyle = '#333355';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`#${i + 1}`, PANEL_X + 6, cursor + 14);

      ctx.fillStyle = '#3a3a55';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(line.score.toFixed(1), PANEL_X + 28, cursor + 14);

      if (line.garbageSent > 0) {
        ctx.fillStyle = '#664444';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`+${line.garbageSent}↑`, PANEL_X + PANEL_W - 6, cursor + 14);
      }

      cursor += COMPACT_ROW_H;
    }

    // Row separator
    ctx.strokeStyle = '#1a1a33';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PANEL_X + 4, cursor);
    ctx.lineTo(PANEL_X + PANEL_W - 4, cursor);
    ctx.stroke();
  }

  // Footer
  ctx.fillStyle = '#3a3a66';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('↑↓ cycle lines', PANEL_X + PANEL_W / 2, PANEL_Y + PANEL_H - 6);
}
