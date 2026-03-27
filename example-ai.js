// Example custom AI for Tetris upload feature.
// Implements a simple greedy one-piece lookahead: tries every (rotation, column)
// placement, simulates the result, and picks the best score.
//
// Interface required by the game:
//   function getBestMove(bot, pendingGarbage) => { rotationIndex, x, y, useHold }
//
// bot = {
//   board:      number[][],    // 20 rows × 10 cols, 0 = empty, non-zero = filled
//   active:     { type, rotationIndex, x, y },
//   nextQueue:  string[],      // next 5 piece types
//   hold:       string | null,
//   holdUsed:   boolean,
//   bagState:   string[],
//   lines:      number,
//   dead:       boolean,
// }

// ---- Piece rotation matrices ----
// Each piece has 4 rotations, each rotation is a 2D array of 0/1.
const PIECES = {
  I: [
    [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
    [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
    [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
  ],
  O: [
    [[1,1],[1,1]],
    [[1,1],[1,1]],
    [[1,1],[1,1]],
    [[1,1],[1,1]],
  ],
  T: [
    [[0,1,0],[1,1,1],[0,0,0]],
    [[0,1,0],[0,1,1],[0,1,0]],
    [[0,0,0],[1,1,1],[0,1,0]],
    [[0,1,0],[1,1,0],[0,1,0]],
  ],
  S: [
    [[0,1,1],[1,1,0],[0,0,0]],
    [[0,1,0],[0,1,1],[0,0,1]],
    [[0,0,0],[0,1,1],[1,1,0]],
    [[1,0,0],[1,1,0],[0,1,0]],
  ],
  Z: [
    [[1,1,0],[0,1,1],[0,0,0]],
    [[0,0,1],[0,1,1],[0,1,0]],
    [[0,0,0],[1,1,0],[0,1,1]],
    [[0,1,0],[1,1,0],[1,0,0]],
  ],
  J: [
    [[1,0,0],[1,1,1],[0,0,0]],
    [[0,1,1],[0,1,0],[0,1,0]],
    [[0,0,0],[1,1,1],[0,0,1]],
    [[0,1,0],[0,1,0],[1,1,0]],
  ],
  L: [
    [[0,0,1],[1,1,1],[0,0,0]],
    [[0,1,0],[0,1,0],[0,1,1]],
    [[0,0,0],[1,1,1],[1,0,0]],
    [[1,1,0],[0,1,0],[0,1,0]],
  ],
};

const BOARD_ROWS = 20;
const BOARD_COLS = 10;

function getRotation(type, rotIndex) {
  return PIECES[type][rotIndex % 4];
}

// Returns true if piece at (x, y) collides with board or walls.
function collides(board, rotation, x, y) {
  for (let r = 0; r < rotation.length; r++) {
    for (let c = 0; c < rotation[r].length; c++) {
      if (!rotation[r][c]) continue;
      const row = y + r;
      const col = x + c;
      if (col < 0 || col >= BOARD_COLS || row >= BOARD_ROWS) return true;
      if (row >= 0 && board[row][col] !== 0) return true;
    }
  }
  return false;
}

// Returns the lowest y the piece can reach by dropping straight down.
function hardDropY(board, rotation, x) {
  let y = 0;
  while (!collides(board, rotation, x, y + 1)) y++;
  return y;
}

// Returns a new board with the piece locked in and full rows cleared.
function lockAndClear(board, rotation, x, y) {
  const newBoard = board.map(row => [...row]);
  for (let r = 0; r < rotation.length; r++) {
    for (let c = 0; c < rotation[r].length; c++) {
      if (!rotation[r][c]) continue;
      const row = y + r;
      if (row >= 0 && row < BOARD_ROWS) newBoard[row][x + c] = 1;
    }
  }
  // Clear full rows
  const cleared = newBoard.filter(row => row.some(cell => cell === 0));
  const linesCleared = BOARD_ROWS - cleared.length;
  const empty = Array.from({ length: linesCleared }, () => new Array(BOARD_COLS).fill(0));
  return { board: [...empty, ...cleared], linesCleared };
}

// ---- Board evaluation ----
// Score a board state — higher is better.
function evaluate(board, linesCleared) {
  const heights = new Array(BOARD_COLS).fill(0);
  let holes = 0;

  for (let c = 0; c < BOARD_COLS; c++) {
    let foundTop = false;
    for (let r = 0; r < BOARD_ROWS; r++) {
      if (board[r][c] !== 0) {
        if (!foundTop) { heights[c] = BOARD_ROWS - r; foundTop = true; }
      } else if (foundTop) {
        holes++;
      }
    }
  }

  const aggHeight = heights.reduce((a, b) => a + b, 0);
  let bumpiness = 0;
  for (let c = 0; c < BOARD_COLS - 1; c++) {
    bumpiness += Math.abs(heights[c] - heights[c + 1]);
  }

  const lineClearScore = [0, -0.5, 1.0, 2.5, 5.0][Math.min(linesCleared, 4)];

  return -0.5 * aggHeight
       + lineClearScore
       - 0.8 * holes
       - 0.3 * bumpiness;
}

// ---- Move finder ----
// Try all rotations and x positions for a piece type and return the best placement.
function bestPlacementFor(board, pieceType) {
  let best = null;
  let bestScore = -Infinity;

  for (let rotIndex = 0; rotIndex < 4; rotIndex++) {
    const rotation = getRotation(pieceType, rotIndex);
    const pieceW = rotation[0].length;

    for (let x = 0; x <= BOARD_COLS - pieceW; x++) {
      if (collides(board, rotation, x, 0)) continue;

      const y = hardDropY(board, rotation, x);
      const { board: resultBoard, linesCleared } = lockAndClear(board, rotation, x, y);
      const score = evaluate(resultBoard, linesCleared);

      if (score > bestScore) {
        bestScore = score;
        best = { rotationIndex: rotIndex, x, y, score };
      }
    }
  }

  return best;
}

// ---- Main entry point ----
function getBestMove(bot, pendingGarbage) {
  const active = bestPlacementFor(bot.board, bot.active.type);

  // Consider using hold if it would give a better placement.
  if (!bot.holdUsed) {
    const holdType = bot.hold ?? bot.nextQueue[0];
    const fromHold = bestPlacementFor(bot.board, holdType);
    if (fromHold && active && fromHold.score > active.score + 0.5) {
      return { rotationIndex: fromHold.rotationIndex, x: fromHold.x, y: fromHold.y, useHold: true };
    }
  }

  if (active) {
    return { rotationIndex: active.rotationIndex, x: active.x, y: active.y, useHold: false };
  }

  // Fallback: drop in place
  return { rotationIndex: 0, x: bot.active.x, y: bot.active.y, useHold: false };
}
