import { ActivePiece, CellValue, PieceType } from './types';
import { Bag } from './bag';
import {
  emptyBoard, collides, lockPiece, clearLines, isGameOver, hardDropY,
  addGarbageLines, countTSpinCorners, BOARD_COLS,
} from './board';
import { setLockHook, spawnPiece, NEXT_QUEUE_SIZE } from './game';

// Jstris combo bonus table (0-indexed: 0 = first consecutive clear).
export const COMBO_TABLE = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5] as const;

export interface CombatState {
  combo: number;           // -1 = no streak; 0+ = consecutive-clear index (0 = first clear)
  b2bActive: boolean;      // previous qualifying clear enables B2B bonus on this one
  pendingGarbage: number;  // incoming garbage queued to arrive on next lock
}

export interface BotBoard {
  board: CellValue[][];
  active: ActivePiece;
  nextQueue: PieceType[];
  hold: PieceType | null;
  holdUsed: boolean;
  bagState: PieceType[];
  // pieceIndex >= 0: bot-vs-bot mode — draws from shared bvbSeq by index so
  // both bots always see the same piece sequence regardless of play speed.
  // pieceIndex === -1: versus/watch mode — uses botBag + bagState as before.
  pieceIndex: number;
  lines: number;
  dead: boolean;
  // Combat state mirrored here so the AI worker receives it with each move request.
  combo: number;      // -1 = no streak; mirrors CombatState.combo
  b2bActive: boolean; // mirrors CombatState.b2bActive
}

export interface BotVsBotData {
  bot1: BotBoard;
  bot1Combat: CombatState;
  bot2: BotBoard;
  bot2Combat: CombatState;
  bot1ThinkAccumMs: number;
  bot2ThinkAccumMs: number;
  winner: 'bot1' | 'bot2' | 'draw' | null;
  pendingMove1: { rotationIndex: number; x: number; y: number; useHold: boolean } | null;
  pendingMove2: { rotationIndex: number; x: number; y: number; useHold: boolean } | null;
}

export interface VersusData {
  playerCombat: CombatState;
  bot: BotBoard;
  botCombat: CombatState;
  botThinkAccumMs: number;
  winner: 'player' | 'bot' | null;
  pendingMove: { rotationIndex: number; x: number; y: number; useHold: boolean } | null;
}

// ---- Bot bag — used only for versus/watch mode (single bot, pieceIndex = -1) ----
let botBag = new Bag();

// ---- Shared piece sequence for bot-vs-bot ----
// Both bots draw from bvbSeq[bot.pieceIndex], advancing their own index.
// The sequence is grown lazily by a single Bag so both bots always receive
// the same piece at the same sequence position, even at different play speeds.
let bvbSeq: PieceType[] = [];
let bvbSeqBag = new Bag();

function resetBvbSequence(): void {
  bvbSeq = [];
  bvbSeqBag = new Bag();
}

function getBvbPiece(index: number): PieceType {
  while (bvbSeq.length <= index) bvbSeq.push(bvbSeqBag.next());
  return bvbSeq[index];
}

// How many future pieces to send to the AI worker as its bagState lookahead.
// Beam search depth 4 × worst-case 2 draws/level = 8, plus slack.
const WORKER_LOOKAHEAD = 30;

function initBotBoard(): BotBoard {
  botBag = new Bag();
  const all = botBag.peek(NEXT_QUEUE_SIZE + 1);
  for (let i = 0; i < NEXT_QUEUE_SIZE + 1; i++) botBag.next();
  return {
    board: emptyBoard(),
    active: spawnPiece(all[0]),
    nextQueue: all.slice(1),
    hold: null,
    holdUsed: false,
    bagState: botBag.getState(),
    pieceIndex: -1,
    lines: 0,
    dead: false,
    combo: -1,
    b2bActive: false,
  };
}

function makeCombat(): CombatState {
  return { combo: -1, b2bActive: false, pendingGarbage: 0 };
}

// Pass playerSnapshot to start the bot with the same piece sequence as the player.
export function initVersusData(playerSnapshot?: Pick<import('./types').Snapshot, 'active' | 'nextQueue' | 'bagState'>): VersusData {
  let bot: BotBoard;
  if (playerSnapshot) {
    bot = {
      board: emptyBoard(),
      active: spawnPiece(playerSnapshot.active.type),
      nextQueue: [...playerSnapshot.nextQueue],
      hold: null,
      holdUsed: false,
      bagState: [...playerSnapshot.bagState],
      pieceIndex: -1,
      lines: 0,
      dead: false,
      combo: -1,
      b2bActive: false,
    };
  } else {
    bot = initBotBoard();
  }
  return {
    playerCombat: makeCombat(),
    bot,
    botCombat: makeCombat(),
    botThinkAccumMs: 0,
    winner: null,
    pendingMove: null,
  };
}

export function initBotVsBotData(): BotVsBotData {
  // Fresh shared sequence — both bots draw from bvbSeq[pieceIndex] going forward.
  resetBvbSequence();
  // Pre-generate the first NEXT_QUEUE_SIZE + 1 pieces (active + visible next queue).
  for (let i = 0; i <= NEXT_QUEUE_SIZE; i++) getBvbPiece(i);
  const startIndex = NEXT_QUEUE_SIZE + 1; // index of next piece to draw on first lock

  const makeBot = (): BotBoard => ({
    board: emptyBoard(),
    active: spawnPiece(bvbSeq[0]),
    nextQueue: [...bvbSeq.slice(1, NEXT_QUEUE_SIZE + 1)],
    hold: null,
    holdUsed: false,
    bagState: [],    // unused in pieceIndex mode
    pieceIndex: startIndex,
    lines: 0,
    dead: false,
    combo: -1,
    b2bActive: false,
  });

  return {
    bot1: makeBot(), bot1Combat: makeCombat(),
    bot2: makeBot(), bot2Combat: makeCombat(),
    bot1ThinkAccumMs: 0,
    bot2ThinkAccumMs: 0,
    winner: null,
    pendingMove1: null,
    pendingMove2: null,
  };
}

// ---- Garbage math ----

function computeGarbage(
  linesCleared: number,
  isTSpin: boolean,
  b2bActive: boolean,
  comboIndex: number,
): number {
  if (linesCleared === 0) return 0;
  let base: number;
  if (isTSpin) {
    base = ([2, 4, 6] as const)[linesCleared - 1] ?? 6;
  } else {
    base = ([0, 1, 2, 4] as const)[linesCleared - 1] ?? 4;
  }
  const qualifying = isTSpin || linesCleared === 4;
  if (qualifying && b2bActive) base++;
  base += COMBO_TABLE[Math.min(comboIndex, COMBO_TABLE.length - 1)];
  return base;
}

// Shared lock-event handler: updates combat state, exchanges garbage between
// the locker and opponent, and injects any remaining pending garbage into the
// locker's board. Returns the (possibly modified) board.
function handleLock(
  lockerCombat: CombatState,
  opponentCombat: CombatState,
  lockerBoard: CellValue[][],
  linesCleared: number,
  isTSpin: boolean,
): CellValue[][] {
  let garbageOut = 0;
  if (linesCleared > 0) {
    lockerCombat.combo = lockerCombat.combo < 0 ? 0 : lockerCombat.combo + 1;
    garbageOut = computeGarbage(linesCleared, isTSpin, lockerCombat.b2bActive, lockerCombat.combo);
    lockerCombat.b2bActive = isTSpin || linesCleared === 4;
  } else {
    lockerCombat.combo = -1;
  }

  // Cancel outgoing against incoming, route remainder to opponent
  const netOut = Math.max(0, garbageOut - lockerCombat.pendingGarbage);
  lockerCombat.pendingGarbage = Math.max(0, lockerCombat.pendingGarbage - garbageOut);
  opponentCombat.pendingGarbage += netOut;

  // Inject remaining pending garbage into the locker's own board
  if (lockerCombat.pendingGarbage > 0) {
    const holeCol = Math.floor(Math.random() * BOARD_COLS);
    lockerBoard = addGarbageLines(lockerBoard, lockerCombat.pendingGarbage, holeCol);
    lockerCombat.pendingGarbage = 0;
  }
  return lockerBoard;
}

// ---- Player lock hook ----

export function setupPlayerLockHook(data: VersusData): void {
  setLockHook((state, linesCleared, landedPiece, preLockBoard, wasRotation) => {
    const isTSpin =
      landedPiece.type === 'T' &&
      wasRotation &&
      countTSpinCorners(preLockBoard, landedPiece) >= 3;
    state.board = handleLock(
      data.playerCombat,
      data.botCombat,
      state.board,
      linesCleared,
      isTSpin,
    );
    // Check if post-garbage board triggers game over (board overflow)
    if (isGameOver(state.board)) state.mode = 'gameover';
  });
}

// Send bot state to the AI worker for async move computation.
export function requestBotMove(
  worker: Worker,
  bot: BotBoard,
  pendingGarbage: number,
  aiParams?: { beamWidth: number; searchDepth: number; advancedEval?: boolean },
): void {
  if (bot.pieceIndex >= 0) {
    // Extend shared sequence and pass a slice as bagState so the beam search
    // looks ahead into the same pieces both bots will actually receive.
    getBvbPiece(bot.pieceIndex + WORKER_LOOKAHEAD - 1);
    const bagState = bvbSeq.slice(bot.pieceIndex, bot.pieceIndex + WORKER_LOOKAHEAD);
    worker.postMessage({ bot: { ...bot, bagState }, pendingGarbage, ...aiParams });
  } else {
    worker.postMessage({ bot, pendingGarbage, ...aiParams });
  }
}

// Apply a pre-computed move to the bot board. Garbage routing is handled
// internally via handleLock (bot's outgoing goes to playerCombat.pendingGarbage).
// Returns null if the move was valid, 'occupied' if the position collided with existing
// cells, or 'floating' if the piece was not at the lowest valid resting row.
// In either error case a hard-drop fallback is applied automatically.
export function applyBotMove(
  move: { rotationIndex: number; x: number; y: number; useHold: boolean },
  bot: BotBoard,
  botCombat: CombatState,
  playerCombat: CombatState,
): 'occupied' | 'floating' | null {
  if (bot.dead) return null;

  const useBvb = bot.pieceIndex >= 0;
  let invalidReason: 'occupied' | 'floating' | null = null;

  // Draw the next piece from the appropriate source and update bot state.
  function drawNext(): PieceType {
    if (useBvb) {
      return getBvbPiece(bot.pieceIndex++);
    }
    const p = botBag.next();
    bot.bagState = botBag.getState();
    return p;
  }

  // Apply hold if requested
  if (move.useHold) {
    const swapIn = bot.hold ?? bot.nextQueue[0];
    if (!bot.hold) {
      bot.nextQueue.shift();
      if (!useBvb) botBag.restoreState(bot.bagState);
      bot.nextQueue.push(drawNext());
    }
    bot.hold = bot.active.type;
    bot.holdUsed = true;
    bot.active = spawnPiece(swapIn);
  }

  // Place at the exact position determined by the BFS (handles T-spins and slides under overhangs).
  // Fall back to a straight hard-drop if the board changed since the search (e.g., garbage added).
  const piece: ActivePiece = {
    type: bot.active.type,
    rotationIndex: move.rotationIndex,
    x: move.x,
    y: move.y,
  };
  if (collides(bot.board, piece, 0, 0)) {
    invalidReason = 'occupied';
  } else if (!collides(bot.board, piece, 0, 1)) {
    invalidReason = 'floating';
  }
  if (invalidReason) {
    piece.y = hardDropY(bot.board, { ...piece, y: 0 });
  }

  // T-spin check (bot always "rotates" to reach target; use 3-corner rule)
  const isTSpin = piece.type === 'T' && countTSpinCorners(bot.board, piece) >= 3;

  // Lock and clear
  const locked = lockPiece(bot.board, piece);
  const { board: clearedBoard, linesCleared } = clearLines(locked);
  bot.board = clearedBoard;
  bot.lines += linesCleared;

  // Garbage exchange
  bot.board = handleLock(botCombat, playerCombat, bot.board, linesCleared, isTSpin);
  // Mirror combat state onto bot so the AI worker sees up-to-date combo/b2b on its next request.
  bot.combo = botCombat.combo;
  bot.b2bActive = botCombat.b2bActive;

  // Spawn next piece
  if (!useBvb) botBag.restoreState(bot.bagState);
  const nextType = bot.nextQueue.shift()!;
  bot.nextQueue.push(drawNext());
  bot.active = spawnPiece(nextType);
  bot.holdUsed = false;

  if (isGameOver(bot.board) || collides(bot.board, bot.active, 0, 0)) {
    bot.dead = true;
  }

  return invalidReason;
}
