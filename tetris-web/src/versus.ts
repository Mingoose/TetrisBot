import { ActivePiece, CellValue, PieceType } from './types';
import { Bag } from './bag';
import {
  emptyBoard, collides, lockPiece, clearLines, hardDropY,
  addGarbageLines, isLockOut, spawnOrBlockOut, visibleRows, BOARD_COLS, SPAWN_Y,
} from './board';
import { setLockHook, spawnPiece, NEXT_QUEUE_SIZE } from './game';
import { SpinKind, RULES, resolveClear } from './rules';
import { placementSpin } from './ai';
import { ALL_PIECE_TYPES } from './pieces';
import { searchStateFor } from './valueBot';

// One received attack waiting to land, TETR.IO style: it can't land until it
// has travelled (readyAt), and all its rows share a hole column unless
// messinessWithin says otherwise.
export interface GarbageChunk {
  amount: number;
  readyAt: number;  // ms timestamp (performance.now() in the game, a virtual clock in the harness)
  column: number;
}

export interface CombatState {
  combo: number;           // -1 = no streak; 0+ = consecutive-clear index (0 = first clear)
  b2b: number;             // B2B level: -1 = none, 0 = first qualifying clear, ...
  incoming: GarbageChunk[]; // oldest first
  pendingGarbage: number;  // total rows in `incoming`; kept in sync by the functions below
}

// What one lock produced, for stats and the self-play harness.
export interface LockOutcome {
  lines: number;
  spin: SpinKind;
  perfectClear: boolean;
  attack: number;     // garbage generated, before cancelling incoming
  surge: number;      // part of attack released by breaking a charged B2B
  garbageIn: number;  // garbage rows that landed on the locker's board
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

// Shared BotBoard factory — all initialization paths use this.
function makeBotBoard(
  activeType: PieceType,
  nextQueue: PieceType[],
  bagState: PieceType[],
  pieceIndex: number = -1,
): BotBoard {
  return {
    board: emptyBoard(),
    active: spawnPiece(activeType),
    nextQueue: [...nextQueue],
    hold: null,
    holdUsed: false,
    bagState: [...bagState],
    pieceIndex,
    lines: 0,
    dead: false,
  };
}

function makeCombat(): CombatState {
  return { combo: -1, b2b: -1, incoming: [], pendingGarbage: 0 };
}

// Pass playerSnapshot to start the bot with the same piece sequence as the player.
export function initVersusData(playerSnapshot?: Pick<import('./types').Snapshot, 'active' | 'nextQueue' | 'bagState'>): VersusData {
  let bot: BotBoard;
  if (playerSnapshot) {
    bot = makeBotBoard(playerSnapshot.active.type, playerSnapshot.nextQueue, playerSnapshot.bagState);
  } else {
    botBag = new Bag();
    const all = botBag.peek(NEXT_QUEUE_SIZE + 1);
    for (let i = 0; i < NEXT_QUEUE_SIZE + 1; i++) botBag.next();
    bot = makeBotBoard(all[0], all.slice(1), botBag.getState());
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

  return {
    bot1: makeBotBoard(bvbSeq[0], bvbSeq.slice(1, NEXT_QUEUE_SIZE + 1), [], startIndex),
    bot1Combat: makeCombat(),
    bot2: makeBotBoard(bvbSeq[0], bvbSeq.slice(1, NEXT_QUEUE_SIZE + 1), [], startIndex),
    bot2Combat: makeCombat(),
    bot1ThinkAccumMs: 0,
    bot2ThinkAccumMs: 0,
    winner: null,
    pendingMove1: null,
    pendingMove2: null,
  };
}

// ---- Garbage math ----

// comboIndex: -1 means no streak; 0+ is the consecutive-clear index after incrementing.
function randomHoleColumn(): number {
  return Math.floor(Math.random() * BOARD_COLS);
}

// Queue an attack against `combat`; it can land once it has travelled.
export function receiveGarbage(combat: CombatState, amount: number, now: number): void {
  if (amount <= 0) return;
  const last = combat.incoming[combat.incoming.length - 1];
  const column = last && Math.random() >= RULES.messinessChange ? last.column : randomHoleColumn();
  combat.incoming.push({ amount, readyAt: now + RULES.garbageSpeedMs, column });
  combat.pendingGarbage += amount;
}

// Cancel up to `amount` queued rows, oldest first. Returns what was left over.
function cancelGarbage(combat: CombatState, amount: number): number {
  while (amount > 0 && combat.incoming.length > 0) {
    const chunk = combat.incoming[0];
    const n = Math.min(chunk.amount, amount);
    chunk.amount -= n;
    amount -= n;
    combat.pendingGarbage -= n;
    if (chunk.amount === 0) combat.incoming.shift();
  }
  return amount;
}

// Land up to the garbage cap of rows that have finished travelling.
function tankGarbage(combat: CombatState, board: CellValue[][], now: number): { board: CellValue[][]; rows: number } {
  let rows = 0;
  while (rows < RULES.garbageCap && combat.incoming.length > 0 && combat.incoming[0].readyAt <= now) {
    const chunk = combat.incoming[0];
    if (rows > 0 && Math.random() < RULES.messinessWithin) chunk.column = randomHoleColumn();
    board = addGarbageLines(board, 1, chunk.column);
    chunk.amount--;
    combat.pendingGarbage--;
    rows++;
    if (chunk.amount === 0) combat.incoming.shift();
  }
  return { board, rows };
}

// Shared lock-event handler, following TETR.IO: a clearing lock attacks, its
// attack first cancelling queued garbage; a lock that clears nothing lets
// queued garbage land. `board` is the locker's board after line clears.
export function handleLock(
  lockerCombat: CombatState,
  opponentCombat: CombatState,
  board: CellValue[][],
  linesCleared: number,
  spin: SpinKind,
  now: number,
): { board: CellValue[][]; outcome: LockOutcome } {
  const perfectClear = linesCleared > 0 && board.every(row => row.every(c => c === 0));
  const clear = resolveClear(lockerCombat.combo, lockerCombat.b2b, linesCleared, spin, perfectClear);
  lockerCombat.combo = clear.combo;
  lockerCombat.b2b = clear.b2b;

  let garbageIn = 0;
  if (linesCleared > 0) {
    receiveGarbage(opponentCombat, cancelGarbage(lockerCombat, clear.attack), now);
  } else {
    const tanked = tankGarbage(lockerCombat, board, now);
    board = tanked.board;
    garbageIn = tanked.rows;
  }
  return {
    board,
    outcome: { lines: linesCleared, spin, perfectClear, attack: clear.attack, surge: clear.surge, garbageIn },
  };
}

// ---- Player lock hook ----

export function setupPlayerLockHook(data: VersusData): void {
  setLockHook((state, linesCleared, _landedPiece, spin) => {
    // Landed garbage can't top the player out by itself: it pushes the stack
    // into the buffer, and the next spawn decides (game.ts block out).
    state.board = handleLock(data.playerCombat, data.botCombat, state.board, linesCleared, spin, performance.now()).board;
  });
}

// The next n pieces a bot-vs-bot bot will draw (after its visible queue),
// extending the shared sequence as needed.
export function bvbLookahead(bot: BotBoard, n: number): PieceType[] {
  getBvbPiece(bot.pieceIndex + n - 1);
  return bvbSeq.slice(bot.pieceIndex, bot.pieceIndex + n);
}

// Pieces left in the bot's current bag as a bit set (bit i = ALL_PIECE_TYPES[i]);
// 0x7f when the next draw starts a new bag. Bot-vs-bot bots draw from the shared
// sequence in 7-piece bags, so their bag is what is left before the next boundary.
export function botBagMask(bot: BotBoard): number {
  const left = bot.pieceIndex >= 0 ? bvbLookahead(bot, (7 - bot.pieceIndex % 7) % 7) : bot.bagState;
  if (left.length === 0) return 0x7f;
  let mask = 0;
  for (const p of left) mask |= 1 << ALL_PIECE_TYPES.indexOf(p);
  return mask;
}

// What the AIs see of a bot: its visible rows only (the searches, the value net
// and uploaded AIs all work on a 20-row board). Moves come back in the same
// coordinates, since row 0 is the top visible row either way.
function aiView(bot: BotBoard): BotBoard {
  return { ...bot, board: visibleRows(bot.board) };
}

// Send bot state to the AI worker for async move computation.
// The value-net bot (aiParams.valueNet = search widths) gets the full search
// position instead, built here because garbage readiness uses this thread's clock.
// combat is spread into the message so the worker has current combo/b2b without
// those fields living on BotBoard.
export function requestBotMove(
  worker: Worker,
  bot: BotBoard,
  combat: CombatState,
  aiParams?: { beamWidth: number; searchDepth: number; advancedEval?: boolean; valueNet?: number[]; searchMs?: number },
): void {
  if (aiParams?.valueNet) {
    worker.postMessage({
      valueNet: aiParams.valueNet, searchMs: aiParams.searchMs ?? 1000,
      state: searchStateFor(aiView(bot), combat, botBagMask(bot), performance.now()),
    });
    return;
  }
  const { pendingGarbage, combo, b2b } = combat;
  // b2bActive is kept for uploaded AIs written against the older message format.
  const b2bActive = b2b >= 0;
  if (bot.pieceIndex >= 0) {
    // Pass a slice as bagState so the beam search looks ahead into the same
    // pieces both bots will actually receive.
    const bagState = bvbLookahead(bot, WORKER_LOOKAHEAD);
    worker.postMessage({ bot: { ...aiView(bot), bagState }, pendingGarbage, combo, b2b, b2bActive, ...aiParams });
  } else {
    worker.postMessage({ bot: aiView(bot), pendingGarbage, combo, b2b, b2bActive, ...aiParams });
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
  onLock?: (outcome: LockOutcome) => void,
  now: number = performance.now(),
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
    piece.y = hardDropY(bot.board, { ...piece, y: SPAWN_Y });
  }

  // Bots send only a final position; credit the best spin a rotation into it can earn.
  const spin = invalidReason ? 0 : placementSpin(bot.board, piece);

  // Lock and clear
  const locked = lockPiece(bot.board, piece);
  const { board: clearedBoard, linesCleared } = clearLines(locked);
  bot.board = clearedBoard;
  bot.lines += linesCleared;

  // Garbage exchange
  const lock = handleLock(botCombat, playerCombat, bot.board, linesCleared, spin, now);
  bot.board = lock.board;
  onLock?.(lock.outcome);

  if (isLockOut(piece, linesCleared)) {
    bot.dead = true;
    return invalidReason;
  }

  // Spawn next piece; block out if it has nowhere to appear (left overlapping)
  if (!useBvb) botBag.restoreState(bot.bagState);
  const nextType = bot.nextQueue.shift()!;
  bot.nextQueue.push(drawNext());
  const spawned = spawnOrBlockOut(bot.board, nextType, linesCleared > 0);
  bot.active = spawned ?? spawnPiece(nextType);
  bot.holdUsed = false;
  if (!spawned) bot.dead = true;

  return invalidReason;
}
