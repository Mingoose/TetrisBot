import { GameState, GameVariant, ActivePiece, PieceType, CellValue, Snapshot } from './types';
import { Settings } from './settings';
import {
  emptyBoard,
  collides,
  lockPiece,
  clearLines,
  isGameOver,
  scoreForLines,
  gravityInterval,
  hardDropY,
  attemptRotation,
  BOARD_COLS,
} from './board';
import { getRotation } from './pieces';
import { Bag } from './bag';
import { pushHistory, rewind } from './rewind';
import { InputState, wasJustPressed, isHeld } from './input';
import { enterEditor, exitEditor } from './editor';

export const NEXT_QUEUE_SIZE = 5;

// Hook called inside lockAndSpawn after line clears, before the next piece spawns.
// Used by versus mode to calculate and exchange garbage.
export type LockHook = (
  state: GameState,
  linesCleared: number,
  landedPiece: ActivePiece,
  preLockBoard: CellValue[][],
  wasRotation: boolean,
) => void;

let lockHook: LockHook | null = null;
export function setLockHook(fn: LockHook | null): void { lockHook = fn; }

// Hook called at the START of lockAndSpawn (before mutation) for replay recording.
// Receives the pre-lock snapshot (already deep-cloned by pushHistory) and elapsed ms.
export type PreLockCallback = (snapshot: Snapshot, elapsedMs: number) => void;
let preLockCallback: PreLockCallback | null = null;
export function setPreLockCallback(fn: PreLockCallback | null): void { preLockCallback = fn; }
export const LOCK_DELAY_MS = 500;
const COUNTDOWN_MS = 1500;
const LOCK_RESET_MAX = 15;

// Reset the three transient fields that must be cleared whenever a new piece spawns
// or the game state is restored from a snapshot.
export function resetTransientState(state: GameState): void {
  state.lockDelayMs = LOCK_DELAY_MS;
  state.lockResetCount = 0;
  state.gravityAccumMs = 0;
}
const SOFT_DROP_FACTOR = 10;

// Module-level bag (lives outside GameState so it isn't cloned into snapshots directly,
// but its remaining state IS snapshotted via bagState for rewind accuracy).
let bag = new Bag();

export function initGameState(variant: GameVariant): GameState {
  bag = new Bag();
  const nextQueue = bag.peek(NEXT_QUEUE_SIZE + 1);
  for (let i = 0; i < NEXT_QUEUE_SIZE + 1; i++) bag.next();

  const firstType = nextQueue.shift()!;
  const active = spawnPiece(firstType);

  return {
    board: emptyBoard(),
    score: 0,
    lines: 0,
    level: 1,
    nextQueue,
    hold: null,
    holdUsed: false,
    active,
    bagState: bag.getState(),
    mode: (variant === 'sprint' || variant === 'versus') ? 'countdown' : 'playing',
    variant,
    history: [],
    lockDelayMs: LOCK_DELAY_MS,
    lockResetCount: 0,
    gravityAccumMs: 0,
    lastFrameTime: 0,
    rafHandle: 0,
    countdownMs: (variant === 'sprint' || variant === 'versus') ? COUNTDOWN_MS : 0,
    sprintStartTime: 0,
    sprintElapsedMs: 0,
    lastActionRotation: false,
  };
}

export function spawnPiece(type: PieceType): ActivePiece {
  const rotation = getRotation(type, 0);
  const x = Math.floor((BOARD_COLS - rotation[0].length) / 2);
  return { type, rotationIndex: 0, x, y: 0 };
}

function isGrounded(state: GameState): boolean {
  return collides(state.board, state.active, 0, 1);
}

// Reset lock delay after a successful move/rotation while grounded.
function resetLockOnMove(state: GameState): void {
  state.lockDelayMs = LOCK_DELAY_MS;
  if (state.lockResetCount < LOCK_RESET_MAX) state.lockResetCount++;
}

// Handle DAS/ARR for one direction. Returns the updated DAS accumulator.
function handleDasArr(
  state: GameState,
  justPressed: boolean,
  dasAccum: number,
  dx: number,
  dt: number,
  dasMs: number,
  arrMs: number,
): number {
  if (justPressed) {
    if (tryMove(state, dx, 0) && isGrounded(state)) resetLockOnMove(state);
    return 0;
  }
  dasAccum += dt;
  if (dasAccum >= dasMs) {
    let moved = false;
    if (arrMs === 0) {
      while (tryMove(state, dx, 0)) moved = true;
    } else {
      const ticks = Math.floor((dasAccum - dasMs) / arrMs);
      for (let i = 0; i < ticks + 1; i++) {
        if (tryMove(state, dx, 0)) moved = true;
        else break;
      }
      dasAccum = dasMs + ((dasAccum - dasMs) % arrMs);
    }
    if (moved && isGrounded(state)) resetLockOnMove(state);
  }
  return dasAccum;
}

function lockAndSpawn(state: GameState): void {
  // Snapshot BEFORE mutating board
  pushHistory(state);

  // Fire replay callback with the just-pushed snapshot (already a deep clone).
  if (preLockCallback && (state.variant === 'sprint' || state.variant === 'versus') && state.sprintStartTime > 0) {
    preLockCallback(
      state.history[state.history.length - 1],
      state.lastFrameTime - state.sprintStartTime,
    );
  }

  // Capture for lock hook (T-spin detection needs pre-lock board + piece position)
  const landedPiece = { ...state.active };
  const preLockBoard = state.board;
  const wasRotation = state.lastActionRotation;

  const locked = lockPiece(state.board, state.active);
  const { board: clearedBoard, linesCleared } = clearLines(locked);
  state.board = clearedBoard;
  state.score += scoreForLines(linesCleared, state.level);
  state.lines += linesCleared;
  state.level = Math.floor(state.lines / 10) + 1;

  // Fire versus hook (may modify state.board to inject garbage)
  lockHook?.(state, linesCleared, landedPiece, preLockBoard, wasRotation);
  state.lastActionRotation = false;

  if (state.variant === 'sprint' && state.lines >= 40) {
    state.lines = 40;
    state.sprintElapsedMs = state.lastFrameTime - state.sprintStartTime;
    state.mode = 'gameover';
    return;
  }

  if (isGameOver(state.board)) {
    state.mode = 'gameover';
    return;
  }

  // Restore bag from snapshotted state and dequeue next piece
  bag.restoreState(state.bagState);
  const nextType = state.nextQueue.shift()!;
  state.nextQueue.push(bag.next());
  state.bagState = bag.getState();

  state.active = spawnPiece(nextType);
  state.holdUsed = false;
  resetTransientState(state);

  // Immediate game-over: new piece spawns into filled cells
  if (collides(state.board, state.active, 0, 0)) {
    state.mode = 'gameover';
  }
}

function tryMove(state: GameState, dx: number, dy: number): boolean {
  if (collides(state.board, state.active, dx, dy)) return false;
  state.active = { ...state.active, x: state.active.x + dx, y: state.active.y + dy };
  return true;
}

function tryRotate(state: GameState, delta: number): boolean {
  const rotated = attemptRotation(state.board, state.active, delta);
  if (!rotated) return false;
  state.active = rotated;
  state.lastActionRotation = true;
  return true;
}

function tryHold(state: GameState): void {
  if (state.holdUsed) return;

  const incoming = state.hold ?? state.nextQueue.shift()!;
  if (!state.hold) {
    // Refill next queue
    bag.restoreState(state.bagState);
    state.nextQueue.push(bag.next());
    state.bagState = bag.getState();
  }
  state.hold = state.active.type;
  state.holdUsed = true;
  state.active = spawnPiece(incoming);
  resetTransientState(state);
  state.lastActionRotation = false;
}

function hardDrop(state: GameState): void {
  const dropDist = hardDropY(state.board, state.active) - state.active.y;
  state.score += dropDist * 2;
  state.active = { ...state.active, y: state.active.y + dropDist };
  lockAndSpawn(state);
}

export function processFrame(
  state: GameState,
  input: InputState,
  timestamp: number,
  settings: Settings,
): void {
  if (state.mode === 'menu') return;

  // First frame init
  if (state.lastFrameTime === 0) {
    state.lastFrameTime = timestamp;
    return;
  }

  let dt = timestamp - state.lastFrameTime;
  state.lastFrameTime = timestamp;
  // Guard against tab-hidden spikes
  if (dt > 100) dt = 100;

  const kb = settings.keybindings;
  const DAS_MS = settings.das;
  const ARR_MS = settings.arr;

  // --- One-shot key actions ---

  // Pause toggle (Escape always works as fallback; in creative Esc goes directly to menu)
  if (wasJustPressed(input, kb.pause) || wasJustPressed(input, 'Escape')) {
    if (state.mode === 'playing') {
      const escToMenu = wasJustPressed(input, 'Escape') && state.variant === 'creative';
      state.mode = escToMenu ? 'menu' : 'paused';
    }
    else if (state.mode === 'paused') {
      state.mode = wasJustPressed(input, 'Escape') ? 'menu' : 'playing';
      // Reset DAS so a held direction key doesn't fire ARR immediately on the first active frame
      if (state.mode === 'playing') { input.dasLeft = 0; input.dasRight = 0; }
    }
    else if (state.mode === 'gameover') state.mode = 'menu';
    else if (state.mode === 'countdown') state.mode = 'menu';
    return;
  }

  // Rewind / restart
  if (wasJustPressed(input, kb.rewind)) {
    if (state.variant === 'sprint' || state.variant === 'versus') {
      // Sprint/versus: R always restarts (mid-game or from game-over)
      const handle = state.rafHandle;
      Object.assign(state, initGameState(state.variant));
      state.rafHandle = handle;
    } else if (state.mode === 'gameover') {
      const handle = state.rafHandle;
      Object.assign(state, initGameState('creative'));
      state.rafHandle = handle;
    } else {
      rewind(state);
      input.dasLeft = 0;
      input.dasRight = 0;
    }
    return;
  }

  // Countdown tick
  if (state.mode === 'countdown') {
    state.countdownMs -= dt;
    if (state.countdownMs <= 0) {
      state.mode = 'playing';
      if (state.variant === 'sprint' || state.variant === 'versus') state.sprintStartTime = timestamp;
    }
    return;
  }

  // Editor toggle — creative only
  if (state.variant === 'creative' && wasJustPressed(input, kb.editor)) {
    if (state.mode === 'playing') {
      enterEditor(state);
    } else if (state.mode === 'editor') {
      exitEditor(state, (s) => {
        bag.restoreState(s.bagState);
        const nextType = s.nextQueue[0];
        s.active = spawnPiece(nextType);
      });
    }
    return;
  }

  if (state.mode !== 'playing') return;

  // Rotate (CW / CCW / 180°)
  for (const [key, delta] of [[kb.rotateCW, 1], [kb.rotateCCW, -1], [kb.rotate180, 2]] as [string, number][]) {
    if (wasJustPressed(input, key)) {
      if (tryRotate(state, delta) && isGrounded(state)) resetLockOnMove(state);
    }
  }

  // Hold
  if (wasJustPressed(input, kb.hold)) {
    tryHold(state);
  }

  // Hard drop
  if (wasJustPressed(input, kb.hardDrop)) {
    hardDrop(state);
    return;
  }

  // Sonic drop: teleport to ghost position without locking
  if (settings.sonicDrop && wasJustPressed(input, kb.softDrop)) {
    const ghostY = hardDropY(state.board, state.active);
    if (ghostY !== state.active.y) {
      state.active = { ...state.active, y: ghostY };
      resetTransientState(state);
    }
  }

  // DAS/ARR horizontal movement
  const softDropping = !settings.sonicDrop && isHeld(input, kb.softDrop);
  const movingLeft  = isHeld(input, kb.moveLeft);
  const movingRight = isHeld(input, kb.moveRight);

  // Last-key-wins: when both directions are held, only process the one pressed
  // more recently. dasLeft/dasRight start at 0 on key-down and accumulate each
  // frame, so the lower value belongs to the key pressed most recently.
  // Strict < on both sides: ties (simultaneous press) cancel both directions.
  const bothHeld = movingLeft && movingRight;
  const processLeft  = movingLeft  && (!bothHeld || input.dasLeft  < input.dasRight);
  const processRight = movingRight && (!bothHeld || input.dasRight < input.dasLeft);

  if (processLeft)
    input.dasLeft  = handleDasArr(state, wasJustPressed(input, kb.moveLeft),  input.dasLeft,  -1, dt, DAS_MS, ARR_MS);
  if (processRight)
    input.dasRight = handleDasArr(state, wasJustPressed(input, kb.moveRight), input.dasRight,  1, dt, DAS_MS, ARR_MS);

  // Direction-change delay (DCD): after processing the winning just-press, reset the
  // losing direction's DAS so it must re-charge before ARR resumes. Applied after
  // movement so the tap registers first, then the opposing direction re-charges.
  if (!processRight && wasJustPressed(input, kb.moveLeft))  input.dasRight = 0;
  if (!processLeft  && wasJustPressed(input, kb.moveRight)) input.dasLeft  = 0;

  // Gravity
  const interval = gravityInterval(state.level) / (softDropping ? SOFT_DROP_FACTOR : 1);
  state.gravityAccumMs += dt;

  while (state.gravityAccumMs >= interval) {
    state.gravityAccumMs -= interval;
    if (tryMove(state, 0, 1)) {
      if (softDropping) state.score += 1;
    } else {
      // Piece is grounded; lock delay handles it below
      break;
    }
  }

  // Lock delay
  if (isGrounded(state)) {
    if (state.lockResetCount >= LOCK_RESET_MAX) {
      // Max resets exhausted — lock immediately
      lockAndSpawn(state);
    } else {
      state.lockDelayMs -= dt;
      if (state.lockDelayMs <= 0) {
        lockAndSpawn(state);
      }
    }
  } else {
    // Not grounded — reset lock delay timer
    state.lockDelayMs = LOCK_DELAY_MS;
  }
}
