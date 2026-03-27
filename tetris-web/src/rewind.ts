import { GameState, Snapshot } from './types';

const MAX_HISTORY = 50;

export function takeSnapshot(state: GameState): Snapshot {
  return structuredClone({
    board: state.board,
    score: state.score,
    lines: state.lines,
    level: state.level,
    nextQueue: state.nextQueue,
    hold: state.hold,
    holdUsed: state.holdUsed,
    active: state.active,
    bagState: state.bagState,
  });
}

export function pushHistory(state: GameState): void {
  state.history.push(takeSnapshot(state));
  if (state.history.length > MAX_HISTORY) {
    state.history.shift();
  }
}

export function rewind(state: GameState): void {
  if (state.history.length === 0) return;
  const snap = structuredClone(state.history.pop()!);
  state.board = snap.board;
  state.score = snap.score;
  state.lines = snap.lines;
  state.level = snap.level;
  state.nextQueue = snap.nextQueue;
  state.hold = snap.hold;
  state.holdUsed = snap.holdUsed;
  state.active = snap.active;
  state.bagState = snap.bagState;
  // Reset transient timing (lastFrameTime = 0 forces processFrame to reinitialise dt)
  state.lockDelayMs = 500;
  state.lockResetCount = 0;
  state.gravityAccumMs = 0;
  state.lastFrameTime = 0;
  state.mode = 'playing';
}
