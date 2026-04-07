import { initGameState, processFrame, setLockHook } from './game';
import { setupInput, createInputState, flushInput, wasJustPressed } from './input';
import { setupEditor } from './editor';
import {
  draw, CANVAS_H, VERSUS_CANVAS_W,
  MENU_SPRINT_BTN, MENU_CREATIVE_BTN, MENU_VERSUS_BTN, MENU_WATCH_BTN, MENU_BVB_BTN, MENU_UPLOAD_BTN,
  MENU_DIFF_EASY_BTN, MENU_DIFF_MEDIUM_BTN, MENU_DIFF_HARD_BTN,
} from './renderer';
import { AiDifficulty, AI_DIFFICULTY_PARAMS } from './ai';
import { loadSettings } from './storage';
import { supabase } from './supabase';
import { setupAuthUI } from './authUI';
import { Settings } from './settings';
import { setupSettingsUI } from './settingsUI';
import { GameVariant } from './types';
import { VersusData, BotVsBotData, BotBoard, initVersusData, initBotVsBotData, applyBotMove, requestBotMove, setupPlayerLockHook } from './versus';
import { drawEngineOverlay } from './engineOverlay';
import { gameStateToEngineRequest } from './engine';
import type { EngineAnalysis } from './engine';
import { lockPiece, clearLines } from './board';
import type { CellValue } from './types';
import { setPreLockCallback } from './game';
import type { SprintReplay, ReplayEntry, VersusReplay, VersusReplayEntry, BotSnapshot } from './replay';
import { drawReplayScreen, drawVersusReplayScreen } from './renderer';

const canvas = document.getElementById('game') as HTMLCanvasElement;
canvas.width = VERSUS_CANVAS_W;
canvas.height = CANVAS_H;
canvas.focus();

const ctx = canvas.getContext('2d')!;

// Boot in menu mode.
const state = initGameState('creative');
state.mode = 'menu';

const input = createInputState();
setupInput(canvas, input);
setupEditor(canvas, state);

// ---- Saved AI persistence ----

interface SavedAI { id: string; name: string; code: string; }

let savedAIs: SavedAI[] = [];
let selectedAiIndex = -2; // negative = built-in: -3=easy, -2=medium, -1=hard

async function loadSavedAIs(): Promise<void> {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) { savedAIs = []; return; }
  const { data } = await supabase
    .from('user_ais')
    .select('id, name, code')
    .eq('user_id', user.id)
    .order('created_at');
  savedAIs = (data ?? []) as SavedAI[];
}

async function uploadAI(name: string, code: string): Promise<void> {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return;
  const existing = savedAIs.find(a => a.name === name);
  if (existing) {
    await supabase.from('user_ais').update({ code }).eq('id', existing.id);
    existing.code = code;
  } else {
    const { data } = await supabase
      .from('user_ais')
      .insert({ user_id: user.id, name, code })
      .select('id')
      .single();
    if (data) savedAIs.push({ id: data.id as string, name, code });
  }
}

async function deleteAI(index: number): Promise<void> {
  const ai = savedAIs[index];
  if (!ai) return;
  await supabase.from('user_ais').delete().eq('id', ai.id);
  savedAIs.splice(index, 1);
  if (selectedAiIndex === index) selectedAiIndex = -2;
  else if (selectedAiIndex > index) selectedAiIndex--;
}

function builtinDifficultyOf(index: number): AiDifficulty | null {
  if (index === -3) return 'easy';
  if (index === -2) return 'medium';
  if (index === -1) return 'hard';
  return null;
}

// ---- Worker state ----

let versusData: VersusData | null = null;
let aiWorker: Worker | null = null;
let customAiWorker: Worker | null = null;
let customAiName: string | null = null;
let customAiBlobUrl: string | null = null;
let botMoveTimeout: ReturnType<typeof setTimeout> | null = null;
let customAiError: string | null = null;
let customAiWarning: string | null = null;
let versusDifficulty: AiDifficulty = 'medium';
let versusDifficultyPending = false;
let bvbDifficulty: AiDifficulty = 'medium';
let bvbDifficultyPending = false;

// ---- Engine analysis state ----
let engineMode = false;
let engineAnalysis: EngineAnalysis | null = null;
let engineSelectedLine = 0;
let engineWorker: Worker | null = null;
let engineBoardStates: CellValue[][][] = [];
let engineAnimFrame = 0;
let engineAnimLastAdvance = 0;
const ENGINE_ANIM_MS = 700;

function computeEngineBoardStates(
  board: CellValue[][],
  analysis: EngineAnalysis,
  sel: number,
): CellValue[][][] {
  const line = analysis.lines[Math.min(sel, analysis.lines.length - 1)];
  if (!line || line.moves.length === 0) return [board];
  const states: CellValue[][][] = [board];
  let cur = board;
  for (const mv of line.moves) {
    const locked = lockPiece(cur, { type: mv.pieceType, rotationIndex: mv.rotationIndex, x: mv.x, y: mv.y });
    const { board: cleared } = clearLines(locked);
    states.push(cleared);
    cur = cleared;
  }
  return states;
}

function startEngineMode(): void {
  engineMode = true;
  engineAnalysis = null;
  engineSelectedLine = 0;

  if (!engineWorker) {
    engineWorker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
    engineWorker.onerror = (e) => console.error('Engine worker error:', e.message);
    engineWorker.onmessage = (e: MessageEvent) => {
      if (e.data?.type === 'analysis') {
        engineAnalysis = e.data.result as EngineAnalysis;
        engineSelectedLine = 0;
        engineAnimFrame = 0;
        engineAnimLastAdvance = 0;
        engineBoardStates = computeEngineBoardStates(state.board, engineAnalysis, 0);
      }
    };
  }

  const request = gameStateToEngineRequest(state, { beamWidth: 48, searchDepth: 6, topN: 5 });
  engineWorker.postMessage({ type: 'analyze', request });
}

function stopEngineMode(): void {
  engineMode = false;
  engineAnalysis = null;
  engineSelectedLine = 0;
  engineBoardStates = [];
  engineAnimFrame = 0;
  engineAnimLastAdvance = 0;
}

// ---- Sprint replay state ----
let latestSprintReplay: SprintReplay | null = null;
let recordingEntries: ReplayEntry[] = [];
let replayMode = false;
let replayPaused = false;
let replayElapsedMs = 0;

function startSprintRecording(): void {
  recordingEntries = [];
  setPreLockCallback((snapshot, elapsedMs) => {
    recordingEntries.push({ snapshot, elapsedMs });
  });
}

function finishSprintRecording(finalBoard: CellValue[][], finalElapsedMs: number): void {
  latestSprintReplay = {
    entries: recordingEntries,
    finalElapsedMs,
    finalBoard: structuredClone(finalBoard),
  };
}

// ---- Versus replay state ----
let latestVersusReplay: VersusReplay | null = null;
let versusRecordingEntries: VersusReplayEntry[] = [];
let versusReplayMode = false;
let versusReplayPaused = false;
let versusReplayElapsedMs = 0;

function snapshotFromBot(bot: import('./versus').BotBoard): BotSnapshot {
  return {
    board: structuredClone(bot.board),
    active: { ...bot.active },
    nextQueue: [...bot.nextQueue],
    hold: bot.hold,
    holdUsed: bot.holdUsed,
    lines: bot.lines,
    dead: bot.dead,
  };
}

function snapshotFromState(s: import('./types').GameState): import('./types').Snapshot {
  return {
    board: structuredClone(s.board),
    score: s.score,
    lines: s.lines,
    level: s.level,
    nextQueue: [...s.nextQueue],
    hold: s.hold,
    holdUsed: s.holdUsed,
    active: { ...s.active },
    bagState: [...s.bagState],
  };
}

function startVersusRecording(): void {
  versusRecordingEntries = [];
  setPreLockCallback((snapshot, elapsedMs) => {
    if (versusData) {
      versusRecordingEntries.push({ elapsedMs, playerSnapshot: snapshot, botSnapshot: snapshotFromBot(versusData.bot) });
    }
  });
}

function finishVersusRecording(
  winner: 'player' | 'bot',
  finalElapsedMs: number,
  finalPlayerBoard: CellValue[][],
  finalBotBoard: CellValue[][],
): void {
  latestVersusReplay = {
    entries: [...versusRecordingEntries],
    finalElapsedMs,
    winner,
    finalPlayerBoard: structuredClone(finalPlayerBoard),
    finalBotBoard: structuredClone(finalBotBoard),
  };
}

// ---- Bot vs Bot state ----
let botVsBotData: BotVsBotData | null = null;
let bvbWorker1: Worker | null = null;
let bvbWorker2: Worker | null = null;
let bvbBot1Timeout: ReturnType<typeof setTimeout> | null = null;

function getAiWorker(): Worker {
  if (customAiWorker) return customAiWorker;
  if (!aiWorker) {
    aiWorker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
    aiWorker.onerror = (e) => console.error('Built-in AI worker error:', e.message);
  }
  return aiWorker;
}

function isValidMove(m: unknown): m is { rotationIndex: number; x: number; y: number; useHold: boolean } {
  if (!m || typeof m !== 'object') return false;
  const o = m as Record<string, unknown>;
  return typeof o.rotationIndex === 'number' && typeof o.x === 'number' &&
         typeof o.y === 'number' && typeof o.useHold === 'boolean';
}

function wireWorker(w: Worker): void {
  w.onmessage = (e) => {
    if (e.data?.error) {
      console.warn('AI worker error:', e.data.error);
      if (customAiWorker) customAiError = `Runtime error: ${e.data.error}`;
      return;
    }
    if (botMoveTimeout) { clearTimeout(botMoveTimeout); botMoveTimeout = null; }
    if (versusData) {
      if (isValidMove(e.data)) {
        versusData.pendingMove = e.data;
      } else if (customAiWorker) {
        customAiError = 'getBestMove returned an invalid move object (expected { rotationIndex, x, y, useHold })';
      }
    }
  };
}

function requestMoveWithTimeout(bot: BotBoard, pendingGarbage: number): void {
  const w = getAiWorker();
  let aiParams: typeof AI_DIFFICULTY_PARAMS[AiDifficulty] | undefined;
  if (!customAiWorker) {
    const builtinDiff = builtinDifficultyOf(selectedAiIndex);
    aiParams = state.variant === 'versus'
      ? AI_DIFFICULTY_PARAMS[versusDifficulty]
      : AI_DIFFICULTY_PARAMS[builtinDiff ?? 'medium'];
  }
  requestBotMove(w, bot, pendingGarbage, aiParams);
  if (botMoveTimeout) clearTimeout(botMoveTimeout);
  botMoveTimeout = setTimeout(() => {
    botMoveTimeout = null;
    if (state.variant === 'watch' && customAiWorker) {
      // Watch mode: terminate the game and show an error.
      customAiError = 'AI exceeded the 2 s time limit — game terminated';
      customAiWarning = null;
      customAiWorker.terminate();
      customAiWorker = null;
      customAiName = null;
      if (versusData) versusData.bot.dead = true;
    } else {
      // Versus mode: fall back to built-in AI silently.
      if (customAiWorker) {
        customAiWorker.terminate();
        customAiWorker = null;
        customAiName = null;
      }
      const fallback = getAiWorker();
      wireWorker(fallback);
      requestBotMove(fallback, bot, pendingGarbage);
    }
  }, 2000);
}

// Creates a blob worker from the selected saved AI (or clears to built-in).
// Call this at every game start so the correct AI is active.
function setupAiForGame(): void {
  customAiWorker?.terminate();
  customAiWorker = null;
  customAiName = null;
  if (customAiBlobUrl) { URL.revokeObjectURL(customAiBlobUrl); customAiBlobUrl = null; }
  if (selectedAiIndex >= 0 && selectedAiIndex < savedAIs.length) {
    const ai = savedAIs[selectedAiIndex];
    // Neutralise all outbound network APIs before running user code.
    // blob: workers are not subject to the page's connect-src CSP in Chrome,
    // so we override them here as a reliable defence-in-depth measure.
    const networkBlock = `
(function(){
  const blocked = () => { throw new Error('Network access is blocked in AI workers'); };
  Object.defineProperty(self, 'fetch',          { value: () => Promise.reject(blocked()), configurable: false, writable: false });
  Object.defineProperty(self, 'XMLHttpRequest', { value: class { open(){blocked();} }, configurable: false, writable: false });
  Object.defineProperty(self, 'WebSocket',      { value: class { constructor(){blocked();} }, configurable: false, writable: false });
  Object.defineProperty(self, 'importScripts',  { value: blocked, configurable: false, writable: false });
})();`;
    const src = `${networkBlock}\n${ai.code}\nself.onmessage=(e)=>{try{const{bot,pendingGarbage}=e.data;self.postMessage(getBestMove(bot,pendingGarbage));}catch(err){self.postMessage({error:String(err)});}};`;
    const blob = new Blob([src], { type: 'application/javascript' });
    customAiBlobUrl = URL.createObjectURL(blob);
    customAiWorker = new Worker(customAiBlobUrl);
    customAiWorker.onerror = (e) => console.warn('Custom AI worker error:', e.message);
    customAiName = ai.name;
  }
}

// Reinitialise game state while preserving the rAF handle.
function restartState(variant: GameVariant): void {
  const handle = state.rafHandle;
  Object.assign(state, initGameState(variant));
  state.rafHandle = handle;
}

function stopBvbWorkers(): void {
  if (bvbBot1Timeout) { clearTimeout(bvbBot1Timeout); bvbBot1Timeout = null; }
  bvbWorker1?.terminate(); bvbWorker1 = null;
  bvbWorker2?.terminate(); bvbWorker2 = null;
  botVsBotData = null;
}

function requestBvbBot1Move(bot: BotBoard, pendingGarbage: number): void {
  if (!bvbWorker1) return;
  // Custom AI workers ignore difficulty params; built-in bot1 uses the AI manager selection.
  const builtinDiff = builtinDifficultyOf(selectedAiIndex);
  requestBotMove(bvbWorker1, bot, pendingGarbage, customAiName ? undefined : AI_DIFFICULTY_PARAMS[builtinDiff ?? 'medium']);
  if (bvbBot1Timeout) clearTimeout(bvbBot1Timeout);
  if (customAiName) {
    bvbBot1Timeout = setTimeout(() => {
      bvbBot1Timeout = null;
      customAiError = 'AI exceeded the 2 s time limit — game terminated';
      customAiWarning = null;
      if (botVsBotData) botVsBotData.bot1.dead = true;
    }, 2000);
  }
}

function startBotVsBot(): void {
  stopBvbWorkers(); // ensure any prior workers are cleaned up before creating new ones
  customAiError = null;
  customAiWarning = null;
  botVsBotData = initBotVsBotData();

  // Bot 1: use selected custom AI if any, otherwise a fresh built-in worker
  setupAiForGame();
  const bvbBot1IsCustom = !!customAiWorker;
  if (customAiWorker) {
    bvbWorker1 = customAiWorker;
    customAiWorker = null; // bvbWorker1 now owns it
  } else {
    bvbWorker1 = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
  }

  // Bot 2: always a fresh built-in worker instance
  bvbWorker2 = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });

  bvbWorker1.onerror = (e) => console.error('BvB worker 1 error:', e.message);
  bvbWorker2.onerror = (e) => console.error('BvB worker 2 error:', e.message);

  bvbWorker1.onmessage = (e) => {
    if (bvbBot1Timeout) { clearTimeout(bvbBot1Timeout); bvbBot1Timeout = null; }
    if (e.data?.error) {
      console.warn('BvB worker 1 error:', e.data.error);
      if (bvbBot1IsCustom) customAiError = `Runtime error: ${e.data.error}`;
      return;
    }
    if (botVsBotData) {
      if (isValidMove(e.data)) {
        botVsBotData.pendingMove1 = e.data;
      } else if (bvbBot1IsCustom) {
        customAiError = 'getBestMove returned an invalid move object (expected { rotationIndex, x, y, useHold })';
      }
    }
  };
  bvbWorker2.onmessage = (e) => {
    if (botVsBotData && isValidMove(e.data)) botVsBotData.pendingMove2 = e.data;
  };

  requestBvbBot1Move(botVsBotData.bot1, 0);
  requestBotMove(bvbWorker2, botVsBotData.bot2, 0, AI_DIFFICULTY_PARAMS[bvbDifficulty]);
}

// ---- AI manager overlay ----

function renderAiList(): void {
  const list = document.getElementById('ai-list')!;
  list.innerHTML = '';

  // Built-in rows
  list.appendChild(makeAiRow(-3, 'Easy',   'BUILT-IN', false));
  list.appendChild(makeAiRow(-2, 'Medium', 'BUILT-IN', false));
  list.appendChild(makeAiRow(-1, 'Hard',   'BUILT-IN', false));

  // Saved AI rows
  savedAIs.forEach((ai, i) => {
    list.appendChild(makeAiRow(i, ai.name, 'CUSTOM', true));
  });
}

function makeAiRow(index: number, name: string, badge: string, deletable: boolean): HTMLElement {
  const item = document.createElement('div');
  item.className = 'ai-item' + (selectedAiIndex === index ? ' selected' : '');

  const radio = document.createElement('div');
  radio.className = 'ai-radio';

  const nameEl = document.createElement('div');
  nameEl.className = 'ai-name';
  nameEl.textContent = name;
  nameEl.title = name;

  const badgeEl = document.createElement('div');
  badgeEl.className = 'ai-badge';
  badgeEl.textContent = badge;

  item.appendChild(radio);
  item.appendChild(nameEl);
  item.appendChild(badgeEl);

  if (deletable) {
    const del = document.createElement('button');
    del.className = 'ai-delete';
    del.textContent = 'DEL';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      await deleteAI(index);
      renderAiList();
    });
    item.appendChild(del);
  }

  item.addEventListener('click', () => {
    selectedAiIndex = index;
    renderAiList();
  });

  return item;
}

async function openAiManager(): Promise<void> {
  await loadSavedAIs();
  renderAiList();
  const overlay = document.getElementById('ai-manager')!;
  overlay.style.display = 'flex';
  canvas.blur();
}

function closeAiManager(): void {
  document.getElementById('ai-manager')!.style.display = 'none';
  canvas.focus();
}

// Wire AI manager DOM events (file upload, done button, escape key)
document.getElementById('ai-upload')!.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (ev) => {
    const code = ev.target!.result as string;
    const existingIdx = savedAIs.findIndex(a => a.name === file.name);
    await uploadAI(file.name, code);
    selectedAiIndex = existingIdx >= 0 ? existingIdx : savedAIs.length - 1;
    renderAiList();
  };
  reader.onerror = () => {
    console.error('Failed to read AI file:', file.name);
  };
  reader.readAsText(file);
  (e.target as HTMLInputElement).value = '';
});

document.getElementById('ai-done-btn')!.addEventListener('click', closeAiManager);

// Close manager when clicking the backdrop (outside the panel)
document.getElementById('ai-manager')!.addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeAiManager();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('ai-manager')!.style.display !== 'none') {
    closeAiManager();
  }
  if (e.key === 'Escape' && versusDifficultyPending) {
    versusDifficultyPending = false;
  }
  if (e.key === 'Escape' && bvbDifficultyPending) {
    bvbDifficultyPending = false;
  }
});

// Menu click handler — hit-test against exported button rects
canvas.addEventListener('click', (e) => {
  if (state.mode !== 'menu') return;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (VERSUS_CANVAS_W / rect.width);
  const my = (e.clientY - rect.top)  * (CANVAS_H / rect.height);

  const hit = (b: typeof MENU_SPRINT_BTN) =>
    mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h;

  if (hit(MENU_UPLOAD_BTN) && isAdmin) {
    openAiManager();
    return;
  }

  if (hit(MENU_BVB_BTN) && isAdmin) {
    bvbDifficultyPending = true;
    return;
  }

  // Difficulty selection screen (shared by versus and bot-vs-bot)
  if (versusDifficultyPending) {
    let chosen: AiDifficulty | null = null;
    if (hit(MENU_DIFF_EASY_BTN))   chosen = 'easy';
    if (hit(MENU_DIFF_MEDIUM_BTN)) chosen = 'medium';
    if (hit(MENU_DIFF_HARD_BTN))   chosen = 'hard';
    if (!chosen) return;
    versusDifficulty = chosen;
    versusDifficultyPending = false;
    customAiError = null;
    customAiWarning = null;
    restartState('versus');
    versusData = initVersusData(state);
    setupPlayerLockHook(versusData);
    setupAiForGame();
    wireWorker(getAiWorker());
    requestMoveWithTimeout(versusData.bot, 0);
    startVersusRecording();
    return;
  }

  if (bvbDifficultyPending) {
    let chosen: AiDifficulty | null = null;
    if (hit(MENU_DIFF_EASY_BTN))   chosen = 'easy';
    if (hit(MENU_DIFF_MEDIUM_BTN)) chosen = 'medium';
    if (hit(MENU_DIFF_HARD_BTN))   chosen = 'hard';
    if (!chosen) return;
    bvbDifficulty = chosen;
    bvbDifficultyPending = false;
    customAiError = null;
    customAiWarning = null;
    restartState('botvsbot');
    startBotVsBot();
    return;
  }

  let variant: GameVariant | null = null;
  if (hit(MENU_SPRINT_BTN))   variant = 'sprint';
  if (hit(MENU_CREATIVE_BTN)) variant = 'creative';
  if (hit(MENU_VERSUS_BTN))   { versusDifficultyPending = true; return; }
  if (hit(MENU_WATCH_BTN) && isAdmin) variant = 'watch';
  if (!variant) return;

  restartState(variant);

  if (variant === 'sprint') {
    startSprintRecording();
  } else if (variant === 'watch') {
    customAiError = null;
    customAiWarning = null;
    versusData = initVersusData();
    setLockHook(null);
    setupAiForGame();
    wireWorker(getAiWorker());
    requestMoveWithTimeout(versusData.bot, 0);
  } else {
    versusData = null;
    setLockHook(null);
  }
});

let settings: Settings;
let isAdmin = false;
let prevTimestamp = 0;

async function startGame(userId: string): Promise<void> {
  const [loaded, profileResult] = await Promise.all([
    loadSettings(userId),
    supabase.from('profiles').select('is_admin').eq('user_id', userId).single(),
  ]);
  settings = loaded;
  isAdmin = profileResult.data?.is_admin ?? false;

    setupSettingsUI(
      state,
      () => settings,
      (updated) => { settings = updated; },
      userId,
    );

    const signOutBtn = document.createElement('button');
    signOutBtn.textContent = 'Sign Out';
    Object.assign(signOutBtn.style, {
      position: 'fixed',
      bottom: '18px',
      right: '18px',
      background: '#1e1e3a',
      color: '#888899',
      border: '1px solid #3a3a6a',
      padding: '7px 18px',
      fontSize: '13px',
      fontFamily: 'monospace',
      cursor: 'pointer',
      zIndex: '10',
    });
    signOutBtn.addEventListener('mouseover', () => { signOutBtn.style.background = '#2a2a4a'; });
    signOutBtn.addEventListener('mouseout',  () => { signOutBtn.style.background = '#1e1e3a'; });
    signOutBtn.addEventListener('click', async () => {
      await supabase.auth.signOut();
      window.location.reload();
    });
    document.body.appendChild(signOutBtn);

    function gameLoop(timestamp: number): void {
    const dt = prevTimestamp === 0 ? 0 : Math.min(timestamp - prevTimestamp, 100);
    prevTimestamp = timestamp;

    // If we've navigated back to menu from versus, tear down the hook and custom worker
    if (versusData !== null && state.mode === 'menu') {
      setLockHook(null);
      versusData = null;
      customAiWorker?.terminate();
      customAiWorker = null;
      customAiName = null;
      if (customAiBlobUrl) { URL.revokeObjectURL(customAiBlobUrl); customAiBlobUrl = null; }
    }
    if (botVsBotData !== null && state.mode === 'menu') {
      stopBvbWorkers();
    }
    if (engineMode && state.mode === 'menu') {
      stopEngineMode();
      engineWorker?.terminate();
      engineWorker = null;
    }

    // ---- REPLAY MODE (sprint) ----
    if (replayMode && latestSprintReplay) {
      if (wasJustPressed(input, 'Space')) replayPaused = !replayPaused;
      if (wasJustPressed(input, 'KeyR')) { replayElapsedMs = 0; replayPaused = false; }
      if (wasJustPressed(input, 'Escape')) replayMode = false;

      if (!replayPaused) {
        replayElapsedMs = Math.min(replayElapsedMs + dt, latestSprintReplay.finalElapsedMs);
      }

      drawReplayScreen(ctx, latestSprintReplay, replayElapsedMs, replayPaused);
      flushInput(input);
      state.rafHandle = requestAnimationFrame(gameLoop);
      return;
    }

    // ---- REPLAY MODE (versus) ----
    if (versusReplayMode && latestVersusReplay) {
      if (wasJustPressed(input, 'Space')) versusReplayPaused = !versusReplayPaused;
      if (wasJustPressed(input, 'KeyR')) { versusReplayElapsedMs = 0; versusReplayPaused = false; }
      if (wasJustPressed(input, 'Escape')) versusReplayMode = false;

      if (!versusReplayPaused) {
        versusReplayElapsedMs = Math.min(versusReplayElapsedMs + dt, latestVersusReplay.finalElapsedMs);
      }

      drawVersusReplayScreen(ctx, latestVersusReplay, versusReplayElapsedMs, versusReplayPaused);
      flushInput(input);
      state.rafHandle = requestAnimationFrame(gameLoop);
      return;
    }

    // Watch / bot-vs-bot: handle input directly (no player game to advance)
    if ((state.variant === 'watch' || state.variant === 'botvsbot') && state.mode !== 'menu') {
      if (wasJustPressed(input, settings.keybindings.pause) || wasJustPressed(input, 'Escape')) {
        state.mode = 'menu';
      } else if (wasJustPressed(input, settings.keybindings.rewind)) {
        if (state.variant === 'watch' && versusData) {
          customAiError = null;
          customAiWarning = null;
          versusData = initVersusData();
          setupAiForGame();
          wireWorker(getAiWorker());
          requestMoveWithTimeout(versusData.bot, 0);
        } else if (state.variant === 'botvsbot') {
          stopBvbWorkers();
          restartState('botvsbot');
          startBotVsBot();
        }
      }
    } else {
      // Detect versus restart (game.ts does Object.assign(state, initGameState('versus'))
      // which resets lastFrameTime to 0; processFrame sets it on next call)
      const wasLastFrameTime = state.lastFrameTime;
      const wasSprintGameover  = state.variant === 'sprint'  && state.mode === 'gameover';
      const wasVersusGameover  = state.variant === 'versus'  && state.mode === 'gameover';

      // "W" key on sprint complete screen → enter sprint replay
      if (wasSprintGameover && latestSprintReplay && wasJustPressed(input, 'KeyW')) {
        replayMode = true;
        replayElapsedMs = 0;
        replayPaused = false;
      }

      // "W" key on versus result screen → enter versus replay
      if (wasVersusGameover && latestVersusReplay && wasJustPressed(input, 'KeyW')) {
        versusReplayMode = true;
        versusReplayElapsedMs = 0;
        versusReplayPaused = false;
      }

      processFrame(state, input, timestamp, settings);

      // Sprint gameover just happened — finalize replay
      if (state.variant === 'sprint' && !wasSprintGameover && state.mode === 'gameover') {
        finishSprintRecording(state.board, state.sprintElapsedMs);
      }

      // Sprint restart via R key (lastFrameTime resets to 0)
      if (state.variant === 'sprint' && wasLastFrameTime !== 0 && state.lastFrameTime === 0) {
        startSprintRecording();
      }

      if (versusData !== null && state.variant === 'versus') {
        // Restart detected: reinit bot and re-register hook
        if (wasLastFrameTime !== 0 && state.lastFrameTime === 0) {
          versusData = initVersusData(state);
          setupPlayerLockHook(versusData);
          setupAiForGame();
          wireWorker(getAiWorker());
          requestMoveWithTimeout(versusData.bot, 0);
          startVersusRecording();
        }

        // Determine winner
        if (versusData.winner === null) {
          const playerDead = state.mode === 'gameover';
          const botDead = versusData.bot.dead;
          const elapsed = state.sprintStartTime > 0 ? timestamp - state.sprintStartTime : 0;
          if (botDead && !playerDead) {
            versusData.winner = 'player';
            state.mode = 'gameover';
            finishVersusRecording('player', elapsed, state.board, versusData.bot.board);
          } else if (playerDead && !botDead) {
            versusData.winner = 'bot';
            versusData.bot.dead = true;
            finishVersusRecording('bot', elapsed, state.board, versusData.bot.board);
          } else if (playerDead && botDead) {
            // Both died on the same frame — player wins (their clear triggered the kill)
            versusData.winner = 'player';
            finishVersusRecording('player', elapsed, state.board, versusData.bot.board);
          }
        }
      }
    }

    // Engine analysis — creative mode only, while paused
    if (state.variant === 'creative') {
      if (state.mode === 'paused') {
        if (wasJustPressed(input, settings.keybindings.engineAnalysis)) {
          if (engineMode) stopEngineMode();
          else startEngineMode();
        }
        if (engineMode && engineAnalysis && engineAnalysis.lines.length > 0) {
          const lineCount = engineAnalysis.lines.length;
          if (wasJustPressed(input, 'ArrowDown')) {
            engineSelectedLine = (engineSelectedLine + 1) % lineCount;
            engineAnimFrame = 0;
            engineAnimLastAdvance = 0;
            engineBoardStates = computeEngineBoardStates(state.board, engineAnalysis, engineSelectedLine);
          }
          if (wasJustPressed(input, 'ArrowUp')) {
            engineSelectedLine = (engineSelectedLine - 1 + lineCount) % lineCount;
            engineAnimFrame = 0;
            engineAnimLastAdvance = 0;
            engineBoardStates = computeEngineBoardStates(state.board, engineAnalysis, engineSelectedLine);
          }
        }
      }
      // Exit engine mode whenever the game is no longer paused
      if (engineMode && state.mode !== 'paused') stopEngineMode();
    }

    // Step bot for versus (when playing) and watch (until topped out)
    if (versusData !== null && !versusData.bot.dead) {
      const botActive = state.variant === 'watch'
        ? state.mode !== 'menu'
        : state.mode === 'playing';
      if (botActive) {
        versusData.botThinkAccumMs += dt;
        const msPerPiece = 1000 / settings.botPps;
        while (versusData.botThinkAccumMs >= msPerPiece && !versusData.bot.dead) {
          if (versusData.pendingMove === null) break; // wait for worker
          versusData.botThinkAccumMs -= msPerPiece;
          const move = versusData.pendingMove;
          versusData.pendingMove = null;
          const watchResult = applyBotMove(move, versusData.bot, versusData.botCombat, versusData.playerCombat);
          if (watchResult && customAiWorker && !customAiError) {
            customAiError = watchResult === 'floating'
              ? `Floating piece: position (x:${move.x}, y:${move.y}, rot:${move.rotationIndex}) is not the lowest valid row — hard drop used instead`
              : `Invalid position (x:${move.x}, y:${move.y}, rot:${move.rotationIndex}) was occupied — hard drop used instead`;
          }
          // Record bot lock for versus replay
          if (state.variant === 'versus' && state.sprintStartTime > 0) {
            versusRecordingEntries.push({
              elapsedMs: timestamp - state.sprintStartTime,
              playerSnapshot: snapshotFromState(state),
              botSnapshot: snapshotFromBot(versusData.bot),
            });
          }
          if (!versusData.bot.dead) {
            requestMoveWithTimeout(versusData.bot, versusData.botCombat.pendingGarbage);
          }
        }
        if (!versusData.bot.dead && versusData.pendingMove === null &&
            versusData.botThinkAccumMs >= msPerPiece && customAiWorker && !customAiWarning && !customAiError) {
          customAiWarning = `AI is too slow for ${settings.botPps} PPS — move computation is taking longer than ${(1000 / settings.botPps).toFixed(0)} ms`;
        }
      }
    }

    // Step bot-vs-bot
    if (state.variant === 'botvsbot' && state.mode === 'playing' && botVsBotData) {
      const msPerPiece = 1000 / settings.botPps;
      botVsBotData.bot1ThinkAccumMs += dt;
      botVsBotData.bot2ThinkAccumMs += dt;

      while (botVsBotData.bot1ThinkAccumMs >= msPerPiece && !botVsBotData.bot1.dead) {
        if (!botVsBotData.pendingMove1) break;
        botVsBotData.bot1ThinkAccumMs -= msPerPiece;
        const m = botVsBotData.pendingMove1; botVsBotData.pendingMove1 = null;
        const bvbResult = applyBotMove(m, botVsBotData.bot1, botVsBotData.bot1Combat, botVsBotData.bot2Combat);
        if (bvbResult && customAiName && !customAiError) {
          customAiError = bvbResult === 'floating'
            ? `Floating piece: position (x:${m.x}, y:${m.y}, rot:${m.rotationIndex}) is not the lowest valid row — hard drop used instead`
            : `Invalid position (x:${m.x}, y:${m.y}, rot:${m.rotationIndex}) was occupied — hard drop used instead`;
        }
        if (!botVsBotData.bot1.dead && bvbWorker1)
          requestBvbBot1Move(botVsBotData.bot1, botVsBotData.bot1Combat.pendingGarbage);
      }
      if (!botVsBotData.bot1.dead && !botVsBotData.pendingMove1 &&
          botVsBotData.bot1ThinkAccumMs >= msPerPiece && !customAiWarning && !customAiError) {
        const bot1Label = customAiName ?? AI_DIFFICULTY_PARAMS[builtinDifficultyOf(selectedAiIndex) ?? 'medium'].label;
        customAiWarning = `Bot 1 (${bot1Label}) is too slow for ${settings.botPps} PPS — reduce PPS in settings`;
      }

      while (botVsBotData.bot2ThinkAccumMs >= msPerPiece && !botVsBotData.bot2.dead) {
        if (!botVsBotData.pendingMove2) break;
        botVsBotData.bot2ThinkAccumMs -= msPerPiece;
        const m = botVsBotData.pendingMove2; botVsBotData.pendingMove2 = null;
        applyBotMove(m, botVsBotData.bot2, botVsBotData.bot2Combat, botVsBotData.bot1Combat);
        if (!botVsBotData.bot2.dead && bvbWorker2)
          requestBotMove(bvbWorker2, botVsBotData.bot2, botVsBotData.bot2Combat.pendingGarbage, AI_DIFFICULTY_PARAMS[bvbDifficulty]);
      }
      if (!botVsBotData.bot2.dead && !botVsBotData.pendingMove2 &&
          botVsBotData.bot2ThinkAccumMs >= msPerPiece && !customAiWarning && !customAiError) {
        customAiWarning = `Bot 2 (${AI_DIFFICULTY_PARAMS[bvbDifficulty].label}) is too slow for ${settings.botPps} PPS — reduce PPS in settings`;
      }

      // Winner detection
      if (botVsBotData.winner === null) {
        const b1 = botVsBotData.bot1.dead, b2 = botVsBotData.bot2.dead;
        if (b1 || b2) {
          botVsBotData.winner = (b1 && b2) ? 'draw' : b1 ? 'bot2' : 'bot1';
          state.mode = 'gameover';
        }
      }
    }

    // Compute display names for bot labels in versus/watch/BvB.
    const builtinDiff = builtinDifficultyOf(selectedAiIndex);
    const managerBotName = customAiName ?? AI_DIFFICULTY_PARAMS[builtinDiff ?? 'medium'].label;
    const bot1Name = state.variant === 'versus'
      ? (customAiName ?? AI_DIFFICULTY_PARAMS[versusDifficulty].label)
      : managerBotName;
    const bot2Name = AI_DIFFICULTY_PARAMS[bvbDifficulty].label;
    draw(ctx, state, versusData, customAiName, botVsBotData, isAdmin, customAiError, customAiWarning, versusDifficultyPending || bvbDifficultyPending, bot1Name, bot2Name, settings.keybindings);
    if (engineMode && state.mode === 'paused' && state.variant === 'creative') {
      if (engineAnalysis && engineBoardStates.length > 1) {
        const moveCount = engineBoardStates.length - 1;
        if (engineAnimLastAdvance === 0) {
          engineAnimLastAdvance = timestamp;
        } else if (timestamp - engineAnimLastAdvance >= ENGINE_ANIM_MS) {
          engineAnimFrame = (engineAnimFrame + 1) % moveCount;
          engineAnimLastAdvance = timestamp;
        }
      }
      drawEngineOverlay(ctx, engineAnalysis, engineSelectedLine, engineBoardStates, engineAnimFrame, timestamp);
    }
    flushInput(input);
    state.rafHandle = requestAnimationFrame(gameLoop);
  }

  state.rafHandle = requestAnimationFrame(gameLoop);
}

setupAuthUI(startGame);
