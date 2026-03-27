import { initGameState, processFrame, setLockHook } from './game';
import { setupInput, createInputState, flushInput, wasJustPressed } from './input';
import { setupEditor } from './editor';
import {
  draw, CANVAS_H, VERSUS_CANVAS_W,
  MENU_SPRINT_BTN, MENU_CREATIVE_BTN, MENU_VERSUS_BTN, MENU_WATCH_BTN, MENU_BVB_BTN, MENU_UPLOAD_BTN,
} from './renderer';
import { loadSettings } from './storage';
import { supabase } from './supabase';
import { setupAuthUI } from './authUI';
import { Settings } from './settings';
import { setupSettingsUI } from './settingsUI';
import { GameVariant } from './types';
import { VersusData, BotVsBotData, BotBoard, initVersusData, initBotVsBotData, applyBotMove, requestBotMove, setupPlayerLockHook } from './versus';

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
let selectedAiIndex = -1; // -1 = built-in

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
  if (selectedAiIndex === index) selectedAiIndex = -1;
  else if (selectedAiIndex > index) selectedAiIndex--;
}

// ---- Worker state ----

let versusData: VersusData | null = null;
let aiWorker: Worker | null = null;
let customAiWorker: Worker | null = null;
let customAiName: string | null = null;
let customAiBlobUrl: string | null = null;
let botMoveTimeout: ReturnType<typeof setTimeout> | null = null;

// ---- Bot vs Bot state ----
let botVsBotData: BotVsBotData | null = null;
let bvbWorker1: Worker | null = null;
let bvbWorker2: Worker | null = null;

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
      // Worker reported an error — let the timeout handle fallback, don't stall.
      console.warn('AI worker error:', e.data.error);
      return;
    }
    if (botMoveTimeout) { clearTimeout(botMoveTimeout); botMoveTimeout = null; }
    if (versusData && isValidMove(e.data)) {
      versusData.pendingMove = e.data;
    }
  };
}

function requestMoveWithTimeout(bot: BotBoard, pendingGarbage: number): void {
  const w = getAiWorker();
  requestBotMove(w, bot, pendingGarbage);
  if (botMoveTimeout) clearTimeout(botMoveTimeout);
  botMoveTimeout = setTimeout(() => {
    botMoveTimeout = null;
    if (customAiWorker) {
      customAiWorker.terminate();
      customAiWorker = null;
      customAiName = null;
    }
    const fallback = getAiWorker();
    wireWorker(fallback);
    requestBotMove(fallback, bot, pendingGarbage);
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
  bvbWorker1?.terminate(); bvbWorker1 = null;
  bvbWorker2?.terminate(); bvbWorker2 = null;
  botVsBotData = null;
}

function startBotVsBot(): void {
  stopBvbWorkers(); // ensure any prior workers are cleaned up before creating new ones
  botVsBotData = initBotVsBotData();

  // Bot 1: use selected custom AI if any, otherwise a fresh built-in worker
  setupAiForGame();
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
    if (botVsBotData && isValidMove(e.data)) botVsBotData.pendingMove1 = e.data;
  };
  bvbWorker2.onmessage = (e) => {
    if (botVsBotData && isValidMove(e.data)) botVsBotData.pendingMove2 = e.data;
  };

  requestBotMove(bvbWorker1, botVsBotData.bot1, 0);
  requestBotMove(bvbWorker2, botVsBotData.bot2, 0);
}

// ---- AI manager overlay ----

function renderAiList(): void {
  const list = document.getElementById('ai-list')!;
  list.innerHTML = '';

  // Built-in row
  list.appendChild(makeAiRow(-1, 'Built-in (Beam Search)', 'DEFAULT', false));

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
  reader.readAsText(file);
  (e.target as HTMLInputElement).value = '';
});

document.getElementById('ai-done-btn')!.addEventListener('click', closeAiManager);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('ai-manager')!.style.display !== 'none') {
    closeAiManager();
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

  if (hit(MENU_UPLOAD_BTN)) {
    openAiManager();
    return;
  }

  if (hit(MENU_BVB_BTN)) {
    restartState('botvsbot');
    startBotVsBot();
    return;
  }

  let variant: GameVariant | null = null;
  if (hit(MENU_SPRINT_BTN))   variant = 'sprint';
  if (hit(MENU_CREATIVE_BTN)) variant = 'creative';
  if (hit(MENU_VERSUS_BTN))   variant = 'versus';
  if (hit(MENU_WATCH_BTN))    variant = 'watch';
  if (!variant) return;

  restartState(variant);

  if (variant === 'versus' || variant === 'watch') {
    versusData = initVersusData(variant === 'versus' ? state : undefined);
    if (variant === 'versus') setupPlayerLockHook(state, versusData);
    else setLockHook(null); // no garbage from player in watch mode
    setupAiForGame();
    wireWorker(getAiWorker());
    requestMoveWithTimeout(versusData.bot, 0);
  } else {
    versusData = null;
    setLockHook(null);
  }
});

let settings: Settings;
let prevTimestamp = 0;

function startGame(userId: string): void {
  loadSettings(userId).then((loaded) => {
    settings = loaded;

    setupSettingsUI(
      state,
      () => settings,
      (updated) => { settings = updated; },
      userId,
    );

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

    // Watch / bot-vs-bot: handle input directly (no player game to advance)
    if ((state.variant === 'watch' || state.variant === 'botvsbot') && state.mode !== 'menu') {
      if (wasJustPressed(input, settings.keybindings.pause) || wasJustPressed(input, 'Escape')) {
        state.mode = 'menu';
      } else if (wasJustPressed(input, settings.keybindings.rewind)) {
        if (state.variant === 'watch' && versusData) {
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

      processFrame(state, input, timestamp, settings);

      if (versusData !== null && state.variant === 'versus') {
        // Restart detected: reinit bot and re-register hook
        if (wasLastFrameTime !== 0 && state.lastFrameTime === 0) {
          versusData = initVersusData(state);
          setupPlayerLockHook(state, versusData);
          setupAiForGame();
          wireWorker(getAiWorker());
          requestMoveWithTimeout(versusData.bot, 0);
        }

        // Determine winner
        if (versusData.winner === null) {
          if (versusData.bot.dead && state.mode === 'playing') {
            versusData.winner = 'player';
            state.mode = 'gameover';
          } else if (state.mode === 'gameover') {
            versusData.winner = 'bot';
            versusData.bot.dead = true;
          }
        }
      }
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
          applyBotMove(move, versusData.bot, versusData.botCombat, versusData.playerCombat);
          if (!versusData.bot.dead) {
            requestMoveWithTimeout(versusData.bot, versusData.botCombat.pendingGarbage);
          }
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
        applyBotMove(m, botVsBotData.bot1, botVsBotData.bot1Combat, botVsBotData.bot2Combat);
        if (!botVsBotData.bot1.dead && bvbWorker1)
          requestBotMove(bvbWorker1, botVsBotData.bot1, botVsBotData.bot1Combat.pendingGarbage);
      }

      while (botVsBotData.bot2ThinkAccumMs >= msPerPiece && !botVsBotData.bot2.dead) {
        if (!botVsBotData.pendingMove2) break;
        botVsBotData.bot2ThinkAccumMs -= msPerPiece;
        const m = botVsBotData.pendingMove2; botVsBotData.pendingMove2 = null;
        applyBotMove(m, botVsBotData.bot2, botVsBotData.bot2Combat, botVsBotData.bot1Combat);
        if (!botVsBotData.bot2.dead && bvbWorker2)
          requestBotMove(bvbWorker2, botVsBotData.bot2, botVsBotData.bot2Combat.pendingGarbage);
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

    draw(ctx, state, versusData, customAiName, botVsBotData);
    flushInput(input);
      state.rafHandle = requestAnimationFrame(gameLoop);
    }

    state.rafHandle = requestAnimationFrame(gameLoop);
  });
}

setupAuthUI(startGame);
