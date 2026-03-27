import { Settings, KeyBindings, ACTION_LABELS, keyLabel, DEFAULT_KEYBINDINGS, DEFAULT_DAS, DEFAULT_ARR, DEFAULT_SONIC_DROP, DEFAULT_BOT_PPS } from './settings';
import { saveSettings } from './storage';
import { GameState } from './types';

type ActionKey = keyof KeyBindings;

let listeningFor: ActionKey | null = null;
let pendingBindings: KeyBindings = { ...DEFAULT_KEYBINDINGS };
let pendingDas = DEFAULT_DAS;
let pendingArr = DEFAULT_ARR;
let pendingSonicDrop = DEFAULT_SONIC_DROP;
let pendingBotPps = DEFAULT_BOT_PPS;
let modalEl: HTMLElement | null = null;
let prevMode: GameState['mode'] = 'playing';

export function setupSettingsUI(
  gameState: GameState,
  getSettings: () => Settings,
  onSettingsChange: (s: Settings) => void,
  userId: string,
): void {
  // Build modal DOM
  modalEl = createModal(gameState, getSettings, onSettingsChange, userId);
  document.body.appendChild(modalEl);

  // Settings button
  const btn = document.createElement('button');
  btn.textContent = '⚙ Settings';
  btn.id = 'settings-btn';
  Object.assign(btn.style, {
    position: 'fixed',
    bottom: '18px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: '#1e1e3a',
    color: '#aaaacc',
    border: '1px solid #3a3a6a',
    borderRadius: '6px',
    padding: '7px 18px',
    fontSize: '13px',
    fontFamily: 'monospace',
    cursor: 'pointer',
    zIndex: '10',
  });
  btn.addEventListener('mouseover', () => { btn.style.background = '#2a2a4a'; });
  btn.addEventListener('mouseout',  () => { btn.style.background = '#1e1e3a'; });
  btn.addEventListener('click', () => openModal(gameState, getSettings));
  document.body.appendChild(btn);
}

function openModal(gameState: GameState, getSettings: () => Settings): void {
  if (!modalEl) return;
  const s = getSettings();
  pendingBindings = { ...s.keybindings };
  pendingDas = s.das;
  pendingArr = s.arr;
  pendingSonicDrop = s.sonicDrop;
  pendingBotPps = s.botPps;
  listeningFor = null;
  prevMode = gameState.mode;
  if (gameState.mode === 'playing') gameState.mode = 'paused';
  refreshRows();
  modalEl.style.display = 'flex';
}

function closeModal(gameState: GameState): void {
  if (!modalEl) return;
  listeningFor = null;
  modalEl.style.display = 'none';
  if (gameState.mode === 'paused' && prevMode !== 'paused') {
    gameState.mode = prevMode;
  }
}

function createModal(
  gameState: GameState,
  getSettings: () => Settings,
  onSettingsChange: (s: Settings) => void,
  userId: string,
): HTMLElement {
  const overlay = document.createElement('div');
  Object.assign(overlay.style, {
    display: 'none',
    position: 'fixed',
    inset: '0',
    background: 'rgba(0,0,0,0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: '100',
    fontFamily: 'monospace',
  });

  const box = document.createElement('div');
  Object.assign(box.style, {
    background: '#12122a',
    border: '1px solid #3a3a6a',
    borderRadius: '10px',
    padding: '28px 32px',
    width: '360px',
    maxHeight: '85vh',
    overflowY: 'auto',
    color: '#cccccc',
  });

  // Title
  const title = document.createElement('h2');
  title.textContent = 'Controls';
  Object.assign(title.style, { margin: '0 0 20px', fontSize: '18px', color: '#aaaaff' });
  box.appendChild(title);

  // Rows container
  const rowsEl = document.createElement('div');
  rowsEl.id = 'kb-rows';
  box.appendChild(rowsEl);

  // DAS / ARR sliders
  const movementTitle = document.createElement('h3');
  movementTitle.textContent = 'Movement';
  Object.assign(movementTitle.style, { margin: '18px 0 10px', fontSize: '14px', color: '#aaaaff' });
  box.appendChild(movementTitle);
  const slidersEl = document.createElement('div');
  slidersEl.id = 'slider-rows';
  box.appendChild(slidersEl);

  function refreshSliders(): void {
    slidersEl.innerHTML = '';
    slidersEl.appendChild(makeSlider('DAS — delay before auto-repeat', 'ms', 0, 300, 1, () => pendingDas, (v) => { pendingDas = v; }));
    slidersEl.appendChild(makeSlider('ARR — speed during auto-repeat', 'ms', 0, 100, 1, () => pendingArr, (v) => { pendingArr = v; }));
    slidersEl.appendChild(makeToggle('Sonic drop — soft drop snaps to bottom', () => pendingSonicDrop, (v) => { pendingSonicDrop = v; }));

    const versusTitle = document.createElement('h3');
    versusTitle.textContent = 'Versus';
    Object.assign(versusTitle.style, { margin: '18px 0 10px', fontSize: '14px', color: '#aaaaff' });
    slidersEl.appendChild(versusTitle);
    slidersEl.appendChild(makeSlider('Bot speed — pieces per second', 'pps', 0.5, 5, 0.1, () => pendingBotPps, (v) => { pendingBotPps = v; }));
  }
  refreshSliders();

  // Separator
  const sep = document.createElement('div');
  Object.assign(sep.style, { borderTop: '1px solid #2a2a4a', margin: '20px 0 16px' });
  box.appendChild(sep);

  // Buttons row
  const btns = document.createElement('div');
  Object.assign(btns.style, { display: 'flex', gap: '10px', justifyContent: 'flex-end' });

  const resetBtn = makeButton('Reset defaults', '#1e1e3a', '#888899');
  resetBtn.addEventListener('click', () => {
    pendingBindings = { ...DEFAULT_KEYBINDINGS };
    pendingDas = DEFAULT_DAS;
    pendingArr = DEFAULT_ARR;
    pendingSonicDrop = DEFAULT_SONIC_DROP;
    pendingBotPps = DEFAULT_BOT_PPS;
    listeningFor = null;
    refreshRows();
    refreshSliders();
  });

  const cancelBtn = makeButton('Cancel', '#1e1e3a', '#888899');
  cancelBtn.addEventListener('click', () => closeModal(gameState));

  const saveBtn = makeButton('Save', '#2a2a6a', '#aaaaff');
  saveBtn.addEventListener('click', async () => {
    const newSettings: Settings = { keybindings: { ...pendingBindings }, das: pendingDas, arr: pendingArr, sonicDrop: pendingSonicDrop, botPps: pendingBotPps };
    onSettingsChange(newSettings);
    await saveSettings(userId, newSettings);
    closeModal(gameState);
  });

  btns.append(resetBtn, cancelBtn, saveBtn);
  box.appendChild(btns);
  overlay.appendChild(box);

  // Global keydown while modal is open
  overlay.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (!listeningFor) {
      if (e.code === 'Escape') closeModal(gameState);
      return;
    }
    e.preventDefault();
    if (e.code === 'Escape') {
      listeningFor = null;
      refreshRows();
      return;
    }
    pendingBindings[listeningFor] = e.code;
    listeningFor = null;
    refreshRows();
  });

  // Make overlay focusable so it captures keys
  overlay.tabIndex = 0;
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeModal(gameState);
  });

  // Re-focus overlay whenever it opens (so keydown works without clicking inside)
  const origDisplay = overlay.style.display;
  const observer = new MutationObserver(() => {
    if (overlay.style.display !== 'none' && overlay.style.display !== origDisplay) {
      overlay.focus();
    }
  });
  observer.observe(overlay, { attributes: true, attributeFilter: ['style'] });

  function refreshRows(): void {
    const container = overlay.querySelector('#kb-rows')!;
    container.innerHTML = '';
    (Object.keys(ACTION_LABELS) as ActionKey[]).forEach((action) => {
      container.appendChild(makeRow(action, getSettings));
    });
  }

  function makeRow(action: ActionKey, _getSettings: () => Settings): HTMLElement {
    const row = document.createElement('div');
    Object.assign(row.style, {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '6px 0',
      borderBottom: '1px solid #1a1a30',
    });

    const label = document.createElement('span');
    label.textContent = ACTION_LABELS[action];
    Object.assign(label.style, { fontSize: '13px', color: '#aaaacc' });

    const keyBtn = document.createElement('button');
    const isListening = listeningFor === action;
    keyBtn.textContent = isListening ? '…press a key…' : keyLabel(pendingBindings[action]);
    Object.assign(keyBtn.style, {
      background: isListening ? '#2a2a6a' : '#1a1a36',
      color: isListening ? '#ffffff' : '#ddddff',
      border: `1px solid ${isListening ? '#6666cc' : '#2a2a4a'}`,
      borderRadius: '4px',
      padding: '4px 10px',
      fontSize: '12px',
      fontFamily: 'monospace',
      cursor: 'pointer',
      minWidth: '80px',
    });

    keyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      listeningFor = action;
      refreshRows();
      overlay.focus();
    });

    row.append(label, keyBtn);
    return row;
  }

  // Store refreshRows on overlay so openModal can call it
  (overlay as HTMLElement & { _refresh?: () => void })._refresh = () => { refreshRows(); refreshSliders(); };

  return overlay;
}

function refreshRows(): void {
  if (!modalEl) return;
  const el = modalEl as HTMLElement & { _refresh?: () => void };
  el._refresh?.();
}

function makeSlider(
  label: string,
  unit: string,
  min: number,
  max: number,
  step: number,
  getValue: () => number,
  setValue: (v: number) => void,
): HTMLElement {
  const row = document.createElement('div');
  Object.assign(row.style, { padding: '8px 0', borderBottom: '1px solid #1a1a30' });

  const header = document.createElement('div');
  Object.assign(header.style, { display: 'flex', justifyContent: 'space-between', marginBottom: '6px' });

  const labelEl = document.createElement('span');
  labelEl.textContent = label;
  Object.assign(labelEl.style, { fontSize: '12px', color: '#aaaacc' });

  const valueEl = document.createElement('span');
  const fmt = (v: number) => (step < 1 ? v.toFixed(1) : String(v));
  valueEl.textContent = `${fmt(getValue())} ${unit}`;
  Object.assign(valueEl.style, { fontSize: '12px', color: '#ddddff', fontFamily: 'monospace' });

  header.append(labelEl, valueEl);

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  slider.value = String(getValue());
  Object.assign(slider.style, { width: '100%', accentColor: '#6666cc', cursor: 'pointer' });

  slider.addEventListener('input', () => {
    const v = Number(slider.value);
    setValue(v);
    valueEl.textContent = `${fmt(v)} ${unit}`;
  });

  row.append(header, slider);
  return row;
}

function makeToggle(
  label: string,
  getValue: () => boolean,
  setValue: (v: boolean) => void,
): HTMLElement {
  const row = document.createElement('div');
  Object.assign(row.style, {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 0',
    borderBottom: '1px solid #1a1a30',
  });

  const labelEl = document.createElement('span');
  labelEl.textContent = label;
  Object.assign(labelEl.style, { fontSize: '12px', color: '#aaaacc' });

  const toggle = document.createElement('button');
  const update = () => {
    const on = getValue();
    toggle.textContent = on ? 'ON' : 'OFF';
    toggle.style.background = on ? '#2a4a2a' : '#1a1a36';
    toggle.style.color = on ? '#66ff66' : '#888899';
    toggle.style.borderColor = on ? '#44aa44' : '#2a2a4a';
  };
  Object.assign(toggle.style, {
    border: '1px solid',
    borderRadius: '4px',
    padding: '4px 12px',
    fontSize: '12px',
    fontFamily: 'monospace',
    cursor: 'pointer',
    minWidth: '50px',
  });
  update();

  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    setValue(!getValue());
    update();
  });

  row.append(labelEl, toggle);
  return row;
}

function makeButton(text: string, bg: string, color: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = text;
  Object.assign(b.style, {
    background: bg,
    color,
    border: `1px solid ${color}44`,
    borderRadius: '5px',
    padding: '6px 14px',
    fontSize: '12px',
    fontFamily: 'monospace',
    cursor: 'pointer',
  });
  return b;
}
