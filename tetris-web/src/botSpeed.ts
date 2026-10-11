// Bot speed, chosen with the opponent (AI picker / bot-vs-bot setup) rather
// than in Settings. Each mode remembers its last choice in this browser.

export interface SpeedPreset { name: string; pps: number }

// For reference, TETR.IO beginners play around 1 piece per second and
// experienced players 2-3.
export const SPEED_PRESETS: readonly SpeedPreset[] = [
  { name: 'Relaxed', pps: 0.5 },
  { name: 'Casual',  pps: 1 },
  { name: 'Steady',  pps: 1.5 },
  { name: 'Fast',    pps: 2 },
  { name: 'Blitz',   pps: 3 },
];
const DEFAULT_PPS = 1;

export type SpeedSlot = 'versus' | 'watch' | 'bvb1' | 'bvb2';
const storageKey = (slot: SpeedSlot) => `tetris.botSpeed.${slot}`;

// The remembered speed for a slot, or the default. Storage can be unavailable
// (private windows, blocked site data), so every access is guarded.
export function loadSpeed(slot: SpeedSlot): number {
  try {
    const pps = Number(localStorage.getItem(storageKey(slot)));
    if (SPEED_PRESETS.some(p => p.pps === pps)) return pps;
  } catch { /* fall through to the default */ }
  return DEFAULT_PPS;
}

export function saveSpeed(slot: SpeedSlot, pps: number): void {
  try { localStorage.setItem(storageKey(slot), String(pps)); } catch { /* not remembered */ }
}

export function presetName(pps: number): string {
  return SPEED_PRESETS.find(p => p.pps === pps)?.name ?? 'Custom';
}

/** e.g. "Casual · 1 PPS" */
export function speedText(pps: number): string {
  return `${presetName(pps)} · ${pps} PPS`;
}

// How long a time-budgeted bot (the value net) may think per move at this
// speed: 85% of the time per piece, leaving room for the rest of the frame,
// and never more than its own budget. Faster speeds mean shallower searches.
export function searchBudgetMs(maxMs: number, pps: number): number {
  return Math.min(maxMs, Math.round(0.85 * 1000 / pps));
}

// A row of preset buttons for choosing a bot's speed (AI picker, bot-vs-bot
// setup). Calls onChange with the new speed; the caller saves and re-renders.
export function speedPickerEl(title: string, pps: number, onChange: (pps: number) => void): HTMLElement {
  const root = document.createElement('div');
  root.className = 'speed-picker';

  const head = document.createElement('div');
  head.className = 'speed-head';
  const titleEl = document.createElement('span');
  titleEl.textContent = title;
  const caption = document.createElement('span');
  caption.className = 'speed-caption';
  caption.textContent = `${pps} piece${pps === 1 ? '' : 's'} per second`;
  head.append(titleEl, caption);

  const options = document.createElement('div');
  options.className = 'speed-options';
  for (const p of SPEED_PRESETS) {
    const btn = document.createElement('button');
    btn.className = 'speed-btn' + (p.pps === pps ? ' selected' : '');
    btn.innerHTML = `<span class="speed-name">${p.name}</span><span class="speed-pps">${p.pps}</span>`;
    btn.title = `${p.pps} pieces per second`;
    btn.addEventListener('click', () => onChange(p.pps));
    options.appendChild(btn);
  }

  root.append(head, options);
  return root;
}
