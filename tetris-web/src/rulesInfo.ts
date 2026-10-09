// The "?" panel in the AI picker: versus rules in plain language. Every number
// is computed from rules.ts, so the panel always matches the game.

import { RULES, SPIN_NONE, SPIN_MINI, SPIN_FULL, SpinKind, resolveClear } from './rules';

// Attack of one clear with no combo or B2B running.
function plain(lines: number, spin: SpinKind): number {
  return resolveClear(-1, -1, lines, spin, false).attack;
}

// What the n-th clear of a combo sends (n = 1 is the clear that starts it).
function comboRun(lines: number, count: number): number[] {
  return Array.from({ length: count }, (_, i) => resolveClear(i - 1, -1, lines, SPIN_NONE, false).attack);
}

function row(name: string, ...cells: (string | number)[]): string {
  return `<tr><td>${name}</td>${cells.map(c => `<td class="n">${c}</td>`).join('')}</tr>`;
}

function rulesHtml(): string {
  const names = ['Single', 'Double', 'Triple', 'Quad'];
  const clears = [1, 2, 3, 4].map(n => row(names[n - 1], plain(n, SPIN_NONE))).join('');
  const spins = [1, 2, 3].map(n =>
    row(names[n - 1], plain(n, SPIN_FULL), plain(n, SPIN_MINI))).join('');
  const surgeFrom = RULES.surgeAt;
  const surges = [surgeFrom, surgeFrom + 2, surgeFrom + 6]
    .map(level => `B2B ×${level} → +${resolveClear(-1, level, 1, SPIN_NONE, false).surge}`)
    .join(', ');
  const travelMs = Math.round(RULES.garbageSpeedMs);

  return `
<p>Both players get the same pieces. Clearing lines sends garbage to your opponent;
the first player who can't continue loses. The scoring follows TETR.IO's multiplayer rules.</p>

<h3>LINE CLEARS</h3>
<p>Lines of garbage sent by each clear on its own:</p>
<table>${clears}</table>

<h3>SPINS</h3>
<p>A spin is a piece rotated into a spot it couldn't slide into. The last thing you do
before the piece locks must be a rotation.</p>
<ul>
  <li><b>T-spin:</b> a T with 3 of its 4 corners blocked (walls and floor count). It is a full T-spin
  when both corners in front of the T are blocked (or it got in with the special "TST" wall kick);
  otherwise it is a <b>mini</b>.</li>
  <li><b>Other pieces (All-Mini+):</b> any piece that ends up unable to move left, right, up or down
  scores as a mini spin.</li>
</ul>
<table><tr><th></th><th>T-spin</th><th>Mini</th></tr>${spins}</table>

<h3>BACK-TO-BACK (B2B)</h3>
<p>Quads and spins (minis included) are <b>difficult clears</b>. Each difficult clear that follows
another, with no normal clear in between, raises your B2B chain, shown as <b>B2B ×N</b>, and adds <b>+${RULES.b2bBonus}</b> to its attack.
A normal single, double or triple ends the chain; pieces that clear nothing don't.</p>

<h3>SURGE</h3>
<p>Once the chain reaches <b>B2B ×${surgeFrom}</b> it is charged. When a charged chain ends,
it releases a surge on top of that clear's attack: <b>as many lines as the chain was long</b>
(${surges}). Keep the chain going as long as you can, then cash it in.</p>

<h3>COMBOS</h3>
<p>Clearing with several pieces in a row is a combo. Each clear in a combo multiplies its attack
by 1 + ${RULES.comboBonus} for each earlier clear in the combo, and even singles start to send something
once the combo is long. Attack is rounded down. One piece that clears nothing ends the combo.
Lines sent by the 1st, 2nd, 3rd … clear of a combo:</p>
<table>
  <tr><th></th>${comboRun(1, 8).map((_, i) => `<th>${i + 1}</th>`).join('')}</tr>
  ${row('Singles', ...comboRun(1, 8))}
  ${row('Doubles', ...comboRun(2, 8))}
</table>

<h3>ALL CLEAR</h3>
<p>Emptying the whole board sends <b>+${RULES.allClearGarbage}</b> lines and raises B2B by
${RULES.allClearB2b} instead of 1.</p>

<h3>INCOMING GARBAGE</h3>
<ul>
  <li>Attacks against you wait in the red bar beside your board.</li>
  <li><b>Clearing lines cancels it first:</b> your attack removes waiting garbage, oldest first,
  and only what is left over is sent.</li>
  <li>Garbage only rises when you place a piece that <b>clears nothing</b>: at most
  <b>${RULES.garbageCap}</b> lines per piece, and only attacks that have been waiting at least
  ${travelMs} ms.</li>
  <li>All lines from one attack share the same hole; the hole moves between attacks.</li>
</ul>

<h3>TOPPING OUT</h3>
<p>There is hidden room above the board, and new pieces appear just above it. You lose when:</p>
<ul>
  <li>the next piece has no room to appear (if you just cleared lines, it may appear higher
  instead), or</li>
  <li>a piece locks completely above the board without clearing a line.</li>
</ul>
<p>Garbage alone never ends the game; it pushes your stack up and the next piece decides.
A red border means you are close to the top, and a faint red shape shows where the next piece
will be blocked.</p>`;
}

export function setupRulesPanel(): void {
  const panel = document.getElementById('rules-panel')!;
  document.getElementById('rules-body')!.innerHTML = rulesHtml();
  document.getElementById('rules-open-btn')!.addEventListener('click', () => {
    panel.style.display = 'flex';
  });
  document.getElementById('rules-close-btn')!.addEventListener('click', () => {
    panel.style.display = 'none';
  });
}

export function isRulesPanelOpen(): boolean {
  return document.getElementById('rules-panel')!.style.display !== 'none';
}

export function closeRulesPanel(): void {
  document.getElementById('rules-panel')!.style.display = 'none';
}
