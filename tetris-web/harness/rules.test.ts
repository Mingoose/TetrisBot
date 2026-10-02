// Checks rules.ts against TETR.IO behaviour.
//
//   npm run test:rules
//
// 1. Attack: resolveClear is compared with Triangle.js's garbageCalcV2 (MIT,
//    github.com/halp1/triangle, src/engine/utils/damageCalc) on random clears,
//    plus hand-picked cases for surge and all clears, which live in its lock code.
// 2. Spins: hand-built boards checked with both the game's detectSpin and the
//    bot's placement search.

import type { CellValue, PieceType } from '../src/types';
import { resolveClear, classifySpin, SPIN_NONE, SPIN_MINI, SPIN_FULL, SpinKind, RULES } from '../src/rules';
import { detectSpin, lockPiece, clearLines } from '../src/board';
import { placementSpin } from '../src/ai';
import { handleLock, receiveGarbage, CombatState } from '../src/versus';
import { emptyBoard } from '../src/board';

let failures = 0;
let passes = 0;
function check(name: string, got: unknown, want: unknown): void {
  if (JSON.stringify(got) === JSON.stringify(want)) { passes++; return; }
  failures++;
  console.log(`FAIL ${name}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

// ---- Reference: Triangle.js garbageCalcV2, trimmed to the options TETR.IO multiplayer uses ----
// (multiplier combo table, B2B charging so no chaining, no target bonus)
const garbageData = {
  single: 0, double: 1, triple: 2, quad: 4, penta: 5,
  tspinMini: 0, tspin: 0, tspinMiniSingle: 0, tspinSingle: 2, tspinMiniDouble: 1,
  tspinMiniTriple: 2, tspinDouble: 4, tspinTriple: 6, tspinQuad: 10, tspinPenta: 12,
  backtobackBonus: 1, comboMinifier: 1, comboMinifierLog: 1.25, comboBonus: 0.25,
};
function triangleGarbage(lines: number, spin: 'mini' | 'normal' | null, b2b: number, combo: number): number {
  let garbage = 0;
  switch (lines) {
    case 0: garbage = spin === 'mini' ? garbageData.tspinMini : spin === 'normal' ? garbageData.tspin : 0; break;
    case 1: garbage = spin === 'mini' ? garbageData.tspinMiniSingle : spin === 'normal' ? garbageData.tspinSingle : garbageData.single; break;
    case 2: garbage = spin === 'mini' ? garbageData.tspinMiniDouble : spin === 'normal' ? garbageData.tspinDouble : garbageData.double; break;
    case 3: garbage = spin === 'mini' ? garbageData.tspinMiniTriple : spin === 'normal' ? garbageData.tspinTriple : garbageData.triple; break;
    case 4: garbage = spin ? garbageData.tspinQuad : garbageData.quad; break;
    case 5: garbage = spin ? garbageData.tspinPenta : garbageData.penta; break;
    default: { const t = lines - 5; garbage = spin ? garbageData.tspinPenta + 2 * t : garbageData.penta + t; }
  }
  if (lines > 0 && b2b > 0) garbage += garbageData.backtobackBonus;
  if (combo > 0) {
    garbage *= 1 + garbageData.comboBonus * combo;
    if (combo > 1) garbage = Math.max(Math.log1p(garbageData.comboMinifier * combo * garbageData.comboMinifierLog), garbage);
  }
  return garbage;
}

// ---- 1a. Random clears vs Triangle.js ----
{
  let seed = 12345;
  const rand = (n: number) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  for (let i = 0; i < 20000; i++) {
    const lines = 1 + rand(4);
    const spin = rand(3) as SpinKind;
    const combo = rand(12) - 1;
    const b2b = rand(4) - 1; // stay below surge so only the attack formula is compared
    const res = resolveClear(combo, b2b, lines, spin, false);
    // Triangle's lock code bumps combo/b2b before calling garbageCalcV2.
    const qualifies = spin !== SPIN_NONE || lines >= 4;
    const tB2b = qualifies ? b2b + 1 : -1;
    const tCombo = combo + 1;
    const want = Math.floor(triangleGarbage(lines, spin === SPIN_FULL ? 'normal' : spin === SPIN_MINI ? 'mini' : null,
      Math.max(tB2b, 0), Math.max(tCombo, 0)));
    if (res.attack !== want || res.b2b !== tB2b || res.combo !== tCombo) {
      check(`random #${i} lines=${lines} spin=${spin} combo=${combo} b2b=${b2b}`,
        [res.attack, res.b2b, res.combo], [want, tB2b, tCombo]);
    } else passes++;
  }
}

// ---- 1b. Hand-picked attack cases ----
const atk = (combo: number, b2b: number, lines: number, spin: SpinKind, pc = false) =>
  resolveClear(combo, b2b, lines, spin, pc);
check('single', atk(-1, -1, 1, SPIN_NONE).attack, 0);
check('double', atk(-1, -1, 2, SPIN_NONE).attack, 1);
check('quad starts B2B', [atk(-1, -1, 4, SPIN_NONE).attack, atk(-1, -1, 4, SPIN_NONE).b2b], [4, 0]);
check('B2B quad', atk(-1, 0, 4, SPIN_NONE).attack, 5);
check('TSD', atk(-1, -1, 2, SPIN_FULL).attack, 4);
check('B2B TSD', atk(-1, 0, 2, SPIN_FULL).attack, 5);
check('T-spin mini single sends 0 but keeps B2B', [atk(-1, 2, 1, SPIN_MINI).attack, atk(-1, 2, 1, SPIN_MINI).b2b], [1, 3]);
check('all-spin mini double', atk(-1, -1, 2, SPIN_MINI).attack, 1);
check('no clear keeps B2B, resets combo', [atk(3, 5, 0, SPIN_NONE).b2b, atk(3, 5, 0, SPIN_NONE).combo], [5, -1]);
check('B2B below surge breaks quietly', [atk(-1, 3, 1, SPIN_NONE).attack, atk(-1, 3, 1, SPIN_NONE).surge], [0, 0]);
check('B2B x4 breaks into surge of 4', atk(-1, RULES.surgeAt, 1, SPIN_NONE).surge, 4);
check('B2B x8 breaks into surge of 8', atk(-1, 8, 2, SPIN_NONE).attack, 1 + 8);
check('all clear: +3 attack, +2 B2B, no break', [atk(-1, 5, 2, SPIN_NONE, true).attack, atk(-1, 5, 2, SPIN_NONE, true).b2b], [1 + 1 + 3, 7]);

// ---- 2a. classifySpin ----
check('no rotation, no spin', classifySpin('T', false, 4, 2, true, false), SPIN_NONE);
check('T 3 corners, both front', classifySpin('T', true, 3, 2, false, false), SPIN_FULL);
check('T 3 corners, one front = mini', classifySpin('T', true, 3, 1, false, false), SPIN_MINI);
check('T mini upgraded by TST kick', classifySpin('T', true, 3, 1, false, true), SPIN_FULL);
check('T 2 corners but immobile = mini', classifySpin('T', true, 2, 1, true, false), SPIN_MINI);
check('S immobile = mini', classifySpin('S', true, 0, 0, true, false), SPIN_MINI);
check('S mobile = none', classifySpin('S', true, 0, 0, false, false), SPIN_NONE);

// ---- 2b. Boards ----
function board(rows: string[]): CellValue[][] {
  const pad = Array.from({ length: 20 - rows.length }, () => '..........');
  return [...pad, ...rows].map(r => [...r].map(c => (c === 'X' ? 'X' : 0)) as CellValue[]);
}
function piece(type: PieceType, rotationIndex: number, x: number, y: number) {
  return { type, rotationIndex, x, y };
}

// T-spin double: T pointing down under the overhang at (17, 3).
const tsd = board([
  'XXX.......',
  'XXXX......',
  'XXX...XXXX',
  'XXXX.XXXXX',
]);
const tsdPiece = piece('T', 2, 3, 17);
check('TSD: game detects full spin', detectSpin(tsd, tsdPiece, true, false), SPIN_FULL);
check('TSD: bot search finds full spin', placementSpin(tsd, tsdPiece), SPIN_FULL);
check('TSD: clears 2', clearLines(lockPiece(tsd, tsdPiece)).linesCleared, 2);

// Same T dropped into an open three-corner spot: reachable by dropping, but the
// bot can still finish with a rotation, so whatever spin that gives counts.
const flat = board(['XXX.XXXXXX']);
check('T on flat ground: no spin', detectSpin(flat, piece('T', 2, 2, 18), true, false), SPIN_NONE);

// S-spin: an S standing upright in a cavity it can't slide out of. The block at
// (15, 3) makes SRS skip the early kicks, so a flat S tucked under the
// overhang at (16, 3) rotates down into the cavity on the 4th test.
const sCave = board([
  'XXXX......',
  'XXX.......',
  'XXXX.XXXXX',
  'XXXX..XXXX',
  'XXXXX.XXXX',
]);
const sPiece = piece('S', 1, 3, 17); // R state, cells (17,4) (18,4) (18,5) (19,5)
check('S in cavity: immobile → mini', detectSpin(sCave, sPiece, true, false), SPIN_MINI);
check('S in cavity: not a spin without rotation', detectSpin(sCave, sPiece, false, false), SPIN_NONE);
check('S in cavity: bot search agrees', placementSpin(sCave, sPiece), SPIN_MINI);
check('S-spin triple clears 3', clearLines(lockPiece(sCave, sPiece)).linesCleared, 3);
check('S-spin mini triple sends 2', resolveClear(-1, -1, 3, SPIN_MINI, false).attack, 2);

// ---- 3. Garbage (TETR.IO: lands only on non-clearing locks, capped, after travel) ----
{
  const fresh = (): CombatState => ({ combo: -1, b2b: -1, incoming: [], pendingGarbage: 0 });
  const filledRows = (b: CellValue[][]) => b.filter(r => r.some(c => c !== 0)).length;

  const me = fresh(), them = fresh();
  receiveGarbage(me, 10, 0);
  let r = handleLock(me, them, emptyBoard(), 0, SPIN_NONE, 100);
  check('garbage still travelling does not land', [r.outcome.garbageIn, me.pendingGarbage], [0, 10]);
  r = handleLock(me, them, emptyBoard(), 0, SPIN_NONE, 400);
  check('cap: 8 of 10 rows land', [r.outcome.garbageIn, filledRows(r.board), me.pendingGarbage], [8, 8, 2]);
  check('rows of one attack share a hole column', new Set(r.board.slice(12).map(row => row.indexOf(0))).size, 1);
  r = handleLock(me, them, r.board, 0, SPIN_NONE, 500);
  check('the rest lands on the next lock', [r.outcome.garbageIn, me.pendingGarbage], [2, 0]);

  const a = fresh(), b = fresh();
  const leftover = board(['X.........']); // not a perfect clear
  receiveGarbage(a, 5, 0);
  r = handleLock(a, b, leftover, 2, SPIN_FULL, 1000);
  check('TSD cancels 4 of 5; clearing lock lands nothing', [r.outcome.garbageIn, a.pendingGarbage, b.pendingGarbage], [0, 1, 0]);
  r = handleLock(a, b, leftover, 2, SPIN_FULL, 2000);
  // B2B TSD at combo 1: (4 + 1) × 1.25 = 6.25 → 6; one row cancels, 5 are sent.
  check('B2B combo TSD cancels the last row and sends 5', [a.pendingGarbage, b.pendingGarbage, b.incoming[0].readyAt], [0, 5, 2000 + RULES.garbageSpeedMs]);
}

console.log(`\n${passes} passed, ${failures} failed`);
if (failures) process.exit(1);
