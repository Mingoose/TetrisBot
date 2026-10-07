// Every placement the bot could have made at each recorded decision, for
// training the value net to rank the teacher's choice first.
//
// For each record of a self-play run this lists the candidates valueBot.ts
// would score (same move generator, spins, attack and context inputs) and marks
// the teacher's. Writes two files into the run directory:
//   cand_pos.bin  one entry per decision: game_id u4, ply u2, first candidate u4,
//                 candidate count u2, teacher's index i2 (-1 if not found), and
//                 for runs recorded by duel.ts --record, the candidates the
//                 value-net search expanded (indices, -1 = none) and their deep scores
//   cand.bin      one entry per candidate (layout in CAND_FIELDS)
// Python reads them with training/rank_data.py.
//
// Usage (from tetris-web/):
//   npm run candidates -- ../training/data/runs/teacher-v2

import { appendFileSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { BOARD_COLS, BOARD_ROWS } from '../src/board';
import { RULES } from '../src/rules';
import { listCandidates } from '../src/valueBot';
import { PIECE_ORDER, NO_PIECE, RECORD_FIELDS, RECORD_SIZE, SEARCH_K, SEARCH_SIZE } from './record';

export const POS_FIELDS: [string, string, number][] = [
  ['game_id', '<u4', 1], ['ply', '<u2', 1], ['first', '<u4', 1], ['count', '<u2', 1], ['teacher', '<i2', 1],
  ['soft_idx', '<i2', SEARCH_K], ['soft_score', '<f4', SEARCH_K],
];
export const CAND_FIELDS: [string, string, number][] = [
  ['board', '<u2', 20],   // after line clears, before garbage; row 0 = top
  ['next', 'u1', 1],
  ['queue4', 'u1', 4],
  ['hold', 'u1', 1],
  ['combo', 'i1', 1],     // after the move
  ['b2b', 'i1', 1],       // after the move
  ['landing', 'u1', 1],   // garbage rows that land on it
  ['attack', 'u1', 1],
  ['lines', 'u1', 1],
  ['dies', 'u1', 1],
];
const POS_SIZE = 14 + SEARCH_K * 6;
const CAND_SIZE = 40 + 12;

// ---- Reading records ----

type Rec = Record<string, number | number[]>;

function readRecords(runDir: string): Rec[] {
  const offsets: [string, string, number, number][] = [];
  let o = 0;
  for (const [name, t, n] of RECORD_FIELDS) {
    offsets.push([name, t, n, o]);
    o += ({ '<u4': 4, '<u2': 2, 'u1': 1, 'i1': 1 } as Record<string, number>)[t] * n;
  }
  const recs: Rec[] = [];
  for (const f of readdirSync(runDir).filter(f => /^w\d+\.bin$/.test(f)).sort()) {
    const buf = readFileSync(join(runDir, f));
    for (let base = 0; base + RECORD_SIZE <= buf.length; base += RECORD_SIZE) {
      const r: Rec = {};
      for (const [name, t, n, off] of offsets) {
        const read = (i: number): number => {
          const at = base + off + i * (t === '<u4' ? 4 : t === '<u2' ? 2 : 1);
          return t === '<u4' ? buf.readUInt32LE(at) : t === '<u2' ? buf.readUInt16LE(at)
            : t === 'i1' ? buf.readInt8(at) : buf.readUInt8(at);
        };
        r[name] = n === 1 ? read(0) : Array.from({ length: n }, (_, i) => read(i));
      }
      recs.push(r);
    }
  }
  return recs.sort((a, b) => (a.game_id as number) - (b.game_id as number) || (a.ply as number) - (b.ply as number));
}

interface SideEntry {
  ready: number;
  expanded: { rot: number; x: number; y: number; hold: boolean; score: number }[];
}

// Side-file entries per (game, ply) from a duel.ts --record run, or null if the run has none.
function readSide(runDir: string): Map<number, SideEntry> | null {
  const files = readdirSync(runDir).filter(f => /^s\d+\.bin$/.test(f));
  if (!files.length) return null;
  const map = new Map<number, SideEntry>();
  for (const f of files) {
    const buf = readFileSync(join(runDir, f));
    for (let o = 0; o + SEARCH_SIZE <= buf.length; o += SEARCH_SIZE) {
      const n = buf.readUInt8(o + 7);
      const at = (field: number, i: number) => o + 8 + field * SEARCH_K + i;
      const expanded = Array.from({ length: n }, (_, i) => ({
        rot: buf.readUInt8(at(0, i)), x: buf.readInt8(at(1, i)), y: buf.readInt8(at(2, i)),
        hold: buf.readUInt8(at(3, i)) === 1, score: buf.readFloatLE(o + 8 + 4 * SEARCH_K + 4 * i),
      }));
      map.set(buf.readUInt32LE(o) * 65536 + buf.readUInt16LE(o + 4), { ready: buf.readUInt8(o + 6), expanded });
    }
  }
  return map;
}

// ---- Main ----

const runDir = process.argv[2];
if (!runDir) throw new Error('usage: candidates RUN_DIR');
const recs = readRecords(runDir);
const side = readSide(runDir);
let softMissing = 0;
const piece = (i: number) => PIECE_ORDER[i];

const posBuf = Buffer.alloc(recs.length * POS_SIZE);
// Candidates are appended to cand.bin in chunks; the file can be several GB.
const candFile = join(runDir, 'cand.bin');
writeFileSync(candFile, '');
const candBuf = Buffer.alloc(1 << 26);
let used = 0;
const flush = () => { appendFileSync(candFile, candBuf.subarray(0, used)); used = 0; };
let nPos = 0, nCand = 0, notFound = 0, landingMismatch = 0, landingChecked = 0;
const t0 = performance.now();

for (let i = 0; i < recs.length; i++) {
  const r = recs[i];
  // The teacher's afterstate comes from the next record, so the last decision of a game has no sample.
  if (i + 1 >= recs.length || recs[i + 1].game_id !== r.game_id) continue;

  // Garbage ready to land: what was still queued after the previous move. In
  // self-play an attack arrives just before a move and needs 20 frames to travel,
  // so everything older than this ply has arrived (selfplay.ts, pps 2).
  // Runs recorded by duel.ts --record store it in their side file instead.
  let ready = 0;
  const p = i > 0 && recs[i - 1].game_id === r.game_id ? recs[i - 1] : null;
  const entry = side?.get((r.game_id as number) * 65536 + (r.ply as number));
  if (side) {
    if (!entry) throw new Error(`no side-file entry for game ${r.game_id} ply ${r.ply}`);
    ready = entry.ready;
  } else if (p) {
    const cancelled = (p.lines as number) > 0 ? Math.min(p.attack as number, p.incoming as number) : 0;
    ready = (p.incoming as number) - (p.garbage_in as number) - cancelled;
  }

  const board = Array.from({ length: BOARD_ROWS }, (_, row) =>
    Array.from({ length: BOARD_COLS }, (_, c) => (((r.board as number[])[row] >> c) & 1 ? 'X' : 0) as 0 | 'X'));
  const hold = r.hold as number;
  const cands = listCandidates({
    board, active: piece(r.active as number), hold: hold === NO_PIECE ? null : piece(hold),
    queue: (r.queue as number[]).map(piece), bagMask: r.bag_mask as number,
    combo: r.combo as number, b2b: r.b2b as number,
    landingCols: new Array(Math.max(0, ready)).fill(0), // hole columns aren't recorded
  });

  const [rot, x, y, useHold] = [r.move_rot, r.move_x, r.move_y, r.move_hold === 1];
  const teacher = cands.findIndex(c => c.move.rotationIndex === rot && c.move.x === x && c.move.y === y
    && c.move.useHold === useHold);
  if (teacher < 0) notFound++;
  else if (cands[teacher].lines === 0) {
    landingChecked++;
    if (cands[teacher].landing !== Math.min(r.garbage_in as number, RULES.garbageCap)) landingMismatch++;
  }

  if (used + cands.length * CAND_SIZE > candBuf.length) flush();
  let o = nPos * POS_SIZE;
  o = posBuf.writeUInt32LE(r.game_id as number, o);
  o = posBuf.writeUInt16LE(r.ply as number, o);
  o = posBuf.writeUInt32LE(nCand, o);
  o = posBuf.writeUInt16LE(cands.length, o);
  o = posBuf.writeInt16LE(teacher, o);
  const soft = (entry?.expanded ?? []).map(e => {
    const k = cands.findIndex(c => c.move.rotationIndex === e.rot && c.move.x === e.x && c.move.y === e.y
      && c.move.useHold === e.hold);
    if (k < 0) softMissing++;
    return { k, score: e.score };
  });
  for (let k = 0; k < SEARCH_K; k++) o = posBuf.writeInt16LE(soft[k]?.k ?? -1, o);
  for (let k = 0; k < SEARCH_K; k++) o = posBuf.writeFloatLE(soft[k]?.score ?? 0, o);
  nPos++;

  for (const c of cands) {
    let q = used;
    for (let row = 0; row < BOARD_ROWS; row++) q = candBuf.writeUInt16LE(c.after[row], q);
    q = candBuf.writeUInt8(c.next, q);
    for (let k = 0; k < 4; k++) q = candBuf.writeUInt8(c.queue4[k], q);
    q = candBuf.writeUInt8(c.hold, q);
    q = candBuf.writeInt8(Math.max(-128, Math.min(127, c.combo)), q);
    q = candBuf.writeInt8(Math.max(-128, Math.min(127, c.b2b)), q);
    q = candBuf.writeUInt8(c.landing, q);
    q = candBuf.writeUInt8(Math.min(255, c.attack), q);
    q = candBuf.writeUInt8(c.lines, q);
    candBuf.writeUInt8(c.dies ? 1 : 0, q);
    used += CAND_SIZE;
    nCand++;
  }
  if (nPos % 50000 === 0) console.log(`${nPos} decisions, ${nCand} candidates (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
}

flush();
writeFileSync(join(runDir, 'cand_pos.bin'), posBuf.subarray(0, nPos * POS_SIZE));
writeFileSync(join(runDir, 'candidates.json'), JSON.stringify({
  pos_fields: POS_FIELDS, pos_size: POS_SIZE, cand_fields: CAND_FIELDS, cand_size: CAND_SIZE,
  decisions: nPos, candidates: nCand, teacher_not_found: notFound,
}, null, 2));
console.log(`${runDir}: ${nPos} decisions, ${nCand} candidates (${(nCand / nPos).toFixed(1)} each) ` +
  `in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
if (side) console.log(`search-expanded moves not found among candidates: ${softMissing}`);
console.log(`teacher move not found: ${notFound}; ready-garbage reconstruction wrong on ` +
  `${landingMismatch}/${landingChecked} non-clearing teacher moves`);
if (landingMismatch || softMissing) process.exitCode = 1;
