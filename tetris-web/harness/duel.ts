// Headless 1v1: two bots on the same piece sequence, exchanging garbage under
// the versus.ts rules, as in the app's bot-vs-bot mode.
//
// Both bots place pieces at the same rate on a virtual clock (--pps), so the
// result measures decisions, not thinking speed. Sides swap every game so
// neither bot always moves first; games are reproducible from (--seed, game id).
//
// Usage (from tetris-web/):
//   npm run duel -- --a net:../training/models/value_v2_c16.json --b hard --games 200
//
// Bot specs: hard (W32 D5) | hard:W:D | net:FILE.json (1-ply value net)
//            | net:FILE.json@K[,K2...] (value net; top K re-searched with the next
//              piece, top K2 of each of those with the piece after; see findBestMoveDeep)
// Options: --games N --workers N --pieces N (cap per bot; draw if both reach it)
//          --pps N --seed N --out FILE.jsonl (one line per game)
//          --record DIR  also write both sides' moves as self-play records
//                        (record.ts; game id = 2 × duel game + board 0/1) and a
//                        side file per worker with ready garbage and, for net
//                        moves, the search's expanded moves and scores

import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { initBotVsBotData, applyBotMove, botBagMask, bvbLookahead, BotBoard, CombatState } from '../src/versus';
import { findBestMoveHard } from '../src/ai';
import { ValueNet, ValueNetFile } from '../src/valueNet';
import type { PieceType } from '../src/types';
import { searchDeep, searchStateFor, SearchResult } from '../src/valueBot';
import {
  NO_PIECE, PIECE_ORDER, RECORD_FIELDS, RECORD_SIZE, SEARCH_FIELDS, SEARCH_SIZE, TERMINAL_DIED, TERMINAL_NONE,
  TERMINAL_TRUNCATED, PositionRecord, writeRecord, writeSearchRecord,
} from './record';

interface Config {
  a: string; b: string; games: number; workers: number; pieces: number;
  pps: number; seed: number; out: string; record: string;
}

const DEFAULTS: Config = {
  a: '', b: 'hard', games: 100, workers: Math.max(1, availableParallelism() - 2),
  pieces: 1000, pps: 2, seed: 1, out: '', record: '',
};

interface SideStats { pieces: number; attack: number; lines: number; thinkMs: number; dead: boolean }
interface GameResult { type: 'done'; gameId: number; aSide: 0 | 1; winner: 'a' | 'b' | 'draw'; a: SideStats; b: SideStats }

// ---- Seeded RNG (as in selfplay.ts) ----

function mulberry32(a: number): () => number {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gameSeed(seed: number, gameId: number): number {
  return (Math.imul(seed, 0x9e3779b1) ^ Math.imul(gameId + 1, 0x85ebca6b)) >>> 0;
}

// ---- Bots ----

type Think = (bot: BotBoard, combat: CombatState, now: number) => SearchResult;

function makeBot(spec: string): Think {
  const [kind, ...rest] = spec.split(':');
  if (kind === 'hard') {
    const [w, d] = rest.length ? rest.map(Number) : [32, 5];
    return (bot, combat) => ({
      move: findBestMoveHard(
        { ...bot, bagState: bvbLookahead(bot, 30) }, combat.pendingGarbage, w, d, combat.combo, combat.b2b),
      expanded: [],
    });
  }
  if (kind === 'net') {
    const [path, k] = rest.join(':').split('@');
    const widths = k ? k.split(',').map(Number) : [];
    const net = new ValueNet(JSON.parse(readFileSync(path, 'utf8')) as ValueNetFile);
    return (bot, combat, now) => searchDeep(searchStateFor(bot, combat, botBagMask(bot), now), net, widths);
  }
  throw new Error(`Unknown bot spec: ${spec}`);
}

// ---- One game ----

interface Recorded { records: Buffer; search: Buffer }

function playGame(cfg: Config, gameId: number, bots: [Think, Think], rec?: Recorded[]): GameResult {
  Math.random = mulberry32(gameSeed(cfg.seed, gameId));
  const d = initBotVsBotData();
  const aSide = (gameId % 2) as 0 | 1;            // which board bot A plays this game
  const boards = [d.bot1, d.bot2];
  const combats = [d.bot1Combat, d.bot2Combat];
  const thinks = aSide === 0 ? bots : [bots[1], bots[0]];
  const stats: SideStats[] = [0, 1].map(() => ({ pieces: 0, attack: 0, lines: 0, thinkMs: 0, dead: false }));
  const first = aSide; // bot A moves first at equal times; swaps with the sides

  const interval = 1000 / cfg.pps;
  // Records per board, finished after the game (the last one's terminal flag depends on how it ends).
  const recs: PositionRecord[][] = [[], []];
  const searches: Parameters<typeof writeSearchRecord>[2][][] = [[], []];
  const pieceIdx = (p: PieceType | null) => (p ? PIECE_ORDER.indexOf(p) : NO_PIECE);
  while (!boards[0].dead && !boards[1].dead) {
    // Next to move: the side with fewer pieces placed (ties go to `first`).
    const s = stats[0].pieces === stats[1].pieces ? first : stats[0].pieces < stats[1].pieces ? 0 : 1;
    if (stats[s].pieces >= cfg.pieces) break;
    const now = stats[s].pieces * interval;
    const bot = boards[s], combat = combats[s];
    const before = rec && {
      board: Uint16Array.from(bot.board.map(row => row.reduce<number>((m, c, i) => (c ? m | (1 << i) : m), 0))),
      active: pieceIdx(bot.active.type), hold: pieceIdx(bot.hold),
      queue: bot.nextQueue.slice(0, 5).map(pieceIdx), bagMask: botBagMask(bot),
      combo: combat.combo, b2b: combat.b2b, incoming: combat.pendingGarbage,
      ready: searchStateFor(bot, combat, 0, now).landingCols.length,
    };
    const t0 = performance.now();
    const { move, expanded } = thinks[s](bot, combat, now);
    stats[s].thinkMs += performance.now() - t0;
    applyBotMove(move, bot, combat, combats[1 - s], o => {
      stats[s].attack += o.attack;
      stats[s].lines += o.lines;
      if (!before) return;
      const id = gameId * 2 + s;
      recs[s].push({
        gameId: id, ply: stats[s].pieces, ...before,
        moveRot: move.rotationIndex, moveX: move.x, moveY: move.y, moveHold: move.useHold, moveRandom: false,
        lines: o.lines, attack: o.attack, spin: o.spin, surge: o.surge, perfectClear: o.perfectClear,
        garbageIn: o.garbageIn, terminal: TERMINAL_NONE,
      });
      searches[s].push({ gameId: id, ply: stats[s].pieces, ready: before.ready, expanded });
    }, now);
    stats[s].pieces++;
  }
  if (rec) {
    for (const s of [0, 1]) {
      const last = recs[s][recs[s].length - 1];
      // A loss is a death; a win or the piece cap just ends the game.
      if (last) last.terminal = boards[s].dead ? TERMINAL_DIED : TERMINAL_TRUNCATED;
      const r = Buffer.alloc(recs[s].length * RECORD_SIZE);
      recs[s].reduce((o, x) => writeRecord(r, o, x), 0);
      const q = Buffer.alloc(searches[s].length * SEARCH_SIZE);
      searches[s].reduce((o, x) => writeSearchRecord(q, o, x), 0);
      rec.push({ records: r, search: q });
    }
  }
  stats[0].dead = boards[0].dead;
  stats[1].dead = boards[1].dead;
  const a = stats[aSide], b = stats[1 - aSide];
  const winner = a.dead === b.dead ? 'draw' : a.dead ? 'b' : 'a';
  return { type: 'done', gameId, aSide, winner, a, b };
}

// ---- Worker ----

function runWorker(): void {
  const { cfg, workerId } = workerData as { cfg: Config; workerId: number };
  const bots: [Think, Think] = [makeBot(cfg.a), makeBot(cfg.b)];
  parentPort!.on('message', (msg: { type: 'game'; gameId: number } | { type: 'stop' }) => {
    if (msg.type === 'stop') process.exit(0);
    try {
      const rec: Recorded[] | undefined = cfg.record ? [] : undefined;
      const result = playGame(cfg, msg.gameId, bots, rec);
      for (const r of rec ?? []) {
        appendFileSync(join(cfg.record, `w${workerId}.bin`), r.records);
        appendFileSync(join(cfg.record, `s${workerId}.bin`), r.search);
      }
      parentPort!.postMessage(result);
    } catch (err) {
      parentPort!.postMessage({ type: 'failed', gameId: msg.gameId, error: String((err as Error).stack ?? err) });
    }
  });
  parentPort!.postMessage({ type: 'ready' });
}

// ---- Main ----

function parseArgs(argv: string[]): Config {
  const cfg = { ...DEFAULTS };
  const keys: Record<string, keyof Config> = {
    '--a': 'a', '--b': 'b', '--games': 'games', '--workers': 'workers', '--pieces': 'pieces',
    '--pps': 'pps', '--seed': 'seed', '--out': 'out', '--record': 'record',
  };
  for (let i = 0; i < argv.length; i += 2) {
    const key = keys[argv[i]];
    if (!key || argv[i + 1] === undefined) throw new Error(`Unknown or incomplete option: ${argv[i]}`);
    (cfg as Record<string, unknown>)[key] = ['a', 'b', 'out', 'record'].includes(key) ? argv[i + 1] : Number(argv[i + 1]);
  }
  if (!cfg.a) throw new Error('--a BOT is required');
  return cfg;
}

function summarize(cfg: Config, results: GameResult[]): void {
  const n = results.length;
  const wins = results.filter(r => r.winner === 'a').length;
  const losses = results.filter(r => r.winner === 'b').length;
  const draws = n - wins - losses;
  const score = (wins + draws / 2) / Math.max(1, n);
  const se = Math.sqrt(score * (1 - score) / Math.max(1, n));
  const side = (k: 'a' | 'b') => {
    const pieces = results.reduce((s, r) => s + r[k].pieces, 0);
    const attack = results.reduce((s, r) => s + r[k].attack, 0);
    const ms = results.reduce((s, r) => s + r[k].thinkMs, 0);
    return `attack/piece ${(attack / pieces).toFixed(3)}, ${(ms / pieces).toFixed(0)} ms/move`;
  };
  console.log(`\nA = ${cfg.a}\nB = ${cfg.b}`);
  console.log(`${n} games: A won ${wins}, lost ${losses}, drew ${draws} → A scores ${(100 * score).toFixed(1)}% ± ${(196 * se).toFixed(1)} (95%)`);
  console.log(`A: ${side('a')}\nB: ${side('b')}`);
  console.log(`mean game length ${(results.reduce((s, r) => s + r.a.pieces, 0) / Math.max(1, n)).toFixed(0)} pieces per bot`);
}

function runMain(): void {
  const cfg = parseArgs(process.argv.slice(2));
  if (cfg.out) writeFileSync(cfg.out, '');
  if (cfg.record) {
    const metaPath = join(cfg.record, 'meta.json');
    if (existsSync(metaPath)) throw new Error(`${cfg.record} already contains a run; pick a new --record`);
    mkdirSync(cfg.record, { recursive: true });
    // Same meta layout as selfplay.ts, so selfplay_data.py and candidates.ts read it unchanged.
    writeFileSync(metaPath, JSON.stringify({
      config: { ...cfg, net: cfg.a.startsWith('net:') ? cfg.a.slice(4) : '', beam: 32, depth: 5, source: 'duel' },
      record_size: RECORD_SIZE, fields: RECORD_FIELDS, piece_order: PIECE_ORDER,
      search_size: SEARCH_SIZE, search_fields: SEARCH_FIELDS, started: new Date().toISOString(),
    }, null, 2));
  }
  console.log(`${cfg.a} vs ${cfg.b}: ${cfg.games} games on ${cfg.workers} workers, ${cfg.pps} pps, seed ${cfg.seed}`);

  const results: GameResult[] = [];
  let nextGame = 0, live = cfg.workers;
  const t0 = performance.now();
  const workerFile = fileURLToPath(import.meta.url);
  for (let id = 0; id < cfg.workers; id++) {
    const w = new Worker(workerFile, { workerData: { cfg, workerId: id } });
    const dispatch = () => {
      if (nextGame < cfg.games) w.postMessage({ type: 'game', gameId: nextGame++ });
      else {
        w.postMessage({ type: 'stop' });
        if (--live === 0) summarize(cfg, results);
      }
    };
    w.on('message', (msg: { type: 'ready' } | GameResult | { type: 'failed'; gameId: number; error: string }) => {
      if (msg.type === 'failed') {
        console.error(`game ${msg.gameId} failed and was skipped:\n${msg.error}`);
      } else if (msg.type === 'done') {
        results.push(msg);
        if (cfg.out) appendFileSync(cfg.out, JSON.stringify(msg) + '\n');
        const wins = results.filter(r => r.winner === 'a').length;
        const losses = results.filter(r => r.winner === 'b').length;
        console.log(`game ${msg.gameId}: ${msg.winner === 'draw' ? 'draw' : `${msg.winner.toUpperCase()} wins`} after ` +
          `${msg.a.pieces}/${msg.b.pieces} pieces, attack ${msg.a.attack}–${msg.b.attack} — ` +
          `A ${wins}–${losses} after ${results.length} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
      }
      dispatch();
    });
    w.on('error', err => {
      console.error(`worker ${id} failed:`, err);
      process.exitCode = 1;
      if (--live === 0) summarize(cfg, results);
    });
  }
}

if (isMainThread) runMain();
else runWorker();
