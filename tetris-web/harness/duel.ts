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

import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { ALL_PIECE_TYPES } from '../src/pieces';
import { RULES } from '../src/rules';
import { initBotVsBotData, applyBotMove, bvbLookahead, BotBoard, CombatState } from '../src/versus';
import { findBestMoveHard } from '../src/ai';
import { ValueNet, ValueNetFile } from '../src/valueNet';
import { findBestMoveDeep, BotMove } from '../src/valueBot';

interface Config {
  a: string; b: string; games: number; workers: number; pieces: number;
  pps: number; seed: number; out: string;
}

const DEFAULTS: Config = {
  a: '', b: 'hard', games: 100, workers: Math.max(1, availableParallelism() - 2),
  pieces: 1000, pps: 2, seed: 1, out: '',
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

type Think = (bot: BotBoard, combat: CombatState, now: number) => BotMove;

function makeBot(spec: string): Think {
  const [kind, ...rest] = spec.split(':');
  if (kind === 'hard') {
    const [w, d] = rest.length ? rest.map(Number) : [32, 5];
    return (bot, combat) => findBestMoveHard(
      { ...bot, bagState: bvbLookahead(bot, 30) }, combat.pendingGarbage, w, d, combat.combo, combat.b2b);
  }
  if (kind === 'net') {
    const [path, k] = rest.join(':').split('@');
    const widths = k ? k.split(',').map(Number) : [];
    const net = new ValueNet(JSON.parse(readFileSync(path, 'utf8')) as ValueNetFile);
    return (bot, combat, now) => {
      // The bag as a set: the pieces left before the next 7-piece boundary of the shared sequence.
      const left = (7 - bot.pieceIndex % 7) % 7;
      let bagMask = left ? 0 : 0x7f;
      for (const p of bvbLookahead(bot, left)) bagMask |= 1 << ALL_PIECE_TYPES.indexOf(p);
      // Queued garbage rows, and those that would land on a non-clearing lock now
      // (finished travelling, at most the cap; versus.ts tankGarbage).
      const pendingCols: number[] = [];
      let ready = 0;
      for (const chunk of combat.incoming) {
        for (let i = 0; i < chunk.amount; i++) pendingCols.push(chunk.column);
        if (chunk.readyAt <= now && ready === pendingCols.length - chunk.amount) ready = pendingCols.length;
      }
      const landingCols = pendingCols.slice(0, Math.min(ready, RULES.garbageCap));
      return findBestMoveDeep({
        board: bot.board, active: bot.active.type, hold: bot.hold, queue: bot.nextQueue,
        bagMask, combo: combat.combo, b2b: combat.b2b, landingCols, pendingCols,
      }, net, widths);
    };
  }
  throw new Error(`Unknown bot spec: ${spec}`);
}

// ---- One game ----

function playGame(cfg: Config, gameId: number, bots: [Think, Think]): GameResult {
  Math.random = mulberry32(gameSeed(cfg.seed, gameId));
  const d = initBotVsBotData();
  const aSide = (gameId % 2) as 0 | 1;            // which board bot A plays this game
  const boards = [d.bot1, d.bot2];
  const combats = [d.bot1Combat, d.bot2Combat];
  const thinks = aSide === 0 ? bots : [bots[1], bots[0]];
  const stats: SideStats[] = [0, 1].map(() => ({ pieces: 0, attack: 0, lines: 0, thinkMs: 0, dead: false }));
  const first = aSide; // bot A moves first at equal times; swaps with the sides

  const interval = 1000 / cfg.pps;
  while (!boards[0].dead && !boards[1].dead) {
    // Next to move: the side with fewer pieces placed (ties go to `first`).
    const s = stats[0].pieces === stats[1].pieces ? first : stats[0].pieces < stats[1].pieces ? 0 : 1;
    if (stats[s].pieces >= cfg.pieces) break;
    const now = stats[s].pieces * interval;
    const t0 = performance.now();
    const move = thinks[s](boards[s], combats[s], now);
    stats[s].thinkMs += performance.now() - t0;
    applyBotMove(move, boards[s], combats[s], combats[1 - s], o => {
      stats[s].attack += o.attack;
      stats[s].lines += o.lines;
    }, now);
    stats[s].pieces++;
  }
  stats[0].dead = boards[0].dead;
  stats[1].dead = boards[1].dead;
  const a = stats[aSide], b = stats[1 - aSide];
  const winner = a.dead === b.dead ? 'draw' : a.dead ? 'b' : 'a';
  return { type: 'done', gameId, aSide, winner, a, b };
}

// ---- Worker ----

function runWorker(): void {
  const { cfg } = workerData as { cfg: Config };
  const bots: [Think, Think] = [makeBot(cfg.a), makeBot(cfg.b)];
  parentPort!.on('message', (msg: { type: 'game'; gameId: number } | { type: 'stop' }) => {
    if (msg.type === 'stop') process.exit(0);
    try {
      parentPort!.postMessage(playGame(cfg, msg.gameId, bots));
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
    '--pps': 'pps', '--seed': 'seed', '--out': 'out',
  };
  for (let i = 0; i < argv.length; i += 2) {
    const key = keys[argv[i]];
    if (!key || argv[i + 1] === undefined) throw new Error(`Unknown or incomplete option: ${argv[i]}`);
    (cfg as Record<string, unknown>)[key] = ['a', 'b', 'out'].includes(key) ? argv[i + 1] : Number(argv[i + 1]);
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
  console.log(`${cfg.a} vs ${cfg.b}: ${cfg.games} games on ${cfg.workers} workers, ${cfg.pps} pps, seed ${cfg.seed}`);

  const results: GameResult[] = [];
  let nextGame = 0, live = cfg.workers;
  const t0 = performance.now();
  const workerFile = fileURLToPath(import.meta.url);
  for (let id = 0; id < cfg.workers; id++) {
    const w = new Worker(workerFile, { workerData: { cfg } });
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
