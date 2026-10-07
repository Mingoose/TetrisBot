// Headless self-play data generator.
//
// Plays games with the real ai.ts beam search (the "teacher") across worker
// threads and writes one fixed-width record per decision point (see record.ts).
// Each worker appends whole games to its own file, so an interrupted run keeps
// every finished game.
//
// Usage (from tetris-web/):
//   npm run harness -- --out ../training/data/runs/teacher-v1 --games 200
//
// Options (defaults in DEFAULTS below):
//   --out DIR            output directory (must not already contain a run)
//   --games N            games to play
//   --first-game N       first game id, for adding a second batch to a new directory
//   --workers N          worker threads
//   --pieces N           piece cap per game (game is marked truncated at the cap)
//   --beam N --depth N   teacher search settings
//   --eps P              probability of a random straight-drop move instead of the teacher's
//   --garbage-rate R     expected incoming garbage rows per piece (arrives in 1–4 row bursts)
//   --pps N              pieces per second on the virtual clock that times garbage travel
//   --seed N             base seed; each game's pieces and garbage are a function of (seed, game id)
//   --weights FILE       JSON of hard-mode weight overrides (see setWeights in ai.ts)
//   --net FILE.json@K,…  teacher = the value-net bot instead of the beam search
//                        (export_value_ts.py JSON; widths as in valueBot.ts findBestMoveDeep)

import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { CellValue, PieceType } from '../src/types';
import { BOARD_COLS, BOARD_ROWS, collides, hardDropY } from '../src/board';
import { getRotation } from '../src/pieces';
import { initVersusData, applyBotMove, receiveGarbage, BotBoard, CombatState, LockOutcome } from '../src/versus';
import { findBestMoveHard, setWeights } from '../src/ai';
import { ValueNet, ValueNetFile } from '../src/valueNet';
import { findBestMoveDeep, searchStateFor } from '../src/valueBot';
import {
  PIECE_ORDER, NO_PIECE, RECORD_FIELDS, RECORD_SIZE, TERMINAL_DIED, TERMINAL_NONE,
  TERMINAL_TRUNCATED, PositionRecord, writeRecord,
} from './record';

interface Config {
  out: string;
  games: number;
  firstGame: number;
  workers: number;
  pieces: number;
  beam: number;
  depth: number;
  eps: number;
  garbageRate: number;
  pps: number;
  seed: number;
  weights: string;
  net: string;
}

const DEFAULTS: Config = {
  out: '',
  games: 20,
  firstGame: 0,
  workers: Math.max(1, availableParallelism() - 2),
  pieces: 300,
  beam: 32,
  depth: 5,
  eps: 0.05,
  garbageRate: 0.2,
  pps: 2,
  seed: 1,
  weights: '',
  net: '',
};

interface GameSummary {
  type: 'done';
  gameId: number;
  plies: number;
  ms: number;
  attack: number;
  lines: number;
  died: boolean;
}

// ---- Seeded RNG ----
// The game code draws from Math.random (bag shuffles, garbage hole columns), so
// each game swaps in a seeded generator to make (seed, game id) reproducible.

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

// ---- Encoding helpers ----

const pieceIdx = (p: PieceType | null): number => (p ? PIECE_ORDER.indexOf(p) : NO_PIECE);

function boardToBits(board: CellValue[][]): Uint16Array {
  const bits = new Uint16Array(BOARD_ROWS);
  for (let r = 0; r < BOARD_ROWS; r++) {
    let m = 0;
    for (let c = 0; c < BOARD_COLS; c++) if (board[r][c] !== 0) m |= 1 << c;
    bits[r] = m;
  }
  return bits;
}

// Pieces left in the current bag, as a set. An empty bag means the next draw
// starts a fresh bag, so all seven pieces are available.
function bagMask(bagState: PieceType[]): number {
  if (bagState.length === 0) return 0x7f;
  let m = 0;
  for (const p of bagState) m |= 1 << pieceIdx(p);
  return m;
}

// ---- Exploration ----

type Move = { rotationIndex: number; x: number; y: number; useHold: boolean };
type Teacher = (bot: BotBoard, combat: CombatState, now: number) => Move;

function makeTeacher(cfg: Config): Teacher {
  if (!cfg.net) {
    return (bot, combat) => findBestMoveHard(bot, combat.pendingGarbage, cfg.beam, cfg.depth, combat.combo, combat.b2b);
  }
  const [path, k] = cfg.net.split('@');
  const widths = k ? k.split(',').map(Number) : [];
  const net = new ValueNet(JSON.parse(readFileSync(path, 'utf8')) as ValueNetFile);
  return (bot, combat, now) => findBestMoveDeep(searchStateFor(bot, combat, bagMask(bot.bagState), now), net, widths);
}

// A uniformly random straight drop of the active piece, or null if none fits.
function randomDrop(board: CellValue[][], type: PieceType, rng: () => number): Move | null {
  const options: Move[] = [];
  for (let rot = 0; rot < 4; rot++) {
    // Matrices are up to 4 wide with empty edge columns, so x can start left of the wall.
    for (let x = -getRotation(type, rot)[0].length + 1; x < BOARD_COLS; x++) {
      const piece = { type, rotationIndex: rot, x, y: 0 };
      if (collides(board, piece, 0, 0)) continue;
      options.push({ rotationIndex: rot, x, y: hardDropY(board, piece), useHold: false });
    }
  }
  return options.length ? options[Math.floor(rng() * options.length)] : null;
}

// ---- One game ----

function playGame(cfg: Config, gameId: number, teacher: Teacher): { records: Buffer; summary: GameSummary } {
  const rng = mulberry32(gameSeed(cfg.seed, gameId));
  Math.random = rng;

  const v = initVersusData();
  const bot = v.bot;
  const combat = v.botCombat;
  const burstChance = cfg.garbageRate / 2.5; // bursts are 1–4 rows, mean 2.5

  const buf = Buffer.alloc(cfg.pieces * RECORD_SIZE);
  let offset = 0;
  let totalAttack = 0;
  const t0 = performance.now();
  let ply = 0;

  for (; ply < cfg.pieces && !bot.dead; ply++) {
    const now = ply * 1000 / cfg.pps;
    if (rng() < burstChance) receiveGarbage(combat, 1 + Math.floor(rng() * 4), now);

    const rec: Partial<PositionRecord> = {
      gameId,
      ply,
      board: boardToBits(bot.board),
      active: pieceIdx(bot.active.type),
      hold: pieceIdx(bot.hold),
      queue: bot.nextQueue.slice(0, 5).map(pieceIdx),
      bagMask: bagMask(bot.bagState),
      combo: combat.combo,
      b2b: combat.b2b,
      incoming: combat.pendingGarbage,
    };

    let move: Move | null = null;
    let isRandom = false;
    if (rng() < cfg.eps) {
      move = randomDrop(bot.board, bot.active.type, rng);
      isRandom = move !== null;
    }
    if (!move) move = teacher(bot, combat, now);

    let out!: LockOutcome;
    applyBotMove(move, bot, combat, v.playerCombat, o => { out = o; }, now);
    v.playerCombat.incoming = []; // no opponent; discard what was sent
    v.playerCombat.pendingGarbage = 0;
    totalAttack += out.attack;

    rec.moveRot = move.rotationIndex;
    rec.moveX = move.x;
    rec.moveY = move.y;
    rec.moveHold = move.useHold;
    rec.moveRandom = isRandom;
    rec.lines = out.lines;
    rec.attack = out.attack;
    rec.spin = out.spin;
    rec.surge = out.surge;
    rec.perfectClear = out.perfectClear;
    rec.garbageIn = out.garbageIn;
    rec.terminal = bot.dead ? TERMINAL_DIED
      : ply === cfg.pieces - 1 ? TERMINAL_TRUNCATED
      : TERMINAL_NONE;

    offset = writeRecord(buf, offset, rec as PositionRecord);
  }

  return {
    records: buf.subarray(0, offset),
    summary: {
      type: 'done', gameId, plies: ply, ms: performance.now() - t0,
      attack: totalAttack, lines: bot.lines, died: bot.dead,
    },
  };
}

// ---- Worker ----

function runWorker(): void {
  const { cfg, workerId, weights } = workerData as { cfg: Config; workerId: number; weights: Record<string, number> };
  setWeights(weights);
  const teacher = makeTeacher(cfg);
  const file = join(cfg.out, `w${workerId}.bin`);
  parentPort!.on('message', (msg: { type: 'game'; gameId: number } | { type: 'stop' }) => {
    if (msg.type === 'stop') { process.exit(0); }
    // An engine error costs one game, not the worker. Nothing from the failed game is written.
    try {
      const { records, summary } = playGame(cfg, msg.gameId, teacher);
      appendFileSync(file, records);
      parentPort!.postMessage(summary);
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
    '--out': 'out', '--games': 'games', '--first-game': 'firstGame', '--workers': 'workers',
    '--pieces': 'pieces', '--beam': 'beam', '--depth': 'depth', '--eps': 'eps',
    '--garbage-rate': 'garbageRate', '--pps': 'pps', '--seed': 'seed', '--weights': 'weights',
    '--net': 'net',
  };
  for (let i = 0; i < argv.length; i += 2) {
    const key = keys[argv[i]];
    if (!key || argv[i + 1] === undefined) throw new Error(`Unknown or incomplete option: ${argv[i]}`);
    (cfg as Record<string, unknown>)[key] = key === 'out' || key === 'weights' || key === 'net' ? argv[i + 1] : Number(argv[i + 1]);
  }
  if (!cfg.out) throw new Error('--out DIR is required');
  return cfg;
}

function runMain(): void {
  const cfg = parseArgs(process.argv.slice(2));
  const metaPath = join(cfg.out, 'meta.json');
  if (existsSync(metaPath)) throw new Error(`${cfg.out} already contains a run; pick a new --out`);
  mkdirSync(cfg.out, { recursive: true });

  const weights = cfg.weights ? JSON.parse(readFileSync(cfg.weights, 'utf8')) : {};
  setWeights(weights); // validate keys before starting workers

  const meta = {
    config: cfg,
    weights,
    record_size: RECORD_SIZE,
    fields: RECORD_FIELDS,
    piece_order: PIECE_ORDER,
    started: new Date().toISOString(),
    finished: null as string | null,
    totals: null as Record<string, number> | null,
  };
  writeFileSync(metaPath, JSON.stringify(meta, null, 2));

  console.log(`Teacher ${cfg.net ? `net ${cfg.net}` : `W${cfg.beam} D${cfg.depth}`}, eps ${cfg.eps}, garbage ${cfg.garbageRate}/piece, ` +
    `${cfg.games} games × ≤${cfg.pieces} pieces on ${cfg.workers} workers → ${cfg.out}`);

  let nextGame = cfg.firstGame;
  const endGame = cfg.firstGame + cfg.games;
  let done = 0, plies = 0, attack = 0, deaths = 0, live = cfg.workers;
  const failed: number[] = [];
  const t0 = performance.now();

  const finish = () => {
    const secs = (performance.now() - t0) / 1000;
    meta.finished = new Date().toISOString();
    meta.totals = { games: done, positions: plies, attack, deaths, seconds: Math.round(secs) };
    (meta as Record<string, unknown>).failed_games = failed;
    writeFileSync(metaPath, JSON.stringify(meta, null, 2));
    console.log(`\nDone: ${done} games, ${plies} positions in ${secs.toFixed(0)} s ` +
      `(${(plies / secs).toFixed(1)} positions/s, ~${Math.round(plies / secs * 3600).toLocaleString()}/hour)`);
    console.log(`Attack per piece ${(attack / Math.max(1, plies)).toFixed(3)}, deaths ${deaths}/${done}`);
    if (failed.length) console.log(`FAILED games (skipped): ${failed.join(', ')}`);
  };

  const workerFile = fileURLToPath(import.meta.url);
  for (let id = 0; id < cfg.workers; id++) {
    const w = new Worker(workerFile, { workerData: { cfg, workerId: id, weights } });
    const dispatch = () => {
      if (nextGame < endGame) w.postMessage({ type: 'game', gameId: nextGame++ });
      else {
        w.postMessage({ type: 'stop' });
        if (--live === 0) finish();
      }
    };
    w.on('message', (msg: { type: 'ready' } | GameSummary | { type: 'failed'; gameId: number; error: string }) => {
      if (msg.type === 'failed') {
        failed.push(msg.gameId);
        console.error(`game ${msg.gameId} failed and was skipped:\n${msg.error}`);
      } else if (msg.type === 'done') {
        done++; plies += msg.plies; attack += msg.attack; if (msg.died) deaths++;
        const rate = plies / ((performance.now() - t0) / 1000);
        console.log(`game ${msg.gameId}: ${msg.plies} pieces, ${msg.lines} lines, attack ${msg.attack}` +
          `${msg.died ? ', died' : ''} (${(msg.ms / msg.plies).toFixed(0)} ms/piece) — ` +
          `${done}/${cfg.games} games, ${rate.toFixed(1)} positions/s overall`);
      }
      dispatch();
    });
    w.on('error', err => {
      // The game in progress is lost; games this worker already finished are on disk.
      console.error(`worker ${id} failed:`, err);
      process.exitCode = 1;
      if (--live === 0) finish();
    });
  }
}

if (isMainThread) runMain();
else runWorker();
