import { findBestMove, findBestMoveHard, analyzePositionHard } from './ai';
import type { BotBoard } from './versus';
import type { EngineRequest } from './engine';
import { ValueNet, ValueNetManifest, valueNetFromBinary } from './valueNet';
import { findBestMoveTimed, ValueSearchState } from './valueBot';

// Load the value net (experimental difficulty) as soon as the worker starts, and
// tell main.ts whether it is available so the menu can enable the difficulty.
const netReady: Promise<ValueNet> = (async () => {
  const base = `${import.meta.env.BASE_URL}models/`;
  const get = async (name: string) => {
    const res = await fetch(base + name);
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    return res;
  };
  const [manifest, blob] = await Promise.all([
    get('value_net.json').then(r => r.json() as Promise<ValueNetManifest>),
    get('value_net.bin').then(r => r.arrayBuffer()),
  ]);
  return new ValueNet(valueNetFromBinary(manifest, blob));
})();
netReady.then(
  () => self.postMessage({ type: 'value_ready' }),
  err => { console.error('[worker] value net unavailable:', err); self.postMessage({ type: 'value_unavailable' }); },
);

self.onmessage = (e: MessageEvent) => {
  const data = e.data as
    | { type: 'analyze'; request: EngineRequest }
    | { type?: undefined; valueNet: number[]; searchMs: number; state: ValueSearchState }
    | { type?: undefined; valueNet?: undefined; bot: BotBoard; pendingGarbage: number; combo?: number; b2b?: number; beamWidth?: number; searchDepth?: number; advancedEval?: boolean };

  if (data.type === 'analyze') {
    const result = analyzePositionHard(data.request);
    self.postMessage({ type: 'analysis', result });
    return;
  }

  if (data.valueNet) {
    const { valueNet: widths, searchMs, state } = data;
    netReady
      .then(net => {
        const { move, widths: used, ms } = findBestMoveTimed(state, net, widths, searchMs);
        if (import.meta.env.DEV) console.debug(`[worker] value net move @${used.join(',')} in ${ms.toFixed(0)} ms`);
        self.postMessage(move);
      })
      .catch(err => self.postMessage({ error: String(err) }));
    return;
  }

  const { bot, pendingGarbage, combo = -1, b2b = -1, beamWidth, searchDepth, advancedEval } = data;
  const fn = advancedEval ? findBestMoveHard : findBestMove;
  self.postMessage(fn(bot, pendingGarbage, beamWidth, searchDepth, combo, b2b));
};
