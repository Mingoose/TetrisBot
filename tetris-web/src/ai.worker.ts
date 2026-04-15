import { findBestMove, findBestMoveHard, findBestMoveCNN, analyzePositionHard } from './ai';
import type { BotBoard } from './versus';
import type { EngineRequest } from './engine';
import { loadCnnModel, evaluateBoardsBatch } from './cnnEvaluator';

// Begin loading the CNN model immediately. Once ready (or if unavailable), post a
// status message so main.ts can enable/disable the experimental difficulty button.
loadCnnModel()
  .then(() => self.postMessage({ type: 'cnn_ready' }))
  .catch(() => self.postMessage({ type: 'cnn_unavailable' }));

self.onmessage = (e: MessageEvent) => {
  const data = e.data as
    | { type: 'analyze'; request: EngineRequest }
    | { type?: undefined; bot: BotBoard; pendingGarbage: number; combo?: number; b2bActive?: boolean; beamWidth?: number; searchDepth?: number; advancedEval?: boolean; cnnEval?: boolean };

  console.log('[worker] onmessage type:', data.type, 'cnnEval:', (data as any).cnnEval);

  if (data.type === 'analyze') {
    const result = analyzePositionHard(data.request);
    self.postMessage({ type: 'analysis', result });
    return;
  }

  const { bot, pendingGarbage, combo = -1, b2bActive = false, beamWidth, searchDepth, advancedEval, cnnEval } = data;

  if (cnnEval) {
    console.log('[worker] starting cnnEval move');
    findBestMoveCNN(bot, pendingGarbage, beamWidth, searchDepth, combo, b2bActive, evaluateBoardsBatch)
      .then(move => { console.log('[worker] cnnEval move done:', move); self.postMessage(move); })
      .catch(e => console.error('[worker] cnnEval move error:', e));
    return;
  }

  const fn = advancedEval ? findBestMoveHard : findBestMove;
  self.postMessage(fn(bot, pendingGarbage, beamWidth, searchDepth, combo, b2bActive));
};
