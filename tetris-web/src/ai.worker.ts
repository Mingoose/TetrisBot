import { findBestMove, findBestMoveHard, analyzePositionHard } from './ai';
import type { BotBoard } from './versus';
import type { EngineRequest } from './engine';

self.onmessage = (e: MessageEvent) => {
  const data = e.data as
    | { type: 'analyze'; request: EngineRequest }
    | { type?: undefined; bot: BotBoard; pendingGarbage: number; combo?: number; b2bActive?: boolean; beamWidth?: number; searchDepth?: number; advancedEval?: boolean };

  if (data.type === 'analyze') {
    const result = analyzePositionHard(data.request);
    self.postMessage({ type: 'analysis', result });
    return;
  }

  // Existing bot move path — response shape unchanged so wireWorker() keeps working.
  const { bot, pendingGarbage, combo = -1, b2bActive = false, beamWidth, searchDepth, advancedEval } = data;
  const fn = advancedEval ? findBestMoveHard : findBestMove;
  self.postMessage(fn(bot, pendingGarbage, beamWidth, searchDepth, combo, b2bActive));
};
