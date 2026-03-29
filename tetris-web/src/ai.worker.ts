import { findBestMove, findBestMoveHard, analyzePositionHard } from './ai';
import type { BotBoard } from './versus';
import type { EngineRequest } from './engine';

self.onmessage = (e: MessageEvent) => {
  const data = e.data as
    | { type: 'analyze'; request: EngineRequest }
    | { type?: undefined; bot: BotBoard; pendingGarbage: number; beamWidth?: number; searchDepth?: number; advancedEval?: boolean };

  if (data.type === 'analyze') {
    const result = analyzePositionHard(data.request);
    self.postMessage({ type: 'analysis', result });
    return;
  }

  // Existing bot move path — response shape unchanged so wireWorker() keeps working.
  const { bot, pendingGarbage, beamWidth, searchDepth, advancedEval } = data;
  const fn = advancedEval ? findBestMoveHard : findBestMove;
  self.postMessage(fn(bot, pendingGarbage, beamWidth, searchDepth));
};
