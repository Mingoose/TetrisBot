import { findBestMove, findBestMoveHard } from './ai';
import type { BotBoard } from './versus';

self.onmessage = (e: MessageEvent<{
  bot: BotBoard;
  pendingGarbage: number;
  beamWidth?: number;
  searchDepth?: number;
  advancedEval?: boolean;
}>) => {
  const { bot, pendingGarbage, beamWidth, searchDepth, advancedEval } = e.data;
  const fn = advancedEval ? findBestMoveHard : findBestMove;
  self.postMessage(fn(bot, pendingGarbage, beamWidth, searchDepth));
};
