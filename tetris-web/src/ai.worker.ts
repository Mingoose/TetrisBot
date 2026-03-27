import { findBestMove } from './ai';
import type { BotBoard } from './versus';

self.onmessage = (e: MessageEvent<{ bot: BotBoard; pendingGarbage: number }>) => {
  const { bot, pendingGarbage } = e.data;
  self.postMessage(findBestMove(bot, pendingGarbage));
};
