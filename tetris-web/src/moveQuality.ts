import type { EngineAnalysis } from './engine';
import type { ActivePiece } from './types';

export type MoveQuality = 'great' | 'good' | 'mistake' | 'blunder';

export interface ClassificationResult {
  quality: MoveQuality;
  bestScore: number | null;
  playerScore: number | null;
  /** Index of the player's move in analysis.lines, or -1 if not found in topN. */
  playerLineIndex: number;
  /** bestScore - playerScore; null if player's move was not found. */
  delta: number | null;
}

/**
 * Find the player's actual placement in the engine lines and classify quality.
 *
 * Matching is done on the first move of each line: pieceType + x + y + rotationIndex
 * must all match the piece that was actually played (snapshot.active).
 */
export function classifyMove(
  analysis: EngineAnalysis,
  active: ActivePiece,
): ClassificationResult {
  if (analysis.lines.length === 0) {
    return { quality: 'blunder', bestScore: null, playerScore: null, playerLineIndex: -1, delta: null };
  }

  const bestScore = analysis.lines[0].score;

  const playerLineIndex = analysis.lines.findIndex(line => {
    const m = line.moves[0];
    if (!m) return false;
    return (
      m.pieceType      === active.type &&
      m.x              === active.x &&
      m.y              === active.y &&
      m.rotationIndex  === active.rotationIndex
    );
  });

  if (playerLineIndex === -1) {
    return { quality: 'blunder', bestScore, playerScore: null, playerLineIndex: -1, delta: null };
  }

  const playerScore = analysis.lines[playerLineIndex].score;
  const delta = bestScore - playerScore;

  let quality: MoveQuality;
  if (delta <= 5)       quality = 'great';
  else if (delta <= 25) quality = 'good';
  else if (delta <= 60) quality = 'mistake';
  else                  quality = 'blunder';

  return { quality, bestScore, playerScore, playerLineIndex, delta };
}

export function qualityColor(q: MoveQuality): string {
  switch (q) {
    case 'great':   return '#66ffaa';
    case 'good':    return '#aaaaff';
    case 'mistake': return '#ffcc44';
    case 'blunder': return '#ff5555';
  }
}

export function qualityLabel(q: MoveQuality): string {
  return q.charAt(0).toUpperCase() + q.slice(1);
}
