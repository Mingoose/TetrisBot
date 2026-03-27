import { PieceType } from './types';
import { ALL_PIECE_TYPES } from './pieces';

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export class Bag {
  private remaining: PieceType[];

  constructor(state?: PieceType[]) {
    this.remaining = state ? [...state] : shuffle(ALL_PIECE_TYPES);
  }

  private refillIfNeeded(): void {
    if (this.remaining.length === 0) {
      this.remaining = shuffle(ALL_PIECE_TYPES);
    }
  }

  next(): PieceType {
    this.refillIfNeeded();
    return this.remaining.shift()!;
  }

  // Peek at the next N pieces without consuming them.
  // May look across bag boundaries (generates a second bag temporarily).
  peek(n: number): PieceType[] {
    const result: PieceType[] = [];
    let extra: PieceType[] = [];
    for (let i = 0; i < n; i++) {
      if (i < this.remaining.length) {
        result.push(this.remaining[i]);
      } else {
        if (extra.length === 0) extra = shuffle(ALL_PIECE_TYPES);
        result.push(extra.shift()!);
      }
    }
    return result;
  }

  getState(): PieceType[] {
    return [...this.remaining];
  }

  restoreState(state: PieceType[]): void {
    this.remaining = [...state];
  }
}
