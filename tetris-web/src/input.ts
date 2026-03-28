export interface InputState {
  keys: Set<string>;
  dasLeft: number;   // ms left key has been held
  dasRight: number;  // ms right key has been held
  // Prevent key repeat for one-shot actions
  justPressed: Set<string>;
  justReleased: Set<string>;
}

export const DAS_MS = 133;
export const ARR_MS = 10;

export function createInputState(): InputState {
  return {
    keys: new Set(),
    dasLeft: 0,
    dasRight: 0,
    justPressed: new Set(),
    justReleased: new Set(),
  };
}

export function setupInput(canvas: HTMLCanvasElement, input: InputState): void {
  canvas.addEventListener('keydown', (e) => {
    if (!input.keys.has(e.code)) {
      input.justPressed.add(e.code);
    }
    input.keys.add(e.code);
    // Prevent browser scroll/defaults for game keys
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
      e.preventDefault();
    }
  });

  canvas.addEventListener('keyup', (e) => {
    input.keys.delete(e.code);
    input.justReleased.add(e.code);
    if (e.code === 'ArrowLeft') input.dasLeft = 0;
    if (e.code === 'ArrowRight') input.dasRight = 0;
  });
}

export function wasJustPressed(input: InputState, code: string): boolean {
  return input.justPressed.has(code);
}

export function isHeld(input: InputState, code: string): boolean {
  return input.keys.has(code);
}

// Call at end of each frame to clear per-frame edge state
export function flushInput(input: InputState): void {
  input.justPressed.clear();
  input.justReleased.clear();
}
