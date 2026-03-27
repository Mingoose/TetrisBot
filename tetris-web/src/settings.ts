export interface KeyBindings {
  moveLeft: string;
  moveRight: string;
  softDrop: string;
  hardDrop: string;
  rotateCW: string;
  rotateCCW: string;
  rotate180: string;
  hold: string;
  rewind: string;
  editor: string;
  pause: string;
}

export interface Settings {
  keybindings: KeyBindings;
  das: number;              // ms before auto-repeat kicks in
  arr: number;              // ms between repeated moves (0 = instant)
  sonicDrop: boolean;       // soft drop instantly moves to bottom without locking
  botPps: number;           // versus bot speed in pieces per second
}

export const DEFAULT_DAS = 133;
export const DEFAULT_ARR = 10;
export const DEFAULT_SONIC_DROP = false;
export const DEFAULT_BOT_PPS = 1.5;

export const DEFAULT_KEYBINDINGS: KeyBindings = {
  moveLeft:  'ArrowLeft',
  moveRight: 'ArrowRight',
  softDrop:  'ArrowDown',
  hardDrop:  'Space',
  rotateCW:  'ArrowUp',
  rotateCCW: 'KeyZ',
  rotate180: 'KeyA',
  hold:      'KeyC',
  rewind:    'KeyR',
  editor:    'KeyE',
  pause:     'KeyP',
};

export const ACTION_LABELS: Record<keyof KeyBindings, string> = {
  moveLeft:  'Move Left',
  moveRight: 'Move Right',
  softDrop:  'Soft Drop',
  hardDrop:  'Hard Drop',
  rotateCW:  'Rotate CW',
  rotateCCW: 'Rotate CCW',
  rotate180: 'Rotate 180°',
  hold:      'Hold Piece',
  rewind:    'Rewind',
  editor:    'Editor Mode',
  pause:     'Pause',
};

// Human-readable key code labels
export function keyLabel(code: string): string {
  const map: Record<string, string> = {
    ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓',
    Space: 'Space', Escape: 'Esc',
    ShiftLeft: 'L-Shift', ShiftRight: 'R-Shift',
    ControlLeft: 'L-Ctrl', ControlRight: 'R-Ctrl',
    AltLeft: 'L-Alt', AltRight: 'R-Alt',
    Enter: 'Enter', Backspace: 'Backspace', Tab: 'Tab',
  };
  if (map[code]) return map[code];
  // KeyA → A, Digit1 → 1, etc.
  if (code.startsWith('Key')) return code.slice(3);
  if (code.startsWith('Digit')) return code.slice(5);
  if (code.startsWith('Numpad')) return `Num${code.slice(6)}`;
  return code;
}
