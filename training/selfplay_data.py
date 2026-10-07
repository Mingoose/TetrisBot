"""
Load self-play records written by tetris-web/harness/selfplay.ts.

A run directory holds meta.json (config + record layout) and one w<N>.bin file
per worker. Each worker file is a sequence of whole games, so records for one
game are contiguous and in ply order.

Usage:
    from selfplay_data import load_run, boards_to_grid
    recs, meta = load_run('data/runs/teacher-v1')
    grid = boards_to_grid(recs['board'])   # (N, 20, 10) uint8
"""

import glob
import json
import os

import numpy as np


def run_dtype(meta: dict) -> np.dtype:
    dtype = np.dtype([(name, t, (n,)) if n > 1 else (name, t) for name, t, n in meta['fields']])
    assert dtype.itemsize == meta['record_size'], 'record layout does not match meta.json'
    return dtype


def load_run(run_dir: str) -> tuple[np.ndarray, dict]:
    """Return (records sorted by game_id then ply, meta)."""
    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)
    dtype = run_dtype(meta)
    parts = []
    for path in sorted(glob.glob(os.path.join(run_dir, 'w*.bin'))):
        # A run in progress may be mid-write; drop any trailing partial record.
        raw = np.fromfile(path, dtype=np.uint8)
        whole = len(raw) // dtype.itemsize * dtype.itemsize
        parts.append(raw[:whole].view(dtype))
    recs = np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    recs = recs[np.lexsort((recs['ply'], recs['game_id']))]
    return recs, meta


def boards_to_grid(board_bits: np.ndarray) -> np.ndarray:
    """(N, 20) uint16 row bitmasks → (N, 20, 10) uint8 occupancy, row 0 = top."""
    cols = np.arange(10, dtype=np.uint16)
    return ((board_bits[..., None] >> cols) & 1).astype(np.uint8)


def summarize(run_dir: str) -> None:
    recs, meta = load_run(run_dir)
    games = np.unique(recs['game_id'])
    teacher = recs['move_random'] == 0
    print(f'{run_dir}: {len(recs)} positions from {len(games)} games')
    cfg = meta['config']
    who = f"net {cfg['net']}" if cfg.get('net') else f"W{cfg['beam']} D{cfg['depth']}"
    print(f"  teacher {who}, "
          f"random moves {np.mean(~teacher):.1%}")
    if 'spin' in recs.dtype.names:  # current format (TETR.IO rules)
        spins = f"spins {int((recs['spin'] == 2).sum())} full / {int((recs['spin'] == 1).sum())} mini, " \
                f"surge attack {int(recs['surge'].sum())}, perfect clears {int(recs['perfect_clear'].sum())}"
    else:                           # teacher-v1 format (old JStris-style rules)
        spins = f"T-spins {int(recs['tspin'].sum())}"
    print(f"  attack/piece {recs['attack'].mean():.3f}, lines/piece {recs['lines'].mean():.3f}, {spins}")
    print(f"  deaths {int((recs['terminal'] == 1).sum())}, truncated {int((recs['terminal'] == 2).sum())}")


if __name__ == '__main__':
    import sys
    for d in sys.argv[1:]:
        summarize(d)
