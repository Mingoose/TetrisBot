"""
Load the candidate placements written by tetris-web/harness/candidates.ts.

For every decision in a self-play run, cand_pos.bin says where its candidates
start in cand.bin, how many there are and which one the teacher played. Each
candidate holds the afterstate board and the context inputs the bot would feed
the value net for it (see tetris-web/src/valueBot.ts listCandidates).

Usage:
    from rank_data import load_candidates
    pos, cand = load_candidates('data/runs/teacher-v2')
"""

import json
import os

import numpy as np


def _dtype(fields, size):
    dt = np.dtype([(name, t, (n,)) if n > 1 else (name, t) for name, t, n in fields])
    assert dt.itemsize == size, 'candidate layout does not match candidates.json'
    return dt


def load_candidates(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (decisions sorted by game_id then ply, candidates)."""
    path = os.path.join(run_dir, 'candidates.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing; run `npm run candidates -- {run_dir}` in tetris-web/')
    with open(path) as f:
        meta = json.load(f)
    pos = np.fromfile(os.path.join(run_dir, 'cand_pos.bin'), dtype=_dtype(meta['pos_fields'], meta['pos_size']))
    cand = np.fromfile(os.path.join(run_dir, 'cand.bin'), dtype=_dtype(meta['cand_fields'], meta['cand_size']))
    assert len(pos) == meta['decisions'] and len(cand) == meta['candidates'], f'{run_dir}: candidate files are incomplete'
    return pos, cand


def gather(pos: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate indices for decisions `sel`, and each candidate's row in `sel`."""
    counts = pos['count'][sel].astype(np.int64)
    starts = pos['first'][sel].astype(np.int64)
    row = np.repeat(np.arange(len(sel)), counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return starts[row] + within, row
