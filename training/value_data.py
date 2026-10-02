"""
Turn self-play records into (afterstate, piece context) → value training samples.

The value net scores the position right after the bot places a piece, using only
what the bot would know at that moment. That is what a 1-ply search can feed it:
place each candidate, then ask the net how good the result is given the pieces
still to come.

Sample t is built from records t and t+1 of the same game:
  board    the board after move t's line clears, before incoming garbage lands
           (record t+1's board with the garbage rows removed)
  context  next piece to play, the next 4 queue pieces (unknowns = none), hold,
           pieces left in the bag, combo, B2B level, and garbage about to land
  label    discounted attack over the following `horizon` pieces, minus a
           discounted penalty if the bot dies within that window

The immediate attack of move t is not in the label: the search computes it
exactly and adds it to the net's score.

Usage:
    from value_data import build_samples
    boards, ctx, labels, game_ids = build_samples(recs, horizon=12)
"""

import numpy as np

from selfplay_data import boards_to_grid

NO_PIECE = 7
CONTEXT_SIZE = 7 + 4 * 8 + 8 + 7 + 3  # next, queue[4], hold, bag, combo/b2b/garbage


def encode_context(next_piece, queue4, hold, bag_mask, combo, b2b, garbage_landing) -> np.ndarray:
    """Vectorised context encoding; every argument is an (N,) or (N, 4) array."""
    n = len(next_piece)
    ctx = np.zeros((n, CONTEXT_SIZE), dtype=np.float32)
    rows = np.arange(n)
    ctx[rows, next_piece] = 1.0                              # 0..6
    for i in range(4):
        ctx[rows, 7 + i * 8 + queue4[:, i]] = 1.0             # 7..38 (index 7 = none)
    ctx[rows, 39 + hold] = 1.0                               # 39..46
    for p in range(7):
        ctx[:, 47 + p] = (bag_mask >> p) & 1                 # 47..53
    ctx[:, 54] = (combo.astype(np.float32) + 1) / 10.0
    ctx[:, 55] = (np.clip(b2b, -1, 19) + 1) / 10.0         # B2B level; surge charges from 4
    ctx[:, 56] = garbage_landing / 10.0
    return ctx


def build_samples(recs: np.ndarray, horizon: int = 12, gamma: float = 0.97,
                  death_penalty: float = 10.0):
    """Return (boards uint8 (N,20,10), ctx float32 (N,C), labels float32 (N,), game_ids)."""
    n = len(recs)
    game = recs['game_id'].astype(np.int64)

    # Index of the last record of each record's game (records are sorted by game, ply).
    boundaries = np.flatnonzero(np.diff(game)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]]) - 1
    last = np.repeat(ends, ends - starts + 1)
    died = recs['terminal'][last] == 1

    t = np.arange(n)
    # Need record t+1 to see the afterstate.
    keep = t < last
    # Without a death, the whole label window has to exist (capped or partly written games).
    keep &= died | (t + horizon <= last)
    t = t[keep]
    t1 = t + 1

    # Afterstate board: drop the garbage rows pushed in from the bottom.
    g = recs['garbage_in'][t].astype(np.int64)
    after = recs['board'][t1]
    rows = np.arange(20)[None, :] - g[:, None]           # source row for each target row
    after = np.where(rows >= 0, np.take_along_axis(after, np.clip(rows, 0, 19), axis=1), 0)
    boards = boards_to_grid(after.astype(np.uint16))

    # Context known right after move t.
    queue4 = recs['queue'][t1][:, :4].copy()
    held_from_empty = (recs['move_hold'][t] == 1) & (recs['hold'][t] == NO_PIECE)
    queue4[held_from_empty, 3] = NO_PIECE                 # two pieces were drawn; the later one is unseen
    ctx = encode_context(
        next_piece=recs['active'][t1].astype(np.int64),
        queue4=queue4.astype(np.int64),
        hold=recs['hold'][t1].astype(np.int64),
        bag_mask=recs['bag_mask'][t].astype(np.int64),   # before the unseen draw
        combo=recs['combo'][t1],
        b2b=recs['b2b'][t1].astype(np.float32),        # B2B level (-1 = none)
        garbage_landing=g.astype(np.float32),
    )

    # Label: discounted future attack, and a death penalty inside the window.
    attack = recs['attack'].astype(np.float32)
    lab = np.zeros(len(t), dtype=np.float32)
    lt = last[t]
    for k in range(1, horizon + 1):
        idx = t + k
        valid = idx <= lt
        lab[valid] += gamma ** (k - 1) * attack[idx[valid]]
    dies_in_window = died[t] & (lt - t <= horizon)
    lab[dies_in_window] -= death_penalty * gamma ** (lt[dies_in_window] - t[dies_in_window] - 1)

    return boards, ctx, lab, game[t]
