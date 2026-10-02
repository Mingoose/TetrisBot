"""
Compare self-play runs that used the same seeds (paired A/B for bot weights).

Because every run plays the same piece sequences and incoming garbage, each game
is compared with its twin from the baseline run, which removes most of the luck.

Usage:
    python compare_runs.py BASELINE_DIR OTHER_DIR [OTHER_DIR ...]
"""

import sys

import numpy as np

from selfplay_data import load_run


def per_game(run_dir):
    recs, meta = load_run(run_dir)
    games = {}
    for g in np.unique(recs['game_id']):
        r = recs[recs['game_id'] == g]
        games[int(g)] = dict(
            pieces=len(r),
            attack=int(r['attack'].sum()),
            died=bool((r['terminal'] == 1).any()),
            full=int((r['spin'] == 2).sum()),
            mini=int((r['spin'] == 1).sum()),
            mini_clears=int(((r['spin'] == 1) & (r['lines'] > 0)).sum()),
            surge=int(r['surge'].sum()),
            max_b2b=int(r['b2b'].max()),
        )
    return games, meta


def summary(name, games):
    pieces = sum(g['pieces'] for g in games.values())
    attack = sum(g['attack'] for g in games.values())
    return (f"{name:22s} APP {attack / pieces:.3f}  deaths {sum(g['died'] for g in games.values()):2d}/{len(games)}  "
            f"spins {sum(g['full'] for g in games.values()):4d} full / {sum(g['mini_clears'] for g in games.values()):3d} "
            f"mini clears  surge {sum(g['surge'] for g in games.values()):4d}  "
            f"max B2B {max(g['max_b2b'] for g in games.values())}")


def main(base_dir, others):
    base, _ = per_game(base_dir)
    print(summary(base_dir.rstrip('/').split('/')[-1], base))
    for d in others:
        other, _ = per_game(d)
        print(summary(d.rstrip('/').split('/')[-1], other))
        shared = sorted(set(base) & set(other))
        # Paired difference in attack per piece, game by game.
        diffs = np.array([other[g]['attack'] / other[g]['pieces'] - base[g]['attack'] / base[g]['pieces']
                          for g in shared])
        se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
        wins = int((diffs > 0).sum())
        print(f"{'':22s} vs baseline: APP {diffs.mean():+.3f} ± {se:.3f} (paired SE), "
              f"better in {wins}/{len(shared)} games")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
