"""
Export a trained value net for the TypeScript evaluator (tetris-web/src/valueNet.ts).

Writes one JSON file with the config, label scaling, every weight tensor, and a
set of check positions from the validation games. Each check carries the
decision the teacher faced (record t), the teacher's move, and what Python
built for the resulting sample (afterstate board, context vector, net output),
so tetris-web/harness/valuecheck.ts can confirm the TypeScript side matches
and measure how often the net picks the teacher's move.

With --app DIR it instead writes the compact files the web app loads:
DIR/value_net.json (config, label scaling, tensor shapes and offsets) and
DIR/value_net.bin (all weights as little-endian float32), no check positions.

Usage:
    .venv/bin/python export_value_ts.py models/value_v2_c16.pt \\
        --out models/value_v2_c16.json data/runs/teacher-v2 data/runs/teacher-v2b
    .venv/bin/python export_value_ts.py models/rank_v2_c16.pt --app ../tetris-web/public/models
"""

import argparse
import json
import os

import numpy as np
import torch

from selfplay_data import load_run
from value_data import build_samples
from value_net import ValueNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('runs', nargs='*', help='the runs the net was trained on (to find validation games)')
    ap.add_argument('--out', help='JSON with weights and check positions (needs the runs)')
    ap.add_argument('--app', help='directory to write value_net.json + value_net.bin for the web app')
    ap.add_argument('--checks', type=int, default=1000, help='validation positions to include')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = ValueNet(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    a = ckpt['args']

    if args.app:
        export_app(ckpt, args.app)
    if not args.out:
        return
    assert args.runs, '--out needs the training runs to pick check positions'

    # Rebuild samples exactly as train_value.py does, keeping each sample's record.
    per_run = []
    games = []
    for i, d in enumerate(args.runs):
        recs, _ = load_run(d)
        b, c, _, g, t = build_samples(recs, a['horizon'], a['gamma'], a['death_penalty'], return_index=True)
        per_run.append((recs, b, c, t))
        games.append(g + i * 10_000_000)
    games = np.concatenate(games)
    uniq = np.unique(games)
    val_games = set(np.random.default_rng(0).choice(uniq, size=max(1, len(uniq) // 10), replace=False).tolist())

    # Pick check samples: validation games, teacher (non-random) moves only.
    pool = []
    offset = 0
    for ri, (recs, b, c, t) in enumerate(per_run):
        g = games[offset:offset + len(t)]
        ok = np.array([x in val_games for x in g]) & (recs['move_random'][t] == 0)
        pool += [(ri, j) for j in np.flatnonzero(ok)]
        offset += len(t)
    rng = np.random.default_rng(1)
    picks = [pool[k] for k in rng.choice(len(pool), size=min(args.checks, len(pool)), replace=False)]

    checks = []
    for ri, j in picks:
        recs, b, c, t = per_run[ri]
        r = recs[t[j]]
        with torch.no_grad():
            out = model(torch.from_numpy(b[j:j + 1]).float(), torch.from_numpy(c[j:j + 1])).item()
        bits = (b[j].astype(np.uint16) << np.arange(10, dtype=np.uint16)).sum(axis=1)
        checks.append({
            'board': r['board'].tolist(), 'active': int(r['active']), 'hold': int(r['hold']),
            'queue': r['queue'].tolist(), 'bag_mask': int(r['bag_mask']),
            'combo': int(r['combo']), 'b2b': int(r['b2b']),
            'move': [int(r['move_rot']), int(r['move_x']), int(r['move_y']), int(r['move_hold'])],
            'lines': int(r['lines']), 'attack': int(r['attack']), 'garbage_in': int(r['garbage_in']),
            'after': bits.tolist(), 'ctx': c[j].tolist(),
            'value': out * ckpt['label_std'] + ckpt['label_mean'],
        })

    weights = {k: {'shape': list(v.shape), 'data': v.flatten().tolist()} for k, v in ckpt['state_dict'].items()}
    with open(args.out, 'w') as f:
        json.dump({'config': ckpt['config'], 'label_mean': ckpt['label_mean'], 'label_std': ckpt['label_std'],
                   'weights': weights, 'checks': checks}, f)
    print(f'{args.out}: {sum(np.prod(w["shape"]) for w in weights.values()):,} weights, {len(checks)} checks')


def export_app(ckpt, out_dir):
    tensors, offset, blobs = [], 0, []
    for name, v in ckpt['state_dict'].items():
        data = v.detach().cpu().numpy().astype('<f4').ravel()
        tensors.append({'name': name, 'shape': list(v.shape), 'offset': offset})
        offset += data.size
        blobs.append(data)
    os.makedirs(out_dir, exist_ok=True)
    np.concatenate(blobs).tofile(os.path.join(out_dir, 'value_net.bin'))
    with open(os.path.join(out_dir, 'value_net.json'), 'w') as f:
        json.dump({'config': ckpt['config'], 'label_mean': ckpt['label_mean'], 'label_std': ckpt['label_std'],
                   'tensors': tensors}, f, indent=1)
    print(f'{out_dir}/value_net.json + value_net.bin: {offset:,} weights ({offset * 4 / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
