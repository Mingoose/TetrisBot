"""
Check whether a trained value net values a T-slot more when a T is coming.

Scores two boards, a T-spin double setup and a similar board with no slot,
under different piece contexts. If the net uses the queue, the slot board's
value should depend on where (or whether) the next T is far more than the
plain board's does.

Usage:
    python probe_value.py models/value_v1.pt
"""

import sys

import numpy as np
import torch

from value_data import NO_PIECE, encode_context
from value_net import ValueNet

I, O, T, S, Z, J, L = range(7)

# Row 0 = top. T-spin double: a T pointing down fits rows 18–19 under the
# overhang at (17, 3) and clears both rows.
TSD = ['..........'] * 16 + [
    'XXX.......',
    'XXXX......',
    'XXX...XXXX',
    'XXXX.XXXXX',
]
# Same cell count, no slot: a plain stack with a right-side well.
FLAT = ['..........'] * 16 + [
    '..........',
    'XXXXX.....',
    'XXXXXXXXX.',
    'XXXXXXXXX.',
]

CONTEXTS = {
    'T is next':            dict(next_piece=T, queue=[S, Z, O, I], hold=NO_PIECE, bag=[I, O, S, Z, J, L]),
    'T in hold':            dict(next_piece=S, queue=[Z, O, I, J], hold=T,        bag=[I, O, S, Z, J, L]),
    'T 4th in queue':       dict(next_piece=S, queue=[Z, O, T, J], hold=NO_PIECE, bag=[I, O, S, Z, J, L]),
    'no T seen, T in bag':  dict(next_piece=S, queue=[Z, O, I, J], hold=NO_PIECE, bag=[T, L]),
    'no T until next bag':  dict(next_piece=S, queue=[Z, O, I, J], hold=NO_PIECE, bag=[L]),
}


def grid(rows):
    return np.array([[c == 'X' for c in r] for r in rows], dtype=np.float32)


def context_vec(c):
    mask = sum(1 << p for p in c['bag'])
    return encode_context(
        next_piece=np.array([c['next_piece']]), queue4=np.array([c['queue']]),
        hold=np.array([c['hold']]), bag_mask=np.array([mask]),
        combo=np.array([-1]), b2b=np.array([-1.0]), garbage_landing=np.array([0.0]))


def main(path):
    ckpt = torch.load(path)
    model = ValueNet(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    mean, std = ckpt['label_mean'], ckpt['label_std']

    boards = {'T-slot': grid(TSD), 'flat': grid(FLAT)}
    assert boards['T-slot'].sum() == boards['flat'].sum()
    print(f'{"context":24s}' + ''.join(f'{b:>10s}' for b in boards) + '   slot − flat')
    for name, c in CONTEXTS.items():
        ctx = torch.from_numpy(context_vec(c))
        vals = []
        with torch.no_grad():
            for b in boards.values():
                vals.append(model(torch.from_numpy(b)[None], ctx).item() * std + mean)
        print(f'{name:24s}' + ''.join(f'{v:10.2f}' for v in vals) + f'   {vals[0] - vals[1]:+.2f}')
    print('\nValues are predicted future attack (≈ garbage lines over the next ~12 pieces).')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'models/value_v1.pt')
