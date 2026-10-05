"""
Train the queue-aware value net on self-play runs.

Splits by game (never by position), so validation boards come from games the
net has not seen. Reports, besides loss:
  - R² against predicting the mean (does the net explain anything?)
  - context ablation: validation loss with the piece context shuffled between
    samples. A big jump means the net really uses the upcoming pieces.

Usage:
    python train_value.py data/runs/teacher-v1 [more runs...]
        [--epochs 20] [--patience 4] [--horizon 12] [--channels 32]
        [--device auto|cpu|mps|cuda] [--out models/value_v1.pt]
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from selfplay_data import load_run
from value_data import build_samples
from value_net import ValueNet


def load_samples(run_dirs, horizon, gamma, death_penalty):
    boards, ctxs, labels, games = [], [], [], []
    for i, d in enumerate(run_dirs):
        recs, _ = load_run(d)
        b, c, l, g = build_samples(recs, horizon, gamma, death_penalty)
        boards.append(b); ctxs.append(c); labels.append(l)
        games.append(g + i * 10_000_000)  # keep game ids distinct across runs
        print(f'{d}: {len(recs)} records → {len(l)} samples')
    return (np.concatenate(boards), np.concatenate(ctxs),
            np.concatenate(labels), np.concatenate(games))


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():  # Apple GPU; needs a recent macOS
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, shuffle_context=False):
    model.eval()
    total, n, preds, targets = 0.0, 0, [], []
    gen = torch.Generator().manual_seed(0)
    for b, c, y in loader:
        if shuffle_context:
            c = c[torch.randperm(len(c), generator=gen)]
        b, c, y = b.to(device), c.to(device), y.to(device)
        p = model(b, c)
        total += loss_fn(p, y).item() * len(y)
        n += len(y)
        preds.append(p); targets.append(y)
    p, y = torch.cat(preds), torch.cat(targets)
    r2 = 1 - ((p - y) ** 2).mean() / y.var()
    return total / n, r2.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--gamma', type=float, default=0.97)
    ap.add_argument('--death-penalty', type=float, default=10.0)
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--squeeze', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--train-fraction', type=float, default=1.0,
                    help='train on this fraction of the training games; validation games stay the same')
    ap.add_argument('--patience', type=int, default=4,
                    help='stop after this many epochs without a better validation loss')
    ap.add_argument('--out', default='models/value_v1.pt')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f'device: {device}')
    torch.manual_seed(0)

    boards, ctx, labels, games = load_samples(args.runs, args.horizon, args.gamma, args.death_penalty)
    mean, std = float(labels.mean()), float(labels.std() + 1e-6)
    print(f'{len(labels)} samples, label mean {mean:.3f} std {std:.3f}, '
          f'{(labels < 0).mean():.1%} include a death')

    # Split by game.
    uniq = np.unique(games)
    rng = np.random.default_rng(0)
    val_games = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False).tolist())
    is_val = np.array([g in val_games for g in games])
    is_train = ~is_val
    if args.train_fraction < 1.0:
        # Subsample whole games, after the split, so runs with different
        # fractions are scored on identical validation games.
        train_games = np.array(sorted(set(uniq.tolist()) - val_games))
        keep = set(np.random.default_rng(1).choice(
            train_games, size=max(1, int(len(train_games) * args.train_fraction)), replace=False).tolist())
        is_train &= np.array([g in keep for g in games])

    def make_ds(mask):
        return TensorDataset(torch.from_numpy(boards[mask]).float(),
                             torch.from_numpy(ctx[mask]),
                             torch.from_numpy((labels[mask] - mean) / std))
    train_dl = DataLoader(make_ds(is_train), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(make_ds(is_val), batch_size=args.batch_size * 4)
    n_train_games = len(np.unique(games[is_train]))
    print(f'train {int(is_train.sum())} / val {int(is_val.sum())} samples '
          f'({n_train_games} / {len(val_games)} games)')

    model = ValueNet(args.channels, args.squeeze, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: {n_params:,} params, {model.macs_per_board():,} MACs/board')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()
    best, best_epoch = float('inf'), 0
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b, c, y in train_dl:
            b, c, y = b.to(device), c.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(b, c), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y); n += len(y)
        sched.step()
        val_loss, r2 = evaluate(model, val_dl, loss_fn, device)
        flag = ''
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            # Save CPU tensors so the checkpoint loads on any machine.
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({'state_dict': state, 'config': model.config,
                        'label_mean': mean, 'label_std': std, 'args': vars(args)}, args.out)
            flag = ' *'
        print(f'epoch {epoch:3d}  train {total / n:.4f}  val {val_loss:.4f}  R² {r2:.3f}  '
              f'({time.time() - t0:.0f}s){flag}')
        if epoch - best_epoch >= args.patience:
            print(f'no improvement for {args.patience} epochs; stopping')
            break

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    val_loss, r2 = evaluate(model, val_dl, loss_fn, device)
    shuf_loss, shuf_r2 = evaluate(model, val_dl, loss_fn, device, shuffle_context=True)
    print(f'\nbest checkpoint → {args.out}')
    print(f'  val loss {val_loss:.4f}, R² {r2:.3f}')
    print(f'  with piece context shuffled: loss {shuf_loss:.4f}, R² {shuf_r2:.3f} '
          f'(bigger drop = net relies more on upcoming pieces)')


if __name__ == '__main__':
    main()
