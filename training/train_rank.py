"""
Train the value net to rank the teacher's move first among all placements.

train_value.py fits the net only on afterstates the teacher chose, so it never
sees a bad placement; a 1-ply search over ~70 candidates then finds boards it
misjudges (it lost 0–100 to hard mode). This trainer scores every candidate
the way the bot does, attack + value, and adds a softmax ranking loss whose
target is the teacher's move. The value loss on the teacher's candidate stays,
so the output keeps meaning "future attack" and adding the immediate attack to
it stays sensible.

    loss = SmoothL1(value of teacher's candidate, label) + rank_weight × CE(scores / temp, teacher)

Candidates come from `npm run candidates -- RUN_DIR` (tetris-web/), read with
rank_data.py. Validation games are the same as train_value.py's, so the
checkpoint can be exported and compared with export_value_ts.py / valuecheck.

Usage:
    .venv/bin/python train_rank.py data/runs/teacher-v2 data/runs/teacher-v2b \\
        --channels 16 --out models/rank_v2_c16.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rank_data import gather, load_candidates
from selfplay_data import boards_to_grid, load_run
from train_value import pick_device
from value_data import build_samples, encode_context
from value_net import ValueNet


def load(run_dirs, horizon, gamma, death_penalty):
    """Decisions and candidates of all runs, with each decision's label (nan if none)."""
    parts, cands, sample_games, sample_labels = [], [], [], []
    cand_offset = 0
    for i, d in enumerate(run_dirs):
        recs, _ = load_run(d)
        _, _, lab, g, t = build_samples(recs, horizon, gamma, death_penalty, return_index=True)
        sample_games.append(g + i * 10_000_000)
        sample_labels.append(lab)
        full = np.full(len(recs), np.nan, dtype=np.float32)
        full[t] = lab

        pos, cand = load_candidates(d)
        rec_key = recs['game_id'].astype(np.int64) * 65536 + recs['ply']
        pos_key = pos['game_id'].astype(np.int64) * 65536 + pos['ply']
        ridx = np.searchsorted(rec_key, pos_key)
        assert (rec_key[ridx] == pos_key).all(), f'{d}: candidates do not match the records'
        parts.append(dict(
            game=pos['game_id'].astype(np.int64) + i * 10_000_000,
            first=pos['first'].astype(np.int64) + cand_offset, count=pos['count'].astype(np.int64),
            teacher=pos['teacher'].astype(np.int64), label=full[ridx],
            random=recs['move_random'][ridx] == 1, bag=recs['bag_mask'][ridx].astype(np.int64)))
        cands.append(cand)
        cand_offset += len(cand)
        print(f'{d}: {len(pos)} decisions, {len(cand)} candidates, {np.isfinite(full[ridx]).sum()} with labels')
    pos = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return pos, np.concatenate(cands), np.concatenate(sample_games), np.concatenate(sample_labels)


class Batcher:
    def __init__(self, pos, cand, mean, std, device):
        self.pos, self.cand, self.mean, self.std, self.device = pos, cand, mean, std, device

    def __call__(self, sel):
        p = self.pos
        idx, row = gather({'first': p['first'], 'count': p['count']}, sel)
        c = self.cand[idx]
        within = np.arange(len(idx)) - np.repeat(np.cumsum(p['count'][sel]) - p['count'][sel], p['count'][sel])
        ctx = encode_context(c['next'].astype(np.int64), c['queue4'].astype(np.int64), c['hold'].astype(np.int64),
                             p['bag'][sel][row], c['combo'], c['b2b'].astype(np.float32),
                             c['landing'].astype(np.float32))
        teacher = p['teacher'][sel]
        is_teacher = within == teacher[row]
        dev = self.device
        return dict(
            boards=torch.from_numpy(boards_to_grid(c['board'])).to(dev).float(),
            ctx=torch.from_numpy(ctx).to(dev),
            attack=torch.from_numpy(c['attack'].astype(np.float32)).to(dev),
            # Moves that top out are never played; keep the teacher's in any case.
            playable=torch.from_numpy((c['dies'] == 0) | is_teacher).to(dev),
            row=torch.from_numpy(row).to(dev), within=torch.from_numpy(within).to(dev),
            k=int(p['count'][sel].max()), n=len(sel),
            teacher=torch.from_numpy(teacher).to(dev),
            rank_rows=torch.from_numpy(~p['random'][sel] & (teacher >= 0)).to(dev),
            teacher_flat=torch.from_numpy(np.flatnonzero(is_teacher)).to(dev),
            label_rows=torch.from_numpy(np.isfinite(p['label'][sel][row[is_teacher]])).to(dev),
            label=torch.from_numpy(((p['label'][sel] - self.mean) / self.std).astype(np.float32)).to(dev),
        )


def forward(model, b, mean, std, temp):
    """Return (value loss, rank loss, value preds/targets, padded scores) for one batch."""
    v = model(b['boards'], b['ctx'])
    scores = b['attack'] + v * std + mean
    pad = torch.full((b['n'], b['k']), float('-inf'), device=v.device)
    pad[b['row'], b['within']] = torch.where(b['playable'], scores, torch.full_like(scores, float('-inf')))

    # Value: the teacher's candidate is the value sample (same board and context).
    tv = v[b['teacher_flat']]
    # teacher_flat is in row order; rows without a found teacher move have no entry.
    has_teacher = b['teacher'] >= 0
    labels = b['label'][has_teacher]
    lr = b['label_rows']
    value_loss = F.smooth_l1_loss(tv[lr], labels[lr]) if lr.any() else v.sum() * 0

    rr = b['rank_rows']
    rank_loss = F.cross_entropy(pad[rr] / temp, b['teacher'][rr]) if rr.any() else v.sum() * 0
    return value_loss, rank_loss, tv[lr], labels[lr], pad


@torch.no_grad()
def evaluate(model, batcher, sel, batch, mean, std, temp):
    model.eval()
    vl = rl = 0.0
    n = 0
    preds, targets = [], []
    top1 = top3 = ranked = 0
    for s in range(0, len(sel), batch):
        part = sel[s:s + batch]
        b = batcher(part)
        v_loss, r_loss, p, y, pad = forward(model, b, mean, std, temp)
        vl += v_loss.item() * len(part); rl += r_loss.item() * len(part); n += len(part)
        preds.append(p); targets.append(y)
        rr = b['rank_rows']
        order = pad[rr].argsort(dim=1, descending=True)
        t = b['teacher'][rr]
        top1 += (order[:, 0] == t).sum().item()
        top3 += (order[:, :3] == t[:, None]).any(dim=1).sum().item()
        ranked += int(rr.sum().item())
    p, y = torch.cat(preds), torch.cat(targets)
    r2 = 1 - ((p - y) ** 2).mean() / y.var()
    return vl / n, rl / n, r2.item(), top1 / ranked, top3 / ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--positions-per-epoch', type=int, default=200_000,
                    help='training decisions sampled per epoch (each brings ~70 candidates)')
    ap.add_argument('--batch', type=int, default=128, help='decisions per batch')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rank-weight', type=float, default=1.0)
    ap.add_argument('--temp', type=float, default=1.0, help='softmax temperature, in attack units')
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--gamma', type=float, default=0.97)
    ap.add_argument('--death-penalty', type=float, default=10.0)
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--squeeze', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--init', default='', help='start from this value-net checkpoint')
    ap.add_argument('--val-positions', type=int, default=20_000)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--out', default='models/rank_v2.pt')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f'device: {device}')
    torch.manual_seed(0)

    pos, cand, sample_games, labels = load(args.runs, args.horizon, args.gamma, args.death_penalty)
    mean, std = float(labels.mean()), float(labels.std() + 1e-6)

    # Same validation games as train_value.py.
    uniq = np.unique(sample_games)
    val_games = set(np.random.default_rng(0).choice(uniq, size=max(1, len(uniq) // 10), replace=False).tolist())
    is_val = np.array([g in val_games for g in pos['game']])
    train_sel = np.flatnonzero(~is_val)
    rng = np.random.default_rng(0)
    val_sel = np.sort(rng.choice(np.flatnonzero(is_val), size=min(args.val_positions, int(is_val.sum())), replace=False))
    print(f'{len(train_sel)} training / {is_val.sum()} validation decisions '
          f'(scoring {len(val_sel)}), label mean {mean:.3f} std {std:.3f}')

    model = ValueNet(args.channels, args.squeeze, args.hidden).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location='cpu')
        assert ckpt['config'] == model.config, f'--init config {ckpt["config"]} != {model.config}'
        model.load_state_dict(ckpt['state_dict'])
        print(f'initialised from {args.init}')
    print(f'model: {sum(p.numel() for p in model.parameters()):,} params')

    batcher = Batcher(pos, cand, mean, std, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    def report(tag, res):
        vl, rl, r2, t1, t3 = res
        return f'{tag} value {vl:.4f} R² {r2:.3f} rank {rl:.3f} top1 {t1:.1%} top3 {t3:.1%}'

    print(report('before training: val', evaluate(model, batcher, val_sel, args.batch, mean, std, args.temp)))
    best, best_epoch = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sel = rng.choice(train_sel, size=min(args.positions_per_epoch, len(train_sel)), replace=False)
        tv = tr = 0.0
        nb = 0
        for s in range(0, len(sel), args.batch):
            b = batcher(np.sort(sel[s:s + args.batch]))
            v_loss, r_loss, *_ = forward(model, b, mean, std, args.temp)
            loss = v_loss + args.rank_weight * r_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tv += v_loss.item(); tr += r_loss.item(); nb += 1
        sched.step()
        res = evaluate(model, batcher, val_sel, args.batch, mean, std, args.temp)
        val_total = res[0] + args.rank_weight * res[1]
        flag = ''
        if val_total < best:
            best, best_epoch = val_total, epoch
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({'state_dict': state, 'config': model.config,
                        'label_mean': mean, 'label_std': std, 'args': vars(args)}, args.out)
            flag = ' *'
        print(f'epoch {epoch:3d}  train value {tv / nb:.4f} rank {tr / nb:.3f} | '
              + report('val', res) + f'  ({time.time() - t0:.0f}s){flag}', flush=True)
        if epoch - best_epoch >= args.patience:
            print(f'no improvement for {args.patience} epochs; stopping')
            break

    print(f'\nbest checkpoint (epoch {best_epoch}) → {args.out}')


if __name__ == '__main__':
    main()
