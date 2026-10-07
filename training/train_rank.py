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

Runs recorded by `duel.ts --record` (games between the value-net bot and another
bot) carry the net's own search results instead: the top moves it expanded and
their deeper scores. Those decisions get no teacher target; instead the net's
1-ply scores over the expanded moves are trained toward the search's (soft
targets, cross-entropy against softmax(search scores / soft_temp)):

    + soft_weight × CE(softmax(search / soft_temp), softmax(scores[expanded] / temp))

Other runs (hard-mode self-play) act as an anchor that keeps the ranking of all
other candidates; --anchor-fraction sets their share of each epoch.

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
        n = len(pos)
        soft = 'soft_idx' in pos.dtype.names and (pos['soft_idx'][:, 0] >= 0).any()
        rec_key = recs['game_id'].astype(np.int64) * 65536 + recs['ply']
        pos_key = pos['game_id'].astype(np.int64) * 65536 + pos['ply']
        ridx = np.searchsorted(rec_key, pos_key)
        assert (rec_key[ridx] == pos_key).all(), f'{d}: candidates do not match the records'
        parts.append(dict(
            game=pos['game_id'].astype(np.int64) + i * 10_000_000,
            first=pos['first'].astype(np.int64) + cand_offset, count=pos['count'].astype(np.int64),
            teacher=pos['teacher'].astype(np.int64), label=full[ridx],
            # Runs with search results train on those, never on the move played.
            hard_target=(recs['move_random'][ridx] == 0) & (pos['teacher'] >= 0) & (not soft),
            soft_idx=pos['soft_idx'].astype(np.int64) if soft else np.full((n, 3), -1, np.int64),
            soft_score=pos['soft_score'].astype(np.float32) if soft else np.zeros((n, 3), np.float32),
            anchor=np.full(n, not soft), bag=recs['bag_mask'][ridx].astype(np.int64)))
        cands.append(cand)
        cand_offset += len(cand)
        kind = f'{(pos["soft_idx"][:, 1] >= 0).sum()} with search targets' if soft else 'teacher targets (anchor)'
        print(f'{d}: {len(pos)} decisions, {len(cand)} candidates, {np.isfinite(full[ridx]).sum()} with labels, {kind}')
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
            rank_rows=torch.from_numpy(p['hard_target'][sel]).to(dev),
            # Search targets need at least two expanded moves to say anything.
            soft_idx=torch.from_numpy(p['soft_idx'][sel]).to(dev),
            soft_score=torch.from_numpy(p['soft_score'][sel]).to(dev),
            soft_rows=torch.from_numpy((p['soft_idx'][sel] >= 0).sum(axis=1) >= 2).to(dev),
            teacher_flat=torch.from_numpy(np.flatnonzero(is_teacher)).to(dev),
            label_rows=torch.from_numpy(np.isfinite(p['label'][sel][row[is_teacher]])).to(dev),
            label=torch.from_numpy(((p['label'][sel] - self.mean) / self.std).astype(np.float32)).to(dev),
        )


def forward(model, b, mean, std, temp, soft_temp):
    """Return (value loss, rank loss, soft loss, value preds/targets, padded scores, soft logits) for one batch."""
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

    # Search targets: the net's scores over the moves the search expanded vs the search's own scores.
    sr = b['soft_rows']
    soft_logits = None
    soft_loss = v.sum() * 0
    if sr.any():
        idx = b['soft_idx'][sr]
        valid = idx >= 0
        rows = torch.arange(len(pad), device=v.device)[sr][:, None].expand_as(idx)
        neg = torch.full(idx.shape, float('-inf'), device=v.device)
        soft_logits = torch.where(valid, pad[rows, idx.clamp(min=0)], neg) / temp
        target = torch.softmax(torch.where(valid, b['soft_score'][sr] / soft_temp, neg), dim=1)
        logp = torch.log_softmax(soft_logits, dim=1)
        soft_loss = -(target * torch.where(valid, logp, torch.zeros_like(logp))).sum(dim=1).mean()
    return value_loss, rank_loss, soft_loss, tv[lr], labels[lr], pad, soft_logits


@torch.no_grad()
def evaluate(model, batcher, sel, batch, mean, std, temp, soft_temp):
    model.eval()
    vl = rl = sl = 0.0
    n = n_rank = n_soft = 0
    preds, targets = [], []
    top1 = top3 = ranked = 0
    soft_agree = soft_n = 0
    for s in range(0, len(sel), batch):
        part = sel[s:s + batch]
        b = batcher(part)
        v_loss, r_loss, s_loss, p, y, pad, soft_logits = forward(model, b, mean, std, temp, soft_temp)
        # Each loss is a mean over the rows it applies to, so weight it by those rows.
        k_rank, k_soft = int(b['rank_rows'].sum()), int(b['soft_rows'].sum())
        vl += v_loss.item() * len(part); rl += r_loss.item() * k_rank; sl += s_loss.item() * k_soft
        n += len(part); n_rank += k_rank; n_soft += k_soft
        if soft_logits is not None:
            # Does the net's 1-ply favourite among the expanded moves match the search's choice?
            sc = b['soft_score'][b['soft_rows']].masked_fill(b['soft_idx'][b['soft_rows']] < 0, float('-inf'))
            soft_agree += (soft_logits.argmax(dim=1) == sc.argmax(dim=1)).sum().item()
            soft_n += len(sc)
        preds.append(p); targets.append(y)
        rr = b['rank_rows']
        order = pad[rr].argsort(dim=1, descending=True)
        t = b['teacher'][rr]
        top1 += (order[:, 0] == t).sum().item()
        top3 += (order[:, :3] == t[:, None]).any(dim=1).sum().item()
        ranked += int(rr.sum().item())
    p, y = torch.cat(preds), torch.cat(targets)
    r2 = 1 - ((p - y) ** 2).mean() / y.var()
    return dict(value=vl / n, rank=rl / max(1, n_rank), soft=sl / max(1, n_soft), r2=r2.item(), top1=top1 / max(1, ranked),
                top3=top3 / max(1, ranked), soft_agree=soft_agree / max(1, soft_n), ranked=ranked, soft_n=soft_n)


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
    ap.add_argument('--soft-weight', type=float, default=1.0, help='weight of the search-target loss')
    ap.add_argument('--soft-temp', type=float, default=1.0, help='temperature of the search-score targets')
    ap.add_argument('--anchor-fraction', type=float, default=0.5,
                    help="share of each epoch's decisions drawn from teacher-target (anchor) runs")
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
    ap.add_argument('--patience', type=int, default=3,
                    help='stop after this many epochs without a better validation loss (0 = never)')
    ap.add_argument('--out', default='models/rank_v2.pt')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f'device: {device}')
    torch.manual_seed(0)

    pos, cand, sample_games, labels = load(args.runs, args.horizon, args.gamma, args.death_penalty)
    mean, std = float(labels.mean()), float(labels.std() + 1e-6)
    init = torch.load(args.init, map_location='cpu') if args.init else None
    if init:
        # Keep the starting net's label scaling: scores are attack + value × std + mean,
        # so new scaling would change how much the value counts against the attack.
        mean, std = init['label_mean'], init['label_std']

    # Same validation games as train_value.py.
    uniq = np.unique(sample_games)
    val_games = set(np.random.default_rng(0).choice(uniq, size=max(1, len(uniq) // 10), replace=False).tolist())
    is_val = np.array([g in val_games for g in pos['game']])
    train_sel = np.flatnonzero(~is_val)
    rng = np.random.default_rng(0)
    val_sel = np.sort(rng.choice(np.flatnonzero(is_val), size=min(args.val_positions, int(is_val.sum())), replace=False))
    print(f'{len(train_sel)} training / {is_val.sum()} validation decisions '
          f'(scoring {len(val_sel)}), label mean {mean:.3f} std {std:.3f}')
    train_anchor = train_sel[pos['anchor'][train_sel]]
    train_new = train_sel[~pos['anchor'][train_sel]]
    if len(train_anchor) and len(train_new):
        n_anchor = int(args.positions_per_epoch * args.anchor_fraction)
        print(f'per epoch: {n_anchor} anchor + {args.positions_per_epoch - n_anchor} new decisions')

    model = ValueNet(args.channels, args.squeeze, args.hidden).to(device)
    if init:
        assert init['config'] == model.config, f'--init config {init["config"]} != {model.config}'
        model.load_state_dict(init['state_dict'])
        print(f'initialised from {args.init}')
    print(f'model: {sum(p.numel() for p in model.parameters()):,} params')

    batcher = Batcher(pos, cand, mean, std, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    last_path = os.path.splitext(args.out)[0] + '.last.pt'

    def report(tag, r):
        out = f'{tag} value {r["value"]:.4f} R² {r["r2"]:.3f}'
        if r['ranked']:
            out += f' rank {r["rank"]:.3f} top1 {r["top1"]:.1%} top3 {r["top3"]:.1%}'
        if r['soft_n']:
            out += f' soft {r["soft"]:.3f} agree {r["soft_agree"]:.1%}'
        return out

    def val_loss(r):
        return r['value'] + args.rank_weight * r['rank'] + args.soft_weight * r['soft']

    def epoch_sample():
        if not (len(train_anchor) and len(train_new)):
            return rng.choice(train_sel, size=min(args.positions_per_epoch, len(train_sel)), replace=False)
        n_anchor = int(args.positions_per_epoch * args.anchor_fraction)
        return np.concatenate([
            rng.choice(train_anchor, size=min(n_anchor, len(train_anchor)), replace=False),
            rng.choice(train_new, size=min(args.positions_per_epoch - n_anchor, len(train_new)), replace=False)])

    evaluate_val = lambda: evaluate(model, batcher, val_sel, args.batch, mean, std, args.temp, args.soft_temp)
    print(report('before training: val', evaluate_val()))
    best, best_epoch = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sel = rng.permutation(epoch_sample())
        tv = tr = ts = 0.0
        nb = 0
        for s in range(0, len(sel), args.batch):
            b = batcher(np.sort(sel[s:s + args.batch]))
            v_loss, r_loss, s_loss, *_ = forward(model, b, mean, std, args.temp, args.soft_temp)
            loss = v_loss + args.rank_weight * r_loss + args.soft_weight * s_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tv += v_loss.item(); tr += r_loss.item(); ts += s_loss.item(); nb += 1
        sched.step()
        res = evaluate_val()
        val_total = val_loss(res)
        flag = ''
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt = {'state_dict': state, 'config': model.config,
                'label_mean': mean, 'label_std': std, 'args': vars(args), 'epoch': epoch}
        torch.save(ckpt, last_path)
        if val_total < best:
            best, best_epoch = val_total, epoch
            torch.save(ckpt, args.out)
            flag = ' *'
        print(f'epoch {epoch:3d}  train value {tv / nb:.4f} rank {tr / nb:.3f} soft {ts / nb:.3f} | '
              + report('val', res) + f'  ({time.time() - t0:.0f}s){flag}', flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            print(f'no improvement for {args.patience} epochs; stopping')
            break

    print(f'\nbest checkpoint (epoch {best_epoch}) → {args.out}; last epoch → {last_path}')


if __name__ == '__main__':
    main()
