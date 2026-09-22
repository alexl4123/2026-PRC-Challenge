"""A parameterised copy of prc/models/embed_nn.py, for the hyperparameter sweep.

WHY A COPY RATHER THAN EXTRA KEYWORD ARGUMENTS ON embed_nn.fit_predict

prc/models/embed_nn.py is the network that produced the 286.66 reference, and
cluster/nn_task.py -- which may not be modified -- calls it. Adding sweep knobs
there would put the reference path and the experimental path in the same
function, where a default that quietly changes meaning breaks a number we have
already reported. So the sweep gets its own module, and pays for that with one
obligation:

    *** THE INVARIANT: at the defaults in nnsweep_configs.BASE, fit_sweep()
        must be BIT-FOR-BIT identical to embed_nn.fit_predict(). ***

That is not a comment, it is a test: cluster/nnsweep_smoke.py --check-identity
trains both on the same subsample with the same seed and asserts the predictions
are exactly equal. If a future edit breaks it, the `base` arm stops being a
control and every paired comparison in the sweep becomes meaningless, so run it
after touching anything in here.

Keeping the invariant means the ORDER OF CONSTRUCTION matters as much as the
arithmetic. Every nn.Embedding and nn.Linear consumes the global torch RNG when
it initialises, so building the same modules in a different order gives
different (equally valid, but different) initial weights. SweepMLP therefore
builds exactly what EmbedMLP builds, in exactly the same sequence.

WHAT IS PARAMETERISED BEYOND embed_nn
  emb_cap / emb_mult / emb_pow / emb_floor   the embedding-width rule
  schedule / pct_start                       'onecycle' or 'cosine'
  loss / huber_delta                         'mse' or 'huber'
  epochs / batch / lr / weight_decay / hidden / dropout   as before
`missing` is not here: it belongs to altfam.numeric_design, which the prep stage
applies before the arrays ever reach this module.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from prc.models.embed_nn import _predict, remap_unseen, resolve_device


def emb_dims(cards, cap=32, mult=1.6, pow_=0.56, floor=3):
    """Embedding width per table.

    The shipped rule is min(32, max(3, round(1.6 * card**0.56))). The cap is a
    guess and binds on exactly two tables: STAND_mvt (1,899 levels -> 110) and
    AIRCRAFT_OPERATOR_flt (676 -> 61). Everything else is already under 32, so
    moving the cap changes those two tables and nothing else.
    """
    return [int(min(cap, max(floor, round(mult * c ** pow_)))) for c in cards]


class SweepMLP(nn.Module):
    """EmbedMLP with the embedding widths supplied rather than computed.

    Construction order is copied from prc.models.embed_nn.EmbedMLP so that, given
    the same widths, the same seed produces the same initial weights.
    """

    def __init__(self, cards, dims, n_num, hidden=(512, 256, 128), dropout=0.10):
        super().__init__()
        if len(dims) != len(cards):
            raise ValueError(f'{len(dims)} embedding widths for {len(cards)} tables')
        self.dims = list(dims)
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        for e in self.embs:
            nn.init.normal_(e.weight, 0.0, 0.05)
            with torch.no_grad():
                e.weight[0].zero_()          # the missing/unseen slot starts neutral
        d_in = sum(dims) + n_num
        layers, d = [nn.BatchNorm1d(d_in)], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())
        self.n_emb_params = sum(p.numel() for p in self.embs.parameters())

    def forward(self, c, z):
        e = [emb(c[:, j]) for j, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [z], dim=1)).squeeze(1)


def _make_sched(opt, schedule, lr, steps, pct_start):
    """OneCycle (the shipped schedule) or warmup+cosine.

    The OneCycle branch is constructed with exactly the arguments embed_nn uses,
    so at pct_start=0.25 it is the same object with the same state.

    The cosine branch exists because OneCycleLR does two things at once: it
    warms up AND it anneals to max_lr/25/1e4, i.e. to zero for all practical
    purposes. A plain warmup-then-cosine separates "does the network want a long
    warmup" from "does it want to finish at a dead stop".
    """
    if schedule == 'onecycle':
        # OneCycleLR computes its warmup phase as `float(pct_start*total_steps) - 1`
        # and then divides by (end_step - start_step). When pct_start*steps <= 1
        # that denominator is 0 and the FIRST optimiser step dies with a bare
        # `ZeroDivisionError: float division by zero`, which says nothing about
        # the cause. Only reachable at small scale -- the smallest stage-A arm
        # (b32k) gets ~175 warmup steps on the real folds -- but a subsampled
        # smoke run walks straight into it, so name the cause here.
        if int(pct_start * steps) < 2:
            raise ValueError(
                f'OneCycleLR degenerates: pct_start={pct_start} x total_steps='
                f'{steps} gives fewer than 2 warmup steps. total_steps = epochs x '
                f'ceil(n_train/batch), so raise epochs, lower batch, or use more '
                f'training rows.')
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=steps, pct_start=pct_start)
    if schedule == 'cosine':
        warm = max(1, int(round(pct_start * steps)))

        def f(step):
            if step < warm:
                return (step + 1) / warm
            t = (step - warm) / max(1, steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(opt, f)
    raise ValueError(f'unknown schedule {schedule!r}')


def _make_loss(loss, huber_delta):
    if loss == 'mse':
        return nn.MSELoss()
    if loss == 'huber':
        # On the STANDARDISED target, so delta=1.0 is about one sd, ~450 s.
        return nn.HuberLoss(delta=float(huber_delta))
    raise ValueError(f'unknown loss {loss!r}')


def fit_sweep(codes, Z, y, train_mask, pred_mask, *, epochs=12, batch=8192,
              lr=3e-3, weight_decay=1e-5, hidden=(512, 256, 128), dropout=0.10,
              emb_cap=32, emb_mult=1.6, emb_pow=0.56, emb_floor=3,
              schedule='onecycle', pct_start=0.25, loss='mse', huber_delta=1.0,
              seed=0, threads=4, log=print, eval_hook=None, device=None,
              already_remapped=False):
    """Train on `train_mask`, return predictions for `pred_mask`, in seconds.

    `already_remapped` skips remap_unseen when the caller has pre-applied it
    (the prep stage does, and caches the result). remap_unseen is idempotent, so
    passing False on pre-remapped codes is merely wasteful, never wrong.

    `eval_hook(epoch, pred)` is called after every epoch with the CURRENT
    predictions for `pred_mask`. The sweep uses it to record the learning curve.
    IT IS A DIAGNOSTIC, NOT A SELECTOR: the reported prediction is always the
    final epoch's, because choosing the best epoch on the held-out fold is
    selection on the thing being measured.
    """
    dev = resolve_device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not already_remapped:
        codes = remap_unseen(codes, train_mask)
    cards = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
    dims = emb_dims(cards, cap=emb_cap, mult=emb_mult, pow_=emb_pow, floor=emb_floor)

    ytr = y[train_mask].astype(np.float32)
    mu, sd = float(ytr.mean()), float(ytr.std()) or 1.0

    Ctr = torch.from_numpy(np.ascontiguousarray(codes[train_mask])).long().to(dev)
    Ztr = torch.from_numpy(np.ascontiguousarray(Z[train_mask])).to(dev)
    Ttr = torch.from_numpy((ytr - mu) / sd).to(dev)
    Cpr = torch.from_numpy(np.ascontiguousarray(codes[pred_mask])).long().to(dev)
    Zpr = torch.from_numpy(np.ascontiguousarray(Z[pred_mask])).to(dev)

    net = SweepMLP(cards, dims, Z.shape[1], hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = Ctr.shape[0]
    steps = epochs * ((n + batch - 1) // batch)
    sched = _make_sched(opt, schedule, lr, steps, pct_start)
    log(f'    net: {net.n_params:,} params ({net.n_emb_params:,} in embeddings), '
        f'cards {cards}, dims {dims}, {Z.shape[1]} numeric, {n:,} train rows, '
        f'{steps:,} steps, device={dev}')

    lossf = _make_loss(loss, huber_delta)
    g = torch.Generator().manual_seed(seed)
    curve = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch].to(dev)
            if idx.numel() < 2:          # BatchNorm needs >1 row
                continue
            opt.zero_grad(set_to_none=True)
            l = lossf(net(Ctr[idx], Ztr[idx]), Ttr[idx])
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(l.detach()) * idx.numel()
        curve.append(tot / n)
        if eval_hook is not None:
            eval_hook(ep + 1, tot / n, _predict(net, Cpr, Zpr, mu, sd, batch))
        else:
            log(f'    epoch {ep + 1}/{epochs}  train loss(std) {tot / n:.5f}')
    pred = _predict(net, Cpr, Zpr, mu, sd, batch)
    return pred, {'n_params': int(net.n_params), 'n_emb_params': int(net.n_emb_params),
                  'cards': cards, 'dims': dims, 'steps': int(steps),
                  'train_loss': curve, 'mu': mu, 'sd': sd}
