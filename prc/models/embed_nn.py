"""Entity-embedding network for the bulk regressor. CPU torch.

WHY THIS SHAPE

The dataset is 2.08M rows wide by 72 features, of which 8 are categorical and
two of those are high cardinality (STAND_mvt 1,899, AIRCRAFT_OPERATOR_flt 676).
That is the classic entity-embedding setting: a learned dense vector per level,
concatenated with the continuous block and fed to a small MLP. 2.08M rows is
ample for it and the whole thing is ~600 MB as float32, so no data generator is
needed -- the tensors live in memory and batches are taken by slicing a
permutation.

The reason to bother is not that it should beat LightGBM. It is that it gets
things wrong in a DIFFERENT WAY: a smooth global function of the inputs rather
than an axis-aligned partition. Decorrelated residuals at similar accuracy are
worth more in a blend than a small solo win.

THREE CHOICES THAT ACTUALLY MATTER HERE, all learned the hard way:

1. ROBUST SCALING, NOT Z-SCORES. `off_sched` ranges over +-130,000 s because a
   handful of rows have an absurd schedule offset. Standardising by the standard
   deviation puts 99.9% of the mass inside a pinhead near zero and the first
   layer cannot resolve it. Median/IQR with a +-8 clip keeps the bulk spread
   out and still marks the outliers as far away. (altfam.numeric_design)

2. UNSEEN LEVELS GO TO SLOT 0. Codes are built from the whole-file vocabulary,
   so a stand that appears only in the held-out month would otherwise carry a
   randomly-initialised embedding that never received a gradient -- noise
   injected exactly where the model is least sure. `remap_unseen` folds every
   level absent from the fold's training rows into the missing slot, which is
   what LightGBM effectively does with an unseen category.

3. STANDARDISED TARGET. Taxi-out is ~1,000 +- 450 s. Regressing it raw makes
   the initial loss ~1e6 and the first gradient step enormous; the network
   spends its budget fixing the output bias. Fit on (y - mu) / sigma and undo it
   at predict time.
"""
import numpy as np
import torch
import torch.nn as nn


def emb_dim(card: int) -> int:
    """Embedding width for a categorical with `card` levels.

    The usual 1.6 * n^0.56 rule, floored at 3 and capped at 32. The cap matters:
    uncapped it asks for 110 dimensions for STAND_mvt, which is more parameters
    in one table than in the rest of the network and overfits 1,899 levels whose
    median frequency is about 500 rows.
    """
    return int(min(32, max(3, round(1.6 * card ** 0.56))))


def remap_unseen(codes: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Send codes never seen in the fold's training rows to slot 0 (missing)."""
    out = codes.copy()
    for j in range(codes.shape[1]):
        seen = np.zeros(int(codes[:, j].max()) + 1, bool)
        seen[np.unique(codes[train_mask, j])] = True
        bad = ~seen[out[:, j]]
        out[bad, j] = 0
    return out


class EmbedMLP(nn.Module):
    def __init__(self, cards, n_num, hidden=(512, 256, 128), dropout=0.10):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cards])
        for e in self.embs:
            nn.init.normal_(e.weight, 0.0, 0.05)
            with torch.no_grad():
                e.weight[0].zero_()          # the missing/unseen slot starts neutral
        d_in = sum(emb_dim(c) for c in cards) + n_num
        layers, d = [nn.BatchNorm1d(d_in)], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, c, z):
        e = [emb(c[:, j]) for j, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [z], dim=1)).squeeze(1)


def resolve_device(device=None):
    """'cuda' if asked for and available, else 'cpu'.

    On the cluster GPU node a job submitted WITHOUT `--gres=gpu:N` sees no
    devices at all -- the cluster confines them per cgroup -- so
    `torch.cuda.is_available()` returns False and this silently falls back to
    CPU, which would be a very slow surprise. The caller is told which device
    was chosen so that mistake is visible in the log rather than inferred from
    the wall time.
    """
    if device in (None, 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'cuda requested but torch.cuda.is_available() is False. On the '
            'cluster this almost always means the job was submitted to -p gpu '
            'without --gres=gpu:N, so the cgroup exposes no devices.')
    return torch.device(device)


def fit_predict(codes, Z, y, train_mask, pred_mask, *, epochs=8, batch=4096,
                lr=3e-3, weight_decay=1e-5, hidden=(512, 256, 128), dropout=0.10,
                seed=0, threads=4, log=print, eval_hook=None, device=None):
    """Train on `train_mask`, return predictions for `pred_mask` (seconds).

    `eval_hook(epoch, pred)` is called after each epoch with the current
    predictions for `pred_mask`, so a caller can watch the learning curve
    without a second training run.

    `device` is 'cpu', 'cuda', or None/'auto' to pick whatever is present.
    """
    dev = resolve_device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    codes = remap_unseen(codes, train_mask)
    cards = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]

    ytr = y[train_mask].astype(np.float32)
    mu, sd = float(ytr.mean()), float(ytr.std()) or 1.0

    # The whole training set lives on the device: 2.08M x ~72 float32 is under
    # 700 MB, trivial against an L40S's 46 GB, and it removes the per-batch
    # host-to-device copy that would otherwise dominate at this batch size.
    Ctr = torch.from_numpy(np.ascontiguousarray(codes[train_mask])).long().to(dev)
    Ztr = torch.from_numpy(np.ascontiguousarray(Z[train_mask])).to(dev)
    Ttr = torch.from_numpy((ytr - mu) / sd).to(dev)
    Cpr = torch.from_numpy(np.ascontiguousarray(codes[pred_mask])).long().to(dev)
    Zpr = torch.from_numpy(np.ascontiguousarray(Z[pred_mask])).to(dev)

    net = EmbedMLP(cards, Z.shape[1], hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = Ctr.shape[0]
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.25)
    log(f'    net: {net.n_params:,} params, cards {cards}, {Z.shape[1]} numeric, '
        f'{n:,} train rows, {steps:,} steps, device={dev}')

    lossf = nn.MSELoss()
    # The permutation generator stays on CPU; indices are moved per batch, which
    # is a few thousand int64 and not worth keeping a device generator for.
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch].to(dev)
            if idx.numel() < 2:          # BatchNorm needs >1 row
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Ctr[idx], Ztr[idx]), Ttr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * idx.numel()
        if eval_hook is not None:
            eval_hook(ep + 1, _predict(net, Cpr, Zpr, mu, sd, batch))
        else:
            log(f'    epoch {ep + 1}/{epochs}  train MSE(std) {tot / n:.5f}')
    return _predict(net, Cpr, Zpr, mu, sd, batch)


def _predict(net, C, Z, mu, sd, batch=16384):
    net.eval()
    out = np.empty(C.shape[0], dtype=np.float64)
    with torch.no_grad():
        for i in range(0, C.shape[0], batch):
            out[i:i + batch] = net(C[i:i + batch], Z[i:i + batch]).numpy()
    return np.clip(out * sd + mu, 0, None)
