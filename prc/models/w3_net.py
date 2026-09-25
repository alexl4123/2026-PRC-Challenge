"""W3: a second, structurally different network for the bulk regressor -- for ensemble diversity.

The 22-arm sweep of 2026-09-23 found the residual correlation between the entity-embedding MLP
(prc/models/embed_nn.py) and LightGBM stuck at 0.91-0.95 whatever was tuned WITHIN that
architecture. A blend gains from members that are wrong in different ways, so this network changes
the parts the sweep could not:

1. NUMERIC INPUTS BY POSITION IN THEIR OWN DISTRIBUTION. Each numeric feature is cut into (up to)
   16 quantile bins from the fold's training rows and piecewise-linearly encoded (PLE; Gorishniy,
   Rubachev & Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS
   2022): bin t is 0 below it, 1 above it and linear inside. A missing value gets its own
   indicator instead of the median, so "absent" is a separate region (embed_nn median-fills,
   which exp47 measured as the wrong choice for off_aobt3). Each feature then has its own small
   linear projection (17 -> 8), so the network learns a per-feature non-linear response before
   any interaction.
2. RESIDUAL BLOCKS instead of a plain funnel (the tabular ResNet of Gorishniy et al., 2021).
3. TWO OUTPUT HEADS: "clock" airports, where the NM record sees most of the delay band (EHAM,
   EGLL, LFPG, EDDF), and the six NM-blind airports (strategy report, section 3 Q3).

Unchanged from embed_nn, on purpose: the target standardisation, unseen categories folded into
slot 0, AdamW + OneCycle, MSE on the bulk rows (y <= 7200), whole training set on the device.
"""
import numpy as np
import torch
import torch.nn as nn

from .embed_nn import emb_dim, remap_unseen, resolve_device

CLOCK_AIRPORTS = ('EHAM', 'EGLL', 'LFPG', 'EDDF')


def design(X, cols, cats):
    """(codes int32 [n, n_cat] with 0 = missing, raw numerics float32 [n, n_num] with NaN, num_cols)."""
    cat_cols = [c for c in cols if c in cats]
    num_cols = [c for c in cols if c not in cats]
    codes = np.empty((len(X), len(cat_cols)), dtype=np.int32)
    for j, c in enumerate(cat_cols):
        codes[:, j] = X[c].cat.codes.to_numpy() + 1
    num = np.empty((len(X), len(num_cols)), dtype=np.float32)
    for j, c in enumerate(num_cols):
        num[:, j] = X[c].to_numpy(dtype=np.float32, na_value=np.nan)
    return codes, num, num_cols


def head_index(apt):
    return np.isin(np.asarray(apt).astype(str), CLOCK_AIRPORTS).astype(np.int64)


def ple_edges(num, fit_mask, n_bins=16):
    """(F, n_bins + 1) bin edges from the fit rows' quantiles. Features with fewer distinct values
    get repeated edges (zero-width bins, guarded in the encoder); all-NaN features get [0, 1]."""
    F = num.shape[1]
    edges = np.zeros((F, n_bins + 1), dtype=np.float32)
    q = np.linspace(0, 1, n_bins + 1)
    for j in range(F):
        v = num[fit_mask, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            edges[j] = np.linspace(0, 1, n_bins + 1)
            continue
        e = np.unique(np.quantile(v, q))
        if e.size == 1:
            e = np.array([e[0], e[0] + 1.0])
        edges[j] = np.concatenate([e, np.full(n_bins + 1 - e.size, e[-1])])
    return edges


class PLE(nn.Module):
    """Piecewise-linear encoding + missing indicator + a per-feature linear map to d dims."""
    def __init__(self, edges, d=8):
        super().__init__()
        e = torch.as_tensor(edges)
        self.register_buffer('left', e[:, :-1])
        self.register_buffer('width', torch.clamp(e[:, 1:] - e[:, :-1], min=1e-6))
        F, T = self.left.shape
        self.w = nn.Parameter(torch.randn(F, T + 1, d) * (1.0 / np.sqrt(T + 1)))
        self.b = nn.Parameter(torch.zeros(F, d))
        self.out_dim = F * d

    def forward(self, x):
        miss = torch.isnan(x)
        xz = torch.where(miss, torch.zeros_like(x), x)
        enc = torch.clamp((xz.unsqueeze(-1) - self.left) / self.width, 0.0, 1.0)
        enc = enc * (~miss).unsqueeze(-1)
        enc = torch.cat([enc, miss.unsqueeze(-1).to(enc.dtype)], dim=-1)            # (B, F, T+1)
        h = torch.einsum('bft,ftd->bfd', enc, self.w) + self.b
        return torch.relu(h).flatten(1)


class Block(nn.Module):
    def __init__(self, d, mult=2, dropout=0.10):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, d * mult), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(d * mult, d), nn.Dropout(dropout / 2))

    def forward(self, x):
        return x + self.f(x)


class ResPLENet(nn.Module):
    def __init__(self, cards, edges, d_num=8, width=384, n_blocks=3, dropout=0.10, emb_cap=16, n_heads=2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c, emb_cap)) for c in cards])
        for e in self.embs:
            nn.init.normal_(e.weight, 0.0, 0.05)
            with torch.no_grad():
                e.weight[0].zero_()
        self.ple = PLE(edges, d_num)
        d_in = sum(emb_dim(c, emb_cap) for c in cards) + self.ple.out_dim
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[Block(width, 2, dropout) for _ in range(n_blocks)])
        self.norm = nn.BatchNorm1d(width)
        self.heads = nn.Linear(width, n_heads)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, c, x, h):
        z = torch.cat([emb(c[:, j]) for j, emb in enumerate(self.embs)] + [self.ple(x)], dim=1)
        z = torch.relu(self.norm(self.blocks(self.inp(z))))
        return self.heads(z).gather(1, h.unsqueeze(1)).squeeze(1)


def fit_predict(codes, num, heads, y, train_mask, pred_mask, *, epochs=12, batch=8192, lr=2e-3,
                weight_decay=1e-5, width=384, n_blocks=3, dropout=0.10, n_bins=16, d_num=8, emb_cap=16,
                seed=0, threads=4, log=print, device=None):
    """Train on `train_mask`, return predictions (seconds) for `pred_mask`."""
    dev = resolve_device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    codes = remap_unseen(codes, train_mask)
    cards = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
    edges = ple_edges(num, train_mask, n_bins)
    ytr = y[train_mask].astype(np.float32)
    mu, sd = float(ytr.mean()), float(ytr.std()) or 1.0

    t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a)).to(dev) if dt is None else \
        torch.from_numpy(np.ascontiguousarray(a)).to(dt).to(dev)
    Ctr, Xtr, Htr = t(codes[train_mask], torch.long), t(num[train_mask]), t(heads[train_mask], torch.long)
    Ttr = t((ytr - mu) / sd)
    Cpr, Xpr, Hpr = t(codes[pred_mask], torch.long), t(num[pred_mask]), t(heads[pred_mask], torch.long)

    net = ResPLENet(cards, edges, d_num=d_num, width=width, n_blocks=n_blocks, dropout=dropout,
                    emb_cap=emb_cap).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = Ctr.shape[0]
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    log(f'    w3 net: {net.n_params:,} params, cards {cards}, {num.shape[1]} numeric x {n_bins} bins, '
        f'heads clock/blind {int(Htr.sum())}/{int(n - Htr.sum())}, {n:,} train rows, {steps:,} steps, device={dev}')
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch].to(dev)
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Ctr[idx], Xtr[idx], Htr[idx]), Ttr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * idx.numel()
        log(f'    epoch {ep + 1}/{epochs}  train MSE(std) {tot / n:.5f}')
    net.eval()
    out = np.empty(Cpr.shape[0], dtype=np.float64)
    with torch.no_grad():
        for i in range(0, Cpr.shape[0], 16384):
            out[i:i + 16384] = net(Cpr[i:i + 16384], Xpr[i:i + 16384], Hpr[i:i + 16384]).detach().cpu().numpy()
    return np.clip(out * sd + mu, 0, None)
