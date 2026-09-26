"""Bet D: a set model over the movements around each departure (the 'seq' member).

WHAT IT IS

For every departure (the target) the context is the K movements nearest in time (take-off / landing
time, MVT_TIME) at the same airport within +-`window` seconds, departures AND arrivals, never crossing
a calendar month (the ranking file holds only Jan and Jul 2026, so a context that crossed a month
boundary in training could not be reproduced there). Each neighbour is one token of LABEL-FREE fields:

  time relative to the target   take-off/landing, AOBT_3, SCHED, EOBT_1, in-block (arrivals), and the
                                same differences against the target's own AOBT_3 / SCHED
  what it is                    phase, wake category, runway (embedding), same runway / stand / pier
  its own clocks                MVT - AOBT_3 (the NM taxi proxy of a departure), MVT - SCHED, MVT - EOBT_1
  arrivals' ground time         in-block - landing (taxi-in: provided data in the ranking file too),
                                in-block - SCHED
  its adsb.lol observation      implied taxi (adsb_off_best = MVT - pushback/rest end) and the other
                                off-block candidates, hold / stop / dwell / speed, coverage flags

NEVER a departure's BLOCK_TIME or TAXITIME: `movement_table` blanks BLOCK_TIME on every departure row
before anything else touches it and `assert_label_free` checks it again, because in the training file
those columns are filled and a silent leak would look like a spectacular model.

The target's own 143 member columns (cluster/w2_features.py) go through a robust-scaled MLP branch
(median / IQR from the fold's fit rows, clipped at +-8, NaN -> 0 PLUS a missing flag -- the review's
point 3 against embed_nn's median fill). The tokens go through a small pre-norm transformer encoder
with a CLS token projected from the target branch, so attention is conditioned on the target. The
head reads [target branch, CLS output]. `use_context=False` drops the context branch entirely: the
same network is then a plain tabular MLP, which is the ablation the bet is judged against.

Target: standardised taxi seconds on the bulk rows (y <= 7200), like embed_nn / w3; `offset` lets a
caller fit a residual instead (target y - offset, prediction offset + net) for diagnostics.

All token features are deterministic transforms (signed log of seconds / fixed scales), so training
and ranking contexts are built by the same code with no fitted state other than the categorical
vocabularies, which `movement_table` takes from the training file.
"""
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')

MV_COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
           'AOBT_3_flt', 'EOBT_1_flt', 'RUNWAY_mvt', 'STAND_mvt', 'WK_TBL_CAT_flt']
# adsb.lol per-departure columns used as NEIGHBOUR fields (the target's own are in its 143 columns)
ADSB_DUR = ['adsb_off_best', 'adsb_off_first', 'adsb_off_move', 'adsb_off_push', 'adsb_off_roll',
            'adsb_off_last', 'adsb_hold_s', 'adsb_stop_s', 'adsb_dwell_s', 'adsb_stat_s', 'adsb_gap_last_to_air']
ADSB_RAW = ['adsb_n_stops', 'adsb_n_push', 'adsb_gs_med_moving', 'adsb_has_ground', 'adsb_day_ok', 'adsb_at_park',
            'adsb_mlat_frac', 'adsb_path_m']
ADSB_TOK = ADSB_DUR + ADSB_RAW
TIME_KEYS = ['t_mvt', 't_a3', 't_sched', 't_eobt', 't_blk']


def slog(x):
    """Signed log of seconds, in minutes-ish units: sign(x) * log1p(|x| / 60). Works on numpy and torch."""
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x) / 60.0)
    return np.sign(x) * np.log1p(np.abs(x) / 60.0)


def _secs(s):
    return np.array((s - EPOCH).dt.total_seconds().to_numpy(), dtype=np.float64)   # writable copy


def _codes(values, vocab):
    """values -> 1..len(vocab) by vocab position, 0 = missing / unseen."""
    idx = pd.Index(vocab).get_indexer(pd.Series(values).astype(object).where(pd.notna(values), None))
    return (idx + 1).astype(np.int64)


def _str(s):
    """String view with NaN -> 'NA' (pandas' str dtype keeps NaN through astype(str))."""
    return s.astype(object).where(s.notna(), 'NA').astype(str)


def _pier(stand):
    return _str(stand).str.extract(r'^([A-Za-z]+|\d)', expand=False).fillna('NA')


def vocabularies(full):
    """Categorical vocabularies for the token fields, from the TRAINING movements (reused for ranking)."""
    apt = _str(full.APT)
    return {
        'rwy': sorted((apt + '|' + _str(full.RUNWAY_mvt)).unique()),
        'stand': sorted((apt + '|' + _str(full.STAND_mvt)).unique()),
        'pier': sorted((apt + '|' + _pier(full.STAND_mvt)).unique()),
        'wake': sorted(_str(full.WK_TBL_CAT_flt[full.WK_TBL_CAT_flt.notna()]).unique()),
    }


def movement_table(full, adsb=None, vocab=None):
    """Compact per-movement table (both phases, file order) for context building.

    full  : load(which) restricted to MV_COLS (both phases)
    adsb  : cache/adsb_lol/features_{which}.parquet (keyed by MVT_ID_mvt; departures only) or None
    vocab : vocabularies(training movements); built from `full` when None (training)

    Returns (dict of numpy arrays, vocab). Departure BLOCK_TIME is blanked HERE, first.
    """
    is_dep = full.PHASE_mvt.astype(str).eq('DEP').to_numpy()
    t_blk = _secs(full.BLOCK_TIME_UTC_mvt)
    t_blk[is_dep] = np.nan                                  # departure label: never used
    t_mvt = _secs(full.MVT_TIME_UTC_mvt)
    if not np.isfinite(t_mvt).all():
        raise ValueError('MVT_TIME missing on some movements')
    ts = full.MVT_TIME_UTC_mvt.dt.tz_convert('UTC')
    ym = (ts.dt.year * 12 + ts.dt.month - 1).to_numpy()
    month_start = np.array([(pd.Timestamp(year=int(v // 12), month=int(v % 12) + 1, day=1, tz='UTC') - EPOCH)
                            .total_seconds() for v in range(ym.min(), ym.max() + 1)])[ym - ym.min()]
    apt_codes = pd.factorize(_str(full.APT).to_numpy(), sort=True)[0]
    # group = (airport, calendar month of MVT); context never crosses it
    grp = apt_codes.astype(np.int64) * 10_000 + (ym - ym.min())

    if vocab is None:
        vocab = vocabularies(full)
    ap = _str(full.APT)
    mv = {
        'id': full.MVT_ID_mvt.to_numpy(), 'is_dep': is_dep, 'grp': grp, 't_abs': t_mvt,
        'rwy': _codes((ap + '|' + _str(full.RUNWAY_mvt)).to_numpy(), vocab['rwy']),
        'stand': _codes((ap + '|' + _str(full.STAND_mvt)).to_numpy(), vocab['stand']),
        'pier': _codes((ap + '|' + _pier(full.STAND_mvt)).to_numpy(), vocab['pier']),
        'wake': _codes(full.WK_TBL_CAT_flt.astype(object).to_numpy(), vocab['wake']),
    }
    mv['cards'] = [len(vocab['wake']) + 1, len(vocab['rwy']) + 1]      # fixed by the training vocabulary
    t_a3, t_sched, t_eobt = _secs(full.AOBT_3_flt), _secs(full.SCHED_TIME_UTC_mvt), _secs(full.EOBT_1_flt)
    t_a3[~is_dep] = np.nan                                  # an arrival's AOBT/EOBT are at its ORIGIN airport
    t_eobt[~is_dep] = np.nan
    # times relative to the start of the movement's own month (float32 exact to ~0.25 s)
    T = np.stack([t_mvt, t_a3, t_sched, t_eobt, t_blk], axis=1) - month_start[:, None]
    mv['T'] = T.astype(np.float32)

    # static token features (deterministic transforms, NaN kept here; flags added in static_tensor)
    S = {
        'is_dep': is_dep.astype(np.float32),
        'proxy': slog(t_mvt - t_a3),                       # DEP: MVT - AOBT_3 (NM taxi proxy)
        'mvt_m_sched': slog(t_mvt - t_sched),             # DEP: take-off vs sched off-block; ARR: landing vs SIBT
        'mvt_m_eobt': slog(t_mvt - t_eobt),               # DEP only
        'taxi_in': slog(t_blk - t_mvt),                   # ARR only (provided data)
        'blk_m_sched': slog(t_blk - t_sched),             # ARR only
    }
    if adsb is not None:
        a = adsb.drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(mv['id'])
        for c in ADSB_DUR:
            S[c] = slog(a[c].to_numpy(dtype=float))
        for c in ADSB_RAW:
            v = a[c].to_numpy(dtype=float)
            S[c] = np.log1p(np.clip(v, 0, None)) if c in ('adsb_n_stops', 'adsb_n_push', 'adsb_path_m') else \
                (v / 10.0 if c == 'adsb_gs_med_moving' else v)
        for c in ADSB_TOK:                                  # arrivals carry no departure ADS-B row
            S[c] = np.where(is_dep, S[c], np.nan)
    mv['S_names'] = list(S)
    mv['S'] = np.stack([np.asarray(v, dtype=np.float32) for v in S.values()], axis=1)
    assert_label_free(mv)
    return mv, vocab


def assert_label_free(mv):
    """Fail loudly if any departure carries an in-block time in the token tables."""
    blk = mv['T'][:, TIME_KEYS.index('t_blk')]
    if np.isfinite(blk[mv['is_dep']]).any():
        raise RuntimeError('LEAK: a departure row carries BLOCK_TIME in the movement table')
    for nm in ('taxi_in', 'blk_m_sched'):
        v = mv['S'][:, mv['S_names'].index(nm)]
        if np.isfinite(v[mv['is_dep']]).any():
            raise RuntimeError(f'LEAK: departure rows carry {nm}')


def build_context(mv, K=64, window=5400, log=print):
    """Neighbour indices for every DEPARTURE of `mv`, in mv's departure order.

    nb[i, :] = row indices into mv of the K movements nearest in MVT time to departure i within the same
    (airport, month) group and |dt| <= window, excluding i itself; -1 = empty slot. Vectorised per group:
    in a group sorted by time the K nearest lie within K positions either side.
    """
    t, grp, is_dep = mv['t_abs'], mv['grp'], mv['is_dep']
    order = np.lexsort((t, grp))
    gs = grp[order]
    cut = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    nb_all = np.full((len(t), K), -1, dtype=np.int32)
    offs = np.r_[np.arange(-K, 0), np.arange(1, K + 1)]
    for s, e in zip(cut[:-1], cut[1:]):
        rows = order[s:e]
        tt = t[rows]
        n = e - s
        dpos = np.flatnonzero(is_dep[rows])
        if dpos.size == 0:
            continue
        cand = dpos[:, None] + offs[None, :]
        ok = (cand >= 0) & (cand < n)
        cc = np.clip(cand, 0, n - 1)
        d = np.abs(tt[cc] - tt[dpos][:, None])
        d[~ok | (d > window)] = np.inf
        kk = min(K, d.shape[1])
        sel = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dsel = np.take_along_axis(d, sel, 1)
        # order the chosen slots by signed time for readability (the encoder is permutation-invariant)
        sgn = np.take_along_axis(tt[cc] - tt[dpos][:, None], sel, 1)
        o = np.argsort(np.where(np.isfinite(dsel), sgn, np.inf), axis=1)
        sel, dsel = np.take_along_axis(sel, o, 1), np.take_along_axis(dsel, o, 1)
        idx = rows[np.take_along_axis(cc, sel, 1)]
        nb_all[rows[dpos], :kk] = np.where(np.isfinite(dsel), idx, -1)
    nb = nb_all[is_dep]
    fill = (nb >= 0).sum(1)
    log(f'    context: {len(nb):,} departures, K={K}, window {window} s; filled slots mean {fill.mean():.1f}, '
        f'p10 {np.percentile(fill, 10):.0f}, empty contexts {(fill == 0).mean():.2%}')
    return nb


# ----------------------------------------------------------------------------------------------------
# target branch design
# ----------------------------------------------------------------------------------------------------

def target_design(X, cols, cats, fit_mask):
    """(codes int64 [n, n_cat] 0 = missing/unseen-later, Z float32 [n, n_num + n_flags], names).

    Robust scaling (median / IQR over the fit rows, clip +-8), NaN -> 0 plus a missing flag for every
    column that has any NaN on the fit rows or elsewhere (fitted on the union, which only decides WHICH
    flags exist, never their values).
    """
    cat_cols = [c for c in cols if c in cats]
    num_cols = [c for c in cols if c not in cats]
    codes = np.empty((len(X), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        codes[:, j] = X[c].cat.codes.to_numpy() + 1
    fit = np.asarray(fit_mask, bool)
    has_na = [bool(X[c].isna().any()) for c in num_cols]
    names = num_cols + [c + '__na' for c, h in zip(num_cols, has_na) if h]
    Z = np.zeros((len(X), len(names)), dtype=np.float32)          # preallocated: one copy, not two
    jf = len(num_cols)
    for j, c in enumerate(num_cols):
        v = X[c].to_numpy(dtype=np.float32, na_value=np.nan)
        ok = np.isfinite(v)
        f = v[fit & ok]
        if f.size:
            med = float(np.median(f)); q1, q3 = np.percentile(f, [25, 75]); sc = float(q3 - q1) or 1.0
        else:
            med, sc = 0.0, 1.0
        Z[:, j] = np.where(ok, np.clip((v - med) / sc, -8, 8), 0.0)
        if has_na[j]:
            Z[:, jf] = ~ok
            jf += 1
    return codes, Z, cat_cols, names


def remap_unseen(codes, train_mask):
    out = codes.copy()
    for j in range(codes.shape[1]):
        seen = np.zeros(int(codes[:, j].max()) + 1, bool)
        seen[np.unique(codes[train_mask, j])] = True
        out[~seen[out[:, j]], j] = 0
    return out


def emb_dim(card, cap=16):
    return int(min(cap, max(3, round(1.6 * card ** 0.56))))


# ----------------------------------------------------------------------------------------------------
# token featurisation (torch, device-agnostic)
# ----------------------------------------------------------------------------------------------------

class Context:
    """Device tensors for the movement table + a batch featuriser.

    tok_num = static features of the neighbour (with missing flags) + relative-time features against the
    target + same-runway / stand / pier flags; tok_cat = (wake, runway) codes; pad = empty slot.
    """
    N_REL = 10

    def __init__(self, mv, device):
        S = mv['S']
        na = ~np.isfinite(S)
        flag_cols = np.flatnonzero(na.any(0))
        self.S = torch.from_numpy(np.concatenate([np.nan_to_num(S, nan=0.0), na[:, flag_cols].astype(np.float32)],
                                                 axis=1)).to(device)
        self.T = torch.from_numpy(mv['T']).to(device)
        self.keys = torch.from_numpy(np.stack([mv['rwy'], mv['stand'], mv['pier']], 1)).to(device)
        self.cat = torch.from_numpy(np.stack([mv['wake'], mv['rwy']], 1)).to(device)
        self.cat_cards = list(mv['cards'])
        self.n_tok = self.S.shape[1] + 2 * self.N_REL + 3
        self.device = device

    def tokens(self, tgt, nb):
        """tgt: [B] mv rows of the targets; nb: [B, K] mv rows (-1 = pad). -> (num [B,K,F], cat [B,K,2], pad)."""
        pad = nb < 0
        j = nb.clamp_min(0)
        Tn, Tt = self.T[j], self.T[tgt]                        # [B,K,5], [B,5]
        mvt_t, a3_t, sch_t = Tt[:, 0:1], Tt[:, 1:2], Tt[:, 2:3]
        rel = [
            (Tn[..., 0] - mvt_t) / 1800.0,                     # take-off/landing vs our take-off (linear)
            slog(Tn[..., 0] - mvt_t),
            slog(Tn[..., 1] - mvt_t),                          # its AOBT_3 vs our take-off
            slog(Tn[..., 2] - mvt_t),                          # its SCHED vs our take-off
            slog(Tn[..., 3] - mvt_t),                          # its EOBT_1 vs our take-off
            slog(Tn[..., 4] - mvt_t),                          # its in-block (arrivals) vs our take-off
            slog(Tn[..., 0] - a3_t),                           # its take-off/landing vs OUR AOBT_3
            slog(Tn[..., 1] - a3_t),                           # its AOBT_3 vs our AOBT_3
            slog(Tn[..., 2] - sch_t),                          # its SCHED vs our SCHED
            slog(Tn[..., 0] - sch_t),                          # its take-off/landing vs our SCHED
        ]
        rel = torch.stack(rel, -1)
        rna = torch.isnan(rel)
        rel = torch.nan_to_num(rel, nan=0.0).clamp(-12, 12)
        kn, kt = self.keys[j], self.keys[tgt][:, None, :]
        same = ((kn == kt) & (kt > 0)).float()                  # same runway / stand / pier
        num = torch.cat([self.S[j], rel, rna.float(), same], -1)
        num = num.masked_fill(pad[..., None], 0.0)
        cat = self.cat[j].masked_fill(pad[..., None], 0)
        return num, cat, pad


# ----------------------------------------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------------------------------------

class SeqNet(nn.Module):
    def __init__(self, cards, n_num, n_tok=0, tok_cards=(), d=64, layers=2, heads=4, hidden=(512, 256),
                 dropout=0.10, emb_cap=16, use_context=True, enc_dropout=0.0):
        super().__init__()
        self.use_context = use_context
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c, emb_cap)) for c in cards])
        for e in self.embs:
            nn.init.normal_(e.weight, 0.0, 0.05)
            with torch.no_grad():
                e.weight[0].zero_()
        d_in = sum(emb_dim(c, emb_cap) for c in cards) + n_num
        trunk, dd = [nn.LayerNorm(d_in)], d_in
        for h in hidden:
            trunk += [nn.Linear(dd, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            dd = h
        self.trunk = nn.Sequential(*trunk)
        d_head = dd
        if use_context:
            self.tok_embs = nn.ModuleList([nn.Embedding(c, 8) for c in tok_cards])
            self.tok_in = nn.Sequential(nn.Linear(n_tok + 8 * len(tok_cards), d), nn.GELU(), nn.Linear(d, d))
            self.cls = nn.Linear(dd, d)
            # no dropout inside the encoder: on CPU the Bernoulli masks of the attention maps were 40% of the
            # step time (profiled), and weight decay + the trunk/head dropout regularise enough at this size
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, enc_dropout, activation='gelu', batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.enc_norm = nn.LayerNorm(d)
            d_head += d
        self.head = nn.Sequential(nn.Linear(d_head, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, c, z, tok_num=None, tok_cat=None, pad=None):
        h = self.trunk(torch.cat([emb(c[:, j]) for j, emb in enumerate(self.embs)] + [z], 1))
        if not self.use_context:
            return self.head(h).squeeze(1)
        te = [emb(tok_cat[..., j]) for j, emb in enumerate(self.tok_embs)]
        tok = self.tok_in(torch.cat([tok_num] + te, -1))
        seq = torch.cat([self.cls(h)[:, None, :], tok], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        ctx = self.enc_norm(self.enc(seq, src_key_padding_mask=mask)[:, 0])
        return self.head(torch.cat([h, ctx], 1)).squeeze(1)


def resolve_device(device=None):
    if device in (None, 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('cuda requested but torch.cuda.is_available() is False -- on the cluster this means the '
                           'job ran on -p gpu WITHOUT --gres=gpu:1')
    return torch.device(device)


def fit_predict(codes, Z, y, train_mask, pred_mask, *, ctx=None, tgt_mv=None, nb=None, offset=None,
                use_context=True, epochs=8, batch=2048, lr=2e-3, weight_decay=1e-5, d=64, layers=2, heads=4,
                hidden=(512, 256), dropout=0.10, emb_cap=16, seed=0, threads=4, device=None, amp=True,
                log=print, eval_hook=None):
    """Train on `train_mask` rows, return predictions (seconds) for `pred_mask` rows.

    codes / Z : target branch design for all rows; y: labels (NaN allowed off the train rows)
    ctx       : Context (movement tensors on the device); tgt_mv [n] mv row of each target; nb [n, K]
    offset    : optional per-row baseline -> residual fit (target y - offset, prediction offset + net)
    eval_hook(epoch, pred) : called after each epoch with the current predictions for pred_mask
    """
    dev = resolve_device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_amp = amp and dev.type == 'cuda'
    codes = remap_unseen(codes, train_mask)
    cards = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
    off = np.zeros(len(y), np.float64) if offset is None else np.asarray(offset, np.float64)
    r = (np.asarray(y, np.float64) - off)[train_mask]
    mu, sd = float(r.mean()), float(r.std()) or 1.0

    def put(a, dt=None):
        t = torch.from_numpy(np.ascontiguousarray(a))
        return (t if dt is None else t.to(dt)).to(dev)

    tr, pr = np.flatnonzero(train_mask), np.flatnonzero(pred_mask)
    Ctr, Ztr, Ttr = put(codes[tr]), put(Z[tr], torch.float32), put(((r - mu) / sd).astype(np.float32))
    Cpr, Zpr = put(codes[pr]), put(Z[pr], torch.float32)
    if use_context:
        if ctx is None or nb is None or tgt_mv is None:
            raise ValueError('use_context needs ctx, tgt_mv and nb')
        NBtr, NBpr = put(nb[tr].astype(np.int64)), put(nb[pr].astype(np.int64))
        Gtr, Gpr = put(tgt_mv[tr].astype(np.int64)), put(tgt_mv[pr].astype(np.int64))
    net = SeqNet(cards, Z.shape[1], ctx.n_tok if use_context else 0, ctx.cat_cards if use_context else (),
                 d=d, layers=layers, heads=heads, hidden=hidden, dropout=dropout, emb_cap=emb_cap,
                 use_context=use_context).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(tr)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    log(f'    seq net: {net.n_params:,} params, context={use_context}, {Z.shape[1]} target numerics, '
        f'{n:,} train rows, {steps:,} steps, device={dev}, amp={use_amp}')

    def run(C, Zb, NB, G, idx):
        if use_context:
            tn, tc, pad = ctx.tokens(G[idx], NB[idx])
            return net(C[idx], Zb[idx], tn, tc, pad)
        return net(C[idx], Zb[idx])

    def predict():
        net.eval()
        out = np.empty(len(pr), np.float64)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            for i in range(0, len(pr), 4 * batch):
                idx = torch.arange(i, min(i + 4 * batch, len(pr)), device=dev)
                out[i:i + len(idx)] = run(Cpr, Zpr, NBpr if use_context else None, Gpr if use_context else None,
                                          idx).float().cpu().numpy()
        return off[pr] + out * sd + mu

    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    t_start = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch].to(dev)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = run(Ctr, Ztr, NBtr if use_context else None, Gtr if use_context else None, idx)
                loss = lossf(out.float(), Ttr[idx])
                loss.backward()
            except RuntimeError as e:
                # the CUDA bf16 path cannot be exercised on the laptop; if it breaks, it breaks on the very first
                # step -- fall back to fp32 once, visibly, instead of losing the cluster task
                if not (use_amp and ep == 0 and i == 0):
                    raise
                log(f'    WARNING: bf16 autocast failed on the first step ({str(e)[:300]}); continuing in fp32')
                use_amp = False
                opt.zero_grad(set_to_none=True)
                out = run(Ctr, Ztr, NBtr if use_context else None, Gtr if use_context else None, idx)
                loss = lossf(out.float(), Ttr[idx])
                loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * idx.numel()
        if not np.isfinite(tot):
            raise RuntimeError(f'training diverged: non-finite loss in epoch {ep + 1}')
        msg = f'    epoch {ep + 1}/{epochs}  train MSE(std) {tot / n:.5f}  [{time.time() - t_start:.0f} s]'
        if eval_hook is not None:
            msg += '  ' + str(eval_hook(ep + 1, predict()))
        log(msg)
    return predict()
