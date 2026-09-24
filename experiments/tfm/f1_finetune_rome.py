"""F1 -- fine-tune TabICL as Rome's tail-zone classifier ("connectors"), against plain in-context TabICL.

v13 uses TabICL in context: the 627 labelled Rome P flights of 2025 are pasted into the input and
the pretrained weights stay frozen. Here the weights are also trained on those flights, with
TabICL's own single-dataset fine-tuning (tabicl 2.2.0, FinetunedTabICLClassifier). The three
stages can be frozen separately, which gives the variants:

  icl      no fine-tuning (= v13; the reference every variant is judged against)
  ft_col   train only the column-embedding stage: how our 38 features are read ("input connector")
  ft_cr    train column embedding + row interaction, keep the in-context stage frozen
  ft_full  train everything

Leave-one-month-out on Rome's P rows of 2025: each fold fine-tunes on the other 11 months and
predicts the held-out month. Inside a fold, TabICL takes a stratified 10% validation split for
early stopping on log-loss, and it restores the pretrained weights when no epoch beats them, so a
variant can only differ from `icl` where fine-tuning helped on that split. Hyperparameters are the
package defaults (30 epochs, lr 1e-5, patience 8) except the log-loss metric, which tracks the
Brier/RMSE objective better than the default AUC. Nothing is tuned on the out-of-fold results.

PRE-REGISTERED DECISION: a variant replaces v13's in-context p only if it beats `icl` on both
12 months and the scored months, with the 12-month paired-bootstrap interval below 0. With three
variants tried, the winner is also reported next to the other two; if they disagree in sign, the
result is treated as noise.

Resumable: each (variant, month, seed) is saved under cache/tfm_f1/parts/ and skipped if present.

usage (laptop): HF_HUB_OFFLINE=1 systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
                    nice -n 5 python experiments/tfm/f1_finetune_rome.py --threads 3
      (GPU, one task per month):  --device cuda --months 7 --seeds 0 1 2   (see cluster/f1_array.sbatch)
      report from saved parts:    --report --seeds 0 1 2
"""
import argparse, logging, os, sys, time, tempfile, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from prc.config import CACHE
from prc.validation import paired_bootstrap

OUT = CACHE / 'tfm_f1'
PARTS = OUT / 'parts'
KEYS = ('k_op', 'k_stand', 'k_pier', 'k_ades', 'apt', 'month')
VARIANTS = {
    'icl': None,
    'ft_col': dict(freeze_row=True, freeze_icl=True),
    'ft_cr': dict(freeze_icl=True),
    'ft_full': dict(),
}


def load():
    M = CACHE / 'major'
    d = pd.read_parquet(M / 'dep_training.parquet', columns=['month', 'y', 'off_sched', 'off_aobt3'])
    y, so, mon = d.y.to_numpy(), d.off_sched.to_numpy(), d.month.to_numpy()
    Pidx = np.flatnonzero(~np.isfinite(d.off_aobt3.to_numpy()) & (so > 7200))
    cand = pd.read_parquet(CACHE / 'exp62_cand_training.parquet').reset_index(drop=True)
    if len(cand) != len(Pidx) or not np.array_equal(cand.month.to_numpy(), mon[Pidx]):
        sys.exit('FATAL: exp62_cand_training.parquet is not aligned with the P rows of dep_training')
    num = [c for c in cand.columns if c not in KEYS and cand[c].dtype != object]
    rome = cand.apt.astype(str).to_numpy() == 'LIRF'
    return dict(y=y, so=so, mon=mon, Pidx=Pidx, rome=rome, pm=cand.month.to_numpy(),
                X=cand[num].astype(float).to_numpy(), t=(y[Pidx] > 7200).astype(int),
                p_cur=np.load(CACHE / 'exp62_spec_P.npy')[1], v9=np.load(M / 'v9s_oof.npy'),
                pb=np.load(CACHE / 'exp62_pb.npy').astype(float), nfeat=len(num))


def fit_predict(variant, Xtr, ttr, Xte, seed, device, threads, batch_size, logfile=None):
    if VARIANTS[variant] is None:
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(device=device, random_state=seed, n_jobs=threads, batch_size=batch_size)
        clf.fit(Xtr, ttr)
        return clf.predict_proba(Xte)[:, 1], {}
    from tabicl import FinetunedTabICLClassifier
    clf = FinetunedTabICLClassifier(device=device, random_state=seed, eval_metric='log_loss',
                                    amp=(device != 'cpu'),
                                    extra_classifier_kwargs=dict(batch_size=batch_size, n_jobs=threads),
                                    **VARIANTS[variant])
    lg = logging.getLogger('tabicl')                 # per-epoch validation summaries -> one log per fold
    lg.setLevel(logging.INFO)
    h = logging.FileHandler(logfile, mode='w') if logfile else logging.NullHandler()
    lg.addHandler(h)
    try:
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(Xtr, ttr, output_dir=tmp)
            p = clf.predict_proba(Xte)[:, 1]
    finally:
        lg.removeHandler(h); h.close()
    return p, {'best_val_metric': round(float(getattr(clf, '_best_metric_', float('nan'))), 4)}


def run(D, variants, months, seeds, device, threads, batch_size):
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed in seeds:
        for m in months:
            tr = D['rome'] & (D['pm'] != m)
            te = D['rome'] & (D['pm'] == m)
            for v in variants:
                f = PARTS / f'{v}_m{m:02d}_s{seed}.npy'
                if f.exists():
                    continue
                t1 = time.time()
                (OUT / 'logs').mkdir(parents=True, exist_ok=True)
                p, info = fit_predict(v, D['X'][tr], D['t'][tr], D['X'][te], seed, device, threads, batch_size,
                                      logfile=OUT / 'logs' / f'{v}_m{m:02d}_s{seed}.log')
                np.save(f, p)
                print(f'seed {seed} month {m:2d} {v:7s}: context {tr.sum()} query {te.sum()}  '
                      f'{time.time() - t1:5.0f} s (total {time.time() - t0:5.0f} s)  {info}', flush=True)


def gather(D, v, seeds):
    ps = []
    for seed in seeds:
        files = [PARTS / f'{v}_m{m:02d}_s{seed}.npy' for m in range(1, 13)]
        if not all(f.exists() for f in files):
            continue
        p = np.full(len(D['Pidx']), np.nan)
        for m, f in zip(range(1, 13), files):
            p[D['rome'] & (D['pm'] == m)] = np.load(f)
        ps.append(p)
    return (np.mean(ps, axis=0), len(ps)) if ps else (None, 0)


def report(D, seeds):
    y, so, mon, Pidx, t, rome = D['y'], D['so'], D['mon'], D['Pidx'], D['t'], D['rome']
    S = np.isin(mon, [1, 7])

    def full(p):
        o = D['v9'].copy()
        pp = np.where(rome, p, D['p_cur'])
        o[Pidx] = np.clip(pp * so[Pidx] + (1 - pp) * D['pb'][Pidx], 0, None)
        return o

    arms = {'cur (v7)': (D['p_cur'], 0)}
    for v in VARIANTS:
        p, k = gather(D, v, seeds)
        if p is not None:
            arms[v] = (p, k)
    if 'icl' not in arms:
        print('no complete icl reference yet'); return
    print(f'\nRome P rows {rome.sum()}, tails {t[rome].sum()}')
    for nm, (p, k) in arms.items():
        print(f'  {nm:9s} seeds {k}  Brier {np.mean((p[rome] - t[rome]) ** 2):.4f}  AUC {roc_auc_score(t[rome], p[rome]):.3f}')
    ref = full(arms['icl'][0])
    print('\nRMSE change over ALL 2025 departures vs in-context TabICL (= v13), paired bootstrap 95%:')
    for nm, (p, _) in arms.items():
        if nm == 'icl':
            continue
        f = full(p)
        out = []
        for scope, mm in (('12m', np.ones(len(y), bool)), ('scored', S)):
            b = paired_bootstrap(y[mm], f[mm], ref[mm])
            out.append(f'{scope} {b["diff"]:+.2f} [{b["lo"]:+.2f},{b["hi"]:+.2f}]')
        per = [np.sqrt(np.mean((y[mon == m] - f[mon == m]) ** 2)) - np.sqrt(np.mean((y[mon == m] - ref[mon == m]) ** 2))
               for m in range(1, 13)]
        print(f'  {nm:9s} ' + '   '.join(out) + f'   months better {sum(v < 0 for v in per)}/12')
    np.savez(OUT / 'f1_oof.npz', Pidx=Pidx, rome=rome, **{f'p_{k}': p for k, (p, _) in arms.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--months', nargs='+', type=int, default=list(range(1, 13)))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1, help='inference ensemble members per pass (memory)')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    if a.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('FATAL: --device cuda but no GPU is visible (on -p gpu, --gres=gpu:1 is mandatory)')
    D = load()
    print(f'Rome P rows {D["rome"].sum()} (tails {D["t"][D["rome"]].sum()}), features {D["nfeat"]}, device {a.device}')
    if not a.report:
        run(D, a.variants, a.months, a.seeds, a.device, a.threads, a.batch_size)
    report(D, a.seeds)


if __name__ == '__main__':
    main()
