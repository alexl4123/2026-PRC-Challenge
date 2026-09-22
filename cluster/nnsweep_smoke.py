"""Exercise every code path of the sweep locally, on CPU, before it goes near Slurm.

    /home/thinklex/prc2026-venv/bin/python cluster/nnsweep_smoke.py --all

We have lost array jobs to a `numeric_design` unpacking error, a `.numpy()` on a
CUDA tensor and a CatBoost NaN. Every one of them was a path that looked fine and
had never been executed. So: run each of these before submitting anything.

  --check-mapping   the array index -> (config, seed, month) mapping is a
                    bijection onto the full product, and no config name contains
                    the '__' that cluster/collect.py splits filenames on.

  --check-identity  *** THE IMPORTANT ONE ***  cluster/nnsweep_model.fit_sweep at
                    the defaults in nnsweep_configs.BASE is BIT-FOR-BIT identical
                    to prc.models.embed_nn.fit_predict. If it is not, the `base`
                    arm is not a control and every paired comparison in the sweep
                    is against the wrong thing. Runs on synthetic arrays of the
                    real shapes and dtypes, so it is fast and tests the code
                    rather than the data.

  --check-configs   every config in the registry builds its network, its
                    schedule and its loss, and trains two epochs on synthetic
                    data. Catches a typo'd schedule name or an embedding rule
                    that produces a zero-width table without waiting for a GPU.

  --pipeline        the real thing end to end on real data: prep one fold with
                    --subsample, sweep every stage-A config through the actual
                    CLI, then score it with nnsweep_collect.py. Slowest (~20 min)
                    and the only check that exercises argument parsing, the
                    prerequisite checks, the file layout and the collector.

Memory: the pipeline check loads the cached altfam design (~1.6 GB) and builds
the numeric design over all 2.08M rows (~0.6 GB per mode), so it peaks around
3 GB. Run it inside a cap anyway:

  systemd-run --user --scope -p MemoryMax=7G -p MemorySwapMax=0 \
    systemd-inhibit --what=idle:sleep:handle-lid-switch \
    /home/thinklex/prc2026-venv/bin/python cluster/nnsweep_smoke.py --all
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from cluster import nnsweep_configs as NC

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_CAMPAIGN = 'nnsweep_smoke'
N_ROWS, N_CAT, N_NUM = 6000, 8, 64


def fake(seed=0, n=N_ROWS):
    """Synthetic arrays with the real shapes, dtypes and cardinality spread.

    The cardinalities mimic the real tables -- APT 10 up to STAND_mvt 1,899 --
    because the embedding-width rule is one of the things under test and a
    uniform toy cardinality would not exercise the cap at all.
    """
    rng = np.random.default_rng(seed)
    cards = [11, 54, 40, 1900, 300, 677, 9, 5][:N_CAT]
    codes = np.stack([rng.integers(0, c, n) for c in cards], axis=1).astype(np.int32)
    Z = rng.standard_normal((n, N_NUM)).astype(np.float32)
    y = (1000 + 400 * rng.standard_normal(n) + 30 * Z[:, 0]).astype(np.float64)
    y = np.clip(y, 0, None)
    trn = np.zeros(n, bool); trn[: int(0.8 * n)] = True
    va = ~trn
    return codes, Z, y, trn, va


def check_mapping():
    print('== index mapping ==')
    ok = NC.verify()
    print('   PASS' if ok else '   FAIL')
    return ok


def check_identity():
    print('== fit_sweep(BASE defaults) == embed_nn.fit_predict, bit for bit ==')
    from cluster.nnsweep_model import fit_sweep
    from prc.models.embed_nn import fit_predict
    codes, Z, y, trn, va = fake()
    kw = dict(epochs=2, batch=512, lr=3e-3, weight_decay=1e-5,
              hidden=(512, 256, 128), dropout=0.10, seed=7, threads=2,
              device='cpu', log=lambda s: None)
    a = fit_predict(codes, Z, y, trn, va, **kw)
    b, info = fit_sweep(codes, Z, y, trn, va, **kw)
    same = np.array_equal(a, b)
    d = float(np.abs(a - b).max())
    print(f'   n={len(a)}  max|diff| {d:.3e}  identical={same}')
    print(f'   dims {info["dims"]}  (cap 32 binds on STAND-like 1,900 and 677)')
    if not same:
        print('   FAIL -- the base arm would not be a control. Check the module '
              'construction order in SweepMLP against EmbedMLP.')
    else:
        print('   PASS')
    return same


def check_configs():
    print('== every registry config builds and trains ==')
    from cluster.nnsweep_model import fit_sweep
    codes, Z, y, trn, va = fake(seed=1, n=4000)
    ok = True
    n_train = int(trn.sum())
    for name in NC.STAGE_A:
        cfg = {k: v for k, v in NC.CONFIGS[name].items() if k not in ('note', 'missing')}
        # Shorten the run, but keep every config's REAL batch, lr and schedule.
        # Epochs are set so each arm still gets >= 40 optimiser steps: below
        # that, OneCycleLR's warmup phase collapses to zero width and the run
        # dies for a reason that has nothing to do with the config being valid.
        # On the real folds the smallest arm (b32k) gets ~700 steps
        # (12 epochs x ceil(~1.92M bulk train rows / 32768)).
        per_epoch = -(-n_train // cfg['batch'])
        cfg['epochs'] = max(2, -(-40 // per_epoch))
        cfg['hidden'] = tuple(cfg['hidden'])
        try:
            t = time.time()
            p, info = fit_sweep(codes, Z, y, trn, va, seed=0, threads=2,
                                device='cpu', log=lambda s: None, **cfg)
            bad = (not np.isfinite(p).all()) or (p < 0).any() or min(info['dims']) < 1
            print(f'   {"FAIL" if bad else "ok  "} {name:9s} params {info["n_params"]:>9,d} '
                  f'dims {str(info["dims"]):<34s} pred {p.mean():7.1f} '
                  f'({time.time() - t:.1f}s)')
            ok &= not bad
        except Exception as e:                                   # noqa: BLE001
            print(f'   FAIL {name:9s} {type(e).__name__}: {e}')
            ok = False
    print('   PASS' if ok else '   FAIL')
    return ok


def check_numeric_design_contract():
    """numeric_design's docstring says it returns four values; it returns six.

    That mismatch has already cost one array job. Assert the real arity here so
    a future change to altfam.py breaks this test rather than the campaign.
    """
    print('== altfam.numeric_design returns six values, not the four it documents ==')
    import inspect

    from prc.models import altfam
    src = inspect.getsource(altfam.numeric_design)
    n = src.count('return ')
    ok = 'return codes, cards, Z, cat_cols, num_cols, stats' in src and n == 1
    print(f'   single return of six values: {ok}')
    print('   PASS' if ok else '   FAIL -- update cluster/nnsweep_task.stage_fixed')
    return ok


def run(cmd, **kw):
    print(f'   $ {" ".join(cmd[1:])}', flush=True)
    r = subprocess.run(cmd, cwd=REPO, **kw)
    return r.returncode == 0


def check_pipeline(subsample, keep, out_root, n_configs=None):
    """The real CLI, on real data, writing OUTSIDE the repository.

    WHY out_root. The prep stage caches ~1.1 GB per fold, and this repository
    lives inside a Dropbox-synced tree. During the first run of this test the
    sync daemon removed the freshly written arrays mid-run and then restored
    them with their original mtimes after an `rm -rf`, which looked exactly like
    a bug in the atomic writer and was not. Local smoke runs therefore write to
    a scratch directory; on the cluster --out-root is unused and everything
    lands in cache/cv as usual.
    """
    print('== end-to-end pipeline on real data (subsampled) ==')
    d = Path(out_root) / SMOKE_CAMPAIGN
    print(f'   writing to {d}')
    if not keep and d.exists():
        import shutil
        shutil.rmtree(d)
    task = [PY, 'cluster/nnsweep_task.py', '--out-root', str(out_root)]

    print('   -- the prerequisite check must fire BEFORE the prep stage runs')
    r = subprocess.run(task + ['--campaign', SMOKE_CAMPAIGN, '--stage', 'sweep',
                               '--group', 'smoke', '--index', '0', '--device', 'cpu'],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode == 0 or 'MISSING PREREQUISITE' not in (r.stdout + r.stderr):
        print(f'   FAIL: a sweep task without prep exited {r.returncode} without '
              f'the prerequisite error')
        return False
    print('   ok   sweep refuses to run before the prep stage')

    print(f'   -- prep, one fold, --subsample {subsample:,}')
    if not run(task + ['--campaign', SMOKE_CAMPAIGN, '--stage', 'fixed',
                       '--group', 'smoke', '--index', '0', '--design', 'altfam',
                       '--subsample', str(subsample), '--threads', '4']):
        print('   FAIL: prep stage')
        return False

    n = NC.n_units('smoke')
    if n_configs:
        # --check-configs already builds and trains EVERY arm, so the pipeline's
        # unique coverage is the plumbing (CLI, prerequisite checks, file layout,
        # idempotency, the CUDA guard, both collectors) plus the two arms that
        # take a different path through the TASK rather than the model: `sent`
        # reads a different cached Z, `huber` a different loss. Those are in the
        # first indices only by luck, so a truncated run is a convenience, not a
        # substitute -- run the full one before submitting to Slurm.
        n = min(n, n_configs)
        print(f'   NOTE: limited to the first {n} of {NC.n_units("smoke")} arms')
    print(f'   -- sweep, {n} stage-A config(s) through the real CLI')
    for i in range(n):
        if not run(task + ['--campaign', SMOKE_CAMPAIGN, '--stage', 'sweep',
                           '--group', 'smoke', '--index', str(i),
                           '--device', 'cpu', '--threads', '4']):
            print(f'   FAIL: sweep index {i} ({NC.unit("smoke", i)})')
            return False

    print('   -- idempotency: re-running a finished unit must skip')
    r = subprocess.run(task + ['--campaign', SMOKE_CAMPAIGN, '--stage', 'sweep',
                               '--group', 'smoke', '--index', '0', '--device', 'cpu'],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or 'skipping' not in r.stdout:
        print(f'   FAIL: re-run did not skip (exit {r.returncode})')
        return False
    print('   ok   re-run skipped')

    print('   -- the CUDA guard must fire on a machine with no GPU')
    r = subprocess.run(task + ['--campaign', SMOKE_CAMPAIGN, '--stage', 'sweep',
                               '--group', 'smoke', '--index', '1', '--device', 'cuda'],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode == 0 or '--gres' not in (r.stdout + r.stderr):
        print(f'   FAIL: --device cuda exited {r.returncode} without the --gres error')
        return False
    print('   ok   --device cuda refuses when no device is visible')

    print('   -- collector')
    if not run([PY, 'cluster/nnsweep_collect.py', '--campaign', SMOKE_CAMPAIGN,
                '--group', 'smoke', '--boot', '100', '--per-fold',
                '--out-root', str(out_root)]):
        print('   FAIL: collector')
        return False

    print('   -- cluster/collect.py must ignore the _-prefixed subdirectories')
    # collect.py has no --out-root, so mirror the flat layout into cache/cv for
    # this one check. Only the small mixture .npy files are copied; the GB-scale
    # _fixed arrays stay in the scratch directory.
    from prc.config import CACHE
    mirror = CACHE / 'cv' / SMOKE_CAMPAIGN
    mirror.mkdir(parents=True, exist_ok=True)
    for f in sorted(d.glob('*.npy')):
        (mirror / f.name).write_bytes(f.read_bytes())
    for sub in ('_bulk', '_meta', '_fixed'):
        (mirror / sub).mkdir(exist_ok=True)
    r = subprocess.run([PY, 'cluster/collect.py', SMOKE_CAMPAIGN],
                       cwd=REPO, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if 'INCOMPLETE' not in out:
        print(f'   FAIL: collect.py did not report the 1-fold smoke arms as '
              f'INCOMPLETE:\n{out[-800:]}')
        return False
    print('   ok   collect.py parses the arm names and reports 1-fold arms '
          'INCOMPLETE, as it should')
    if not keep:
        import shutil
        shutil.rmtree(mirror, ignore_errors=True)
    print('   PASS')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--check-mapping', action='store_true')
    ap.add_argument('--check-identity', action='store_true')
    ap.add_argument('--check-configs', action='store_true')
    ap.add_argument('--pipeline', action='store_true')
    ap.add_argument('--subsample', type=int, default=150_000)
    ap.add_argument('--keep', action='store_true', help='do not wipe the smoke campaign')
    ap.add_argument('--pipeline-configs', type=int, default=None,
                    help='run only the first N arms through the pipeline check. '
                         'Convenience for a loaded laptop; the full 22-arm run is '
                         'what should pass before a Slurm submission.')
    ap.add_argument('--out-root', default=str(Path(tempfile.gettempdir()) / 'prc_nnsweep_smoke'),
                    help='scratch directory for the pipeline check; keep it OUTSIDE '
                         'the Dropbox-synced repository')
    a = ap.parse_args()
    if not any([a.all, a.check_mapping, a.check_identity, a.check_configs, a.pipeline]):
        a.all = True

    results = {}
    if a.all or a.check_mapping:
        results['mapping'] = check_mapping(); print()
    if a.all or a.check_identity:
        results['identity'] = check_identity(); print()
    if a.all or a.check_configs:
        results['numeric_design'] = check_numeric_design_contract(); print()
        results['configs'] = check_configs(); print()
    if a.all or a.pipeline:
        results['pipeline'] = check_pipeline(a.subsample, a.keep, a.out_root, a.pipeline_configs); print()

    from prc.io import peak_rss_gb
    print('=' * 60)
    for k, v in results.items():
        print(f'  {"PASS" if v else "FAIL"}  {k}')
    print(f'  peak RSS {peak_rss_gb():.2f} GB')
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
