"""Build cache_final/ for the final phase (PRC_RANKING=final) from the initial-phase cache/.

The final phase rebuilds everything that depends on the ranking file (features of the ranking rows, every final fit,
the production tail classifier, TabICL probabilities, the route-D corrector chain) from the organisers'
final_ranking.parquet. Everything that depends only on the 2025 training data (training features, out-of-fold
predictions, blend weights, the 2025 TabICL context, ...) is mirrored from cache/:
  * big read-only folders are symlinked as a whole (cv/, external/, adsb_lol/points|raw|events|meta|match);
  * training.parquet is symlinked (read-only use; the final run never rebuilds it);
  * every other training-side file is COPIED, so no final-phase script can write through a link into cache/.
Ranking-derived files are NOT mirrored: anything with 'ranking' in its path, the *_final folders, final/,
major/m11d_pred_trailD.npy, major/m11f_candidates.npz, strategy/, tfm_w4/, plus pure analysis folders.

    python tools/make_final_cache.py            # idempotent; existing entries in cache_final/ are left alone
"""
import os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC, DST = REPO / 'cache', REPO / 'cache_final'
LINK_DIRS = ['cv', 'external', 'adsb_lol/points', 'adsb_lol/raw', 'adsb_lol/events', 'adsb_lol/meta', 'adsb_lol/match']
LINK_FILES = ['training.parquet']
SKIP_TOP = {'final', 'strategy', 'tfm_w4', 'hunt', 'tail', 'plan_1001', 'plan_1006', 'routed_0926', 'exp63_smoke', 'exp63_screen',
            'deice_0926', 'drivers_0926', 'ground_sources', 'tfm_f1', 'tfm_f2', 'tfm_r1', 'drift', 'research_0925', 'altfam'}
SKIP_FILES = {'major/m11d_pred_trailD.npy', 'major/m11f_candidates.npz', 'exp64_p6_p7.npy', 'exp64_pred_v6recipe_v7.npy'}
SKIP_PREFIX = ('exp64_seeds/', 'major/m11b_pred_')   # ranking-shaped arrays whose names lack 'ranking' (found by shape scan)


def ranking_derived(rel: str) -> bool:
    top = rel.split('/')[0]
    return ('ranking' in rel or top.endswith('_final') or top.endswith('_final_smoke') or top in SKIP_TOP
            or rel in SKIP_FILES or rel.startswith(SKIP_PREFIX) or rel.startswith('adsb_lol/features_v2'))


def main():
    if not SRC.is_dir():
        sys.exit(f'FATAL: {SRC} missing')
    DST.mkdir(exist_ok=True)
    n_link = n_copy = n_skip = 0
    for rel in LINK_DIRS + LINK_FILES:
        s, d = SRC / rel, DST / rel
        if s.exists() and not d.exists() and not d.is_symlink():
            d.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(s, d); n_link += 1
    linked = tuple(r + '/' for r in LINK_DIRS)
    for p in SRC.rglob('*'):
        if p.is_dir():
            continue
        rel = p.relative_to(SRC).as_posix()
        if rel in LINK_FILES or rel.startswith(linked):
            continue
        if ranking_derived(rel):
            n_skip += 1
            continue
        d = DST / rel
        if d.exists() or d.is_symlink():
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d); n_copy += 1
    print(f'cache_final: {n_link} symlinks, {n_copy} files copied, {n_skip} ranking-derived files skipped -> {DST}')


if __name__ == '__main__':
    main()
