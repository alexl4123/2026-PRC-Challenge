"""Loading with a memory budget.

The laptop has 29 GiB and is the user's working machine, so nothing here is
allowed to be careless. Always prefer `columns=` and `phase=` over loading
everything and subsetting afterwards.

Peak RSS of a run is available from `peak_rss_gb()`. For anything long, wrap the
process so a runaway fails small rather than taking the desktop with it:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
        systemd-inhibit --what=idle:sleep:handle-lid-switch \
        /home/thinklex/prc2026-venv/bin/python experiments/expNN_....py
"""
import glob
import resource

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATA, CACHE, CAT_COLS

EXPECTED_ROWS = {'training': 4_167_797, 'ranking': 689_534}


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLS:
        if c in df.columns and str(df[c].dtype) != 'category':
            df[c] = df[c].astype('category')
    return df


def _with_apt(df: pd.DataFrame) -> pd.DataFrame:
    """Reporting airport: ADEP for departures, ADES for arrivals."""
    df['APT'] = np.where(df.PHASE_mvt.eq('DEP'), df.ADEP_mvt, df.ADES_mvt)
    return df


def _build_cache():
    """Convert the raw parquets into compact cached parquets.

    Streamed one monthly file at a time through a ParquetWriter so the peak stays
    near one month's worth (~0.6 GB) instead of holding all twelve at once.

    Every table after the first is cast to the first table's schema. Without that,
    pyarrow infers `string` for some files and `large_string` for others depending
    on the data, the writer rejects the mismatch, and the output ends up SILENTLY
    TRUNCATED. The row-count assertion at the end is the backstop.
    """
    files = sorted(glob.glob(str(DATA / 'training_*.parquet')))
    if not files:
        raise FileNotFoundError(f'no training_*.parquet under {DATA}')

    writer = schema = None
    try:
        for f in files:
            tbl = pa.Table.from_pandas(_with_apt(pd.read_parquet(f)), preserve_index=False)
            if schema is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(CACHE / 'training.parquet', schema, compression='zstd')
            elif tbl.schema != schema:
                tbl = tbl.cast(schema)
            writer.write_table(tbl)
            del tbl
    finally:
        if writer is not None:
            writer.close()

    rk = pa.Table.from_pandas(_with_apt(pd.read_parquet(DATA / 'ranking.parquet')),
                              preserve_index=False)
    pq.write_table(rk, CACHE / 'ranking.parquet', compression='zstd')
    del rk

    for name, want in EXPECTED_ROWS.items():
        got = pq.ParquetFile(CACHE / f'{name}.parquet').metadata.num_rows
        if got != want:
            (CACHE / f'{name}.parquet').unlink(missing_ok=True)
            raise RuntimeError(f'cache build for {name} produced {got} rows, expected {want}; '
                               'cache deleted rather than left corrupt')


def load(which: str, phase: str | None = None, columns=None) -> pd.DataFrame:
    """Load 'training' or 'ranking'.

    phase   : 'DEP', 'ARR', or None for both.
    columns : column subset; 'PHASE_mvt' is pulled in automatically when filtering.
    """
    path = CACHE / f'{which}.parquet'
    if not path.exists():
        _build_cache()
    cols = None
    if columns is not None:
        cols = list(dict.fromkeys(list(columns) + (['PHASE_mvt'] if phase else [])))
    df = pd.read_parquet(path, columns=cols)
    if phase is not None:
        df = df[df.PHASE_mvt.eq(phase)]
        if columns is not None and 'PHASE_mvt' not in columns:
            df = df.drop(columns='PHASE_mvt')
    df = _downcast(df.reset_index(drop=True))
    for c in df.columns:
        if str(df[c].dtype) == 'category':
            df[c] = df[c].cat.remove_unused_categories()
    return df
