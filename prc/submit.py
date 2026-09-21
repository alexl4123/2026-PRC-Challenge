"""Building and validating a submission file.

The organizers' validator rejects a submission if any MVT_ID_mvt does not match,
any row is missing, or any extra row is present. Submission dtypes are not
documented anywhere, so the safe construction is to start from the official
template and overwrite only the value column, leaving its dtype untouched.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATA, OUTPUTS

TEAM = 'understanding-accordion'
BUCKET = 'prc-2026-understanding-accordion'
N_ROWS = 344_841


def build(mvt_id, pred, version: int, clip=(0.0, 7200.0)):
    """Write <team>_v<N>.parquet into outputs/ and validate it. Returns the path."""
    template = pd.read_parquet(DATA / 'submitting.parquet')
    # NOTE: the template's value column is int32 ON DISK but all-null, so pandas
    # hands it back as float64. Writing that back out would give the submission a
    # different arrow schema from the official template. Read the real schema
    # from the parquet metadata instead and write to it explicitly.
    tmpl_schema = pq.ParquetFile(DATA / 'submitting.parquet').schema_arrow

    p = np.clip(np.asarray(pred, float), *clip)
    s = pd.Series(p, index=pd.Index(np.asarray(mvt_id), name='MVT_ID_mvt'))
    if s.index.duplicated().any():
        raise ValueError('duplicate MVT_ID_mvt in predictions')

    out = template.copy()
    mapped = out.MVT_ID_mvt.map(s)
    if mapped.isna().any():
        raise ValueError(f'{int(mapped.isna().sum())} template ids have no prediction')
    # Round before casting: the target is an integer second count, and rounding
    # costs ~0.29 s of RMSE in quadrature, i.e. nothing at a 240 s scale.
    out['TAXITIME_SEC_mvt'] = np.rint(mapped.to_numpy()).astype(np.int32)

    path = OUTPUTS / f'{TEAM}_v{version}.parquet'
    pq.write_table(pa.Table.from_pandas(out, schema=tmpl_schema, preserve_index=False), path)
    validate(path)
    return path


def validate(path):
    """Re-read from disk and check everything the official validator checks."""
    sub = pd.read_parquet(path)
    template = pd.read_parquet(DATA / 'submitting.parquet')
    ranking = pd.read_parquet(DATA / 'ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt'])
    dep_ids = set(ranking.loc[ranking.PHASE_mvt.eq('DEP'), 'MVT_ID_mvt'])

    problems = []
    if list(sub.columns) != ['MVT_ID_mvt', 'TAXITIME_SEC_mvt']:
        problems.append(f'columns are {list(sub.columns)}, expected '
                        "['MVT_ID_mvt', 'TAXITIME_SEC_mvt']")
    if len(sub) != N_ROWS:
        problems.append(f'{len(sub)} rows, expected {N_ROWS}')
    if set(sub.MVT_ID_mvt) != dep_ids:
        problems.append(f'id set mismatch vs ranking departures: '
                        f'{len(set(sub.MVT_ID_mvt) - dep_ids)} extra, '
                        f'{len(dep_ids - set(sub.MVT_ID_mvt))} missing')
    if sub.MVT_ID_mvt.duplicated().any():
        problems.append('duplicate ids')
    if sub.TAXITIME_SEC_mvt.isna().any():
        problems.append(f'{int(sub.TAXITIME_SEC_mvt.isna().sum())} null predictions')
    if sub.MVT_ID_mvt.dtype != template.MVT_ID_mvt.dtype:
        problems.append(f'id dtype {sub.MVT_ID_mvt.dtype} != template '
                        f'{template.MVT_ID_mvt.dtype}')
    got = pq.ParquetFile(path).schema_arrow
    want = pq.ParquetFile(DATA / 'submitting.parquet').schema_arrow
    for f in ('MVT_ID_mvt', 'TAXITIME_SEC_mvt'):
        if got.field(f).type != want.field(f).type:
            problems.append(f'on-disk arrow type for {f} is {got.field(f).type}, '
                            f'template has {want.field(f).type}')
    if problems:
        raise ValueError('submission INVALID:\n  ' + '\n  '.join(problems))

    print(f'  VALID: {path.name}  {len(sub):,} rows  dtype={sub.TAXITIME_SEC_mvt.dtype}  '
          f'pred min/med/max = {sub.TAXITIME_SEC_mvt.min()}/'
          f'{int(sub.TAXITIME_SEC_mvt.median())}/{sub.TAXITIME_SEC_mvt.max()}')
    print(f'  upload: mc cp {path} osn/{BUCKET}/')
    return True
