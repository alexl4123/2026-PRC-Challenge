"""Count OPDI surface events within R km of the ten ARPs, reading only the row groups holding the wanted
event types (files are sorted by type) and only the type/lat/lon columns, over HTTP range requests."""
import sys, collections, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
from httprange import HTTPRange
ARP = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.4775, -0.4614), 'EHAM': (52.3086, 4.7639),
       'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626), 'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389),
       'LSZH': (47.4647, 8.5492), 'LTFM': (41.2622, 28.7425)}
WANT = {'exit-parking_position', 'entry-parking_position', 'take-off', 'entry-taxiway', 'exit-taxiway', 'entry-runway', 'exit-apron'}
def main(interval, size, R=5.0):
    url = f'https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{interval}.parquet'
    f = HTTPRange(url, size=size); pf = pq.ParquetFile(f)
    res = collections.defaultdict(collections.Counter)
    for i in range(pf.metadata.num_row_groups):
        types = set(pf.read_row_group(i, columns=['type']).column('type').to_pylist())
        if not types & WANT: continue
        t = pf.read_row_group(i, columns=['type', 'latitude', 'longitude'])
        ty = np.array(t.column('type').to_pylist()); la = t.column('latitude').to_numpy(); lo = t.column('longitude').to_numpy()
        for a, (a_la, a_lo) in ARP.items():
            dk = 6371 * np.hypot(np.radians(lo - a_lo) * np.cos(np.radians(a_la)), np.radians(la - a_la))
            m = dk <= R
            for k, v in collections.Counter(ty[m]).items():
                if k in WANT: res[a][k] += v
    print(interval, 'bytes read', f.nbytes)
    cols = ['exit-parking_position', 'exit-apron', 'entry-taxiway', 'entry-runway', 'take-off']
    print('APT  ' + ' '.join(f'{c:>22s}' for c in cols))
    for a in ARP: print(a, ' '.join(f'{res[a][c]:22d}' for c in cols))
    f.log(f'OPDI surface events {interval}')
if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
