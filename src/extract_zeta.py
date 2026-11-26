"""
Extract sea-surface elevation (zeta) at the Kazhydromet gauge cells
from the hourly surface files data.NNN.vtk of the Sep-Nov 2025 run.

Usage (from a terminal, Python 3 with numpy):
    python extract_zeta.py  <folder with data.*.vtk>  <heights.vtk>  <stations.csv>

Output: zeta_stations.csv in the current folder
    columns: hour, time_utc, t_s, zeta_mean_m, zeta_absmax_m, <station>_cm ...
    hour 0 = 2025-09-01 00:00 UTC (first ERA5 record)
"""
import sys, csv, glob, os, re, datetime as dt
import numpy as np

T0 = dt.datetime(2025, 9, 1, 0, 0)          # first ERA5 record, UTC


def read_block(text, key, n):
    """Return the n numbers that follow the header line starting with `key`
    and the LOOKUP_TABLE line."""
    p = text.find(key)
    if p < 0:
        raise KeyError(key)
    p = text.find('LOOKUP_TABLE', p)
    p = text.find('\n', p) + 1
    return np.array(text[p:].split(None, n)[:n], dtype=float)


def main(folder, heights, stations):
    with open(heights) as f:
        htext = f.read()
    nx, ny = map(int, re.search(r'DIMENSIONS\s+(\d+)\s+(\d+)', htext).groups())
    n = nx * ny
    h = read_block(htext, 'SCALARS h', n)
    wet = h > 0

    st = list(csv.DictReader(open(stations)))
    idx = [int(s['index']) for s in st]
    for s in st:
        if not wet[int(s['index'])]:
            sys.exit(f"station {s['station']}: cell is land - check stations.csv")

    files = sorted(glob.glob(os.path.join(folder, 'data.*.vtk')),
                   key=lambda p: int(re.search(r'data\.(\d+)\.vtk$', p).group(1)))
    if not files:
        sys.exit('no data.*.vtk files found')

    out = open('zeta_stations.csv', 'w', newline='')
    w = csv.writer(out)
    w.writerow(['hour', 'time_utc', 't_s', 'zeta_mean_m', 'zeta_absmax_m'] +
               [s['station'] + '_cm' for s in st])

    for k, path in enumerate(files):
        m = int(re.search(r'data\.(\d+)\.vtk$', path).group(1))
        with open(path) as f:
            text = f.read()
        t = float(re.search(r'TIME\s+([-\d.eE+]+)', text).group(1))
        z = read_block(text, 'SCALARS z', n)
        zw = z[wet]
        w.writerow([m, (T0 + dt.timedelta(seconds=t)).strftime('%Y-%m-%d %H:%M'),
                    f'{t:.0f}', f'{zw.mean():.3e}', f'{np.abs(zw).max():.4f}'] +
                   [f'{100 * z[i]:.3f}' for i in idx])
        if k % 100 == 0:
            print(f'{k + 1}/{len(files)}  {path}')
    out.close()
    print('written zeta_stations.csv')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
