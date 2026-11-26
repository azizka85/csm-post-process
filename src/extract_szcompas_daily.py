r"""
Extract daily SZ-COMPAS currents (one record per day at 00:00 UTC) from the zipped NetCDF archives.

The instantaneous files hold 2190 records of 2003, every 4 h from 2003-01-01 04:00 UTC; the record at
00:00 UTC of 2003-01-01 + d days (model day d, d = 1 ... 365) has index 6 d - 1.

Usage (Windows), in a folder with at least about 15 GB free:
    python extract_szcompas_daily.py "<folder with the three .nc.zip files>"
Options:
    --keep-nc     keep the unzipped NetCDF files (otherwise each is deleted after extraction)
    --every 6     record step (6 = daily; 1 = every 4 h, about 6 times more disk)
Writes, for each level L = 1, 7, 15 m:
    sz_daily_<L>m_u.npy, sz_daily_<L>m_v.npy   float32 arrays (days, 589, 299), cm/s, land = NaN
    sz_daily_<L>m_idx.npy                         record indices
Latitude is south to north (row 0 = 36.5092 N), longitude west to east (column 0 = 46.7625 E).
Needs numpy and netCDF4 (pip install netCDF4).
"""
import sys, os, glob, re, zipfile, shutil
import numpy as np

FILL = 9999.0


def find_uv(ds):
    cand = [n for n, v in ds.variables.items() if v.ndim == 3 and v.shape[1:] == (589, 299)]
    un = [n for n in cand if n.lower().startswith('u')]
    vn = [n for n in cand if n.lower().startswith('v')]
    if not un or not vn:
        sys.exit('cannot find u, v of shape (time, 589, 299); 3-D variables found: '
                 + ', '.join(f'{n}{ds.variables[n].shape}' for n in ds.variables if ds.variables[n].ndim == 3))
    return ds.variables[un[0]], ds.variables[vn[0]]


def orientation(ds):
    """True if the latitude coordinate is stored north to south (then rows are flipped)."""
    for n, v in ds.variables.items():
        if v.ndim == 1 and v.shape[0] == 589 and 'lat' in n.lower():
            lat = np.array(v[:])
            return lat[0] > lat[-1]
    return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        print(__doc__); sys.exit(1)
    keep = '--keep-nc' in sys.argv
    every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 6
    import netCDF4
    zips = sorted(glob.glob(os.path.join(args[0], '*uv_at_*_m*.zip')))
    if not zips:
        sys.exit('no *uv_at_*_m*.zip files in ' + args[0])
    for z in zips:
        if '_mm' in os.path.basename(z).lower():
            continue                                    # monthly means are not used
        lev = int(re.search(r'uv_at_(\d+)_m', os.path.basename(z)).group(1))
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith('.nc')][0]
            nc = os.path.join('.', os.path.basename(name))
            if not os.path.exists(nc):
                print(f'{lev} m: unzipping {name} ...', flush=True)
                with zf.open(name) as src, open(nc, 'wb') as dst:
                    shutil.copyfileobj(src, dst, length=16 << 20)
        ds = netCDF4.Dataset(nc)
        U, V = find_uv(ds)
        U.set_auto_mask(False); V.set_auto_mask(False)
        flip = orientation(ds)
        nrec = U.shape[0]
        idx = np.arange(every - 1, nrec, every)
        print(f'{lev} m: {nrec} records, extracting {len(idx)} (every {every}), latitude {"N-S, flipped" if flip else "S-N"}', flush=True)
        uo = np.lib.format.open_memmap(f'sz_daily_{lev}m_u.npy', mode='w+', dtype=np.float32, shape=(len(idx), 589, 299))
        vo = np.lib.format.open_memmap(f'sz_daily_{lev}m_v.npy', mode='w+', dtype=np.float32, shape=(len(idx), 589, 299))
        for k, r in enumerate(idx):
            u = np.array(U[r], dtype=np.float32); v = np.array(V[r], dtype=np.float32)
            if flip:
                u, v = u[::-1], v[::-1]
            land = ~np.isfinite(u) | (np.abs(u) >= FILL) | ~np.isfinite(v) | (np.abs(v) >= FILL)
            uo[k] = np.where(land, np.nan, u); vo[k] = np.where(land, np.nan, v)
            if k % 60 == 0:
                print(f'  record {r} ({k + 1}/{len(idx)}), sea points {int((~land).sum())}', flush=True)
        uo.flush(); vo.flush(); del uo, vo
        np.save(f'sz_daily_{lev}m_idx.npy', idx)
        ds.close()
        if not keep:
            os.remove(nc)
        print(f'{lev} m: written sz_daily_{lev}m_u.npy, _v.npy, _idx.npy', flush=True)


if __name__ == '__main__':
    main()
