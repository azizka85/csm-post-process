r"""
Comparison of the 2003 run with the SZ-COMPAS reconstruction (Table 11 and Figure 11).

SZ-COMPAS: instantaneous currents (cm/s) every 4 h of 2003 on a uniform geographic grid,
46.7625-54.2125 E (299 points) x 36.5092-47.2892 N (589 points), first record 2003-01-01 04:00 UTC,
at the 1st, 4th and 8th sigma levels: 1, 7, 15 m where the depth exceeds 30 m, 3, 23, 50 % of the depth
where it is less. Land is filled with 10000.

Method (Section 2.8, experiment 3): at days 30, 60, ..., 330, 365 the SZ-COMPAS currents are averaged over
3 x 3 cells (about 6 km), interpolated bilinearly to the model grid (only where all four surrounding nodes
are sea), and compared with the model currents interpolated linearly in depth to the same levels and rotated
from grid to east-north components. Vector correlation c = sum(w1 conj(w2)) / sqrt(sum|w1|^2 sum|w2|^2),
w1 = model, w2 = SZ-COMPAS; arg c > 0 means the model current is anticlockwise of SZ-COMPAS.
Regions by model depth: North Caspian shelf h <= 10 m, shelf 10-50 m, intermediate 50-200 m, deep > 200 m.

Usage (Windows):
    python table11_szcompas.py --run "<2003 output folder>" --sz 1:sz_daily_1m 7:sz_daily_7m 15:sz_daily_15m
<2003 output folder> holds heights.vtk, surface\ and volume\ (files matched by their TIME).
SZ-COMPAS input per level, any of:
    sz_daily_<L>m          prefix of the .npy files written by extract_szcompas_daily.py (every day)
    sz_inst_12_<L>m.npz    the 12-instant extracts (arrays idx, u, v)
    <file>.nc              an unzipped NetCDF file (needs netCDF4; only the --days records are read)
--days: model days to compare (default: every day present in both the model output and SZ-COMPAS);
e.g. --days 30 60 90 120 150 180 210 240 270 300 330 365 for the 20 Nov table.
Writes table11.csv (all days), table11_seasons.csv (DJF, MAM, JJA, SON), table11_instants.csv,
figure11_points.csv and Figure11.png/.pdf. Needs numpy (+ matplotlib for the figure).
"""
import argparse, glob, os, csv, sys
import numpy as np

DAYS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
LON = np.linspace(46.7625, 54.2125, 299)
LAT = np.linspace(36.5092, 47.2892, 589)
FILL = 9999.0
REGIONS = [('North Caspian shelf, h <= 10 m', lambda h: h <= 10), ('Shelf, 10 < h <= 50 m', lambda h: (h > 10) & (h <= 50)),
           ('Intermediate, 50 < h <= 200 m', lambda h: (h > 50) & (h <= 200)), ('Deep basin, h > 200 m', lambda h: h > 200)]


def record(day):
    return day * 6 - 1                      # 4-hourly records, first at 04:00 on 1 January


# ---------------------------------------------------------------- model files
def read_text(path):
    with open(path) as fh:
        return fh.read()


def read_surface(path):
    L = read_text(path).split('\n')
    nx, ny, _ = map(int, L[4].split()[1:]); n = nx * ny
    out = {'t': float(L[1].split()[1]),
           'lonlat': np.array(' '.join(L[6:6 + n]).split(), float).reshape(ny, nx, 3)[..., :2]}
    i = 6 + n
    while i < len(L):
        w = L[i].split()
        if w and w[0] == 'VECTORS':
            out[w[1]] = np.array(' '.join(L[i + 1:i + 1 + n]).split(), float).reshape(ny, nx, -1)[..., :2]; i += 1 + n
        elif w and w[0] == 'SCALARS':
            out[w[1]] = np.array(L[i + 2:i + 2 + n], float).reshape(ny, nx); i += 2 + n
        else:
            i += 1
    return out


def read_volume(path):
    txt = read_text(path)
    head = txt[:400].split('\n')
    nx, ny, nz = map(int, head[4].split()[1:]); n = nx * ny * nz
    p0 = txt.index('\n', txt.index('POINTS')) + 1
    k = txt.index('VECTORS v')
    p1 = txt.rfind('FIELD', 0, k); p1 = p1 if p1 > 0 else txt.rfind('POINT_DATA', 0, k)
    z = np.fromstring(txt[p0:p1], sep=' ')[:3 * n].reshape(nz, ny, nx, 3)[..., 2]
    k = txt.index('\n', k) + 1
    v = np.fromstring(txt[k:], sep=' ')[:3 * n].reshape(nz, ny, nx, 3)[..., :2]
    return z, v


def by_day(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, '*.vtk')):
        with open(p) as fh:
            fh.readline(); t = float(fh.readline().split()[1])
        out[int(round(t / 86400))] = p
    return out


def grid_angle(LL):
    """Angle of the grid x axis from east (rad), from the longitude/latitude of the nodes."""
    dlon = np.gradient(LL[..., 0], axis=1); dlat = np.gradient(LL[..., 1], axis=1)
    return np.arctan2(dlat, dlon * np.cos(np.radians(LL[..., 1])))


def model_levels(z, v, H, wet):
    """Model current (grid components as complex) at the SZ-COMPAS levels, linear in depth."""
    h = H[wet]
    zc = z[:, wet]; w = v[:, wet, 0] + 1j * v[:, wet, 1]
    nz = zc.shape[0]; cols = np.arange(h.size)
    out = {}
    for lev, frac in ((1, 0.03), (7, 0.23), (15, 0.50)):
        t = np.where(h > 30, float(lev), frac * h)
        k = np.clip((zc < t).sum(0), 1, nz - 1)
        z0, z1 = zc[k - 1, cols], zc[k, cols]
        a = np.clip((t - z0) / (z1 - z0), 0, 1)            # constant above the first and below the last centre
        a = np.where(t <= zc[0], 0.0, a)
        k0 = np.where(t <= zc[0], 1, k)
        out[lev] = (1 - a) * w[k0 - 1, cols] + a * w[k0, cols]
    return out


# ---------------------------------------------------------------- SZ-COMPAS
class SZ:
    """Access to one SZ-COMPAS level: get(record) -> (u, v) in m/s with NaN on land."""

    def __init__(self, spec, days):
        lev, path = spec.split(':', 1)
        self.level = int(lev)
        if path.lower().endswith('.npz'):
            d = np.load(path)
            self.idx = [int(i) for i in d['idx']]; self.u = d['u']; self.v = d['v']
        elif path.lower().endswith('.nc'):
            import netCDF4
            ds = netCDF4.Dataset(path)
            cand = [n for n, var in ds.variables.items() if var.ndim == 3 and var.shape[1:] == (589, 299)]
            un = [n for n in cand if n.lower().startswith('u')]; vn = [n for n in cand if n.lower().startswith('v')]
            U, V = ds.variables[un[0]], ds.variables[vn[0]]
            U.set_auto_mask(False); V.set_auto_mask(False)
            self.idx = [record(d) for d in days if record(d) < U.shape[0]]
            self.u = np.stack([np.array(U[i]) for i in self.idx]); self.v = np.stack([np.array(V[i]) for i in self.idx])
        else:
            self.idx = [int(i) for i in np.load(path + '_idx.npy')]
            self.u = np.load(path + '_u.npy', mmap_mode='r'); self.v = np.load(path + '_v.npy', mmap_mode='r')
        self.pos = {i: k for k, i in enumerate(self.idx)}

    def has(self, rec):
        return rec in self.pos

    def get(self, rec):
        k = self.pos[rec]
        u = np.array(self.u[k], float); v = np.array(self.v[k], float)
        land = ~np.isfinite(u) | (np.abs(u) >= FILL) | ~np.isfinite(v) | (np.abs(v) >= FILL)
        return np.where(land, np.nan, u) / 100.0, np.where(land, np.nan, v) / 100.0


def box3(a):
    """3 x 3 mean ignoring land; land nodes stay land."""
    p = np.pad(a, 1, constant_values=np.nan)
    s = np.zeros_like(a); n = np.zeros_like(a)
    for dj in range(3):
        for di in range(3):
            b = p[dj:dj + a.shape[0], di:di + a.shape[1]]
            ok = ~np.isnan(b); s += np.where(ok, b, 0); n += ok
    return np.where(np.isnan(a), np.nan, s / np.maximum(n, 1))


def bilinear(a, lon, lat):
    fi = (lon - LON[0]) / (LON[1] - LON[0]); fj = (lat - LAT[0]) / (LAT[1] - LAT[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, 297); j0 = np.clip(np.floor(fj).astype(int), 0, 587)
    x = fi - i0; y = fj - j0
    val = (a[j0, i0] * (1 - x) * (1 - y) + a[j0, i0 + 1] * x * (1 - y) + a[j0 + 1, i0] * (1 - x) * y
           + a[j0 + 1, i0 + 1] * x * y)
    inside = (fi >= 0) & (fi <= 298) & (fj >= 0) & (fj <= 588)
    return np.where(inside, val, np.nan)               # NaN if any corner is land


def corr(w1, w2):
    c = (w1 * np.conj(w2)).sum() / np.sqrt((np.abs(w1) ** 2).sum() * (np.abs(w2) ** 2).sum())
    return np.abs(c), np.degrees(np.angle(c))


SEASONS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}


def season(day):
    import datetime
    m = (datetime.date(2003, 1, 1) + datetime.timedelta(days=day)).month
    return [s for s, ms in SEASONS.items() if m in ms][0]


def write(name, rows):
    with open(name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{x:.4g}' if isinstance(x, float) else x) for k, x in r.items()})


def summarise(inst, label):
    rows = []
    for name, _ in REGIONS:
        for lev in (1, 7, 15):
            r = [x for x in inst if x['region'] == name and x['level'] == lev]
            if not r:
                continue
            c = np.array([x['abs_c'] for x in r]); ang = np.array([x['arg_c_deg'] for x in r])
            q = np.array([x['speed_ratio'] for x in r])
            rows.append(dict(subset=label, region=name, level=lev, instants=len(r),
                             abs_c_median=np.median(c), abs_c_p10=np.percentile(c, 10), abs_c_p90=np.percentile(c, 90),
                             abs_c_min=c.min(), abs_c_max=c.max(), arg_c_median_deg=np.median(ang),
                             speed_ratio_median=np.median(q)))
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--run', required=True)
    a.add_argument('--sz', nargs=3, required=True)
    a.add_argument('--days', nargs='*', type=int, default=None)
    o = a.parse_args()

    Hs = read_surface(os.path.join(o.run, 'heights.vtk'))
    H, LL = Hs['h'], Hs['lonlat']; wet = H > 0; h = H[wet]
    theta = grid_angle(LL)[wet]
    lon, lat = LL[..., 0][wet], LL[..., 1][wet]
    surf = by_day(os.path.join(o.run, 'surface')); vol = by_day(os.path.join(o.run, 'volume'))
    days = o.days or sorted(d for d in vol if d in surf and 1 <= d <= 365)
    sz = {s.level: s for s in (SZ(spec, days) for spec in o.sz)}
    days = [d for d in days if d in vol and d in surf and all(sz[L].has(record(d)) for L in sz)]
    if not days:
        sys.exit('no day is present in both the model output and the SZ-COMPAS input')
    print(f'{len(days)} days: {days[0]} ... {days[-1]}', flush=True)

    inst = []; fig = []
    for day in days:
        z, v = read_volume(vol[day]); S = read_surface(surf[day])
        M = model_levels(z, v, H, wet)
        M = {lev: (w.real * np.cos(theta) - w.imag * np.sin(theta)) + 1j * (w.real * np.sin(theta) + w.imag * np.cos(theta))
             for lev, w in M.items()}
        W = np.abs(S['w10'][wet][:, 0] + 1j * S['w10'][wet][:, 1])
        for lev in (1, 7, 15):
            u, vv = sz[lev].get(record(day))
            s = bilinear(box3(u), lon, lat) + 1j * bilinear(box3(vv), lon, lat)
            for name, sel in REGIONS:
                m = sel(h) & ~np.isnan(s)
                c, ang = corr(M[lev][m], s[m])
                inst.append(dict(day=day, season=season(day), region=name, level=lev, n=int(m.sum()), abs_c=c,
                                 arg_c_deg=ang, speed_ratio=np.median(np.abs(M[lev][m])) / np.median(np.abs(s[m])),
                                 model_median_speed=np.median(np.abs(M[lev][m])), sz_median_speed=np.median(np.abs(s[m]))))
                if lev == 1 and name.startswith('North'):
                    Wm = np.median(W[m])
                    fig.append(dict(day=day, season=season(day), wind=Wm, model_pct=100 * np.median(np.abs(M[1][m])) / Wm,
                                    sz_pct=100 * np.median(np.abs(s[m])) / Wm))
        if day % 30 == 0 or day == days[-1]:
            print(f'day {day} done', flush=True)

    write('table11_instants.csv', inst)
    table = summarise(inst, 'all')
    write('table11.csv', table)
    write('table11_seasons.csv', [r for s_ in SEASONS for r in summarise([x for x in inst if x['season'] == s_], s_)])
    write('figure11_points.csv', fig)
    print(f"\n{'region':32s} lev  |c| median (10-90 %)    arg c   speed ratio   [{len(days)} days]")
    for r in table:
        print(f"{r['region']:32s} {r['level']:3d}  {r['abs_c_median']:.2f} ({r['abs_c_p10']:.2f}-{r['abs_c_p90']:.2f})"
              f"   {r['arg_c_median_deg']:+5.0f}   {r['speed_ratio_median']:.2f}")

    sh = [x for x in inst if x['region'].startswith('North')]
    for lev in (7, 15):
        rm = np.median([x['model_median_speed'] for x in sh if x['level'] == lev]) / np.median([x['model_median_speed'] for x in sh if x['level'] == 1])
        rs = np.median([x['sz_median_speed'] for x in sh if x['level'] == lev]) / np.median([x['sz_median_speed'] for x in sh if x['level'] == 1])
        print(f'North Caspian shelf, median speed at {lev} m / 1 m: model {rm:.2f}, SZ-COMPAS {rs:.2f}')

    if len(fig) < 3:
        return
    Wv = np.array([f['wind'] for f in fig])
    bm = np.polyfit(np.log(Wv), np.log([f['model_pct'] for f in fig]), 1)
    bs = np.polyfit(np.log(Wv), np.log([f['sz_pct'] for f in fig]), 1)
    print(f'Figure 11 power laws: model ~ W^{bm[0]:.2f}, SZ-COMPAS ~ W^{bs[0]:.2f}')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=(4.8, 4.5))
        ms = 4 if len(fig) > 30 else 7
        x = np.linspace(Wv.min() * 0.95, Wv.max() * 1.05, 50)
        ax.loglog(Wv, [p['model_pct'] for p in fig], 'o', ms=ms, alpha=0.6, label=f'Present model ($\\propto W^{{{bm[0]:.2f}}}$)')
        ax.loglog(Wv, [p['sz_pct'] for p in fig], 's', ms=ms, alpha=0.6, label=f'SZ-COMPAS ($\\propto W^{{{bs[0]:.2f}}}$)')
        ax.loglog(x, np.exp(np.polyval(bm, np.log(x))), 'C0-'); ax.loglog(x, np.exp(np.polyval(bs, np.log(x))), 'C1-')
        ax.set_xlabel('Wind speed (m s$^{-1}$)'); ax.set_ylabel('Current / wind speed at 1 m (%)')
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
        f.tight_layout(); f.savefig('Figure11.png', dpi=200); f.savefig('Figure11.pdf')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
