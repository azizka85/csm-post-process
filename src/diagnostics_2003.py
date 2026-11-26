r"""
Daily diagnostics of the one-year 2003 Caspian run (Tables 10 and the stability text, Sections 3.4.2 and 4.3).

Usage (Windows, quotes matter):
    python diagnostics_2003.py "<path>"
<path> is the deepest output folder, the one that contains heights.vtk, surface\ and volume\.
Options:
    --surface-only   skip the volume files (no kinetic energy columns), runs in about a minute
Needs only Python 3 and numpy. Reading the 365 volume files (about 70 MB each) takes roughly 5-10 minutes.

Writes, in the current folder:
    daily_2003.csv     one row per day: volume, kinetic energy by depth class, deviation energy in the
                       deep basin, maximum speeds, checkerboard amplitudes, Table 10 quantities
    table10_2003.csv   the same columns for days 30, 60, ..., 330, 365 only
"""
import sys, os, glob, csv
import numpy as np

F = 0.98e-4          # Coriolis parameter, 1/s
DX = 6500.0          # grid spacing, m
RHO = 1025.0         # water density, kg/m3
DAYS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
CLASSES = {'deep': lambda h: h > 200, 'mid': lambda h: (h > 50) & (h <= 200), 'shelf': lambda h: h <= 50}


def header_time(path):
    with open(path) as fh:
        fh.readline()
        w = fh.readline().split()
    return float(w[1])


def read_surface(path):
    with open(path) as fh:
        L = fh.read().split('\n')
    nx, ny, _ = map(int, L[4].split()[1:])
    n = nx * ny
    out = {'t': float(L[1].split()[1])}
    i = 6 + n
    while i < len(L):
        w = L[i].split()
        if w and w[0] == 'VECTORS':
            out[w[1]] = np.array(' '.join(L[i + 1:i + 1 + n]).split(), float).reshape(ny, nx, -1)[..., :2]
            i += 1 + n
        elif w and w[0] == 'SCALARS':
            out[w[1]] = np.array(L[i + 2:i + 2 + n], float).reshape(ny, nx)
            i += 2 + n
        else:
            i += 1
    return out


def read_volume(path, want_points=False):
    """Velocity at the layer centres, shape (nz, ny, nx, 2); optionally the depths of the layer centres."""
    with open(path) as fh:
        txt = fh.read()
    head = txt[:400].split('\n')
    nx, ny, nz = map(int, head[4].split()[1:])
    n = nx * ny * nz
    k = txt.index('VECTORS v')
    k = txt.index('\n', k) + 1
    v = np.fromstring(txt[k:], sep=' ')[:3 * n].reshape(nz, ny, nx, 3)[..., :2]
    z = None
    if want_points:
        p0 = txt.index('\n', txt.index('POINTS')) + 1
        p1 = txt.index('FIELD') if 'FIELD' in txt[:k] else txt.index('POINT_DATA')
        z = np.fromstring(txt[p0:p1], sep=' ')[:3 * n].reshape(nz, ny, nx, 3)[..., 2]
    return v, z


def layer_thickness(zc, H):
    """Dimensionless layer thicknesses from the layer-centre depths of the deepest column."""
    j, i = np.unravel_index(np.argmax(H), H.shape)
    sig = zc[:, j, i] / H[j, i]
    s = np.zeros(len(sig) + 1)
    for k in range(len(sig)):
        s[k + 1] = 2 * sig[k] - s[k]
    if abs(s[-1] - 1) > 1e-6 or np.any(np.diff(s) <= 0):
        sys.exit('could not reconstruct the layer thicknesses from the volume file')
    return np.diff(s)


def checkerboard(phi, wet):
    """Amplitude of the (-1)^(i+j) mode on 3x3 wet blocks: exact for a pure checkerboard,
    while a smooth field contributes only through its fourth derivatives."""
    a = (phi[:-1, :-1] - phi[1:, :-1] - phi[:-1, 1:] + phi[1:, 1:]) / 4.0
    jj, ii = np.indices(a.shape)
    b = np.where((ii + jj) % 2 == 0, a, -a)
    c = (b[:-1, :-1] + b[1:, :-1] + b[:-1, 1:] + b[1:, 1:]) / 4.0
    m = np.ones(c.shape, bool)
    for dj in range(3):
        for di in range(3):
            m &= wet[dj:dj + c.shape[0], di:di + c.shape[1]]
    return np.abs(c[m])


def advection_ratio(u, v, wet):
    """|u.grad u| / |f u| for the surface current, central differences, interior wet cells only."""
    m = wet[1:-1, 1:-1] & wet[2:, 1:-1] & wet[:-2, 1:-1] & wet[1:-1, 2:] & wet[1:-1, :-2]
    uc, vc = u[1:-1, 1:-1], v[1:-1, 1:-1]
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * DX); uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * DX)
    vx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * DX); vy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * DX)
    ax, ay = uc * ux + vc * uy, uc * vx + vc * vy
    sp = np.hypot(uc, vc)
    m = m & (sp > 1e-3)
    return np.hypot(ax, ay)[m] / (F * sp[m])


def files_by_day(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, '*.vtk')):
        out[int(round(header_time(p) / 86400.0))] = p
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        print(__doc__); sys.exit(1)
    root = args[0]
    surface_only = '--surface-only' in sys.argv
    H = read_surface(os.path.join(root, 'heights.vtk'))['h']
    wet = H > 0
    A = DX * DX
    surf = files_by_day(os.path.join(root, 'surface'))
    vol = {} if surface_only else files_by_day(os.path.join(root, 'volume'))
    if not surf:
        sys.exit('no surface files in ' + os.path.join(root, 'surface'))
    ds = None
    vol0 = None
    rows = []
    for day in sorted(surf):
        S = read_surface(surf[day])
        z = S['z']
        h, zw = H[wet], z[wet]
        us = np.hypot(*S['vs'][wet].T); ua = np.hypot(*S['va'][wet].T)
        volume = ((h + zw).sum()) * A
        if vol0 is None:
            vol0 = volume
        r = np.abs(zw) / h
        cbz = checkerboard(z, wet)
        cbu = np.hypot(checkerboard(S['va'][..., 0], wet), checkerboard(S['va'][..., 1], wet))
        adv = advection_ratio(S['vs'][..., 0], S['vs'][..., 1], wet)
        dev = np.hypot(*(S['vs'][wet] - S['va'][wet]).T)
        row = dict(day=day,
                   wind_mean_ms=np.hypot(*S['w10'][wet].T).mean(),
                   zeta_mean_m=zw.mean(), volume_rel_change=volume / vol0 - 1,
                   zeta_min_cm=100 * zw.min(), zeta_max_cm=100 * zw.max(),
                   max_abs_zeta_over_h=r.max(), pct_cells_zeta_over_h_gt_0p1=100 * (r > 0.1).mean(),
                   us_max_ms=us.max(), ua_max_ms=ua.max())
        for name, sel in CLASSES.items():
            row[f'median_depthmean_{name}_cms'] = 100 * np.median(ua[sel(h)])
        row['median_surface_minus_mean_deep_ms'] = np.median(dev[CLASSES['deep'](h)])
        row['adv_coriolis_median'] = np.median(adv)
        row['adv_coriolis_p95'] = np.percentile(adv, 95)
        row['chk_zeta_p99_cm'] = 100 * np.percentile(cbz, 99); row['chk_zeta_max_cm'] = 100 * cbz.max()
        row['chk_u_p99_cms'] = 100 * np.percentile(cbu, 99); row['chk_u_max_cms'] = 100 * cbu.max()
        if day in vol:
            V, zc = read_volume(vol[day], want_points=ds is None)
            if ds is None:
                ds = layer_thickness(zc, H)
            Vw = V[:, wet, :]                                   # (nz, ncell, 2)
            ubar = np.einsum('k,kcd->cd', ds, Vw)               # depth mean from the profile
            row['depthmean_check_ms'] = np.abs(ubar - S['va'][wet]).max()
            e_tot = 0.5 * RHO * h * np.einsum('k,kc->c', ds, (Vw ** 2).sum(2)) * A        # J per column
            e_dev = 0.5 * RHO * h * np.einsum('k,kc->c', ds, ((Vw - ubar) ** 2).sum(2)) * A
            for name, sel in CLASSES.items():
                m = sel(h)
                row[f'KE_{name}_PJ'] = e_tot[m].sum() / 1e15
                row[f'KEdev_{name}_PJ'] = e_dev[m].sum() / 1e15
            row['KE_total_PJ'] = e_tot.sum() / 1e15
        rows.append(row)
        print(f"day {day:3d}  mean zeta {row['zeta_mean_m']:+.1e} m  us_max {row['us_max_ms']:.2f} m/s  "
              + (f"KE {row['KE_total_PJ']:.3f} PJ  KEdev_deep {row['KEdev_deep_PJ']:.3f} PJ  " if 'KE_total_PJ' in row else '')
              + f"chk_zeta {row['chk_zeta_p99_cm']:.3f} cm  chk_u {row['chk_u_p99_cms']:.3f} cm/s", flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    for name, sel in (('daily_2003.csv', lambda r: True), ('table10_2003.csv', lambda r: r['day'] in DAYS)):
        with open(name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                if sel(r):
                    w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    print('written daily_2003.csv and table10_2003.csv,', len(rows), 'days')


if __name__ == '__main__':
    main()
