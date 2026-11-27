r"""
Vertical grid convergence of the idealised column runs (the metrics of Table 5) for any viscosity case.

For each grid it computes the relative L2 error of the velocity profile against a reference run,
the observed order of accuracy p from the errors of the two finest uniform grids, the apparent order
from three uniform grids with the procedure of Celik et al. (2008) for unequal refinement ratios,
and the grid convergence index (GCI) of the finest uniform grid.

Usage (one case per call; files are the volume output data.300.vtk of each run, at t = 150 h):
    python vertical_convergence.py --name "case 1" --mus 0.005 --mum 0.4 ^
        --ref R1-500\data.300.vtk ^
        --grids R1-50\data.300.vtk R1-125\data.300.vtk R1-250\data.300.vtk ^
        --extra R1-45\data.300.vtk
--grids: three or more uniform grids, any order; --extra: optional non-uniform grids (e.g. TriplePoint Nz = 45),
reported with their error only. --mus/--mum/--ht give the viscosity profile of Eq. (17): mu_s at the surface, decreasing linearly to mu_m
at depth ht and constant below; for constant viscosity give the same value twice. The output reports
l = sqrt(2 mu_s / f), dz1/l, and the largest dz_k/l_k over all layers, with l_k from the smallest
viscosity inside layer k.
--ref is optional: without it only the apparent order and the GCI are computed (they need no reference).
Options: --ht 25, --f 1.2e-4 (Coriolis), --h 260 (depth, m), --fs 1.25 (GCI safety factor).
Appends one row per case to vertical_convergence.csv and prints a summary. Needs Python 3 and numpy.
"""
import argparse, csv, os
import numpy as np


def load_profile(path):
    """Layer-centre depths (m) and horizontal velocity (u, v) of the central column of a volume file."""
    with open(path) as fh:
        txt = fh.read()
    head = txt[:400].split('\n')
    nx, ny, nz = map(int, head[4].split()[1:])
    n = nx * ny * nz
    p0 = txt.index('\n', txt.index('POINTS')) + 1
    k = txt.index('VECTORS v')
    p1 = txt.rfind('FIELD', 0, k)
    p1 = p1 if p1 > 0 else txt.rfind('POINT_DATA', 0, k)
    pts = np.fromstring(txt[p0:p1], sep=' ')[:3 * n].reshape(nz, ny, nx, 3)
    k = txt.index('\n', k) + 1
    vel = np.fromstring(txt[k:], sep=' ')[:3 * n].reshape(nz, ny, nx, 3)
    j, i = ny // 2, nx // 2                     # horizontally uniform problem: any interior column
    return pts[:, j, i, 2], vel[:, j, i, :2]


def thickness(z, h):
    """Layer thicknesses from the layer-centre depths (interfaces rebuilt from the surface down)."""
    s = np.zeros(len(z) + 1)
    for k in range(len(z)):
        s[k + 1] = 2 * z[k] - s[k]
    if abs(s[-1] - h) > 1e-3 * h or np.any(np.diff(s) <= 0):     # fall back to mid-points
        s = np.r_[0, (z[1:] + z[:-1]) / 2, h]
    return np.diff(s)


def max_ratio(z, h, mus, mum, ht, f):
    """Largest dz_k / l_k over the layers, l_k = sqrt(2 mu_min,k / f)."""
    w = thickness(z, h)
    top = np.r_[0, np.cumsum(w)[:-1]]
    bot = top + w
    nu = lambda d: np.where(d <= ht, mus - (mus - mum) * np.minimum(d, ht) / ht, mum)
    numin = np.minimum(nu(top), nu(bot))
    return (w / np.sqrt(2 * numin / f)).max()


def rel_l2(zc, uc, zr, ur, h):
    """Relative L2 norm (%) of the difference between a profile and a reference interpolated to its levels."""
    ui = np.column_stack([np.interp(zc, zr, ur[:, 0]), np.interp(zc, zr, ur[:, 1])])
    w = thickness(zc, h)
    return 100 * np.sqrt((((uc - ui) ** 2).sum(1) * w).sum() / ((ui ** 2).sum(1) * w).sum())


def celik(e21, e32, r21, r32, fs):
    """Apparent order and fine-grid GCI (%) for unequal ratios (Celik et al. 2008), with
    e21 = ||phi2 - phi1||, e32 = ||phi3 - phi2|| as relative norms (%); 1 = finest grid."""
    p = abs(np.log(e32 / e21)) / np.log(r21)
    for _ in range(200):
        q = np.log((r21 ** p - 1) / (r32 ** p - 1))
        p_new = abs(np.log(e32 / e21) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    return p, fs * e21 / (r21 ** p - 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--name', required=True)
    a.add_argument('--ref', default=None)
    a.add_argument('--grids', nargs='+', required=True)
    a.add_argument('--extra', nargs='*', default=[])
    a.add_argument('--mus', type=float, required=True)
    a.add_argument('--mum', type=float, required=True)
    a.add_argument('--ht', type=float, default=25.0)
    a.add_argument('--f', type=float, default=1.2e-4)
    a.add_argument('--h', type=float, default=260.0)
    a.add_argument('--fs', type=float, default=1.25)
    o = a.parse_args()

    ref = load_profile(o.ref) if o.ref else None
    grids = sorted((load_profile(p) for p in o.grids), key=lambda g: len(g[0]))      # coarse -> fine
    nz = [len(g[0]) for g in grids]
    ell = np.sqrt(2 * o.mus / o.f)
    row = {'case': o.name, 'mu_s': o.mus, 'mu_m': o.mum, 'ht_m': o.ht, 'l_m': round(ell, 1),
           'Nz_ref': len(ref[0]) if ref else ''}
    err = []
    for n, (z, u) in zip(nz, grids):
        if ref:
            err.append(rel_l2(z, u, ref[0], ref[1], o.h))
            row[f'L2_Nz{n}_pct'] = f'{err[-1]:.3g}'
        row[f'dz1_over_l_Nz{n}'] = f'{2 * z[0] / ell:.3g}'
        row[f'max_dz_over_lk_Nz{n}'] = f'{max_ratio(z, o.h, o.mus, o.mum, o.ht, o.f):.3g}'
    for p in o.extra:
        z, u = load_profile(p)
        if ref:
            row[f'L2_extra_Nz{len(z)}_pct'] = f'{rel_l2(z, u, ref[0], ref[1], o.h):.3g}'
        row[f'dz1_over_l_extra_Nz{len(z)}'] = f'{2 * z[0] / ell:.3g}'
        row[f'max_dz_over_lk_extra_Nz{len(z)}'] = f'{max_ratio(z, o.h, o.mus, o.mum, o.ht, o.f):.3g}'
    if ref:
        # observed order from the errors of the two finest grids against the reference (as in Table 5)
        row['p_observed'] = f'{np.log(err[-2] / err[-1]) / np.log(nz[-1] / nz[-2]):.2f}'
    # apparent order and GCI from the three finest grids alone (no reference needed)
    (z1, u1), (z2, u2), (z3, u3) = grids[-1], grids[-2], grids[-3]
    e21 = rel_l2(z2, u2, z1, u1, o.h)        # medium against fine
    e32 = rel_l2(z3, u3, z2, u2, o.h)        # coarse against medium
    p_cel, gci = celik(e21, e32, nz[-1] / nz[-2], nz[-2] / nz[-3], o.fs)
    row.update({'p_apparent_Celik': f'{p_cel:.2f}', 'GCI_fine_pct': f'{gci:.3g}',
                'e21_pct': f'{e21:.3g}', 'e32_pct': f'{e32:.3g}'})

    for k, v in row.items():
        print(f'{k:>24}: {v}')
    new = not os.path.exists('vertical_convergence.csv')
    with open('vertical_convergence.csv', 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['key', 'value', 'case'])
        for k, v in row.items():
            w.writerow([k, v, o.name])


if __name__ == '__main__':
    main()
