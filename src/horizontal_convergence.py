r"""
Horizontal grid convergence of the Caspian runs (Table 9).

Compares nested grids (each finer grid has half the spacing, and every second point of it lies on the
coarser grid) at one output time:
  upper part: RMS difference of the surface current, depth-mean current and elevation between successive
              grids at their shared sea points, relative to the RMS value on the finer grid, and the
              apparent order p = ln(e_coarse / e_fine) / ln 2 of each triplet;
  lower part: basin integrals relative to the finest grid: kinetic energy 1/2 rho (h + zeta) |u_bar|^2 dA,
              mean surface speed, mean depth-mean speed, mean bed stress, standard deviation of zeta,
              largest |zeta|, sea area and total volume.

Usage: one "spacing heights surface-file" triple per grid, coarse to fine, e.g. (Windows):
    python horizontal_convergence.py ^
      52000 "dx=52000\heights.vtk" "dx=52000\surface\data.941.vtk" ^
      26000 "dx=26000\heights.vtk" "dx=26000\surface\data.941.vtk" ^
      13000 "dx=13000\heights.vtk" "dx=13000\surface\data.941.vtk" ^
      6500  "dx=6500\heights.vtk"  "dx=6500\surface\data.941.vtk" ^
      3250  "dx=3250\heights.vtk"  "dx=3250\surface\data.941.vtk"
Writes horizontal_convergence.csv. Needs Python 3 and numpy.
"""
import sys, csv
import numpy as np

RHO = 1025.0


def read_vtk(path):
    with open(path) as fh:
        L = fh.read().split('\n')
    nx, ny, _ = map(int, L[4].split()[1:])
    n = nx * ny
    out = {'nx': nx, 'ny': ny, 'time': L[1].strip(),
           'lonlat': np.array(' '.join(L[6:6 + n]).split(), float).reshape(ny, nx, 3)[..., :2]}
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


def shared(C, F):
    """Index slices of the shared points: coarse (j, i) = fine (2j, 2i)."""
    jj = min(C['ny'], (F['ny'] + 1) // 2); ii = min(C['nx'], (F['nx'] + 1) // 2)
    cs = (slice(0, jj), slice(0, ii)); fs = (slice(0, 2 * jj, 2), slice(0, 2 * ii, 2))
    d = np.abs(C['lonlat'][cs] - F['lonlat'][fs]).max()
    if d > 1e-3:
        sys.exit(f'grids are not nested: shared points differ by {d:.4f} degrees')
    return cs, fs


def pair(C, F):
    cs, fs = shared(C, F)
    wet = (C['h'][cs] > 0) & (F['h'][fs] > 0)
    r = {}
    for k in ('vs', 'va'):
        a, b = C[k][cs][wet], F[k][fs][wet]
        r[k] = 100 * np.sqrt(((a - b) ** 2).sum(1).mean() / (b ** 2).sum(1).mean())
    a, b = C['z'][cs][wet], F['z'][fs][wet]
    r['z'] = 100 * np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
    r['n'] = int(wet.sum())
    return r


def integrals(D, dx):
    w = D['h'] > 0
    A = dx * dx
    h, z = D['h'][w], D['z'][w]
    ua = np.hypot(*D['va'][w].T); us = np.hypot(*D['vs'][w].T); tb = np.hypot(*D['tb'][w].T)
    return {'kinetic_energy': 0.5 * RHO * ((h + z) * ua ** 2).sum() * A,
            'mean_surface_speed': us.mean(), 'mean_depthmean_speed': ua.mean(), 'mean_bed_stress': tb.mean(),
            'std_zeta': z.std(), 'max_abs_zeta': np.abs(z).max(), 'sea_area': w.sum() * A,
            'volume': (h + z).sum() * A, 'mean_wind': np.hypot(*D['w10'][w].T).mean()}


def main():
    a = sys.argv[1:]
    if len(a) < 9 or len(a) % 3:
        print(__doc__); sys.exit(1)
    G = []
    for k in range(0, len(a), 3):
        dx = float(a[k]); H = read_vtk(a[k + 1]); D = read_vtk(a[k + 2]); D['h'] = H['h']; D['dx'] = dx
        print(f"{dx / 1000:g} km: {D['nx']} x {D['ny']}, {(D['h'] > 0).sum()} sea points, {D['time']}")
        G.append(D)
    G.sort(key=lambda D: -D['dx'])
    rows = []
    P = [pair(C, F) for C, F in zip(G[:-1], G[1:])]
    print('\nRMS difference between successive grids (% of the finer grid):')
    for (C, F), r in zip(zip(G[:-1], G[1:]), P):
        lab = f"{C['dx'] / 1000:g}-{F['dx'] / 1000:g} km"
        print(f"  {lab:>12}: surface {r['vs']:.1f}  depth-mean {r['va']:.1f}  zeta {r['z']:.1f}  ({r['n']} points)")
        rows += [['difference', lab, k, f'{r[k]:.3g}'] for k in ('vs', 'va', 'z')]
    print('apparent order p:')
    for k in range(len(P) - 1):
        lab = '/'.join(f"{D['dx'] / 1000:g}" for D in G[k:k + 3])
        p = {f: np.log(P[k][f] / P[k + 1][f]) / np.log(2) for f in ('vs', 'va', 'z')}
        print(f"  p({lab}): surface {p['vs']:.2f}  depth-mean {p['va']:.2f}  zeta {p['z']:.2f}")
        rows += [['order', lab, f, f'{p[f]:.3g}'] for f in p]
    I = [integrals(D, D['dx']) for D in G]
    ref = I[-1]
    print(f"\nbasin integrals relative to the {G[-1]['dx'] / 1000:g} km grid (%):")
    for D, v in zip(G, I):
        lab = f"{D['dx'] / 1000:g} km"
        rel = {k: 100 * (v[k] / ref[k] - 1) for k in v}
        print(f"  {lab:>8}: " + '  '.join(f'{k} {rel[k]:+.1f}' for k in v))
        rows += [['integral_rel', lab, k, f'{rel[k]:.3g}'] for k in v]
        rows += [['integral_abs', lab, k, f'{v[k]:.6g}'] for k in v]
    with open('horizontal_convergence.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['kind', 'grid', 'quantity', 'value']); w.writerows(rows)
    print('\nwritten horizontal_convergence.csv')


if __name__ == '__main__':
    main()
