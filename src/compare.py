"""
Compare modelled sea-level changes with the Kazhydromet surge bulletins (Sep-Nov 2025).

Inputs (same folder): zeta_stations.csv, events.csv, monthly_ranges.csv, stations.csv,
                      ERA5 file (for the inverse-barometer estimate, optional)
Outputs: events_compared.csv, ranges_compared.csv, surge_scatter.png, printed statistics
Bulletin dates are local time, UTC+5.
"""
import csv, sys, datetime as dt
import numpy as np

LOCAL = dt.timedelta(hours=5)
MARGIN = dt.timedelta(hours=6)
RHO, G = 1025.0, 9.81


def load_zeta(path='zeta_stations.csv'):
    rows = list(csv.DictReader(open(path)))
    t = np.array([dt.datetime.strptime(r['time_utc'], '%Y-%m-%d %H:%M') for r in rows])
    st = [k[:-3] for k in rows[0] if k.endswith('_cm')]
    z = {s: np.array([float(r[s + '_cm']) for r in rows]) for s in st}
    return t, z


def largest_change(z, sign):
    """Largest rise (sign>0) or fall (sign<0) z[j]-z[i], i<j. Returns value, i, j."""
    best, bi, bj = 0.0, 0, 0
    ext, ie = z[0], 0
    for j in range(1, len(z)):
        d = z[j] - ext
        if sign * d > sign * best:
            best, bi, bj = d, ie, j
        if sign * z[j] < sign * ext:
            ext, ie = z[j], j
    return best, bi, bj


def era5_pressure(path, stations):
    try:
        import netCDF4 as nc
    except ImportError:
        return None
    d = nc.Dataset(path)
    tv = d.variables['valid_time']
    times = np.array([dt.datetime(x.year, x.month, x.day, x.hour)
                      for x in nc.num2date(tv[:], tv.units)])
    lat, lon = d.variables['latitude'][:], d.variables['longitude'][:]
    p = {}
    for s in stations:
        j = np.abs(lat - float(s['lat'])).argmin()
        i = np.abs(lon - float(s['lon'])).argmin()
        p[s['station']] = np.array(d.variables['msl'][:, j, i])
    return times, p


def stats(obs, mod):
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    e = mod - obs
    return dict(n=len(obs), bias=e.mean(), rmse=np.sqrt((e ** 2).mean()),
                r=np.corrcoef(obs, mod)[0, 1] if len(obs) > 2 else np.nan,
                ratio_median=np.median(np.abs(mod) / np.abs(obs)))


def main(era5=None):
    t, z = load_zeta()
    stations = list(csv.DictReader(open('stations.csv')))
    pres = era5_pressure(era5, stations) if era5 else None

    out = []
    for e in csv.DictReader(open('events.csv')):
        s = e['station']
        a = dt.datetime.strptime(e['start_local'], '%Y-%m-%d') - LOCAL - MARGIN
        b = dt.datetime.strptime(e['end_local'], '%Y-%m-%d') + dt.timedelta(days=1) - LOCAL + MARGIN
        m = (t >= a) & (t <= b)
        zz, tt = z[s][m], t[m]
        obs = float(e['obs_cm'])
        mod, i, j = largest_change(zz, np.sign(obs))
        opp, _, _ = largest_change(zz, -np.sign(obs))
        ib = ''
        if pres:
            pt, pp = pres
            k1, k2 = np.where(pt == tt[i])[0], np.where(pt == tt[j])[0]
            if len(k1) and len(k2):
                ib = f"{-(pp[s][k2[0]] - pp[s][k1[0]]) / (RHO * G) * 100:.1f}"
        out.append(dict(e, mod_cm=f'{mod:.1f}', mod_opposite_cm=f'{opp:.1f}',
                        dominant=int(abs(mod) > abs(opp)), mod_from=tt[i].strftime('%m-%d %H'),
                        mod_to=tt[j].strftime('%m-%d %H'), ib_cm=ib))

    with open('events_compared.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    def sel(cond):
        o = [float(x['obs_cm']) for x in out if cond(x)]
        m = [float(x['mod_cm']) for x in out if cond(x)]
        return o, m

    core = lambda x: x['flag'] == ''
    allx = lambda x: x['flag'] not in ('spinup', 'station_far')
    for name, c in [('core (unflagged, without Peshnoy)', core),
                    ('all except spin-up and Peshnoy', allx)]:
        o, m = sel(c)
        # statistics on magnitudes: the modelled change is taken with the observed sign,
        # so a correlation of signed values would be inflated artificially
        s_ = stats(np.abs(o), np.abs(m))
        dom = np.mean([int(x['dominant']) for x in out if c(x)])
        print(f"{name}: n={s_['n']}  bias={s_['bias']:.1f} cm  RMSE={s_['rmse']:.1f} cm  "
              f"r(|obs|,|mod|)={s_['r']:.2f}  median |mod|/|obs|={s_['ratio_median']:.2f}  "
              f"observed direction dominant in model={dom:.0%}")

    # monthly ranges
    rr = []
    for r in csv.DictReader(open('monthly_ranges.csv')):
        y, mth = map(int, r['month'].split('-'))
        a = dt.datetime(y, mth, 1) - LOCAL
        b = dt.datetime(y + (mth == 12), mth % 12 + 1, 1) - LOCAL
        m = (t >= a) & (t < b)
        zz, tt = z[r['station']][m], t[m]
        rr.append(dict(r, obs_range_cm=int(r['max_cm']) - int(r['min_cm']),
                       mod_range_cm=f'{zz.max() - zz.min():.1f}',
                       mod_max_date=(tt[zz.argmax()] + LOCAL).strftime('%Y-%m-%d'),
                       mod_min_date=(tt[zz.argmin()] + LOCAL).strftime('%Y-%m-%d')))
    with open('ranges_compared.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
        w.writeheader(); w.writerows(rr)
    o = [x['obs_range_cm'] for x in rr if x['station'] != 'Peshnoy']
    m = [float(x['mod_range_cm']) for x in rr if x['station'] != 'Peshnoy']
    s_ = stats(o, m)
    print(f"monthly ranges (without Peshnoy): n={s_['n']} bias={s_['bias']:.1f} cm "
          f"RMSE={s_['rmse']:.1f} cm r={s_['r']:.2f} median ratio={s_['ratio_median']:.2f}")

    # scatter plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for flagset, mk, lab in [(lambda x: x['flag'] == '', 'o', 'events'),
                                 (lambda x: x['flag'] not in ('', 'spinup', 'station_far'), 'x', 'flagged')]:
            o, m = sel(flagset)
            ax.plot(o, m, mk, label=lab)
        lim = 40
        ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.8)
        ax.axhline(0, color='0.7', lw=0.5); ax.axvline(0, color='0.7', lw=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('observed change (cm)'); ax.set_ylabel('modelled change (cm)')
        ax.legend(); fig.tight_layout(); fig.savefig('surge_scatter.png', dpi=200)
    except ImportError:
        pass


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
