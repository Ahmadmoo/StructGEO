"""Experiment 3: where can the map place a frame at all? A map-only score (no sensor, no training): the map under a camera
footprint is matched against the map around it, as a perfect sensor would, and the score is how much better the true
place fits than the best place more than loc_excl_m away. It is compared with where the trained runs fail (AUROC,
failure rate per score bin), and drawn over the area as a planning map (PNG + GeoTIFF).
With --ckpt, the same score is also computed on that run's learned map features (zmargin)."""
import argparse, json, os
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from common import load_cfg
from p2_common import KIND, load_frames, load_osm, match_geo, auroc, colorize
from p2_exp_common import Model, map_score, place, r4
import p2_exp_plots as plots

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--ckpt', help='optional run (results/<run>.pt): adds the score on its learned map features')
ap.add_argument('--query', default='thermal')
ap.add_argument('--runs', nargs='+', help='runs to compare with (default loc_runs)')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
menc = None
if a.ckpt:
    M = Model(cfg, a.ckpt)
    cfg, menc = M.cfg, M.menc
frames = load_frames(cfg)
osm, om = load_osm(cfg)
O, mod, ok_m = cfg['out_dir'], KIND[a.query][0], cfg['ok_m']
os.makedirs(f'{O}/exp_figs', exist_ok=True)
fl = [f for f in frames[mod] if f['split'] in ('valid', 'test')]
bs = cfg.get('eval_batch', 32)
kw = dict(excl_m=cfg['loc_excl_m'], beta=cfg['loc_beta'], bs=bs)

# per frame: at the true pose (planning: the drone flies here) and at the stored prior (in flight, before matching)
st = map_score(cfg, osm, om, [dict(E=f['E'], N=f['N'], yaw=f['yaw'], gsd=f['gsd']) for f in fl], mod, menc, **kw)
sp = map_score(cfg, osm, om, [dict(E=f['E'] + f['prior'][0], N=f['N'] + f['prior'][1], yaw=f['yaw'] + f['prior'][2],
                                   gsd=f['gsd']) for f in fl], mod, None, **kw)
rec = [dict(id=f['id'], split=f['split'], scene=f['scene'], place=place(f['scene']), cond=f['cond'],
            **{f'{k}_true': float(v[j]) for k, v in st.items()}, margin_prior=float(sp['margin'][j]))
       for j, f in enumerate(fl)]
scores = [k for k in rec[0] if k.endswith(('_true', '_prior')) and not k.startswith('content')] if rec else []
res = dict(query=a.query, map=cfg.get('map_name') or 'osm', ckpt=a.ckpt, n=len(rec),
           score_median={pc: {s: r4(np.median([r[s] for r in rec if r['place'] == pc])) for s in scores}
                         for pc in sorted({r['place'] for r in rec})},
           runs={})

# against the runs' own test errors
for run in a.runs or cfg['loc_runs']:
    p = f'{O}/{run}_frames.json'
    if not os.path.exists(p):
        print(f'{run}: {p} not found, skipped')
        continue
    pf = json.load(open(p))
    if a.query not in pf:
        continue
    err = {r['id']: r for r in pf[a.query]}
    te = [r for r in rec if r['split'] == 'test' and r['id'] in err]
    ok = np.array([err[r['id']]['err'] <= ok_m for r in te])
    out = {}
    for grp in ['all'] + sorted({r['place'] for r in te}):
        m = np.array([grp == 'all' or r['place'] == grp for r in te])
        if not m.any():
            continue
        out[grp] = dict(n=int(m.sum()), success=r4(ok[m].mean()),
                        auroc={s: r4(auroc(np.array([r[s] for r in te])[m], ok[m])) for s in scores},
                        auroc_run_confidence=r4(auroc(np.array([err[r['id']]['conf'] for r in te])[m], ok[m])))
    res['runs'][run] = out
    plots.binned(f'{O}/exp_figs/loc_binned_{run}_{a.query}.png', [r['margin_true'] for r in te], ~ok, 5,
                 f'{run}: failures vs map distinctiveness (test)', 'map-only score at the true position (bins of equal size)')
    print(run, json.dumps({g: dict(n=v['n'], success=v['success'], auroc_map=v['auroc'].get('margin_true'),
                                   auroc_conf=v['auroc_run_confidence']) for g, v in out.items()}))

# area map
g = match_geo(cfg, mod)
half = g['npx'] * om['res'] / 2
x_lo, x_hi = om['x0'] + half, om['x0'] + om['width'] * om['res'] - half
y_lo, y_hi = om['y0'] - om['height'] * om['res'] + half, om['y0'] - half
tf = [f for f in fl if f['split'] == 'test']
if cfg['loc_area'] == 'test' and tf:
    E, N = np.array([f['E'] for f in tf]), np.array([f['N'] for f in tf])
    x_lo, x_hi = max(x_lo, E.min() - 500), min(x_hi, E.max() + 500)
    y_lo, y_hi = max(y_lo, N.min() - 500), min(y_hi, N.max() + 500)
step = cfg['loc_grid_m']
xs, ys = np.arange(x_lo, x_hi + 1e-6, step), np.arange(y_hi, y_lo - 1e-6, -step)
gsd = float(np.median([f['gsd'] for f in (tf or fl)]))
pts = [dict(E=float(x), N=float(y), yaw=0.0, gsd=gsd) for y in ys for x in xs]
print(f'area map: {len(xs)} x {len(ys)} points every {step} m')
grid = map_score(cfg, osm, om, pts, mod, None, **kw)['margin'].reshape(len(ys), len(xs)).astype(np.float32)
tif = f"{O}/exp_loc_{a.query}_{cfg['loc_area']}.tif"
with rasterio.open(tif, 'w', driver='GTiff', width=len(xs), height=len(ys), count=1, dtype='float32', crs=om['crs'],
                   transform=from_origin(xs[0] - step / 2, ys[0] + step / 2, step, step)) as d:
    d.write(grid[None])
ext = (xs[0] - step / 2, xs[-1] + step / 2, ys[-1] - step / 2, ys[0] + step / 2)
c0, c1 = int((ext[0] - om['x0']) / om['res']), int((ext[1] - om['x0']) / om['res'])
r0, r1 = int((om['y0'] - ext[3]) / om['res']), int((om['y0'] - ext[2]) / om['res'])
sub = np.asarray(osm[:, max(r0, 0):r1, max(c0, 0):c1])
k = max(1, int(np.ceil(max(sub.shape[1:]) / 1600)))
bg = colorize(np.stack([cv2.resize(ch, (max(1, ch.shape[1] // k), max(1, ch.shape[0] // k)), interpolation=cv2.INTER_AREA)
                        for ch in sub]))
first = next(iter(res['runs']), None)
pts_ok = None
if first:
    err = {r['id']: r['err'] for r in json.load(open(f'{O}/{first}_frames.json'))[a.query]}
    tt = [f for f in tf if f['id'] in err]
    pts_ok = ([f['E'] for f in tt], [f['N'] for f in tt], [err[f['id']] <= ok_m for f in tt])
plots.loc_map(f"{O}/exp_figs/loc_area_{a.query}_{cfg['loc_area']}.png", grid, ext, bg, pts_ok,
              f'map-only distinctiveness, {step} m grid' + (f'; dots: {first} test frames' if first else ''))
res['area'] = dict(tif=tif, step_m=step, extent=[r4(x) for x in ext], gsd=r4(gsd),
                   share_below_0p05=r4(float(np.mean(grid < 0.05))), median=r4(float(np.median(grid))))
json.dump(res, open(f'{O}/exp_loc_{a.query}.json', 'w'), indent=1)
json.dump(rec, open(f'{O}/exp_loc_{a.query}_frames.json', 'w'))
print(json.dumps(res['score_median']))
