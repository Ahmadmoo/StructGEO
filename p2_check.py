"""Phase 2 checks before training.
1. Geometry: does the satellite rendered with the metadata pose line up with the frame better than with a wrong
   yaw sign, a 180 deg turn, a 15 deg turn, a +-15% scale or a 20 m shift? (edge correlation, frames from train/valid)
2. Matcher code: OSM rendered into the camera frame from the true pose must be found again by the matcher (test split).
3. Map content per footprint, and the error of the position prior alone."""
import argparse, json, os
import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from common import load_cfg
from p2_common import (load_frames, load_osm, work_size, match_geo, render_sat, load_gray, load_rgb, crop, batch,
                       cam_grid, tmpl_grid, templates, correlate, decode, edges, ncc, colorize, wrap)

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
R, O = cfg['tuav_root'], cfg['out_dir']
os.makedirs(f'{O}/p2_figs', exist_ok=True)
frames = load_frames(cfg)
osm, om = load_osm(cfg)
src = rasterio.open(os.path.join(R, cfg['ref_tif']))
rng = np.random.default_rng(cfg['seed'])
t32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
out = {}


def variants(f):
    y = f['yaw']
    return {'gt': {}, 'yaw_neg': dict(yaw=-y), 'yaw_180': dict(yaw=y + 180), 'yaw_+15': dict(yaw=y + 15),
            'scale_0.85': dict(scale=0.85), 'scale_1.15': dict(scale=1.15), 'shift_20m': dict(dE=20.0)}


for mod, fl in frames.items():
    wh = work_size(cfg, mod)
    g = match_geo(cfg, mod)
    gh, gw = wh[1] // cfg['patch'], wh[0] // cfg['patch']
    load = load_gray if mod == 'thermal' else load_rgb

    # 1. geometry
    pool = [f for f in fl if f['split'] != 'test']
    sel = rng.choice(len(pool), min(cfg['n_check'], len(pool)), replace=False)
    sc = {k: [] for k in variants(pool[0])}
    for j, i in enumerate(sel):
        f = pool[i]
        q = load(os.path.join(R, f['path']), wh)
        eq = edges(q)
        for k, kw in variants(f).items():
            s = render_sat(src, f, wh, **kw)
            m = cv2.erode((s.max(-1) > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            sc[k].append(ncc(eq, edges(s), m))
        if j < cfg['p2_fig']:
            mc, Ew, Nw = crop(osm, om, f['E'], f['N'], g['npx'])
            grid = cam_grid(t32([f['E'] - Ew]), t32([f['N'] - Nw]), t32([f['yaw']]), t32([f['gsd']]), wh, wh[::-1],
                            g['npx'] * g['res'])
            cam = F.grid_sample(torch.from_numpy(mc[None]).float() / 255, grid, align_corners=False)[0].numpy()
            o = colorize(cam)
            over = (q * 0.55 + o * 0.45).astype(np.uint8)
            s = render_sat(src, f, wh)
            cv2.imwrite(f'{O}/p2_figs/check_{mod}_{f["id"]}.png',
                        cv2.cvtColor(np.concatenate([q, s, o, over], 1), cv2.COLOR_RGB2BGR))
    sc = {k: np.array(v) for k, v in sc.items()}
    res = {k: dict(mean_ncc=round(float(np.nanmean(v)), 4),
                   gt_better=round(float(np.nanmean(sc['gt'] > v)), 4) if k != 'gt' else None) for k, v in sc.items()}
    best = max(res, key=lambda k: res[k]['mean_ncc'])
    out[f'geometry_{mod}'] = dict(n=len(sel), best=best, ok=best == 'gt', variants=res)
    print(f'[{mod}] geometry: best={best}', {k: v['mean_ncc'] for k, v in res.items()})
    if best != 'gt':
        hint = {'yaw_neg': 'flip yaw_sign (1 <-> -1)', 'yaw_180': 'add 180 to yaw_offset', 'yaw_+15': 'check yaw_offset',
                'scale_0.85': 'heights look too large (check height_win / rel_height)',
                'scale_1.15': 'heights look too small (check height_win / rel_height)',
                'shift_20m': 'positions look shifted (check TargetLatitude/Longitude vs lat/lon)'}[best]
        print(f'WARNING [{mod}]: the metadata pose is not the best fit -> {hint}, rerun p2_prep.py, and look at p2_figs/check_*')

    # 2. matcher on OSM rendered from the true pose, 3. map content and prior-only error
    test = [f for f in fl if f['split'] == 'test'][:cfg['n_check']]
    rec = []
    for s0 in range(0, len(test), 8):
        fb = test[s0:s0 + 8]
        b = batch(fb, osm, om, g)
        m = torch.from_numpy(b['map']).float() / 255
        m2 = F.avg_pool2d(m, 2)
        grid = cam_grid(t32(b['rel'][:, 0]), t32(b['rel'][:, 1]), t32(b['yaw']), t32(b['gsd']), wh, (gh, gw),
                        g['npx'] * g['res'])
        q = F.grid_sample(m2, grid, mode='bilinear', align_corners=False)
        content = (q.amax(1) > 0.5).float().mean((1, 2)).numpy()
        tg = tmpl_grid(t32(b['gsd']), t32(b['yaw_prior'])[:, None] + t32(g['offs'])[None], wh, g['T'], g['cell'])
        tm, nv = templates(2 * q - 1, tg)
        pos, dyaw, conf, _ = decode(correlate(2 * m2 - 1, tm, nv) * 10, g, cfg['conf_radius_m'])
        err = np.hypot(*(pos.numpy() - b['rel']).T)
        yerr = np.abs(wrap(b['yaw_prior'] + dyaw.numpy() - b['yaw']))
        for j in range(len(fb)):
            rec.append(dict(err=float(err[j]), yaw_err=float(yerr[j]), content=float(content[j]),
                            prior_err=float(np.hypot(*b['rel'][j]))))
    e = np.array([r['err'] for r in rec])
    c = np.array([r['content'] for r in rec])
    pe = np.array([r['prior_err'] for r in rec])
    low = c < cfg['min_content']
    out[f'oracle_{mod}'] = dict(n=len(rec), median_m=round(float(np.median(e)), 2), p90_m=round(float(np.percentile(e, 90)), 2),
                                within_5m=round(float((e <= 5).mean()), 4),
                                yaw_median_deg=round(float(np.median([r['yaw_err'] for r in rec])), 2),
                                low_content_frac=round(float(low.mean()), 4),
                                median_m_low_content=round(float(np.median(e[low])), 2) if low.any() else None)
    out[f'map_content_{mod}'] = dict(median=round(float(np.median(c)), 4), p10=round(float(np.percentile(c, 10)), 4))
    out[f'prior_only_{mod}'] = dict(median_m=round(float(np.median(pe)), 2), mean_m=round(float(pe.mean()), 2),
                                    **{f'r@{t}m': round(float((pe <= t).mean()), 4) for t in (5, 10, 25, 50)})
    print(f'[{mod}] matcher check on OSM-rendered queries:', out[f'oracle_{mod}'])
    print(f'[{mod}] map content in footprint:', out[f'map_content_{mod}'], ' prior only:', out[f'prior_only_{mod}'])
    if (~low).any() and np.median(e[~low]) > 2 * g['cell']:
        print(f'WARNING [{mod}]: the matcher does not recover the true pose from the map itself; check conventions.')

json.dump(out, open(f'{O}/p2_check.json', 'w'), indent=1)
