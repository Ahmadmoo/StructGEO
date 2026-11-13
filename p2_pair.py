"""Experiment 5, prep: pair each thermal frame with the drone RGB frame of the same shot and check that the RGB frame,
resampled into the thermal footprint from the two metadata poses, lines up with the thermal frame (edge correlation,
day frames from train/valid, over pixel offsets within +-pair_shift_px). The best common offset is stored with each
pair, so the vis2th cache (p2_cache.py --kinds vis2th) gets RGB views with the thermal frame's footprint.
Pairs by file name first (.../Thermal/x_T.JPG <-> .../Visible/x_V.JPG or x_W.JPG), else by pose in the same scene."""
import argparse, json, os, re
import cv2
import numpy as np
from common import load_cfg
from p2_common import load_frames, work_size, warp_down, vis_map, read_rgb, load_gray, edges, ncc

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
P, R, O = cfg['p2_cache'], cfg['tuav_root'], cfg['out_dir']
frames = load_frames(cfg)
if 'visible' not in frames:
    raise SystemExit('no drone RGB frames in frames.json. Put the Visible images and their metadata (Visible_info.csv per '
                     'scene, or Data/metadata/*_Visible.json) under tuav_root, then run: p2_prep.py --cfg ... --only visible')
th, vi = frames['thermal'], frames['visible']
by_path = {f['path'].lower(): f for f in vi}
by_scene = {}
for f in vi:
    by_scene.setdefault((f['split'], f['scene']), []).append(f)


def by_name(p):
    q = p.replace('/Thermal/', '/Visible/')
    for suf in ('V', 'W'):
        v = by_path.get(re.sub(r'_T(\.\w+)$', rf'_{suf}\1', q).lower())
        if v:
            return v
    return by_path.get(q.lower())


how = dict(name=0, pose=0, none=0)
for f in th:
    v, kind = by_name(f['path']), 'name'
    if v is None:
        kind, cand = 'pose', by_scene.get((f['split'], f['scene']), [])
        if cand:
            d = np.hypot(np.array([c['E'] for c in cand]) - f['E'], np.array([c['N'] for c in cand]) - f['N'])
            j = int(d.argmin())
            dy = abs((cand[j]['yaw'] - f['yaw'] + 180) % 360 - 180)
            v = cand[j] if d[j] <= cfg['pair_max_m'] and dy <= cfg['pair_max_yaw'] else None
    if v is None:
        f.pop('vis', None)
        how['none'] += 1
        continue
    how[kind] += 1
    f['vis'] = dict(id=v['id'], path=v['path'], E=v['E'], N=v['N'], yaw=v['yaw'], gsd=v['gsd'], how=kind)
print('pairs:', how)
if how['name'] + how['pose'] == 0:
    raise SystemExit('no thermal frame could be paired with an RGB frame (check the Visible file names and poses)')

# offset check: render with a margin once per frame, then score every offset by cropping
wh, vwh = work_size(cfg, 'thermal'), work_size(cfg, 'visible')
W, H = wh
S = int(cfg['pair_shift_px'])
rng = np.random.default_rng(cfg['seed'])
pool = [f for f in th if f.get('vis') and f['split'] != 'test' and f['cond'] == 'day']
sel = [pool[i] for i in rng.choice(len(pool), min(cfg['n_check'], len(pool)), replace=False)] if pool else []
views = []
for f in sel:
    eq = edges(load_gray(os.path.join(R, f['path']), wh))
    im = read_rgb(os.path.join(R, f['vis']['path']))
    big = warp_down(im, vis_map(f, wh, vwh, im.shape[1::-1], (-S, -S)), (W + 2 * S, H + 2 * S))
    views.append((eq, edges(big), cv2.erode((big.max(-1) > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0))


def mean_ncc(dx, dy):
    sl = (slice(S + dy, S + dy + H), slice(S + dx, S + dx + W))
    c = [ncc(eq, eb[sl], ok[sl]) for eq, eb, ok in views]
    c = [x for x in c if np.isfinite(x)]
    return float(np.mean(c)) if c else float('nan')


scores = {}
for dy in range(-S, S + 1, 4):                       # coarse, then +-3 px around the best
    for dx in range(-S, S + 1, 4):
        scores[dx, dy] = mean_ncc(dx, dy)
fin = {k: v for k, v in scores.items() if np.isfinite(v)}
bx, by = max(fin, key=fin.get) if fin else (0, 0)
for dy in range(max(-S, by - 3), min(S, by + 3) + 1):
    for dx in range(max(-S, bx - 3), min(S, bx + 3) + 1):
        if (dx, dy) not in scores:
            scores[dx, dy] = mean_ncc(dx, dy)
fin = {k: v for k, v in scores.items() if np.isfinite(v)}
dx, dy = max(fin, key=fin.get) if fin else (0, 0)
zero = scores.get((0, 0), float('nan'))
use = (dx, dy) if np.hypot(dx, dy) > 2 else (0, 0)
for f in th:
    if f.get('vis'):
        f['vis'].update(dx=float(use[0]), dy=float(use[1]))
r = lambda x: None if not np.isfinite(x) else round(float(x), 4)
info = dict(pairs=how, n_check=len(sel), ncc_at_zero=r(zero), best_shift_px=[dx, dy], ncc_at_best=r(fin.get((dx, dy), np.nan)),
            shift_used_px=list(use), edge=bool(abs(dx) == S or abs(dy) == S))
print(json.dumps(info))
if info['edge']:
    print(f'WARNING: the best offset is at the edge of the search (+-{S} px); raise pair_shift_px or check the poses')
json.dump(frames, open(f'{P}/frames.json.tmp', 'w'))
os.replace(f'{P}/frames.json.tmp', f'{P}/frames.json')
json.dump(info, open(f'{P}/p2_pair.json', 'w'), indent=1)

os.makedirs(f'{O}/p2_figs', exist_ok=True)
from p2_common import render_vis
for f in sel[:cfg['p2_fig']]:
    q = load_gray(os.path.join(R, f['path']), wh)
    v = render_vis(f, wh, R, vwh)
    cv2.imwrite(f"{O}/p2_figs/pair_{f['id']}.png",
                cv2.cvtColor(np.concatenate([q, v, (0.5 * q + 0.5 * v).astype(np.uint8)], 1), cv2.COLOR_RGB2BGR))
