import argparse, glob, json, os
from common import load_cfg

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
O = cfg['out_dir']
f = lambda x, d=1: '–' if x is None else f'{x:.{d}f}'
pc = lambda x: '–' if x is None else f'{100 * x:.0f}%'
lines = ['# Phase 2 results (test split)', '']
ck = f'{O}/p2_check.json'
if os.path.exists(ck):
    c = json.load(open(ck))
    lines += ['## Checks', '', '| Query | Geometry best fit | Matcher on OSM-rendered query: median / ≤5 m | Map content (median) | Prior only: median |',
              '|---|---|---|---|---|']
    for mod in ('thermal', 'visible'):
        if f'geometry_{mod}' in c:
            o, g = c[f'oracle_{mod}'], c[f'geometry_{mod}']
            lines.append(f"| {mod} | {g['best']}{'' if g['ok'] else ' ⚠'} | {f(o['median_m'])} m / {pc(o['within_5m'])} | "
                         f"{pc(c[f'map_content_{mod}']['median'])} | {f(c[f'prior_only_{mod}']['median_m'])} m |")
    lines.append('')
lines += ['## Runs', '',
          '| Run | Train → test | Median (m) | Mean (m) | ≤5 m | ≤10 m | ≤25 m | ≤50 m | Yaw med (°) | AUROC | Coverage | Precision | Day med | Night med |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for p in sorted(glob.glob(f'{O}/p2_*.json')):
    if p.endswith('_frames.json') or p.endswith('p2_check.json'):
        continue
    r = json.load(open(p))
    for kind, res in r['results'].items():
        a, d, n = res['all'], res.get('day') or {}, res.get('night') or {}
        lines.append(f"| {r['name']} | {'+'.join(r['train_queries'])} → {kind} | {f(a['median_m'])} | {f(a['mean_m'])} | "
                     f"{pc(a['r@5m'])} | {pc(a['r@10m'])} | {pc(a['r@25m'])} | {pc(a['r@50m'])} | {f(a['yaw_median_deg'])} | "
                     f"{f(a['auroc'], 2)} | {pc(a['coverage'])} | {pc(a['precision'])} | {f(d.get('median_m'))} | {f(n.get('median_m'))} |")
lines += ['', f"Coverage / precision: frames with confidence ≥ τ (τ set on validation for {pc(cfg['target_precision'])} "
              f"precision at ≤ {cfg['ok_m']} m). Reference: the published satellite-image baseline on this test split "
              f"reports a 9.37 m mean error (different map source: satellite + DSM)."]
open(f'{O}/p2_table.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
