"""Collects the PoC experiment results into results/exp_table.md (missing parts are skipped)."""
import argparse, glob, json, os
import numpy as np
from common import load_cfg
from p2_exp_common import place

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
O, ok_m = cfg['out_dir'], cfg['ok_m']
f1 = lambda x: '–' if x is None else f'{x:.1f}'
f2 = lambda x: '–' if x is None else f'{x:.2f}'
pc = lambda x: '–' if x is None else f'{100 * x:.0f}%'
m_ = lambda x: '–' if x is None else f'{x:.1f} m'
L = ['# PoC experiments (test split)', '']


def cellmr(d):
    return f"{f1(d.get('median_m'))} m / {pc(d.get('r@25m'))}" if d else '–'


# 1. fusion
tr = sorted(glob.glob(f'{O}/exp_trust_*.json'))
tr = [p for p in tr if not p.endswith('_frames.json')]
if tr:
    L += ['## 1. Fusing the last k frames (simulated GPS loss: prior error up to ±100 m, odometry drift between frames)', '',
          'Median error / share within 25 m. "same frames": every k is scored on the frames that have a full window.', '',
          '| Run | Query | k | Frames | Span (m) | All | City | Village | Coverage at 95% precision |', '|---|---|---|---|---|---|---|---|---|']
    for p in tr:
        r = json.load(open(p))
        kmax = max(int(k) for k in r['fusion']['common'])
        st = 'common' if r['fusion']['common'][str(kmax)]['n_test'] >= 50 else 'eligible'
        for k, z in sorted(r['fusion'][st].items(), key=lambda x: int(x[0])):
            t = z['test']
            if not t:
                continue
            L.append(f"| {r['run']} | {r['query']} | {k} | {z['n_test']} | {f1(z['median_span_m'])} | {cellmr(t['all'])} | "
                     f"{cellmr(t.get('city'))} | {cellmr(t.get('village'))} | {pc(t['all'].get('coverage'))} |")
        L.append(f"| | | | | | eps {r['eps']}, drift {r['settings']['odo_drift']:.0%}, frames ≥ "
                 f"{r['settings']['fuse_min_gap_m']} m apart, {st} set | | | |")
    L.append('')

    # 2. protection levels
    L += ['## 2. Protection levels (radius holding 99% / 95% of the probability, scaled on validation)', '',
          'Coverage = share of test frames with error ≤ PL (target = the nominal level). Availability / hazard at the 25 m alert '
          'limit: PL ≤ 25 m / error > 25 m while PL ≤ 25 m.', '',
          '| Run | Query | k | Level | Factor | Uncalibrated coverage | Coverage all / city / village | Per map-score bin | '
          'Per confidence bin | Median PL | Available ≤25 m | Hazard ≤25 m |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in tr:
        r = json.load(open(p))
        ks = sorted(int(k) for k in r['pl'])
        for k, byal in sorted(((k, v) for k, v in r['pl'].items() if int(k) in {ks[0], 5, ks[-1]}), key=lambda x: int(x[0])):
            for al, o in sorted(byal.items(), key=lambda x: -float(x[0])):
                cf, m1, m2 = o['conformal'], o['by_map_score'], o['by_confidence']
                cov = ' / '.join(pc(cf.get(g, {}).get('coverage')) for g in ('all', 'city', 'village'))
                L.append(f"| {r['run']} | {r['query']} | {k} | {1 - float(al):.0%} | {f1(o['q'])} | {pc(o['raw']['all']['coverage'])} | "
                         f"{cov} | {' / '.join(pc(m1.get(g, {}).get('coverage')) for g in ('all', 'city', 'village'))} | "
                         f"{' / '.join(pc(m2.get(g, {}).get('coverage')) for g in ('all', 'city', 'village'))} | "
                         f"{m_(cf['all'].get('median_pl_m'))} | {pc(cf['all'].get('avail@25m'))} | {pc(cf['all'].get('hazard@25m'))} |")
    L.append('')

# 3. localizability
for p in sorted(glob.glob(f'{O}/exp_loc_*.json')):
    if p.endswith('_frames.json'):
        continue
    r = json.load(open(p))
    L += [f"## 3. Map-only distinctiveness vs failures ({r['query']}, map {r['map']})", '',
          'AUROC for "frame placed within 25 m" (0.5 = no signal). Map-only scores need no sensor: at the true position '
          '(planning) or at the prior (in flight). Run confidence is the matcher\'s own (after matching), for reference.', '',
          '| Run | Group | Frames | Success | Map score @ truth | Map score @ prior | Map probability @ truth | Learned map score | '
          'Run confidence |', '|---|---|---|---|---|---|---|---|---|']
    for run, groups in r['runs'].items():
        for gname, v in groups.items():
            au = v['auroc']
            L.append(f"| {run} | {gname} | {v['n']} | {pc(v['success'])} | {f2(au.get('margin_true'))} | {f2(au.get('margin_prior'))} | "
                     f"{f2(au.get('oconf_true'))} | {f2(au.get('zmargin_true'))} | {f2(v['auroc_run_confidence'])} |")
    L += ['', 'Median map score per place: ' + '; '.join(f"{pl}: {v.get('margin_true')}" for pl, v in r['score_median'].items()),
          f"Area map: {r['area']['tif']} ({r['area']['step_m']} m grid; {pc(r['area']['share_below_0p05'])} of it below 0.05).", '']


# 4 + 6. map variants and the m2 control, from the runs' per-frame files
def per_place(run, kind):
    p = f'{O}/{run}_frames.json'
    if not os.path.exists(p):
        return None
    fr = json.load(open(p)).get(kind)
    if not fr:
        return None
    out = {}
    for gname in ['all'] + sorted({place(r['scene']) for r in fr}):
        e = np.array([r['err'] for r in fr if gname == 'all' or place(r['scene']) == gname])
        out[gname] = dict(median_m=float(np.median(e)), **{'r@25m': float((e <= ok_m).mean())}, n=len(e))
    return out


def variants(title, runs):
    rows = [(r, k, per_place(r, k)) for r in runs for k in ('thermal', 'sat')]
    rows = [x for x in rows if x[2]]
    if len({r for r, _, _ in rows}) < 2:
        return
    L.extend([f'## {title}', '', '| Run | Query | All | City | Village |', '|---|---|---|---|---|'])
    L.extend(f"| {r} | {k} | {cellmr(v['all'])} | {cellmr(v.get('city'))} | {cellmr(v.get('village'))} |" for r, k, v in rows)
    L.append('')


variants('4. Richer free maps (m1 settings, different map)', ['p2_m1_thermal', 'p2_m5_dsm', 'p2_m6_wc', 'p2_m7_dsm_wc'])

# 5. one matcher, several sensors
ev = sorted(p for p in glob.glob(f'{O}/exp_evalq_*.json') if not p.endswith('_frames.json'))
if ev:
    L += ['## 5. One matcher, several sensors (stored priors; median / within 25 m)', '',
          '| Run | Query | All | Day | Night | City | Village | Coverage |', '|---|---|---|---|---|---|---|---|']
    for p in ev:
        r = json.load(open(p))
        for kind, v in r['results'].items():
            t = v['test']
            L.append(f"| {r['run']} | {kind} | {cellmr(t['all'])} | {cellmr(t.get('day'))} | {cellmr(t.get('night'))} | "
                     f"{cellmr(t.get('city'))} | {cellmr(t.get('village'))} | {pc(t['all'].get('coverage'))} |")
    L.append('')

variants('6. Where does the satellite -> thermal transfer come from? (trained on satellite crops only)',
         ['p2_m2_sat_only', 'p2_m2r_random_adapter'])

# 7. partial views
p = f'{O}/exp_partial.json'
if os.path.exists(p):
    r = json.load(open(p))
    L += [f"## 7. Partial views ({r['mode']}: part of the frame hidden at the backbone input)", '',
          '| Run | View | Hidden | All | City | Village |', '|---|---|---|---|---|---|']
    for run, conds in r['runs'].items():
        for cond, byf in conds.items():
            for fr, t in byf.items():
                L.append(f"| {run} | {cond} | {pc(float(fr))} | {cellmr(t['all'])} | {cellmr(t.get('city'))} | {cellmr(t.get('village'))} |")
    L.append('')

open(f'{O}/exp_table.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
