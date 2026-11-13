"""Experiment 5: evaluate a trained run on any query kinds with the stored priors, per place (city / village) and
condition (day / night). Main use: the run trained on satellite crops only (m2) on thermal frames, satellite crops and
drone RGB frames in the same footprints (vis2th) -> one matcher, three sensors; RGB at night vs thermal at night."""
import argparse, json
import numpy as np
from common import load_cfg
from p2_common import KIND, wrap, pick_tau
from p2_exp_common import Model, log_post, by_group, place

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--queries', nargs='+', default=['thermal', 'sat', 'vis2th'])
a = ap.parse_args()
M = Model(load_cfg(*a.cfg), a.ckpt)
c, O = M.cfg, M.cfg['out_dir']
EB = c.get('eval_batch', 32)
res, per = dict(run=M.name, map=c['map_name'], results={}), {}
for kind in a.queries:
    mod = KIND[kind][0]
    g, fl = M.G[mod], M.frames[mod]
    rec = {}
    for split in ('valid', 'test'):
        ids = [f['id'] for f in fl if f['split'] == split and (kind != 'vis2th' or f.get('vis'))]
        rec[split] = []
        for s0 in range(0, len(ids), EB):
            fb = [fl[i] for i in ids[s0:s0 + EB]]
            logits, b = M.match(kind, fb)
            lp = log_post(logits)
            S1 = lp.shape[-1]
            k = lp.flatten(1).argmax(1).cpu().numpy()
            a_, sy, sx = k // (S1 * S1), (k // S1) % S1, k % S1
            pos = np.stack([(sx - g['S'] / 2) * g['cell'], -(sy - g['S'] / 2) * g['cell']], 1)
            pxy = lp.exp().sum(1)
            ii = np.arange(S1)
            for j, f in enumerate(fb):
                d2 = (ii[:, None] - sy[j]) ** 2 + (ii[None, :] - sx[j]) ** 2
                conf = float((pxy[j].cpu().numpy() * (d2 * g['cell'] ** 2 <= c['conf_radius_m'] ** 2)).sum())
                rec[split].append(dict(id=f['id'], scene=f['scene'], place=place(f['scene']), cond=f['cond'],
                                       err=float(np.hypot(*(pos[j] - b['rel'][j]))),
                                       yaw_err=float(abs(wrap(b['yaw_prior'][j] + g['offs'][a_[j]] - b['yaw'][j]))),
                                       conf=conf))
    if not rec['test']:
        print(f'{kind}: no test frames (for vis2th: run p2_pair.py and p2_cache.py --kinds vis2th)')
        continue
    tau = pick_tau(rec['valid'], c['ok_m'], c['target_precision']) if rec['valid'] else 0.5
    res['results'][kind] = dict(test=by_group(rec['test'], tau, c['ok_m']), valid=by_group(rec['valid'], tau, c['ok_m'])['all']
                                if rec['valid'] else {})
    per[kind] = rec['test']
    t = res['results'][kind]['test']
    print(f'{M.name} on {kind}:', ', '.join(f"{gname} {v['median_m']:.1f} m / {v['r@25m']:.0%} <=25 m (n={v['n']})"
                                            for gname, v in t.items()))
json.dump(res, open(f'{O}/exp_evalq_{M.name}.json', 'w'), indent=1)
json.dump(per, open(f'{O}/exp_evalq_{M.name}_frames.json', 'w'))
