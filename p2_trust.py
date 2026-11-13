"""Experiments 1 and 2 on a trained run: does a short sequence of frames place the frames one frame cannot, and can each
answer carry an error bound (protection level) that holds?

GPS loss, simulated per evaluated frame t: the drone's own position estimate is off by t's stored evaluation prior
(uniform +-prior_m, as in the single-frame test) and its heading by t's stored heading error. The earlier frames of the
window are placed by odometry with a systematic error (scale and heading, drawn per t, RMS odo_drift x distance), so
their priors differ from t's by that drift. Every frame is matched as usual; the position maps of the earlier frames
(summed over heading) are moved into frame t's map window, blurred by the expected odometry error and multiplied with
t's own heading-and-position map, each mixed with a small uniform floor (eps, picked on validation) so that one
confidently wrong frame cannot veto the rest. The answer is the joint peak, as in p2_train: k = 1 is exactly the
single-frame result of the run (checked against its _frames.json).

Protection level (PL): the radius around the answer that holds 1 - alpha of the probability. It is scaled by a factor
set on validation (split conformal), once for all frames, and per bin of a score known before matching: map
distinctiveness at the prior position (map only) or the model's own confidence."""
import argparse, json, os
import numpy as np
import torch
from common import load_cfg
from p2_common import KIND, wrap, pick_tau
from p2_exp_common import (Model, log_post, offsets, peak_stats, shift_blur, conformal_q, tracks, window, map_score,
                           by_group, place, r4, t32)
import p2_exp_plots as plots

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--ckpt', required=True, help='a trained run, results/<run>.pt')
ap.add_argument('--query', default='thermal', help='query kind: thermal, sat, vis2th')
a = ap.parse_args()
M = Model(load_cfg(*a.cfg), a.ckpt)
c, dev, kind = M.cfg, M.dev, a.query
mod = KIND[kind][0]
g, fl, O = M.G[mod], M.frames[mod], c['out_dir']
os.makedirs(f'{O}/exp_figs', exist_ok=True)
tag = f'{M.name}_{kind}'
K = sorted({int(k) for k in c['fuse_k']} | {1})
kmax, alphas, eps_list = K[-1], [float(x) for x in c['pl_alpha']], [float(e) for e in c['fuse_eps']]
sig_ax = c['odo_drift'] / np.sqrt(2)          # per-axis odometry error per metre (scale and heading terms)
lim = c['search_m'] - 2 * g['cell']           # keep the truth inside the search window
EB = c.get('eval_batch', 32)
BLK = max(1, 16 * EB // kmax)
S1, cell = g['S'] + 1, g['cell']
ok_m, r_conf = c['ok_m'], c['conf_radius_m']
recs = {sp: {e: [] for e in eps_list} for sp in ('valid', 'test')}
info, clipped = {}, 0

for split in ('valid', 'test'):
    trs = tracks(fl, split, c['fuse_max_gap_m'], c['fuse_max_dt_s'])
    if kind == 'vis2th':
        trs = [[i for i in tr if fl[i].get('vis')] for tr in trs]
    ends = [(tr[j], window(tr, j, fl, kmax, c['fuse_min_gap_m'])) for tr in trs for j in range(len(tr))
            if j % c['fuse_every'] == 0]
    steps = [np.hypot(fl[tr[j]]['E'] - fl[tr[j - 1]]['E'], fl[tr[j]]['N'] - fl[tr[j - 1]]['N'])
             for tr in trs for j in range(1, len(tr))]
    info[split] = dict(frames=sum(map(len, trs)), tracks=len(trs), median_track_frames=r4(np.median([len(t) for t in trs])),
                       median_step_m=r4(np.median(steps)) if steps else None, windows=len(ends),
                       full_windows=sum(len(W) >= kmax for _, W in ends))
    print(split, json.dumps(info[split]), flush=True)
    # map-only distinctiveness where the drone believes it is (known before matching)
    pts = [dict(E=fl[t]['E'] + fl[t]['prior'][0], N=fl[t]['N'] + fl[t]['prior'][1], yaw=fl[t]['yaw'] + fl[t]['prior'][2],
                gsd=fl[t]['gsd']) for t, _ in ends]
    loc = map_score(c, M.osm, M.om, pts, mod, excl_m=c['loc_excl_m'], beta=c['loc_beta'], bs=EB)['margin']

    for b0 in range(0, len(ends), BLK):
        blk = ends[b0:b0 + BLK]
        jobs = []                                   # (frame, prior offsets from its truth, is window end)
        for t, W in blk:
            rng = np.random.default_rng([c['seed'], t])
            s_, th = rng.normal(0, sig_ax, 2)
            A = np.array([[s_, -th], [th, s_]])     # odometry: measured displacement = (I + A) x true displacement
            pt, bt, yt = np.array([fl[t]['E'], fl[t]['N']]), np.array(fl[t]['prior'][:2], float), fl[t]['prior'][2]
            for i in W:
                pr = bt - A @ (pt - np.array([fl[i]['E'], fl[i]['N']]))
                if np.abs(pr).max() > lim:
                    clipped += 1
                    pr = np.clip(pr, -lim, lim)
                jobs.append((i, [pr[0], pr[1], yt], i == t))
        P, rel, pri, lpt = [], [], [], []
        for s0 in range(0, len(jobs), EB):
            jb = jobs[s0:s0 + EB]
            logits, b = M.match(kind, [fl[i] for i, _, _ in jb], priors=[p for _, p, _ in jb])
            lp = log_post(logits)
            P.append(lp.exp().sum(1))
            rel.append(b['rel'])
            pri.append(b['prior'])
            lpt += [lp[j].clone() for j, (_, _, e) in enumerate(jb) if e]
        P, rel, pri = torch.cat(P), np.concatenate(rel), np.concatenate(pri)
        j0 = 0
        for wi, (t, W) in enumerate(blk):
            w = len(W)
            Pw, relw, priw, lp0 = P[j0:j0 + w], rel[j0:j0 + w], pri[j0:j0 + w], lpt[wi]
            j0 += w
            r = -relw - priw[:, :2]                  # map-window center minus the dead-reckoned position, per frame
            dE, dN = r[:, 0] - r[0, 0], r[:, 1] - r[0, 1]
            phat = np.array([[fl[i]['E'], fl[i]['N']] for i in W]) + priw[:, :2]
            dist = np.hypot(*(phat - phat[0]).T)     # dead-reckoned distance back from frame t
            Q = Pw.clone()
            if w > 1:
                sig = np.sqrt((sig_ax * dist[1:] / cell) ** 2 + 0.5 ** 2)
                Q[1:] = shift_blur(Pw[1:], t32(-dN[1:] / cell, dev), t32(dE[1:] / cell, dev), t32(sig, dev))
            Q = Q[1:] / Q[1:].flatten(1).sum(1).clamp(min=1e-12)[:, None, None]
            ks = [k for k in K if k <= w]
            nA = lp0.shape[0]
            for eps in eps_list:
                # frame t keeps its (heading, position) map; earlier frames add position evidence. k = 1 is p2_train's answer.
                cum = torch.cat([torch.zeros(1, S1, S1, device=dev), torch.log((1 - eps) * Q + eps / S1 ** 2).cumsum(0)])
                lt = torch.log((1 - eps) * lp0.exp() + eps / (nA * S1 ** 2))
                Fk = torch.stack([lp0 if k == 1 else lt + cum[k - 1][None] for k in ks]).flatten(1).log_softmax(1)
                kk = Fk.argmax(1)
                ab, sy, sx = kk // (S1 * S1), (kk // S1) % S1, kk % S1
                _, _, conf, pl = peak_stats(Fk.exp().view(len(ks), nA, S1, S1).sum(1), g, r_conf, alphas, center=(sy, sx))
                err = np.hypot(*(offsets(sy.cpu().numpy(), sx.cpu().numpy(), g) - relw[0]).T)
                yerr = np.abs(wrap(priw[0, 2] + g['offs'][ab.cpu().numpy()]))
                for q, k in enumerate(ks):
                    recs[split][eps].append(dict(
                        id=t, scene=fl[t]['scene'], place=place(fl[t]['scene']), cond=fl[t]['cond'], k=k, n=w,
                        span_m=round(float(dist[k - 1]), 1), err=float(err[q]), yaw_err=float(yerr[q]), conf=float(conf[q]),
                        loc=float(loc[b0 + wi]), **{f'pl{al}': float(pl[al][q]) for al in alphas}))
        print(f'{split}: {min(b0 + BLK, len(ends))}/{len(ends)} windows', flush=True)


def pick(rr, k, common):
    return [r for r in rr if r['k'] == k and (not common or r['n'] >= kmax)]


# eps: best mean share within ok_m over k > 1 on validation (k = 1 does not depend on it)
eps_score = {e: r4(np.mean([np.mean([r['err'] <= ok_m for r in pick(recs['valid'][e], k, False)] or [np.nan])
                            for k in K if k > 1] or [np.nan])) for e in eps_list}
eps = max(eps_list, key=lambda e: -1 if eps_score[e] is None else eps_score[e])
V, T = recs['valid'][eps], recs['test'][eps]


def pl_stats(rr, pl):
    e, p = np.array([r['err'] for r in rr]), np.asarray(pl, float)
    if not len(e):
        return {}
    one = lambda m: dict(n=int(m.sum()), coverage=r4((e[m] <= p[m]).mean()), median_pl_m=r4(np.median(p[m])),
                         **{f'avail@{al}m': r4((p[m] <= al).mean()) for al in c['pl_alert_m']},
                         **{f'hazard@{al}m': r4(((e[m] > al) & (p[m] <= al)).mean()) for al in c['pl_alert_m']})
    out = {'all': one(np.ones(len(e), bool))}
    for pc in sorted({r['place'] for r in rr}):
        out[pc] = one(np.array([r['place'] == pc for r in rr]))
    return out


def binned_q(v, key, al, nb):
    """Per-bin conformal factors: bin edges from validation quantiles of key."""
    x = np.array([r[key] for r in v])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])) if len(x) else np.array([])
    bv = np.digitize(x, edges)
    qs = [conformal_q([r['err'] / r[f'pl{al}'] for r, j in zip(v, bv) if j == b], al) for b in range(len(edges) + 1)]
    return edges, qs


res = dict(run=M.name, query=kind, map=c['map_name'], eps=eps, eps_valid_r25=eps_score, clipped_priors=clipped, info=info,
           settings={k: c[k] for k in ('fuse_k', 'fuse_min_gap_m', 'fuse_max_gap_m', 'fuse_max_dt_s', 'fuse_every',
                                       'odo_drift', 'fuse_eps', 'pl_alpha', 'pl_alert_m', 'loc_bins', 'prior_m',
                                       'yaw_noise', 'search_m')},
           fusion={'common': {}, 'eligible': {}}, pl={})
for setname, common in (('common', True), ('eligible', False)):
    for k in K:
        v, te = pick(V, k, common), pick(T, k, common)
        tau = pick_tau(v, ok_m, c['target_precision']) if v else 0.5
        res['fusion'][setname][k] = dict(n_test=len(te), median_span_m=r4(np.median([r['span_m'] for r in te])) if te else None,
                                         test=by_group(te, tau, ok_m) if te else {},
                                         valid=by_group(v, tau, ok_m)['all'] if v else {})
k1 = pick(T, 1, False)
res['single_frame_check'] = dict(median_err_m=r4(np.median([r['err'] for r in k1])) if k1 else None)
rf = f'{O}/{M.name}_frames.json'
pf = json.load(open(rf)) if os.path.exists(rf) else {}
if kind in pf:  # k = 1 must reproduce the run's own test errors
    ref = {r['id']: r['err'] for r in pf[kind]}
    d = [abs(r['err'] - ref[r['id']]) for r in k1 if r['id'] in ref]
    res['single_frame_check'].update(frames=len(d), max_diff_to_run_m=r4(max(d)) if d else None)
for k in K:
    v, te = pick(V, k, False), pick(T, k, False)
    if not v or not te:
        continue
    res['pl'][k] = {}
    for al in alphas:
        q = conformal_q([r['err'] / r[f'pl{al}'] for r in v], al)
        o = dict(q=r4(q), raw=pl_stats(te, [r[f'pl{al}'] for r in te]), conformal=pl_stats(te, [q * r[f'pl{al}'] for r in te]))
        for key, nm in (('loc', 'by_map_score'), ('conf', 'by_confidence')):
            edges, qs = binned_q(v, key, al, c['loc_bins'])
            bt = np.digitize([r[key] for r in te], edges)
            o[nm] = dict(edges=[r4(x) for x in edges], q=[r4(x) for x in qs],
                         **pl_stats(te, [qs[j] * r[f'pl{al}'] for r, j in zip(te, bt)]))
        res['pl'][k][al] = o

json.dump(res, open(f'{O}/exp_trust_{tag}.json', 'w'), indent=1)
json.dump({'eps': eps, 'valid': V, 'test': T}, open(f'{O}/exp_trust_{tag}_frames.json', 'w'))

# figures
common = res['fusion']['common'][kmax]['n_test'] >= 50
fz = res['fusion']['common' if common else 'eligible']
ks = [k for k in K if fz[k]['test']]
series = {}
for pc in [p for p in ('city', 'village') if any(p in fz[k]['test'] for k in ks)] or ['all']:
    series[pc] = dict(r25=[fz[k]['test'].get(pc, {}).get('r@25m') for k in ks],
                      median=[fz[k]['test'].get(pc, {}).get('median_m') for k in ks])
plots.fusion_curve(f'{O}/exp_figs/fusion_{tag}.png', ks, series,
                   f'{M.name}, {kind}: fusing the last k frames (odometry drift {c["odo_drift"]:.0%}, '
                   f'{"same" if common else "eligible"} test frames for every k)')
amain = min(alphas)
kk = [k for k in (1, kmax) if k in res['pl']] or list(res['pl'])[:1]
if kk:
    panels = []
    for k in kk:
        te = pick(T, k, False)
        q = res['pl'][k][amain]['q'] or np.inf
        panels.append((f'{"1 frame" if k == 1 else f"{k} frames"}, PL for {1 - amain:.0%}', [r['err'] for r in te],
                       [q * r[f'pl{amain}'] for r in te]))
    plots.stanford(f'{O}/exp_figs/stanford_{tag}.png', panels, c['pl_alert_m'][0])

print(f'eps = {eps} (validation share within {ok_m} m per eps: {eps_score})')
print('single frame check:', res['single_frame_check'])
print(f"{'k':>3} {'n':>6} {'span m':>7} {'median':>7} {'<=25m':>6} | " +
      ' | '.join(f'{p}: median, <=25m' for p in ('city', 'village')))
for k in K:
    z = fz.get(k, {})
    if not z or not z['test']:
        continue
    tt = z['test']
    row = ' | '.join(f"{tt[p]['median_m']:7.1f} {tt[p]['r@25m']:6.2f}" if p in tt else ' ' * 14 for p in ('city', 'village'))
    print(f"{k:>3} {z['n_test']:>6} {z['median_span_m'] or 0:>7.0f} {tt['all']['median_m']:>7.1f} {tt['all']['r@25m']:>6.2f} | {row}")
for k in kk:
    o = res['pl'][k][amain]
    cov = {p: v['coverage'] for p, v in o['conformal'].items()}
    print(f'PL k={k}, {1 - amain:.0%}: factor {o["q"]}, test coverage {cov}; per map-score bin',
          {p: v.get('coverage') for p, v in o['by_map_score'].items() if isinstance(v, dict)},
          '; per confidence bin', {p: v.get('coverage') for p, v in o['by_confidence'].items() if isinstance(v, dict)})
