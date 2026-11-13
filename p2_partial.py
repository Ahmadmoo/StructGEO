"""Experiment 7: partial views. Part of each thermal frame is hidden at the DINOv3 input (mask token, so the visible
tokens never see it), and a run matches with what is left ('masked': hidden tokens carry no template weight), or the
Phase 1 predictor fills in the hidden tokens from the visible ones ('predicted': the predictor guesses the RGB-side
tokens there; its layer-normed output is put back on the scale of the visible tokens). Compared with the full view.
Evaluation only; needs the backbone and the cached thermal images."""
import argparse, json
import numpy as np
import torch
from common import load_cfg, load_backbone, encode, to_input, geometry, RGB, TH
from p2_common import pick_tau, wrap
from p2_exp_common import Model, log_post, offsets, by_group, place

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--ckpts', nargs='+', required=True, help='runs to test, results/<run>.pt')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
models = [Model(cfg, p) for p in a.ckpts]
c = models[0].cfg
for m in models[1:]:
    assert (m.cfg['backbone'], m.cfg['layers']) == (c['backbone'], c['layers']), 'runs must share backbone and layers'
dev, P, O, EB, L = c['device'], c['p2_cache'], c['out_dir'], c.get('eval_batch', 32), c['layers']
bb = load_backbone(c, c['backbone'], dev)
imgs = np.load(f'{P}/p2_img_thermal.npy', mmap_mode='r')
G1 = geometry(c, None, 'same', dev)
gh, gw, uv = G1['gh'], G1['gw'], G1['uv_t'][TH]
fl = models[0].frames['thermal']
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)


def hidden(rng, frac, mode):
    h = np.zeros((gh, gw), bool)
    if mode == 'side':                       # a strip along one side: left, right, top or bottom
        side = int(rng.integers(4))
        n = int(round(frac * (gw if side < 2 else gh)))
        if side == 0:
            h[:, :n] = True
        elif side == 1:
            h[:, gw - n:] = True
        elif side == 2:
            h[:n, :] = True
        else:
            h[gh - n:, :] = True
    else:                                    # random rectangles until frac of the tokens are hidden
        while h.mean() < frac:
            bw, bh = rng.integers(gw // 6, gw // 2 + 1), rng.integers(gh // 6, gh // 2 + 1)
            x0, y0 = rng.integers(0, gw - bw + 1), rng.integers(0, gh - bh + 1)
            h[y0:y0 + bh, x0:x0 + bw] = True
    return h


def record(M, fb, logits, b):
    g = M.G['thermal']
    lp = log_post(logits)
    S1 = lp.shape[-1]
    k = lp.flatten(1).argmax(1).cpu().numpy()
    ab, sy, sx = k // (S1 * S1), (k // S1) % S1, k % S1
    pos = offsets(sy, sx, g)
    pxy = lp.exp().sum(1).cpu().numpy()
    ii = np.arange(S1)
    out = []
    for j, f in enumerate(fb):
        near = ((ii[:, None] - sy[j]) ** 2 + (ii[None, :] - sx[j]) ** 2) * g['cell'] ** 2 <= c['conf_radius_m'] ** 2
        out.append(dict(id=f['id'], scene=f['scene'], place=place(f['scene']), cond=f['cond'],
                        err=float(np.hypot(*(pos[j] - b['rel'][j]))), conf=float((pxy[j] * near).sum()),
                        yaw_err=float(abs(wrap(b['yaw_prior'][j] + g['offs'][ab[j]] - b['yaw'][j])))))
    return out


rec = {(m.name, cond, fr): {'valid': [], 'test': []} for m in models
       for cond, fr in [('full', 0.0)] + [(cd, f) for f in c['partial_frac'] for cd in ('masked', 'predicted')]}
for si, split in enumerate(('valid', 'test')):
    ids = [f['id'] for f in fl if f['split'] == split]
    for s0 in range(0, len(ids), EB):
        idx = ids[s0:s0 + EB]
        fb = [fl[i] for i in idx]
        for M in models:
            rec[(M.name, 'full', 0.0)][split] += record(M, fb, *M.match('thermal', fb))
        x = to_input(imgs[idx], bb['norm'], dev)
        for fr in c['partial_frac']:
            hid = hidden(np.random.default_rng([c['seed'], si, s0, int(round(fr * 1000))]), fr, c['partial_mode'])
            keep, hix = (torch.as_tensor(np.flatnonzero(v.ravel()), device=dev) for v in (~hid, hid))
            with torch.no_grad(), torch.autocast(**amp):
                tk, _ = encode(bb, x, keep=keep, n_last=L, ctx=True)
            vis = torch.as_tensor(~hid, device=dev)[None].expand(len(idx), -1, -1)
            for M in models:
                with torch.no_grad(), torch.autocast(**amp):
                    zv = M.ad(tk.float(), TH).float()
                z = torch.zeros(len(idx), gh * gw, zv.shape[-1], device=dev)
                z[:, keep] = zv
                rec[(M.name, 'masked', fr)][split] += record(M, fb, *M.match('thermal', fb, z=z.view(len(idx), gh, gw, -1),
                                                                               mask=vis))
                with torch.no_grad(), torch.autocast(**amp):
                    pr = M.pr(zv, uv[keep], uv[hix], TH, RGB).float()
                mu, sd = zv.mean(-1).mean(1), zv.std(-1).mean(1)       # put the prediction on the visible tokens' scale
                z[:, hix] = pr * sd[:, None, None] + mu[:, None, None]
                rec[(M.name, 'predicted', fr)][split] += record(M, fb, *M.match('thermal', fb, z=z.view(len(idx), gh, gw, -1)))
        print(f'{split}: {min(s0 + EB, len(ids))}/{len(ids)}', flush=True)

res = dict(mode=c['partial_mode'], fracs=c['partial_frac'], runs={})
print(f"{'run':<24} {'view':<10} {'hidden':>6} {'median':>7} {'<=25m':>6} {'city':>14} {'village':>14}")
for (name, cond, fr), r in rec.items():
    tau = pick_tau(r['valid'], c['ok_m'], c['target_precision']) if r['valid'] else 0.5
    t = by_group(r['test'], tau, c['ok_m'])
    res['runs'].setdefault(name, {}).setdefault(cond, {})[str(fr)] = t
    cell = lambda p: f"{t[p]['median_m']:6.1f} {t[p]['r@25m']:6.0%}" if p in t else ' ' * 14
    print(f"{name:<24} {cond:<10} {fr:>6.0%} {t['all']['median_m']:>7.1f} {t['all']['r@25m']:>6.0%} {cell('city'):>14} "
          f"{cell('village'):>14}")
res['predictor_from'] = {m.name: m.ck_cfg.get('p1_ckpt') for m in models}
json.dump(res, open(f'{O}/exp_partial.json', 'w'), indent=1)
