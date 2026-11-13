"""Phase 1: what only the JEPA predictor can do.
Hide blocks of one sensor's image and ask for the OTHER sensor's vector at every hidden spot. Neither input shows that
ground. The predictor predicts it; a model without a predictor can only use what DINOv3 puts at its mask tokens ('enc')
or copy the nearest visible spot ('copy'). 'seen' is the same spot with nothing hidden (the Phase 1 hit@1 setting).
Score: hit@1 among all target-sensor tokens of the pair, as in Phase 1, split by depth inside the hole.
Writes results/p1_fillin.{json,md} and figures in results/p1_figs/."""
import argparse, json, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import (RGB, TH, load_cfg, load_pairs, geometry, sample_mask, load_backbone, to_input, encode, Adapter,
                    Predictor, interior, chance_hit1)

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--runs', nargs='+', default=['r3_dino_same', 'a1_jepa_only', 'a2_nce_only'])
ap.add_argument('--repeats', type=int, default=4)
ap.add_argument('--batch', type=int, default=16)
args = ap.parse_args()
cfg = load_cfg(*args.cfg)
C, O, dev, P = cfg['cache_dir'], cfg['out_dir'], cfg['device'], cfg['patch']
os.makedirs(f'{O}/p1_figs', exist_ok=True)
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)

ck = {r: torch.load(f'{O}/{r}.pt', map_location='cpu', weights_only=False) for r in args.runs}
c0 = ck[args.runs[0]]['cfg']
bbn, option, L = c0['backbone'], c0['option'], c0['layers']
assert all((k['cfg']['backbone'], k['cfg']['option'], k['cfg']['layers']) == (bbn, option, L) for k in ck.values()), \
    'runs must share backbone, option and layers'
pairs, meta = load_pairs(cfg)
te = [p['id'] for p in pairs if p['test']]
G = geometry(cfg, meta, option, dev)
img = [np.load(f'{C}/img_rgb_{option}.npy', mmap_mode='r'), np.load(f'{C}/img_th.npy', mmap_mode='r')]
feat = [np.load(f'{C}/feat_{bbn}_L{L}_rgb_{option}.npy', mmap_mode='r'),
        np.load(f'{C}/feat_{bbn}_L{L}_th.npy', mmap_mode='r')]
HW = [feat[k].shape[1:3] for k in (RGB, TH)]
D = feat[TH].shape[-1]
bb = load_backbone(cfg, bbn, dev)
assert bb['kind'] == 'dino', 'the no-predictor baseline needs DINOv3 mask tokens'

M = {}
for r, k in ck.items():
    ad, ema = Adapter(D, k['cfg']), Adapter(D, k['cfg'])
    ad.load_state_dict(k['adapter'])
    ema.load_state_dict(k['ema'])
    pr = None
    if k['cfg']['w_jepa'] > 0:
        pr = Predictor(k['cfg'])
        pr.load_state_dict(k['predictor'])
        pr = pr.to(dev).eval()
    M[r] = dict(ad=ad.to(dev).eval(), ema=ema.to(dev).eval(), pr=pr, dim=k['cfg']['dim'])
near = {kt: torch.as_tensor(np.abs(G['uv'][kt][:, None] - G['uv'][1 - kt][None]).max(-1).argmin(1), device=dev)
        for kt in (RGB, TH)}  # context token at each target spot


def score(q, g, tgt, h, w):
    """q (B, n, d): vectors for target spots tgt (flat, target grid h x w); g (B, h*w, d): target tokens.
    Returns hit@1 and rank (B, m) for the m spots of tgt in the interior, and which spots those are."""
    pos, flat = interior(h, w)
    lut = -np.ones(h * w, int)
    lut[flat] = np.arange(len(flat))
    j = lut[tgt]
    ok = j >= 0
    S = F.normalize(q[:, torch.as_tensor(ok, device=dev)].float(), dim=-1) @ F.normalize(g[:, torch.as_tensor(flat, device=dev)].float(), dim=-1).transpose(1, 2)
    jt = torch.as_tensor(j[ok], device=dev)
    pt = torch.as_tensor(pos, device=dev)
    hit = (pt[S.argmax(-1)] - pt[jt]).abs().max(-1).values <= 1
    rank = (S > S.gather(-1, jt.view(1, -1, 1).expand(S.shape[0], -1, 1))).sum(-1) + 1
    return hit.cpu().numpy(), rank.cpu().numpy(), ok


def hole(tgt, keep, kc, kt):
    """Depth bin of each target spot (0: <= 2 tokens from the nearest visible context token, 1: 3-4, 2: 5+) and
    that nearest visible token."""
    d = np.abs(G['uv'][kt][tgt][:, None] - G['uv'][kc][keep][None]).max(-1)
    return np.digitize(d.min(1) / P, [2.5, 4.5]), torch.as_tensor(keep[d.argmin(1)], device=dev)


def fill(idx, kc, kt, tgt, keep):
    """Vectors for the hidden target spots, per method, and the target tokens each is scored against."""
    Fz = [torch.from_numpy(np.ascontiguousarray(f[idx])).to(dev).float().flatten(1, 2) for f in feat]
    keep_t, nc = torch.as_tensor(keep, device=dev), near[kt][torch.as_tensor(tgt, device=dev)]
    _, nvis = hole(tgt, keep, kc, kt)
    with torch.no_grad(), torch.autocast(**amp):
        Zm = encode(bb, to_input(img[kc][idx], bb['norm'], dev), keep=keep_t, n_last=L, ctx=True, all_tokens=True)[0]
        out = {'dino:enc': (Zm[:, nc], Fz[kt]), 'dino:seen': (Fz[kc][:, nc], Fz[kt])}
        for r, m in M.items():
            gt = m['ad'](Fz[kt], kt)
            out[f'{r}:enc'] = (m['ad'](Zm[:, nc], kc), gt)
            out[f'{r}:copy'] = (m['ad'](Zm[:, nvis], kc), gt)
            out[f'{r}:seen'] = (m['ad'](Fz[kc][:, nc], kc), gt)
            if m['pr'] is not None:
                p = m['pr'](m['ad'](Zm[:, keep_t], kc), G['uv_t'][kc][keep_t], G['uv_t'][kt][torch.as_tensor(tgt, device=dev)], kc, kt)
                out[f'{r}:pred'] = (p, F.layer_norm(m['ema'](Fz[kt], kt).float(), (m['dim'],)))
    return {k: (a.float(), b.float()) for k, (a, b) in out.items()}, Fz


SET = {'blocks': c0, 'big': dict(c0, n_blocks=1, block_scale=[0.3, 0.4])}
DIRS = {'rgb>th': (RGB, TH), 'th>rgb': (TH, RGB)}
rec = {}
rng = np.random.default_rng(0)
for s0 in range(0, len(te), args.batch):
    idx = te[s0:s0 + args.batch]
    for sname, sc in SET.items():
        for dname, (kc, kt) in DIRS.items():
            for _ in range(args.repeats):
                tgt, keep = sample_mask(rng, sc, G, kc, kt)
                dep = hole(tgt, keep, kc, kt)[0]
                for mth, (q, g) in fill(idx, kc, kt, tgt, keep)[0].items():
                    hit, rank, ok = score(q, g, tgt, *HW[kt])
                    r = rec.setdefault((mth, sname, dname), dict(hit=[], rank=[], depth=[]))
                    r['hit'].append(hit.ravel())
                    r['rank'].append(rank.ravel())
                    r['depth'].append(np.broadcast_to(dep[ok], hit.shape).ravel())

DEP = ('<=2', '3-4', '5+')


def summary(rs):
    hit, rank, dep = (np.concatenate([a for r in rs for a in r[k]]) for k in ('hit', 'rank', 'depth'))
    return dict(n=int(hit.size), hit1=round(float(hit.mean()), 4), median_rank=float(np.median(rank)),
                by_depth={lab: round(float(hit[dep == i].mean()), 4) if (dep == i).any() else None for i, lab in enumerate(DEP)})


out = {}
for (mth, sname, dname), r in rec.items():
    out.setdefault(mth, {}).setdefault(sname, {})[dname] = summary([r])
for mth in out:
    for sname in SET:
        out[mth][sname]['both'] = summary([rec[(mth, sname, d)] for d in DIRS])
chance = {d: chance_hit1(*HW[kt]) for d, (kc, kt) in DIRS.items()}
json.dump(dict(runs=args.runs, backbone=bbn, option=option, n_pairs=len(te), repeats=args.repeats, chance_hit1=chance,
               results=out), open(f'{O}/p1_fillin.json', 'w'), indent=1)

NAME = {'pred': 'JEPA predictor', 'enc': 'no predictor: DINOv3 mask tokens + adapter',
        'copy': 'no predictor: copy nearest visible spot', 'seen': 'ceiling: spot not hidden'}
lines = ['# Phase 1 fill-in: the other sensor at hidden spots (test pairs)', '',
         'hit@1 among all target-sensor tokens of the pair: best match within 1 token of the true spot '
         f'(chance {chance["rgb>th"]}). Neither image shows the hidden spots, except in the "ceiling" rows.', '',
         '| Run | Method | 4 blocks: RGB→T | 4 blocks: T→RGB | 1 big hole: RGB→T | 1 big hole: T→RGB | Big hole by depth (≤2 / 3–4 / 5+ tokens) |',
         '|---|---|---|---|---|---|---|']
for mth in sorted(out, key=lambda k: (k.split(':')[0] != 'dino', k)):
    run, m = mth.split(':')
    o = out[mth]
    dep = ' / '.join('-' if o['big']['both']['by_depth'][d] is None else f"{o['big']['both']['by_depth'][d]:.2f}" for d in DEP)
    lines.append(f"| {'frozen DINOv3' if run == 'dino' else run} | {NAME[m] if run != 'dino' else NAME[m].replace(' + adapter', '')} | "
                 + ' | '.join(f"{o[s][d]['hit1']:.3f}" for s in SET for d in DIRS) + f' | {dep} |')
open(f'{O}/p1_fillin.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---------------------------------------------------------------- figures (first n_fig test pairs, one big hole)
R0 = args.runs[0]
RB = next((r for r in args.runs if M[r]['pr'] is None), R0)  # a run without a JEPA loss, for the no-predictor tiles
H = 256


def pca_fn(X):
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=3, center=False)
    Q = (X - mu) @ V
    lo, hi = torch.quantile(Q, 0.01, 0), torch.quantile(Q, 0.99, 0)
    return lambda Y: (((Y - mu) @ V - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()


def tile(a, im):
    return cv2.resize(a, (round(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_NEAREST)


def tok(c, k, im, sel=None, dim=0.3):
    c = c.copy()
    if sel is not None:
        m = np.zeros(len(c), bool)
        m[sel] = True
        c[~m] *= dim
    return tile((c.reshape(*HW[k], 3) * 255).astype(np.uint8), im)


def hits(h, sel, k, im):
    a = np.repeat(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2GRAY)[..., None] // 2, 3, -1)
    a = tile(a, im)
    m = np.zeros((*HW[k], 3), np.uint8)
    on = np.zeros(HW[k], bool)
    for i, ok in zip(sel, h):
        m[i // HW[k][1], i % HW[k][1]] = (40, 170, 90) if ok else (220, 60, 50)
        on[i // HW[k][1], i % HW[k][1]] = True
    m, on = tile(m, im), tile(on.astype(np.uint8), im) > 0
    a[on] = (0.3 * a[on] + 0.7 * m[on]).astype(np.uint8)
    return a


def grid(rows):
    w = max(sum(t.shape[1] for t in r) + 8 * (len(r) - 1) for r in rows)
    out = []
    for r in rows:
        x = np.concatenate([np.concatenate([t, np.full((H, 8, 3), 255, np.uint8)], 1) for t in r[:-1]] + [r[-1]], 1)
        out.append(np.pad(x, ((0, 8), (0, w - x.shape[1]), (0, 0)), constant_values=255))
    return np.concatenate(out)[:-8]


def save(name, a):
    cv2.imwrite(f'{O}/p1_figs/{name}.png', cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


rng = np.random.default_rng(1)
for pid in te[:cfg['n_fig']]:
    kc, kt = RGB, TH
    tgt, keep = sample_mask(rng, SET['big'], G, kc, kt)
    vec, Fz = fill([pid], kc, kt, tgt, keep)
    ic, it = img[kc][pid], img[kt][pid]
    hid = np.setdiff1d(np.arange(HW[kc][0] * HW[kc][1]), keep)
    ctx = tile(np.ascontiguousarray(ic), ic)
    m = tile(np.isin(np.arange(HW[kc][0] * HW[kc][1]), hid).reshape(HW[kc]).astype(np.uint8), ic) > 0
    ctx[m] = (0.25 * ctx[m] + 0.75 * 150).astype(np.uint8)
    rows = []
    for key in (f'{R0}:pred', f'{RB}:enc'):
        if key not in vec:
            continue
        q, g = vec[key]
        f = pca_fn(g[0])
        truth, filled = f(g[0]), f(g[0])
        filled[tgt] = f(q[0])
        h, _, ok = score(q, g, tgt, *HW[kt])
        rows.append([ctx if not rows else tile(np.ascontiguousarray(it), it), tok(truth, kt, it), tok(filled, kt, it, tgt), hits(h[0], tgt[ok], kt, it)])
    save(f'fillin_{pid}', grid(rows))
    # shared space: same PCA for both sensors, per model (frozen DINOv3, then each run)
    zs = {'dino': (Fz[RGB][0], Fz[TH][0])}
    with torch.no_grad():
        zs.update({r: (m['ad'](Fz[RGB][0], RGB).float(), m['ad'](Fz[TH][0], TH).float()) for r, m in M.items()})
    sp = []
    for name, (zr, zt) in zs.items():
        f = pca_fn(torch.cat([zr, zt]))
        row = [tile(np.ascontiguousarray(img[RGB][pid]), img[RGB][pid]), tok(f(zr), RGB, img[RGB][pid]),
               tile(np.ascontiguousarray(it), it), tok(f(zt), TH, it)]
        save(f'space_{pid}_{name}', grid([row]))
        sp.append(row)
    save(f'space_{pid}', grid(sp))
    # where does a thermal patch land in the RGB image? frozen DINOv3 vs the first run
    gh, gw = HW[TH]
    rows = []
    for qi in ((gh // 3) * gw + gw // 4, (gh // 2) * gw + gw // 2, (2 * gh // 3) * gw + 3 * gw // 4):
        t = tile(np.ascontiguousarray(it), it)
        s = H / it.shape[0]
        cv2.circle(t, (int(G['uv'][TH][qi][0] * s), int(G['uv'][TH][qi][1] * s)), 7, (255, 60, 40), 2)
        row = [t]
        j = int(np.abs(G['uv'][RGB] - G['uv'][TH][qi]).max(-1).argmin())
        for name in ('dino', R0):
            zr, zt = zs[name]
            sim = (F.normalize(zr.float(), dim=-1) @ F.normalize(zt[qi].float(), dim=-1)).cpu().numpy()
            sim = np.clip((sim - np.median(sim)) / (sim.max() - np.median(sim) + 1e-6), 0, 1)
            heat = cv2.applyColorMap((sim.reshape(HW[RGB]) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
            a = tile(np.ascontiguousarray(img[RGB][pid]), img[RGB][pid])
            a = (0.45 * a + 0.55 * tile(np.ascontiguousarray(heat), img[RGB][pid])).astype(np.uint8)
            sr = H / img[RGB][pid].shape[0]
            cv2.circle(a, (int(((j % HW[RGB][1]) + .5) * P * sr), int(((j // HW[RGB][1]) + .5) * P * sr)), 7, (255, 255, 255), 2)
            row.append(a)
        rows.append(row)
    save(f'match_{pid}', grid(rows))
print('figures:', f'{O}/p1_figs', '(fillin_*, space_*, match_*)')
