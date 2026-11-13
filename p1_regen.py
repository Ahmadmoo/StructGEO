"""Phase 1: rebuild the features of a hidden part of the thermal image from the RGB image.
Case A: only the thermal image is hidden (RGB sees the area). Case B: the area is hidden in both images.
Three ways to rebuild the hidden thermal features from RGB:
  JEPA       - r3: adapter + predictor trained with the JEPA loss (the predictor makes the guess)
  no JEPA    - a2: adapter trained with the anchor only; the RGB vector at the same spot, through the adapter
  no adapter - plain DINOv3; the RGB DINOv3 vector at the same spot
In case B the RGB spot is hidden too, so 'no JEPA' and 'no adapter' get what DINOv3 puts at its mask tokens there.
Each method is scored against the true thermal features in its own space (hit@1 inside the hole, as in Phase 1).
For the figure, PCA colors of the three spaces are aligned on the true features (Procrustes), so tiles compare.
Writes results/p1_regen.{md,json} and results/p1_regen/<pair id>.png."""
import argparse, json, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import (RGB, TH, load_cfg, load_pairs, geometry, sample_mask, load_backbone, to_input, encode, Adapter,
                    Predictor, interior, chance_hit1)

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--jepa', default='r3_dino_same')
ap.add_argument('--nojepa', default='a2_nce_only')
ap.add_argument('--target', default='th', choices=['th', 'rgb'], help='which image gets the hole and is rebuilt')
ap.add_argument('--hole', nargs=2, type=float, default=[0.3, 0.4], help='hole area, fraction of the image')
ap.add_argument('--repeats', type=int, default=4, help='holes per batch for the numbers')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--pairs', nargs='*', type=int, help='pair ids for figures (default: first n_fig test pairs)')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
C, O, dev, P = cfg['cache_dir'], cfg['out_dir'], cfg['device'], cfg['patch']
os.makedirs(f'{O}/p1_regen', exist_ok=True)
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)
kt = TH if a.target == 'th' else RGB
kc = 1 - kt
NAME = {RGB: 'RGB', TH: 'thermal'}

cj = torch.load(f'{O}/{a.jepa}.pt', map_location='cpu', weights_only=False)
cn = torch.load(f'{O}/{a.nojepa}.pt', map_location='cpu', weights_only=False)
bbn, option, L = cj['cfg']['backbone'], cj['cfg']['option'], cj['cfg']['layers']
assert (cn['cfg']['backbone'], cn['cfg']['option'], cn['cfg']['layers']) == (bbn, option, L), 'runs must share backbone, option, layers'
assert cj['cfg']['w_jepa'] > 0, f'{a.jepa} has no trained predictor'
pairs, meta = load_pairs(cfg)
te = [p['id'] for p in pairs if p['test']]
G = geometry(cfg, meta, option, dev)
img = [np.load(f'{C}/img_rgb_{option}.npy', mmap_mode='r'), np.load(f'{C}/img_th.npy', mmap_mode='r')]
feat = [np.load(f'{C}/feat_{bbn}_L{L}_rgb_{option}.npy', mmap_mode='r'), np.load(f'{C}/feat_{bbn}_L{L}_th.npy', mmap_mode='r')]
HW = [feat[k].shape[1:3] for k in (RGB, TH)]
D = feat[TH].shape[-1]
bb = load_backbone(cfg, bbn, dev)
assert bb['kind'] == 'dino', 'case B needs DINOv3 mask tokens'


def build(ck, pred):
    ad, ema = Adapter(D, ck['cfg']), Adapter(D, ck['cfg'])
    ad.load_state_dict(ck['adapter'])
    ema.load_state_dict(ck['ema'])
    pr = None
    if pred:
        pr = Predictor(ck['cfg'])
        pr.load_state_dict(ck['predictor'])
        pr = pr.to(dev).eval()
    return ad.to(dev).eval(), ema.to(dev).eval(), pr


ad_j, ema_j, pr_j = build(cj, True)
ad_n = build(cn, False)[0]
dim_j = cj['cfg']['dim']
hole_cfg = dict(cj['cfg'], n_blocks=1, block_scale=a.hole)
near = torch.as_tensor(np.abs(G['uv'][kt][:, None] - G['uv'][kc][None]).max(-1).argmin(1), device=dev)  # same spot, other image
METHODS = ('JEPA', 'no JEPA', 'no adapter')


def rebuild(idx, tgt, keep, case):
    """{method: (rebuilt vectors at the hidden target spots (B, n, d), true target tokens (B, N, d))}."""
    Fz = [torch.from_numpy(np.ascontiguousarray(f[idx])).to(dev).float().flatten(1, 2) for f in feat]
    t = torch.as_tensor(tgt, device=dev)
    nc = near[t]
    with torch.no_grad(), torch.autocast(**amp):
        if case == 'A':  # the other image is fully visible
            Zc, kp = Fz[kc], torch.arange(Fz[kc].shape[1], device=dev)
        else:            # the other image is hidden at the same place (as in JEPA training)
            kp = torch.as_tensor(keep, device=dev)
            Zc = encode(bb, to_input(img[kc][idx], bb['norm'], dev), keep=kp, n_last=L, ctx=True, all_tokens=True)[0]
        out = {'JEPA': (pr_j(ad_j(Zc[:, kp], kc), G['uv_t'][kc][kp], G['uv_t'][kt][t], kc, kt),
                        F.layer_norm(ema_j(Fz[kt], kt).float(), (dim_j,))),
               'no JEPA': (ad_n(Zc[:, nc], kc), ad_n(Fz[kt], kt)),
               'no adapter': (Zc[:, nc], Fz[kt])}
    return {k: (x.float(), y.float()) for k, (x, y) in out.items()}


def score(q, g, tgt, h, w):
    """hit@1 and rank of each hidden spot among all target tokens of its pair (interior spots only)."""
    pos, flat = interior(h, w)
    lut = -np.ones(h * w, int)
    lut[flat] = np.arange(len(flat))
    j = lut[tgt]
    ok = j >= 0
    S = F.normalize(q[:, torch.as_tensor(ok, device=dev)], dim=-1) @ F.normalize(g[:, torch.as_tensor(flat, device=dev)], dim=-1).transpose(1, 2)
    jt = torch.as_tensor(j[ok], device=dev)
    pt = torch.as_tensor(pos, device=dev)
    hit = (pt[S.argmax(-1)] - pt[jt]).abs().max(-1).values <= 1
    rank = (S > S.gather(-1, jt.view(1, -1, 1).expand(S.shape[0], -1, 1))).sum(-1) + 1
    return hit.cpu().numpy(), rank.cpu().numpy(), ok


# ---------------------------------------------------------------- numbers over all test pairs (same holes for A and B)
rec = {}
rng = np.random.default_rng(0)
for s0 in range(0, len(te), a.batch):
    idx = te[s0:s0 + a.batch]
    for _ in range(a.repeats):
        tgt, keep = sample_mask(rng, hole_cfg, G, kc, kt)
        for case in 'AB':
            for m, (q, g) in rebuild(idx, tgt, keep, case).items():
                hit, rank, _ = score(q, g, tgt, *HW[kt])
                r = rec.setdefault((case, m), ([], []))
                r[0].append(hit.ravel())
                r[1].append(rank.ravel())
res = {case: {m: dict(hit1=round(float(np.concatenate(rec[(case, m)][0]).mean()), 4),
                      median_rank=float(np.median(np.concatenate(rec[(case, m)][1]))),
                      n=int(np.concatenate(rec[(case, m)][0]).size)) for m in METHODS} for case in 'AB'}
chance = chance_hit1(*HW[kt])
json.dump(dict(jepa=a.jepa, nojepa=a.nojepa, target=NAME[kt], hole=a.hole, n_pairs=len(te), repeats=a.repeats,
               chance_hit1=chance, results=res), open(f'{O}/p1_regen.json', 'w'), indent=1)
lines = [f'# Rebuild hidden {NAME[kt]} features from {NAME[kc]} (test pairs, one hole of {a.hole[0]:.0%}-{a.hole[1]:.0%})', '',
         f'hit@1 inside the hole: the rebuilt vector finds the true spot (within 1 token) among all {NAME[kt]} tokens of the pair. '
         f'Chance {chance}.', '',
         f'| Method | A: {NAME[kc]} sees the area | B: hidden in both |', '|---|---|---|']
lab = {'JEPA': f'JEPA ({a.jepa})', 'no JEPA': f'no JEPA, anchor only ({a.nojepa})', 'no adapter': 'no adapter (frozen DINOv3)'}
for m in METHODS:
    lines.append(f"| {lab[m]} | {res['A'][m]['hit1']:.3f} (rank {res['A'][m]['median_rank']:g}) | "
                 f"{res['B'][m]['hit1']:.3f} (rank {res['B'][m]['median_rank']:g}) |")
open(f'{O}/p1_regen.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---------------------------------------------------------------- figures
H, GAP, CAP = 256, 8, 34


def pca3(X):
    mu = X.mean(0)
    return mu, torch.linalg.svd(X - mu, full_matrices=False)[2][:3].T  # exact SVD: same colors on every call


def colorer(ref, X):
    """PCA colors for the space of X (true tokens), rotated and scaled onto the PCA of the reference space."""
    mr, Vr = pca3(ref)
    Pr = (ref - mr) @ Vr
    lo, hi = torch.quantile(Pr, .01, 0), torch.quantile(Pr, .99, 0)
    ms, Vs = pca3(X)
    Ps = (X - ms) @ Vs
    U, S, Wt = torch.linalg.svd(Ps.T @ Pr)
    R, sc = U @ Wt, S.sum() / (Ps ** 2).sum()
    return lambda Y: ((((Y - ms) @ Vs @ R) * sc - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()


def fit(a_, im):
    return cv2.resize(a_, (round(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_NEAREST)


def tok(c, k, im):
    return fit((c.reshape(*HW[k], 3) * 255).astype(np.uint8), im)


def box(t, sel, k, im, gray=False):
    t = t.copy()
    i, j = sel // HW[k][1], sel % HW[k][1]
    s = H / im.shape[0]
    p0, p1 = (int(j.min() * P * s), int(i.min() * P * s)), (int((j.max() + 1) * P * s) - 1, int((i.max() + 1) * P * s) - 1)
    if gray:
        t[p0[1]:p1[1] + 1, p0[0]:p1[0] + 1] = (0.25 * t[p0[1]:p1[1] + 1, p0[0]:p1[0] + 1] + 0.75 * 150).astype(np.uint8)
    cv2.rectangle(t, p0, p1, (255, 255, 255), 2)
    return t


def cap(t, text):
    s = np.full((CAP, t.shape[1], 3), 255, np.uint8)
    cv2.putText(s, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([t, s])


def row(tiles):
    w = sum(t.shape[1] for t in tiles) + GAP * (len(tiles) - 1)
    out = np.full((tiles[0].shape[0], w, 3), 255, np.uint8)
    x = 0
    for t in tiles:
        out[:, x:x + t.shape[1]] = t
        x += t.shape[1] + GAP
    return out


rng = np.random.default_rng(1)
for pid in (a.pairs or te[:cfg['n_fig']]):
    tgt, keep = sample_mask(rng, hole_cfg, G, kc, kt)
    it, ic = np.ascontiguousarray(img[kt][pid]), np.ascontiguousarray(img[kc][pid])
    hidc = np.setdiff1d(np.arange(HW[kc][0] * HW[kc][1]), keep)
    rows = []
    for case in 'AB':
        vec = rebuild([pid], tgt, keep, case)
        ref = vec['no adapter'][1][0]                     # true target DINOv3 tokens: the reference colors
        tiles = [cap(box(fit(it, it), tgt, kt, it, gray=True), f'{case}: {NAME[kt]}, hidden'),
                 cap(box(fit(ic, ic), hidc if case == 'B' else near[torch.as_tensor(tgt, device=dev)].cpu().numpy(), kc, ic, gray=case == 'B'),
                     f'{NAME[kc]} input' + (', hidden too' if case == 'B' else ', sees it')),
                 cap(box(tok(colorer(ref, ref)(ref), kt, it), tgt, kt, it), 'ground truth')]
        for m in METHODS:
            q, g = vec[m]
            f = colorer(ref, g[0])
            c = f(g[0]) * 0.45                              # outside the hole: truth, dimmed
            c[tgt] = f(q[0])                                # inside the hole: rebuilt
            h = score(q, g, tgt, *HW[kt])[0]
            tiles.append(cap(box(tok(c, kt, it), tgt, kt, it), f'{m}: hit@1 {h.mean():.2f}'))
        rows.append(row(tiles))
    out = np.concatenate([rows[0], np.full((GAP, rows[0].shape[1], 3), 255, np.uint8), rows[1]])
    cv2.imwrite(f'{O}/p1_regen/{pid}.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print('figures:', f'{O}/p1_regen/')
