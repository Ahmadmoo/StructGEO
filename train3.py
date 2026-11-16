import argparse, json, math, os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import load_cfg, load_backbone, encode, to_input, Adapter, posenc, var_loss

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--zeroshot', action='store_true')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
C, O, dev, L, p, bbn = cfg['cache_dir'], cfg['out_dir'], cfg['device'], cfg['layers'], cfg['patch'], cfg['backbone']
name = f'zs_{bbn}_L{L}' if a.zeroshot else cfg['name']
os.makedirs(f'{O}/figs', exist_ok=True)
torch.manual_seed(cfg['seed'])
rng = np.random.default_rng(cfg['seed'])

frames = json.load(open(f'{C}/ms2_frames.json'))
meta = json.load(open(f'{C}/ms2_meta.json'))
SENS, PLANE = meta['sensors'], meta['planes']
S, (GH, GW) = len(SENS), meta['grid']['thr']
tr = [f['id'] for f in frames if f['split'] == 'train']
te = [f['id'] for f in frames if f['split'].startswith('test')]
ld = lambda n: np.load(f'{C}/{n}.npy', mmap_mode='r')
feat = [ld(f'ms2_feat_{bbn}_L{L}_{s}') for s in SENS]
img = [ld(f'ms2_img_{s}') for s in SENS]
fwd, fok = ld('ms2_fwd'), ld('ms2_fwd_ok')
bwd, bok = [ld(f'ms2_bwd_{s}') for s in SENS], [ld(f'ms2_bwd_ok_{s}') for s in SENS]
ii, jj = np.meshgrid(np.arange(GH), np.arange(GW), indexing='ij')
POS = np.stack([ii.ravel(), jj.ravel()], 1)
UVn = np.stack([(jj.ravel() + .5) * p, (ii.ravel() + .5) * p], 1)
UV = torch.tensor(UVn, dtype=torch.float32, device=dev)
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)


def T(arr, idx, dt=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(arr[idx])).to(dev).to(dt)


def to_ref(z, fw):
    out = []
    for k, zk in enumerate(z):
        if PLANE[SENS[k]] == 'thr':
            out.append(zk.float().flatten(1, 2))
        else:
            g = F.grid_sample(zk.permute(0, 3, 1, 2).float(), fw[:, k], mode='bilinear', align_corners=False)
            out.append(g.flatten(2).transpose(1, 2))
    return torch.stack(out, 1)


def sigreg(z, step):
    with torch.autocast(amp['device_type'], enabled=False):
        g = torch.Generator(device=z.device).manual_seed(step)
        A = torch.randn(z.shape[1], cfg['sig_proj'], generator=g, device=z.device)
        A = A / A.norm(dim=0)
        t = torch.linspace(-5, 5, 17, device=z.device)
        w = torch.exp(-t ** 2 / 2)
        x = (z.float() @ A)[..., None] * t
        err = ((x.cos().mean(0) - w) ** 2 + x.sin().mean(0) ** 2) * w
        return torch.trapezoid(err, t, dim=-1).mean() * z.shape[0]


def sample(zk):
    x = zk.reshape(-1, zk.shape[-1])
    return x[torch.as_tensor(rng.choice(len(x), min(cfg['sig_n'], len(x)), replace=False), device=dev)]


def nce(Z, OK):
    loss, n = 0, 0
    Pt = torch.as_tensor(POS, device=dev)
    for ka in range(S):
        for kb in range(ka + 1, S):
            m = (OK[:, ka] & OK[:, kb]).cpu().numpy()
            sel = [(b, j) for b in range(len(m)) for j in rng.choice(np.flatnonzero(m[b]), min(cfg['nce_anchors'], m[b].sum()), replace=False)]
            if len(sel) < 2:
                continue
            bi, sj = (torch.as_tensor(v, device=dev) for v in zip(*sel))
            fa, fb = (F.normalize(Z[bi, k, sj].float(), dim=-1) for k in (ka, kb))
            near = (bi[:, None] == bi[None]) & ((Pt[sj][:, None] - Pt[sj][None]).abs().amax(-1) <= 1)
            near &= ~torch.eye(len(bi), dtype=torch.bool, device=dev)
            lg = (fa @ fb.T / cfg['nce_tau']).masked_fill(near, float('-inf'))
            y = torch.arange(len(bi), device=dev)
            loss, n = loss + (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2, n + 1
    return loss / max(n, 1)


class Pred(nn.Module):
    def __init__(s):
        super().__init__()
        w = s.w = cfg['pred_width']
        s.inp = nn.Linear(cfg['dim'], w)
        s.mask = nn.Parameter(torch.randn(1, 1, w) * 0.02)
        s.sens = nn.Parameter(torch.randn(S, w) * 0.02)
        layer = nn.TransformerEncoderLayer(w, cfg['pred_heads'], 4 * w, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        s.tf = nn.TransformerEncoder(layer, cfg['pred_depth'], enable_nested_tensor=False)
        s.norm = nn.LayerNorm(w)
        s.out = nn.Linear(w, cfg['dim'])

    def forward(s, ctx, ctx_uv, pad, q_uv, kc, kt=None):
        c = s.inp(ctx) + posenc(ctx_uv, s.w, p) + s.sens[kc]
        q = (s.mask + posenc(q_uv, s.w, p)[None]).expand(ctx.shape[0], -1, -1)
        if kt is not None:
            q = q + s.sens[kt]
        pad = torch.cat([pad, torch.zeros(len(pad), q.shape[1], dtype=torch.bool, device=pad.device)], 1)
        x = s.norm(s.tf(torch.cat([c, q], 1), src_key_padding_mask=pad))
        return s.out(x[:, c.shape[1]:])


def blocks():
    W, H = GW * p, GH * p
    while True:
        b = []
        for _ in range(cfg['n_blocks']):
            s_ = rng.uniform(*cfg['block_scale']) * W * H
            r = math.exp(rng.uniform(*np.log(cfg['block_aspect'])))
            bw, bh = min(W, math.sqrt(s_ * r)), min(H, math.sqrt(s_ / r))
            x0, y0 = rng.uniform(0, W - bw), rng.uniform(0, H - bh)
            b.append((x0, y0, x0 + bw, y0 + bh))
        b = np.array(b)
        q = ((UVn[:, None, 0] >= b[:, 0]) & (UVn[:, None, 0] < b[:, 2]) & (UVn[:, None, 1] >= b[:, 1]) & (UVn[:, None, 1] < b[:, 3])).any(1)
        if q.sum() >= 4 and q.mean() <= 0.6:
            return b, np.flatnonzero(q)


def hidden(idx, Z, OK, c, cnt, fw):
    kc = int(rng.integers(S))
    q = PLANE[SENS[kc]]
    b, Q = blocks()
    Q = torch.as_tensor(Q, device=dev)
    m = cfg['margin_px'] + p / 2
    bt = torch.as_tensor(b, device=dev, dtype=torch.float32)
    if q == 'thr':
        pos = UV.expand(len(idx), -1, -1)
        okp = torch.ones(pos.shape[:2], dtype=torch.bool, device=dev)
    else:
        pos, okp = T(bwd[kc], idx).flatten(1, 2), T(bok[kc], idx, torch.bool).flatten(1, 2)
    hide = ((pos[..., None, 0] > bt[:, 0] - m) & (pos[..., None, 0] < bt[:, 2] + m) &
            (pos[..., None, 1] > bt[:, 1] - m) & (pos[..., None, 1] < bt[:, 3] + m)).any(-1) & okp
    if q != 'thr':
        Hq, Wq = meta['hw'][q]
        gh, gw = meta['grid'][q]
        cj, ci = np.meshgrid(np.arange(gw), np.arange(gh))
        cq = torch.tensor(np.stack([(cj.ravel() + .5) * p, (ci.ravel() + .5) * p], 1), dtype=torch.float32, device=dev)
        xy = (fw[:, kc].flatten(1, 2)[:, Q] + 1) / 2 * torch.tensor([Wq, Hq], dtype=torch.float32, device=dev)
        d = (cq[None, :, None] - xy[:, None]).abs().amax(-1)
        hide |= ((d <= p + cfg['margin_px']) & OK[:, kc, Q][:, None]).any(-1)
    with torch.no_grad():
        f, _ = encode(bb, to_input(img[kc][idx], bb['norm'], dev), n_last=L, ctx=True, masks=hide)
    zc = ad(f.float(), kc)
    use = ~hide & okp
    n = use.sum(1)
    Lc = int(n.max())
    if Lc == 0:
        return None
    order = torch.argsort((~use).int(), dim=1, stable=True)[:, :Lc]
    ctx = torch.gather(zc, 1, order[..., None].expand(-1, -1, zc.shape[-1]))
    cuv = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
    pad = torch.arange(Lc, device=dev)[None] >= n[:, None]
    if cfg['hid_target'] == 'center':
        tgt, tm, kt = c[:, Q], cnt[:, Q] >= 2, None
    else:
        kt = int((kc + 1 + rng.integers(S - 1)) % S)
        tgt, tm = Z[:, kt, Q], OK[:, kt, Q] & (cnt[:, Q] >= 2)
    if not tm.any():
        return None
    if cfg['hid_detach']:
        tgt = tgt.detach()
    pred = P(ctx, cuv, pad, UV[Q], kc, kt)
    return ((pred.float() - tgt.float()) ** 2).mean(-1)[tm].mean()


D = feat[0].shape[-1]
ad = Adapter(D, cfg, S).to(dev)
P = Pred().to(dev)
bb = load_backbone(cfg, bbn, dev) if cfg['w_hid'] > 0 and not a.zeroshot else None
opt = torch.optim.AdamW(list(ad.parameters()) + list(P.parameters()), lr=cfg['lr'], weight_decay=cfg['wd'])
steps = 0 if a.zeroshot else cfg['epochs'] * max(1, len(tr) // cfg['batch'])
warm = max(1, int(cfg['warmup'] * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

log = []
for st in range(steps):
    idx = np.sort(rng.choice(tr, min(cfg['batch'], len(tr)), replace=False))
    fw, OK = T(fwd, idx), T(fok, idx, torch.bool).flatten(2)
    with torch.autocast(**amp):
        z = [ad(T(feat[k], idx), k) for k in range(S)]
        Z = to_ref(z, fw)
        cnt = OK.sum(1)
        c = (Z * OK[..., None]).sum(1) / cnt.clamp(min=1)[..., None]
        parts = {}
        if cfg['w_inv'] > 0:
            m = OK & (cnt >= 2)[:, None]
            parts['inv'] = ((Z - c[:, None]) ** 2).mean(-1)[m].mean()
        if cfg['w_sig'] > 0:
            parts['sig'] = sum(sigreg(sample(zk), st) for zk in z) / S
        if cfg['w_var'] > 0:
            parts['var'] = sum(var_loss(zk) for zk in z)
        if cfg['w_nce'] > 0:
            parts['nce'] = nce(Z, OK)
        if cfg['w_hid'] > 0:
            h = hidden(idx, Z, OK, c, cnt, fw)
            if h is not None:
                parts['hid'] = h
        loss = sum(cfg['w_' + k] * v for k, v in parts.items())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    log.append({k: v.item() for k, v in parts.items()})
    if (st + 1) % max(1, steps // cfg['epochs']) == 0:
        keys = sorted({k for r in log for k in r})
        print(json.dumps(dict(epoch=(st + 1) * cfg['epochs'] // steps, **{k: round(float(np.mean([r[k] for r in log if k in r])), 4) for k in keys})), flush=True)
        log = []

ad.eval()


def zfun(idx):
    with torch.no_grad(), torch.autocast(**amp):
        return [(T(feat[k], idx) if a.zeroshot else ad(T(feat[k], idx), k)).float() for k in range(S)]


inner = (POS[:, 0] > 0) & (POS[:, 0] < GH - 1) & (POS[:, 1] > 0) & (POS[:, 1] < GW - 1)
pairs = [(ka, kb) for ka in range(S) for kb in range(S) if ka != kb]
keys = [f'{SENS[ka]}→{SENS[kb]}' for ka, kb in pairs]
rows = []
for s0 in range(0, len(te), 16):
    idx = te[s0:s0 + 16]
    z = zfun(idx)
    with torch.no_grad():
        Z = to_ref(z, T(fwd, idx))
    OKn = fok[idx].reshape(len(idx), S, -1).all(1) & inner
    for b, i in enumerate(idx):
        v = np.flatnonzero(OKn[b])
        if len(v) < cfg['min_valid']:
            continue
        near = torch.as_tensor(np.abs(POS[v][:, None] - POS[v][None]).max(-1) <= 1, device=dev)
        f = F.normalize(Z[b][:, torch.as_tensor(v, device=dev)], dim=-1)
        ar = torch.arange(len(v), device=dev)
        hit, rank = {}, {}
        for k, (ka, kb) in zip(keys, pairs):
            Sm = f[ka] @ f[kb].T
            hit[k] = near[ar, Sm.argmax(1)].float().mean().item()
            rank[k] = ((Sm > Sm.diagonal()[:, None]).sum(1) + 1).float().median().item()
        rows.append(dict(cond=frames[i]['cond'], chance=near.float().mean().item(), hit=hit, rank=rank))
    if s0 == 0:
        for b, i in enumerate(idx[:cfg['n_fig']]):
            X = torch.cat([zk[b].reshape(-1, zk.shape[-1]) for zk in z])
            X = X - X.mean(0)
            _, _, V = torch.pca_lowrank(X, q=3, center=False)
            Pc = X @ V
            lo, hi = torch.quantile(Pc, 0.01, 0), torch.quantile(Pc, 0.99, 0)
            Pc = ((Pc - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()
            tiles, o = [], 0
            for k, zk in enumerate(z):
                gh, gw = zk.shape[1:3]
                im = np.ascontiguousarray(img[k][i])
                col = cv2.resize((Pc[o:o + gh * gw].reshape(gh, gw, 3) * 255).astype(np.uint8), (im.shape[1], im.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
                tiles.append(np.concatenate([im, col], 1))
                o += gh * gw
            Wm = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 0), (0, Wm - t.shape[1]), (0, 0))) for t in tiles]
            cv2.imwrite(f'{O}/figs/{name}_{i}.png', cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))

out = dict(name=name, step='zero-shot' if a.zeroshot else 'trained', backbone=bbn, layers=L, sensors=SENS, n_frames=len(rows),
           chance_hit1=round(float(np.mean([r['chance'] for r in rows])), 4),
           weights={k: cfg[k] for k in ('w_inv', 'w_sig', 'w_hid', 'w_nce', 'w_var')}, hid_target=cfg['hid_target'],
           hid_detach=cfg['hid_detach'])
for cond in ['all', *sorted({r['cond'] for r in rows})]:
    sub = [r for r in rows if cond == 'all' or r['cond'] == cond]
    h = {k: round(float(np.mean([r['hit'][k] for r in sub])), 4) for k in keys}
    out[cond] = dict(n=len(sub), hit1_mean=round(float(np.mean(list(h.values()))), 4), hit1=h,
                     medrank={k: round(float(np.mean([r['rank'][k] for r in sub])), 2) for k in keys})
json.dump(out, open(f'{O}/{name}.json', 'w'), indent=1, ensure_ascii=False)
if not a.zeroshot:
    torch.save(dict(adapter=ad.state_dict(), predictor=P.state_dict(), cfg=cfg, sensors=SENS), f'{O}/{name}.pt')
print(json.dumps({k: out['all'][k] for k in ('hit1_mean', 'hit1')}, ensure_ascii=False))
