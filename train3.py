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

pre = cfg.get('data_prefix') or 'ms2'
frames = json.load(open(f'{C}/{pre}_frames.json'))
meta = json.load(open(f'{C}/{pre}_meta.json'))
SALL, PLANE = meta['sensors'], meta['planes']
SENS = cfg.get('use_sensors') or SALL
SEL = [SALL.index(s) for s in SENS]
S, (GH, GW) = len(SENS), meta['grid']['thr']
CNN = 'map' in SENS and cfg.get('map_encoder') == 'cnn'
tr = [f['id'] for f in frames if f['split'] == 'train']
te = [f['id'] for f in frames if f['split'].startswith('test')]


def path(s, kind):
    v = meta.get('files', {}).get(s, {}).get(kind, dict(img=f'{C}/{pre}_img_{s}.npy', feat=f'{C}/{pre}_feat_{bbn}_L{L}_{s}.npy').get(kind))
    return v.format(bbn=bbn, L=L) if v else None


ld = lambda f: np.load(f, mmap_mode='r') if f and os.path.exists(f) else None
feat = [ld(path(s, 'feat')) for s in SENS]
img = [ld(path(s, 'img')) for s in SENS]
lay = ld(path('map', 'layers')) if CNN else None
assert all(f is not None or (CNN and s == 'map' and not a.zeroshot) for s, f in zip(SENS, feat)), 'missing feature cache, run the prep'
fwd, fok = ld(f'{C}/{pre}_fwd.npy'), ld(f'{C}/{pre}_fwd_ok.npy')
bwd, bok = [ld(f'{C}/{pre}_bwd_{s}.npy') for s in SENS], [ld(f'{C}/{pre}_bwd_ok_{s}.npy') for s in SENS]
SO = [] if a.zeroshot else [k for k, s in enumerate(SENS) if s in (cfg.get('struct_only') or [])]
SD = (cfg.get('struct_dim') or cfg['dim']) if SO else cfg['dim']
assert not SO or 0 < SD < cfg['dim'], 'struct_only needs 0 < struct_dim < dim'
WD = torch.ones(S, cfg['dim'], device=dev)
for k in SO:
    WD[k, SD:] = 0
SV = [k for k, s in enumerate(SENS) if s in (cfg.get('sig_valid') or [])]
assert all(PLANE[SENS[k]] == 'thr' for k in SV), 'sig_valid: only sensors on the thermal grid'
dims = lambda ka, kb: slice(0, SD) if ka in SO or kb in SO else slice(None)
ii, jj = np.meshgrid(np.arange(GH), np.arange(GW), indexing='ij')
POS = np.stack([ii.ravel(), jj.ravel()], 1)
UVn = np.stack([(jj.ravel() + .5) * p, (ii.ravel() + .5) * p], 1)
UV = torch.tensor(UVn, dtype=torch.float32, device=dev)
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)


def T(arr, idx, dt=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(arr[idx])).to(dev).to(dt)


def geo(idx):
    return T(fwd, idx)[:, SEL], T(fok, idx, torch.bool)[:, SEL].flatten(2)


class MapCNN(nn.Module):
    def __init__(s, cin, D):
        super().__init__()
        n, c, mods = int(round(math.log2(p // meta['map_down']))), cin, []
        for o in [64, 128, 256, 256][:n]:
            mods += [nn.Conv2d(c, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU()]
            c = o
        s.net = nn.Sequential(*mods, nn.Conv2d(c, 256, 3, 1, 1), nn.GroupNorm(8, 256), nn.GELU(), nn.Conv2d(256, D, 1))

    def forward(s, x):
        return s.net(x).permute(0, 2, 3, 1)


def enc(k, idx, hide=None):
    if CNN and SENS[k] == 'map':
        x = T(lay, idx) / 255
        if hide is not None:
            r = p // meta['map_down']
            m = hide.view(-1, GH, GW).repeat_interleave(r, 1).repeat_interleave(r, 2)
            x = x * (~m)[:, None].float()
        return mcnn(x)
    return T(feat[k], idx)


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


def sample(zk, m=None):
    x = zk.reshape(-1, zk.shape[-1])
    x = x if m is None else x[m.reshape(-1)]
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
            fa, fb = (F.normalize(Z[bi, k, sj][:, dims(ka, kb)].float(), dim=-1) for k in (ka, kb))
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
    if CNN and SENS[kc] == 'map':
        zc = ad(enc(kc, idx, hide), kc).flatten(1, 2)
    else:
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


D = next(f for f in feat if f is not None).shape[-1]
ad = Adapter(D, cfg, S).to(dev)
P = Pred().to(dev)
mcnn = MapCNN(len(meta['map_channels']), D).to(dev) if CNN else None
bb = load_backbone(cfg, bbn, dev) if cfg['w_hid'] > 0 and not a.zeroshot else None
opt = torch.optim.AdamW(list(ad.parameters()) + list(P.parameters()) + (list(mcnn.parameters()) if CNN else []),
                        lr=cfg['lr'], weight_decay=cfg['wd'])
steps = 0 if a.zeroshot else cfg['epochs'] * max(1, len(tr) // cfg['batch'])
warm = max(1, int(cfg['warmup'] * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

log = []
for st in range(steps):
    idx = np.sort(rng.choice(tr, min(cfg['batch'], len(tr)), replace=False))
    fw, OK = geo(idx)
    with torch.autocast(**amp):
        z = [ad(enc(k, idx), k) for k in range(S)]
        Z = to_ref(z, fw)
        M = OK[..., None] * WD[None, :, None]
        cd = M.sum(1)
        c = (Z * M).sum(1) / cd.clamp(min=1)
        cnt = OK.sum(1)
        parts = {}
        if cfg['w_inv'] > 0:
            m = M * (cd >= 2)[:, None]
            parts['inv'] = ((Z - c[:, None]) ** 2 * m).sum() / m.sum().clamp(min=1)
        if cfg['w_sig'] > 0:
            parts['sig'] = sum(sigreg(sample(z[k][..., :SD] if k in SO else z[k], OK[:, k] if k in SV else None), st)
                               for k in range(S)) / S
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
        return [(T(feat[k], idx) if a.zeroshot else ad(enc(k, idx), k)).float() for k in range(S)]


inner = (POS[:, 0] > 0) & (POS[:, 0] < GH - 1) & (POS[:, 1] > 0) & (POS[:, 1] < GW - 1)
pairs = [(ka, kb) for ka in range(S) for kb in range(S) if ka != kb]
keys = [f'{SENS[ka]}→{SENS[kb]}' for ka, kb in pairs]
rows = []
for s0 in range(0, len(te), 16):
    idx = te[s0:s0 + 16]
    z = zfun(idx)
    with torch.no_grad():
        Z = to_ref(z, T(fwd, idx)[:, SEL])
    OKr = fok[idx][:, SEL].reshape(len(idx), S, -1) & inner
    for b, i in enumerate(idx):
        hit, rank, ch = {}, {}, {}
        for k, (ka, kb) in zip(keys, pairs):
            v = np.flatnonzero(OKr[b, ka] & OKr[b, kb])
            if len(v) < cfg['min_valid']:
                continue
            near = torch.as_tensor(np.abs(POS[v][:, None] - POS[v][None]).max(-1) <= 1, device=dev)
            vt, ar = torch.as_tensor(v, device=dev), torch.arange(len(v), device=dev)
            Sm = F.normalize(Z[b, ka][vt][:, dims(ka, kb)], dim=-1) @ F.normalize(Z[b, kb][vt][:, dims(ka, kb)], dim=-1).T
            hit[k] = near[ar, Sm.argmax(1)].float().mean().item()
            rank[k] = ((Sm > Sm.diagonal()[:, None]).sum(1) + 1).float().median().item()
            ch[k] = near.float().mean().item()
        if hit:
            rows.append(dict(cond=frames[i]['cond'], chance=ch, hit=hit, rank=rank))
    if s0 == 0:
        for b, i in enumerate(idx[:cfg['n_fig']]):
            X = torch.cat([zk[b][..., :SD].reshape(-1, SD) if SO else zk[b].reshape(-1, zk.shape[-1]) for zk in z])
            X = X - X.mean(0)
            _, _, V = torch.pca_lowrank(X, q=3, center=False)
            Pc = X @ V
            lo, hi = torch.quantile(Pc, 0.01, 0), torch.quantile(Pc, 0.99, 0)
            Pc = ((Pc - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()
            tiles, o = [], 0
            for k, zk in enumerate(z):
                gh, gw = zk.shape[1:3]
                im = np.ascontiguousarray(img[k][i]) if img[k] is not None else np.zeros((gh * p, gw * p, 3), np.uint8)
                col = cv2.resize((Pc[o:o + gh * gw].reshape(gh, gw, 3) * 255).astype(np.uint8), (im.shape[1], im.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
                tiles.append(np.concatenate([im, col], 1))
                o += gh * gw
            Wm = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 0), (0, Wm - t.shape[1]), (0, 0))) for t in tiles]
            cv2.imwrite(f'{O}/figs/{name}_{i}.png', cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))

avg = lambda x: round(float(np.mean(x)), 4) if len(x) else None
out = dict(name=name, step='zero-shot' if a.zeroshot else 'trained', backbone=bbn, layers=L, sensors=SENS, n_frames=len(rows),
           chance_hit1=avg([v for r in rows for v in r['chance'].values()]),
           weights={k: cfg[k] for k in ('w_inv', 'w_sig', 'w_hid', 'w_nce', 'w_var')}, hid_target=cfg['hid_target'],
           hid_detach=cfg['hid_detach'], struct_only=[SENS[k] for k in SO], struct_dim=SD, map_encoder=cfg.get('map_encoder'))
for cond in ['all', *sorted({r['cond'] for r in rows})]:
    sub = [r for r in rows if cond == 'all' or r['cond'] == cond]
    h = {k: avg([r['hit'][k] for r in sub if k in r['hit']]) for k in keys}
    out[cond] = dict(n=len(sub), hit1_mean=avg([v for v in h.values() if v is not None]), hit1=h,
                     medrank={k: avg([r['rank'][k] for r in sub if k in r['rank']]) for k in keys},
                     chance={k: avg([r['chance'][k] for r in sub if k in r['chance']]) for k in keys})
json.dump(out, open(f'{O}/{name}.json', 'w'), indent=1, ensure_ascii=False)
if not a.zeroshot:
    torch.save(dict(adapter=ad.state_dict(), predictor=P.state_dict(), map_cnn=mcnn.state_dict() if CNN else None, cfg=cfg,
                    sensors=SENS), f'{O}/{name}.pt')
print(json.dumps({k: out['all'][k] for k in ('hit1_mean', 'hit1')}, ensure_ascii=False))
