import json, math, sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SAT = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))
RGB, TH = 0, 1


def load_cfg(*paths):
    cfg = {}
    for p in paths:
        cfg.update(yaml.safe_load(open(p)) or {})
    return cfg


def load_pairs(cfg):
    pairs = json.load(open(f"{cfg['cache_dir']}/pairs.json"))
    meta = json.load(open(f"{cfg['cache_dir']}/meta.json"))
    for p in pairs:
        p['test'] = p['scene'] in cfg['test_scenes']
    return pairs, meta


def homog(H, xy):
    p = np.c_[xy, np.ones(len(xy))] @ np.asarray(H).T
    return p[:, :2] / p[:, 2:]


def work_H(cfg, meta):
    tw, th = cfg['thermal_size']
    S = np.diag([tw / meta['th_size'][0], th / meta['th_size'][1], 1.0])
    return S @ np.asarray(meta['H'])


def geometry(cfg, meta, option, dev='cpu'):
    p = cfg['patch']
    tw, th = cfg['thermal_size']
    gw, gh = tw // p, th // p
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    uv_th = np.stack([(jj + .5) * p, (ii + .5) * p], -1).reshape(-1, 2)
    box_th = np.concatenate([uv_th - p / 2, uv_th + p / 2], 1)
    g = dict(gh=gh, gw=gw, grid=None, rgb_hw=(gh, gw))
    if option == 'same':
        uv_rgb, box_rgb = uv_th, box_th
    else:
        H = work_H(cfg, meta)
        rw, rh = cfg['rgb_native_size']
        s = np.array([meta['rgb_size'][0] / rw, meta['rgb_size'][1] / rh])
        jj, ii = np.meshgrid(np.arange(rw // p), np.arange(rh // p))
        c = np.stack([(jj + .5) * p, (ii + .5) * p], -1).reshape(-1, 2)
        uv_rgb = homog(H, c * s)
        cs = np.stack([homog(H, (c + [dx, dy]) * s) for dx in (-p / 2, p / 2) for dy in (-p / 2, p / 2)])
        box_rgb = np.concatenate([cs.min(0), cs.max(0)], 1)
        q = homog(np.linalg.inv(H), uv_th) / s
        grid = np.stack([q[:, 0] / rw * 2 - 1, q[:, 1] / rh * 2 - 1], -1).reshape(gh, gw, 2)
        g.update(grid=torch.tensor(grid, dtype=torch.float32, device=dev), rgb_hw=(rh // p, rw // p))
    g['uv'] = [uv_rgb, uv_th]
    g['box'] = [box_rgb, box_th]
    g['uv_t'] = [torch.tensor(u, dtype=torch.float32, device=dev) for u in g['uv']]
    return g


def rgb_at_th(zr, grid):
    if grid is None:
        return zr
    out = F.grid_sample(zr.permute(0, 3, 1, 2).float(), grid[None].expand(zr.shape[0], -1, -1, -1),
                        mode='bilinear', align_corners=False)
    return out.permute(0, 2, 3, 1)


def sample_mask(rng, cfg, g, kc, kt):
    W, H = cfg['thermal_size']
    m = cfg['margin_px']
    while True:
        b = []
        for _ in range(cfg['n_blocks']):
            s = rng.uniform(*cfg['block_scale']) * W * H
            a = math.exp(rng.uniform(*np.log(cfg['block_aspect'])))
            bw, bh = min(W, math.sqrt(s * a)), min(H, math.sqrt(s / a))
            x0, y0 = rng.uniform(0, W - bw), rng.uniform(0, H - bh)
            b.append((x0, y0, x0 + bw, y0 + bh))
        b = np.array(b)[None]
        u = g['uv'][kt]
        tgt = ((u[:, None, 0] >= b[..., 0]) & (u[:, None, 0] < b[..., 2]) &
               (u[:, None, 1] >= b[..., 1]) & (u[:, None, 1] < b[..., 3])).any(1)
        x = g['box'][kc]
        hid = ((x[:, None, 0] < b[..., 2] + m) & (x[:, None, 2] > b[..., 0] - m) &
               (x[:, None, 1] < b[..., 3] + m) & (x[:, None, 3] > b[..., 1] - m)).any(1)
        tgt, keep = np.flatnonzero(tgt), np.flatnonzero(~hid)
        if len(tgt) >= 4 and len(keep) >= 0.1 * len(x):
            return tgt, keep


def load_backbone(cfg, name, dev):
    if name == 'ijepa':
        sys.path.insert(0, cfg['ijepa_repo'])
        import src.models.vision_transformer as vt
        ck = torch.load(cfg['ijepa_ckpt'], map_location='cpu', weights_only=False)
        nets = []
        for k in ('target_encoder', 'encoder'):
            m = getattr(vt, cfg['ijepa_arch'])(patch_size=cfg['patch'], img_size=[cfg['ijepa_img']])
            m.load_state_dict({n.replace('module.', '', 1): v for n, v in ck[k].items()})
            nets.append(m.eval().requires_grad_(False).to(dev))
        return dict(kind='ijepa', full=nets[0], ctx=nets[1], norm=IMNET, patch=cfg['patch'])
    sys.path.insert(0, cfg['dinov3_repo'])
    from dinov3.hub import backbones as hb
    m = getattr(hb, cfg['dino_arch'])(pretrained=False, weights=hb.Weights.SAT493M if name == 'dino_sat' else hb.Weights.LVD1689M)
    m.load_state_dict(torch.load(cfg[name], map_location='cpu'))
    m = m.eval().requires_grad_(False).to(dev)
    return dict(kind='dino', full=m, ctx=m, norm=SAT if name == 'dino_sat' else IMNET, patch=cfg['patch'])


def to_input(img, norm, dev):
    x = torch.from_numpy(np.array(img)).to(dev).permute(0, 3, 1, 2).float() / 255
    mean, std = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in norm)
    return (x - mean) / std


def encode(bb, x, keep=None, n_last=1, ctx=False):
    m = bb['ctx'] if ctx else bb['full']
    outs = []
    if bb['kind'] == 'ijepa':
        t = m.patch_embed.proj(x)
        B, D, h, w = t.shape
        t = t.flatten(2).transpose(1, 2)
        n = int(m.pos_embed.shape[1] ** 0.5)
        pe = F.interpolate(m.pos_embed.float().reshape(1, n, n, D).permute(0, 3, 1, 2), size=(h, w),
                           mode='bicubic', align_corners=False)
        t = t + pe.flatten(2).transpose(1, 2).to(t.dtype)
        if keep is not None:
            t = t[:, keep]
        for i, blk in enumerate(m.blocks):
            t = blk(t)
            if i >= len(m.blocks) - n_last:
                outs.append(m.norm(t))
        return torch.stack(outs).mean(0), (h, w)
    B, h, w = x.shape[0], x.shape[2] // bb['patch'], x.shape[3] // bb['patch']
    masks = None
    if keep is not None:
        masks = torch.ones(B, h * w, dtype=torch.bool, device=x.device)
        masks[:, keep] = False
    t, (h, w) = m.prepare_tokens_with_masks(x, masks)
    for i, blk in enumerate(m.blocks):
        t = blk(t, m.rope_embed(H=h, W=w) if m.rope_embed is not None else None)
        if i >= len(m.blocks) - n_last:
            outs.append(m.norm(t[:, m.n_storage_tokens + 1:]))
    out = torch.stack(outs).mean(0)
    return (out if keep is None else out[:, keep]), (h, w)


class Adapter(nn.Module):
    def __init__(s, D, cfg):
        super().__init__()
        s.ln = nn.ModuleList([nn.LayerNorm(D), nn.LayerNorm(D)])
        s.mlp = nn.Sequential(nn.Linear(D, cfg['adapter_hidden']), nn.GELU(), nn.Linear(cfg['adapter_hidden'], cfg['dim']))

    def forward(s, x, k):
        return s.mlp(s.ln[k](x))


def posenc(uv, dim, patch):
    d = dim // 4
    w = 1.0 / 10000 ** (torch.arange(d, device=uv.device, dtype=torch.float32) / d)
    u, v = uv[:, :1] / patch * w, uv[:, 1:] / patch * w
    return torch.cat([u.sin(), u.cos(), v.sin(), v.cos()], 1)


class Predictor(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        w = cfg['pred_width']
        s.w, s.patch = w, cfg['patch']
        s.inp = nn.Linear(cfg['dim'], w)
        s.mask = nn.Parameter(torch.zeros(1, 1, w))
        s.sens = nn.Parameter(torch.zeros(2, w))
        layer = nn.TransformerEncoderLayer(w, cfg['pred_heads'], 4 * w, dropout=0.0, activation='gelu',
                                           batch_first=True, norm_first=True)
        s.tf = nn.TransformerEncoder(layer, cfg['pred_depth'], enable_nested_tensor=False)
        s.norm = nn.LayerNorm(w)
        s.out = nn.Linear(w, cfg['dim'])
        nn.init.normal_(s.mask, std=0.02)
        nn.init.normal_(s.sens, std=0.02)

    def forward(s, ctx, ctx_uv, tgt_uv, kc, kt):
        c = s.inp(ctx) + posenc(ctx_uv, s.w, s.patch) + s.sens[kc]
        q = s.mask.expand(ctx.shape[0], len(tgt_uv), -1) + posenc(tgt_uv, s.w, s.patch) + s.sens[kt]
        x = s.norm(s.tf(torch.cat([c, q], 1)))
        return s.out(x[:, c.shape[1]:])


def var_loss(z):
    z = z.reshape(-1, z.shape[-1]).float()
    return F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()


def interior(gh, gw):
    ii, jj = np.meshgrid(np.arange(1, gh - 1), np.arange(1, gw - 1), indexing='ij')
    pos = np.stack([ii.ravel(), jj.ravel()], 1)
    return pos, pos[:, 0] * gw + pos[:, 1]


def chance_hit1(gh, gw):
    pos, _ = interior(gh, gw)
    return round(float(((np.abs(pos[:, None] - pos[None]).max(-1) <= 1).sum(1) / len(pos)).mean()), 4)


def nce_loss(zt, zr, rng, cfg):
    B, gh, gw, _ = zt.shape
    pos, flat = interior(gh, gw)
    sel = rng.choice(len(pos), min(cfg['nce_anchors'], len(pos)), replace=False)
    p, f = pos[sel], torch.as_tensor(flat[sel], device=zt.device)
    a = F.normalize(zt.flatten(1, 2)[:, f].flatten(0, 1).float(), dim=-1)
    b = F.normalize(zr.flatten(1, 2)[:, f].flatten(0, 1).float(), dim=-1)
    near = (np.abs(p[:, None] - p[None]).max(-1) <= 1) & ~np.eye(len(p), dtype=bool)
    near = torch.kron(torch.eye(B, device=zt.device), torch.as_tensor(near, device=zt.device, dtype=torch.float32)).bool()
    logits = (a @ b.T / cfg['nce_tau']).masked_fill(near, float('-inf'))
    y = torch.arange(len(a), device=zt.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def retrieval(ft, fr):
    gh, gw = ft.shape[1:3]
    pos, flat = interior(gh, gw)
    near = torch.as_tensor(np.abs(pos[:, None] - pos[None]).max(-1) <= 1, device=ft.device)
    ar = torch.arange(len(flat), device=ft.device)
    out = []
    for a, b in zip(ft, fr):
        a = F.normalize(a.reshape(-1, a.shape[-1])[flat].float(), dim=-1)
        b = F.normalize(b.reshape(-1, b.shape[-1])[flat].float(), dim=-1)
        S = a @ b.T
        d = S.diagonal()
        out.append([near[ar, S.argmax(1)].float().mean().item(), near[ar, S.argmax(0)].float().mean().item(),
                    ((S > d[:, None]).sum(1) + 1).float().median().item(), ((S > d[None]).sum(0) + 1).float().median().item()])
    return np.array(out)


def summarize(res, pairs_sub, chance):
    keys = ['hit1_th2rgb', 'hit1_rgb2th', 'medrank_th2rgb', 'medrank_rgb2th']
    out = {'n_pairs': len(res), 'chance_hit1': chance, 'all': dict(zip(keys, res.mean(0).round(4).tolist()))}
    for cond in sorted({p['cond'] for p in pairs_sub}):
        m = np.array([p['cond'] == cond for p in pairs_sub])
        out[cond] = dict(zip(keys, res[m].mean(0).round(4).tolist()))
    return out


def pca_panel(ft, fr, img_th, img_rgb, path, h=256):
    X = torch.cat([ft.reshape(-1, ft.shape[-1]), fr.reshape(-1, fr.shape[-1])]).float()
    X = X - X.mean(0)
    _, _, V = torch.pca_lowrank(X, q=3, center=False)
    P = X @ V
    lo, hi = torch.quantile(P, 0.01, 0), torch.quantile(P, 0.99, 0)
    P = ((P - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()
    n = ft.shape[0] * ft.shape[1]
    tiles = []
    for im, q, s in ((img_rgb, P[n:], fr.shape[:2]), (img_th, P[:n], ft.shape[:2])):
        c = cv2.resize((q.reshape(*s, 3) * 255).astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        for t in (im, c):
            tiles.append(cv2.resize(t, (round(t.shape[1] * h / t.shape[0]), h)))
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
