import argparse, copy, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from common import (RGB, TH, load_cfg, load_pairs, geometry, rgb_at_th, sample_mask, load_backbone, to_input, encode,
                    Adapter, Predictor, var_loss, nce_loss, retrieval, summarize, chance_hit1)

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
C, O, dev, L, name = cfg['cache_dir'], cfg['out_dir'], cfg['device'], cfg['layers'], cfg['name']
bbn, option = cfg['backbone'], cfg['option']
os.makedirs(O, exist_ok=True)
torch.manual_seed(cfg['seed'])
rng = np.random.default_rng(cfg['seed'])

pairs, meta = load_pairs(cfg)
tr = [p['id'] for p in pairs if not p['test']]
te = [p['id'] for p in pairs if p['test']]
G = geometry(cfg, meta, option, dev)
img = [np.load(f'{C}/img_rgb_{option}.npy', mmap_mode='r'), np.load(f'{C}/img_th.npy', mmap_mode='r')]
feat = [np.load(f'{C}/feat_{bbn}_L{L}_rgb_{option}.npy', mmap_mode='r'), np.load(f'{C}/feat_{bbn}_L{L}_th.npy', mmap_mode='r')]
bb = load_backbone(cfg, bbn, dev) if cfg['w_jepa'] > 0 else None

ad = Adapter(feat[TH].shape[-1], cfg).to(dev)
ema = copy.deepcopy(ad).requires_grad_(False)
pr = Predictor(cfg).to(dev)
opt = torch.optim.AdamW(list(ad.parameters()) + list(pr.parameters()), lr=cfg['lr'], weight_decay=cfg['wd'])
steps = cfg['epochs'] * max(1, len(tr) // cfg['batch'])
warm = max(1, int(cfg['warmup'] * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)


def load(arr, idx):
    return torch.from_numpy(np.ascontiguousarray(arr[idx])).to(dev).float()


log = []
for s in range(steps):
    idx = np.sort(rng.choice(tr, min(cfg['batch'], len(tr)), replace=False))
    Fz = [load(f, idx) for f in feat]
    with torch.autocast(**amp):
        z = [ad(Fz[k], k) for k in (RGB, TH)]
        parts = {'var': var_loss(z[RGB]) + var_loss(z[TH])}
        if cfg['w_nce'] > 0:
            parts['nce'] = nce_loss(z[TH], rgb_at_th(z[RGB], G['grid']), rng, cfg)
        if cfg['w_jepa'] > 0:
            X = [to_input(im[idx], bb['norm'], dev) for im in img]
            parts['jepa'] = 0
            for kc, kt in ((RGB, TH), (TH, RGB)):
                tgt, keep = (torch.as_tensor(i, device=dev) for i in sample_mask(rng, cfg, G, kc, kt))
                with torch.no_grad():
                    ctx, _ = encode(bb, X[kc], keep=keep, n_last=L, ctx=True)
                    t = F.layer_norm(ema(Fz[kt], kt).flatten(1, 2)[:, tgt].float(), (cfg['dim'],))
                pred = pr(ad(ctx.float(), kc), G['uv_t'][kc][keep], G['uv_t'][kt][tgt], kc, kt)
                parts['jepa'] = parts['jepa'] + F.smooth_l1_loss(pred.float(), t) / 2
        loss = sum(cfg['w_' + k] * v for k, v in parts.items())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    m = cfg['ema'][0] + (cfg['ema'][1] - cfg['ema'][0]) * s / steps
    with torch.no_grad():
        for pe, po in zip(ema.parameters(), ad.parameters()):
            pe.lerp_(po, 1 - m)
    log.append({k: v.item() for k, v in parts.items()})
    if (s + 1) % max(1, steps // cfg['epochs']) == 0:
        ep = {k: round(float(np.mean([r[k] for r in log])), 4) for k in log[0]}
        print(json.dumps(dict(epoch=(s + 1) * cfg['epochs'] // steps, **ep)), flush=True)
        log = []

ad.eval()
res = []
for s in range(0, len(te), 16):
    idx = te[s:s + 16]
    with torch.no_grad(), torch.autocast(**amp):
        zt, zr = ad(load(feat[TH], idx), TH), ad(load(feat[RGB], idx), RGB)
    with torch.no_grad():
        res.append(retrieval(zt.float(), rgb_at_th(zr.float(), G['grid'])))
out = summarize(np.concatenate(res), [pairs[i] for i in te], chance_hit1(G['gh'], G['gw']))
json.dump(dict(name=name, step='trained', backbone=bbn, option=option, layers=L,
               w_jepa=cfg['w_jepa'], w_nce=cfg['w_nce'], **out), open(f'{O}/{name}.json', 'w'), indent=1)
torch.save(dict(adapter=ad.state_dict(), ema=ema.state_dict(), predictor=pr.state_dict(), cfg=cfg), f'{O}/{name}.pt')
