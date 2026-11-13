"""Phase 2: place a frame on the OSM raster.
Frozen backbone features -> adapter (from Phase 1) -> BEV head -> rotated north-up templates, correlated by FFT with
map-encoder features over the search window and the yaw range -> probability over (yaw, y, x)."""
import argparse, copy, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from common import (RGB, TH, load_cfg, load_backbone, to_input, encode, geometry, sample_mask, Adapter, Predictor,
                    var_loss, nce_loss)
from p2_common import (KIND, cache_key, load_frames, load_osm, work_size, match_geo, batch, cam_grid, tmpl_grid, templates,
                       correlate, decode, MapEncoder, BEVHead, layout_loss, wrap, pick_tau, metrics, figure)

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
P, O, dev, L, name, bbn = cfg['p2_cache'], cfg['out_dir'], cfg['device'], cfg['layers'], cfg['name'], cfg['backbone']
os.makedirs(f'{O}/p2_figs', exist_ok=True)
torch.manual_seed(cfg['seed'])
rng = np.random.default_rng(cfg['seed'])
frames = load_frames(cfg)
osm, om = load_osm(cfg)
cfg['map_ch'] = len(om['channels'])  # saved with the weights, so the experiment scripts rebuild the same model
G = {m: match_geo(cfg, m) for m in frames}
paired = ('thermal', 'sat')
need = set(cfg['train_queries']) | set(cfg['test_queries']) | (set(paired) if cfg['w_nce'] > 0 or cfg['w_jepa'] > 0 else set())
for k in need:
    side = f'{P}/p2_img_{k}.key'
    assert os.path.exists(side) and open(side).read() == cache_key(frames, k, cfg), f'{k}: cache is stale or missing, run p2_cache.py'
feat = {k: np.load(f'{P}/p2_feat_{bbn}_L{L}_{k}.npy', mmap_mode='r') for k in need}
D = feat[cfg['train_queries'][0]].shape[-1]
amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)
f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)

ad, pr = Adapter(D, cfg).to(dev), Predictor(cfg).to(dev)
if cfg.get('p1_ckpt'):
    ck = torch.load(cfg['p1_ckpt'], map_location='cpu', weights_only=False)
    assert (ck['cfg']['backbone'], ck['cfg']['layers']) == (bbn, L), 'p1_ckpt must come from the same backbone and layers'
    ad.load_state_dict(ck['adapter'])
    pr.load_state_dict(ck['predictor'])
train_ad = cfg['adapter_lr'] > 0
assert train_ad or (cfg['w_nce'] == 0 and cfg['w_jepa'] == 0), 'NCE / JEPA need a trainable adapter (adapter_lr > 0)'
ad.requires_grad_(train_ad)
ema = copy.deepcopy(ad).requires_grad_(False)
bev, menc = BEVHead(cfg).to(dev), MapEncoder(cfg).to(dev)
log_scale = torch.nn.Parameter(torch.tensor(math.log(cfg['init_scale']), device=dev))
groups = [dict(params=list(bev.parameters()) + list(menc.parameters()) + [log_scale], lr=cfg['p2_lr'])]
if train_ad:
    groups.append(dict(params=list(ad.parameters()) + (list(pr.parameters()) if cfg['w_jepa'] > 0 else []),
                       lr=cfg['adapter_lr']))
opt = torch.optim.AdamW(groups, weight_decay=cfg['wd'])
tr = {k: [f['id'] for f in frames[KIND[k][0]] if f['split'] == 'train'] for k in cfg['train_queries']}
steps = cfg['p2_epochs'] * max(1, len(tr[cfg['train_queries'][0]]) // cfg['p2_batch'])
warm = max(1, int(cfg['warmup'] * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
if cfg['w_jepa'] > 0:
    bb = load_backbone(cfg, bbn, dev)
    img = {k: np.load(f'{P}/p2_img_{k}.npy', mmap_mode='r') for k in paired}
    G1 = geometry(cfg, None, 'same', dev)


def tokens(kind, idx):
    return torch.from_numpy(np.ascontiguousarray(feat[kind][idx])).to(dev).float()


def run(kind, idx, b, train):
    mod, br = KIND[kind]
    g, wh = G[mod], work_size(cfg, mod)
    with torch.autocast(**amp):
        z = ad(tokens(kind, idx), br)
        mf = menc(torch.from_numpy(b['map']).to(dev).float() / 255)
        bf, lay = bev(z)
    gsd = f32(b['gsd'])
    if train and cfg['scale_jitter'] > 0:
        gsd = gsd * (1 + (torch.rand(len(idx), device=dev) * 2 - 1) * cfg['scale_jitter'])
    grid = tmpl_grid(gsd, f32(b['yaw_prior'])[:, None] + f32(g['offs'])[None], wh, g['T'], g['cell'])
    tm, nv = templates(bf.float(), grid)
    return z, lay, correlate(mf.float(), tm, nv) * log_scale.exp()


def layout_target(kind, b, hw):
    mod = KIND[kind][0]
    g = G[mod]
    m = F.avg_pool2d(torch.from_numpy(b['map']).to(dev).float() / 255, 2)
    grid = cam_grid(f32(b['rel'][:, 0]), f32(b['rel'][:, 1]), f32(b['yaw']), f32(b['gsd']), work_size(cfg, mod), hw,
                    g['npx'] * g['res'])
    return F.grid_sample(m, grid, mode='bilinear', align_corners=False)


log, ep_len = [], max(1, steps // cfg['p2_epochs'])
for s in range(steps):
    kind = cfg['train_queries'][s % len(cfg['train_queries'])]
    mod = KIND[kind][0]
    idx = np.sort(rng.choice(tr[kind], min(cfg['p2_batch'], len(tr[kind])), replace=False))
    b = batch([frames[mod][i] for i in idx], osm, om, G[mod], rng, cfg)
    z, lay, logits = run(kind, idx, b, True)
    gt = torch.as_tensor(b['gt'], device=dev)
    lp = logits.flatten(1).log_softmax(1).view_as(logits)
    parts = {'pose': -lp[torch.arange(len(idx), device=dev), gt[:, 0], gt[:, 1], gt[:, 2]].mean()}
    if cfg['w_layout'] > 0:
        up = F.interpolate(lay.float(), scale_factor=cfg['lay_up'], mode='bilinear', align_corners=False)
        parts['layout'] = layout_loss(up, layout_target(kind, b, up.shape[-2:]))
    if train_ad:
        with torch.autocast(**amp):
            parts['var'] = var_loss(z)
            if cfg['w_nce'] > 0 and kind in paired:
                other = 'sat' if kind == 'thermal' else 'thermal'
                zo = ad(tokens(other, idx), KIND[other][1])
                parts['nce'] = nce_loss(*((z, zo) if kind == 'thermal' else (zo, z)), rng, cfg)
            if cfg['w_jepa'] > 0 and kind in paired:
                X = {RGB: to_input(img['sat'][idx], bb['norm'], dev), TH: to_input(img['thermal'][idx], bb['norm'], dev)}
                Fz = {RGB: tokens('sat', idx), TH: tokens('thermal', idx)}
                parts['jepa'] = 0
                for kc, kt in ((RGB, TH), (TH, RGB)):
                    tgt, keep = (torch.as_tensor(i, device=dev) for i in sample_mask(rng, cfg, G1, kc, kt))
                    with torch.no_grad():
                        ctx, _ = encode(bb, X[kc], keep=keep, n_last=L, ctx=True)
                        t = F.layer_norm(ema(Fz[kt], kt).flatten(1, 2)[:, tgt].float(), (cfg['dim'],))
                    pred = pr(ad(ctx.float(), kc), G1['uv_t'][kc][keep], G1['uv_t'][kt][tgt], kc, kt)
                    parts['jepa'] = parts['jepa'] + F.smooth_l1_loss(pred.float(), t) / 2
    loss = sum(cfg['w_' + k] * v for k, v in parts.items())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if cfg['w_jepa'] > 0:
        m = cfg['ema'][0] + (cfg['ema'][1] - cfg['ema'][0]) * s / steps
        with torch.no_grad():
            for pe, po in zip(ema.parameters(), ad.parameters()):
                pe.lerp_(po, 1 - m)
    log.append({k: torch.as_tensor(v).item() for k, v in parts.items()})
    if (s + 1) % ep_len == 0:
        keys = sorted({k for r in log for k in r})
        print(json.dumps(dict(epoch=(s + 1) // ep_len, scale=round(log_scale.exp().item(), 2),
                              **{k: round(float(np.mean([r[k] for r in log if k in r])), 4) for k in keys})), flush=True)
        log = []

for mdl in (ad, bev, menc):
    mdl.eval()
results, per_frame = {}, {}
for kind in cfg['test_queries']:
    mod = KIND[kind][0]
    g, wh = G[mod], work_size(cfg, mod)
    ip = f'{P}/p2_img_{kind}.npy'
    qimg = np.load(ip, mmap_mode='r') if os.path.exists(ip) else None
    rec = {}
    for split in ('valid', 'test'):
        ids = [f['id'] for f in frames[mod] if f['split'] == split and (kind != 'vis2th' or f.get('vis'))]
        rec[split] = []
        for s0 in range(0, len(ids), cfg['p2_batch']):
            idx = ids[s0:s0 + cfg['p2_batch']]
            fb = [frames[mod][i] for i in idx]
            b = batch(fb, osm, om, g)
            with torch.no_grad():
                logits = run(kind, idx, b, False)[2]
                pos, dyaw, conf, lp = decode(logits, g, cfg['conf_radius_m'])
            pos, dyaw, conf = pos.cpu().numpy(), dyaw.cpu().numpy(), conf.cpu().numpy()
            for j, f in enumerate(fb):
                r = dict(id=f['id'], scene=f['scene'], cond=f['cond'], err=float(np.hypot(*(pos[j] - b['rel'][j]))),
                         yaw_err=float(abs(wrap(b['yaw_prior'][j] + dyaw[j] - b['yaw'][j]))), conf=float(conf[j]),
                         prior_err=float(np.hypot(*b['rel'][j])))
                rec[split].append(r)
                if split == 'test' and qimg is not None and len(rec[split]) <= cfg['p2_fig']:
                    figure(f'{O}/p2_figs/{name}_{kind}_{f["id"]}.png', qimg[f['id']], b['map'][j],
                           lp[j].exp().sum(0).cpu().numpy(), b['rel'][j], pos[j], f['yaw'], f['gsd'], wh, g,
                           f"{kind}  error {r['err']:.1f} m  conf {r['conf']:.2f}")
    tau = pick_tau(rec['valid'], cfg['ok_m'], cfg['target_precision']) if rec['valid'] else 0.5
    te = rec['test']
    pe = np.array([r['prior_err'] for r in te])
    res = {'all': metrics(te, tau, cfg['ok_m']),
           **{c: metrics([r for r in te if r['cond'] == c], tau, cfg['ok_m']) for c in ('day', 'night')},
           'valid': metrics(rec['valid'], tau, cfg['ok_m']),
           'prior_only': dict(median_m=round(float(np.median(pe)), 2), mean_m=round(float(pe.mean()), 2))}
    results[kind] = res
    per_frame[kind] = te
    print(kind, json.dumps(res['all']), flush=True)

json.dump(dict(name=name, backbone=bbn, layers=L, train_queries=cfg['train_queries'], p1_ckpt=cfg.get('p1_ckpt'),
               adapter_lr=cfg['adapter_lr'], **{k: cfg[k] for k in ('w_pose', 'w_layout', 'w_nce', 'w_jepa')},
               results=results), open(f'{O}/{name}.json', 'w'), indent=1)
json.dump(per_frame, open(f'{O}/{name}_frames.json', 'w'))
torch.save(dict(adapter=ad.state_dict(), bev=bev.state_dict(), map_encoder=menc.state_dict(), predictor=pr.state_dict(),
                log_scale=log_scale.detach().cpu(), cfg=cfg), f'{O}/{name}.pt')
