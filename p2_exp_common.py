"""Shared code for the PoC experiments: load a trained Phase 2 run, run its matcher with any position / heading prior,
work with the probability maps it returns, and score how distinctive the map itself is (no sensor)."""
import os, re
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from common import Adapter, Predictor
from p2_common import (KIND, CH, cache_key, load_frames, load_osm, work_size, match_geo, batch, cam_grid, tmpl_grid,
                       templates, correlate, MapEncoder, BEVHead, metrics)

# settings that define a trained model and its map: always taken from the checkpoint, never from the current config
MODEL_KEYS = ('backbone', 'layers', 'patch', 'dim', 'adapter_hidden', 'pred_width', 'pred_depth', 'pred_heads',
              'map_width', 'bev_width', 'match_dim', 'cell_m', 'osm_res', 'tmpl_m', 'tmpl_m_visible', 'search_m',
              'yaw_range', 'yaw_step', 'thermal_size', 'visible_size')


def place(scene):
    """'city1_300_ortho_day' -> 'city', 'village2_300_ortho_day&fog' -> 'village'."""
    m = re.match(r'[A-Za-z]+', scene)
    return m.group(0).lower() if m else scene


def t32(x, dev):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


class Model:
    """A trained Phase 2 run (results/<run>.pt) with the frames, map and cached features it needs."""

    def __init__(s, cfg, ckpt):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        s.ck_cfg = kc = ck['cfg']
        s.name = kc.get('name') or os.path.splitext(os.path.basename(ckpt))[0]
        s.cfg = c = dict(cfg)
        c.update({k: kc[k] for k in MODEL_KEYS if k in kc})
        c['map_name'] = kc.get('map_name') or 'osm'         # runs from before map variants used osm
        c['map_ch'] = kc.get('map_ch', len(CH))
        s.dev = dev = c['device']
        s.frames = load_frames(c)
        s.osm, s.om = load_osm(c)
        assert len(s.om['channels']) == c['map_ch'], \
            f"{s.name}: map {c['map_name']} has {len(s.om['channels'])} channels, the run was trained with {c['map_ch']}"
        s.G = {m: match_geo(c, m) for m in s.frames}
        D = ck['adapter']['ln.0.weight'].shape[0]
        s.ad, s.pr = Adapter(D, c).to(dev).eval(), Predictor(c).to(dev).eval()
        s.ad.load_state_dict(ck['adapter'])
        s.has_pred = 'predictor' in ck
        if s.has_pred:
            s.pr.load_state_dict(ck['predictor'])
        s.bev, s.menc = BEVHead(c).to(dev).eval(), MapEncoder(c).to(dev).eval()
        s.bev.load_state_dict(ck['bev'])
        s.menc.load_state_dict(ck['map_encoder'])
        s.scale = float(torch.as_tensor(ck['log_scale']).exp())
        s.amp = dict(device_type=dev.split(':')[0], dtype=torch.bfloat16)
        s.feat = {}

    def features(s, kind):
        if kind not in s.feat:
            c, P = s.cfg, s.cfg['p2_cache']
            side = f'{P}/p2_img_{kind}.key'
            assert os.path.exists(side) and open(side).read() == cache_key(s.frames, kind, c), \
                f'{kind}: cache is stale or missing, run p2_cache.py --kinds {kind}'
            s.feat[kind] = np.load(f"{P}/p2_feat_{c['backbone']}_L{c['layers']}_{kind}.npy", mmap_mode='r')
        return s.feat[kind]

    def z(s, kind, idx):
        t = torch.from_numpy(np.ascontiguousarray(s.features(kind)[list(idx)])).to(s.dev).float()
        with torch.no_grad(), torch.autocast(**s.amp):
            return s.ad(t, KIND[kind][1]).float()

    def match(s, kind, fb, priors=None, z=None, mask=None):
        """Matcher scores for frames fb (frame dicts) of one query kind.
        priors: (n, 3) prior offsets from the truth [dE, dN, dyaw] (default: the stored evaluation priors).
        z: adapter tokens (n, h, w, dim) instead of the cached frame's; mask: (n, h, w), 1 = token seen.
        Returns logits (n, yaw bins, S+1, S+1) and the batch dict (rel = truth - window center, ...)."""
        mod = KIND[kind][0]
        g, wh, dev = s.G[mod], work_size(s.cfg, mod), s.dev
        if priors is not None:
            fb = [dict(f, prior=[float(x) for x in p]) for f, p in zip(fb, priors)]
        b = batch(fb, s.osm, s.om, g)
        if z is None:
            z = s.z(kind, [f['id'] for f in fb])
        with torch.no_grad():
            with torch.autocast(**s.amp):
                mf = s.menc(torch.from_numpy(b['map']).to(dev).float() / 255)
                bf, _ = s.bev(z)
            bf = bf.float()
            grid = tmpl_grid(t32(b['gsd'], dev), t32(b['yaw_prior'], dev)[:, None] + t32(g['offs'], dev)[None], wh,
                             g['T'], g['cell'])
            if mask is None:
                tm, nv = templates(bf, grid)
            else:
                m = mask.to(dev).float()[:, None]
                tm, _ = templates(bf * m, grid)
                vis, _ = templates(m, grid)
                nv = vis.sum((2, 3, 4)).clamp(min=1)
            logits = correlate(mf.float(), tm, nv) * s.scale
        return logits, b


# ---------------------------------------------------------------- probability maps

def log_post(logits):
    return logits.float().flatten(1).log_softmax(1).view_as(logits)


def offsets(sy, sx, g):
    """Cell indices -> (dE, dN) in m of the image center relative to the map-window center."""
    sy, sx = np.asarray(sy, float), np.asarray(sx, float)
    return np.stack([(sx - g['S'] / 2) * g['cell'], -(sy - g['S'] / 2) * g['cell']], -1)


def peak_stats(p, g, r_conf, alphas, center=None):
    """p (n, S+1, S+1) position probabilities; center (sy, sx) cell tensors, default the peak of p. Returns the center,
    confidence (mass within r_conf of it) and, per alpha, the radius around it that holds 1 - alpha of the mass (plus
    half a cell diagonal, the most a point can be from the center of its cell): the protection level before calibration."""
    n, S1, _ = p.shape
    if center is None:
        k = p.flatten(1).argmax(1)
        sy, sx = k // S1, k % S1
    else:
        sy, sx = center
    i = torch.arange(S1, device=p.device, dtype=torch.float32)
    d = torch.sqrt((i[None, :, None] - sy[:, None, None].float()) ** 2 +
                   (i[None, None, :] - sx[:, None, None].float()) ** 2).flatten(1) * g['cell']
    q = p.flatten(1)
    conf = (q * (d <= r_conf)).sum(1)
    ds, o = d.sort(1)
    cs = q.gather(1, o).cumsum(1)
    pl = {}
    for a in alphas:
        j = (cs < (1 - a) * cs[:, -1:]).sum(1).clamp(max=d.shape[1] - 1)
        pl[a] = (ds.gather(1, j[:, None])[:, 0] + g['cell'] / 2 ** 0.5).cpu().numpy()
    return sy.cpu().numpy(), sx.cpu().numpy(), conf.cpu().numpy(), pl


def shift_blur(p, dy, dx, sig):
    """p (n, S, S) probabilities on the cell grid, moved by (dy, dx) cells and blurred by a Gaussian of sig cells
    (all per map), by FFT on a zero-padded grid (no wrap-around). Mass moved past the edge is dropped."""
    n, S, _ = p.shape
    pad = S // 2 + 1
    q = F.pad(p, (pad, pad, pad, pad))
    f = torch.fft.fftfreq(q.shape[-1], device=p.device)
    fy, fx = f[None, :, None], f[None, None, :]
    sig, dy, dx = (v.float()[:, None, None] for v in (sig, dy, dx))
    H = torch.exp(-2 * np.pi ** 2 * sig ** 2 * (fy ** 2 + fx ** 2)) * torch.exp(-2j * np.pi * (fy * dy + fx * dx))
    out = torch.fft.ifft2(torch.fft.fft2(q) * H).real[:, pad:pad + S, pad:pad + S]
    return out.clamp(min=0)


def conformal_q(scores, alpha):
    """Split-conformal quantile: on exchangeable data, score <= q for a new frame with probability >= 1 - alpha."""
    s = np.sort(np.asarray(scores, float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[k]) if 0 <= k < len(s) else float('inf')


# ---------------------------------------------------------------- tracks (for sequences)

_DJI = re.compile(r'DJI_(\d{14})_\d+_[A-Za-z]+\.\w+$')


def shot_time(path):
    m = _DJI.search(os.path.basename(path))
    try:
        return datetime.strptime(m.group(1), '%Y%m%d%H%M%S').timestamp() if m else None
    except ValueError:
        return None


def tracks(fl, split, max_gap_m, max_dt_s):
    """Frames of one split as flight tracks: same scene, in file order (DJI names sort by time), cut where consecutive
    kept frames jump more than max_gap_m or (from the DJI time stamp in the file name) more than max_dt_s apart."""
    out, cur, prev = [], [], None
    for f in fl:
        if f['split'] != split:
            continue
        if prev is not None:
            t0, t1 = shot_time(prev['path']), shot_time(f['path'])
            if (f['scene'] != prev['scene'] or np.hypot(f['E'] - prev['E'], f['N'] - prev['N']) > max_gap_m
                    or (t0 is not None and t1 is not None and abs(t1 - t0) > max_dt_s)):
                out.append(cur)
                cur = []
        cur.append(f['id'])
        prev = f
    if cur:
        out.append(cur)
    return out


def window(tr, j, fl, kmax, min_gap):
    """Frames used when tr[j] is the newest frame, newest first: walk back along the track and keep a frame when it
    is at least min_gap m from the last kept one, up to kmax frames."""
    out = [tr[j]]
    last = np.array([fl[tr[j]]['E'], fl[tr[j]]['N']])
    for q in range(j - 1, -1, -1):
        if len(out) == kmax:
            break
        p = np.array([fl[tr[q]]['E'], fl[tr[q]]['N']])
        if np.hypot(*(p - last)) >= min_gap:
            out.append(tr[q])
            last = p
    return out


# ---------------------------------------------------------------- map-only localizability

def map_score(cfg, osm, om, pts, mod, menc=None, excl_m=20.0, beta=10.0, bs=16):
    """How distinctive the map is at each point, with no sensor: the map under the camera footprint (as a perfect
    sensor would see it) is matched against the map within +-search_m and the yaw bins, like the real matcher.
    pts: dicts with E, N, yaw, gsd. Returns arrays:
      margin  - (best score within one cell of the point - best score more than excl_m away) / channels;
                0 means another place looks the same, 2 is the maximum
      oconf   - probability within conf_radius_m of the point, scores x beta (the p2_check matcher)
      content - share of the footprint with any mapped feature
      zmargin - with menc (a run's map encoder): the margin on the learned map features / score spread."""
    dev = cfg['device']
    g, wh = match_geo(cfg, mod), work_size(cfg, mod)
    gh, gw = wh[1] // cfg['patch'], wh[0] // cfg['patch']
    S1 = g['S'] + 1
    i = torch.arange(S1, device=dev, dtype=torch.float32)
    out = {k: [] for k in ('margin', 'oconf', 'content') + (('zmargin',) if menc is not None else ())}
    for s0 in range(0, len(pts), bs):
        b = batch([dict(p, prior=[0.0, 0.0, 0.0]) for p in pts[s0:s0 + bs]], osm, om, g)
        m = torch.from_numpy(b['map']).to(dev).float() / 255
        m2 = F.avg_pool2d(m, 2)
        grid = cam_grid(t32(b['rel'][:, 0], dev), t32(b['rel'][:, 1], dev), t32(b['yaw'], dev), t32(b['gsd'], dev), wh,
                        (gh, gw), g['npx'] * g['res'])
        tg = tmpl_grid(t32(b['gsd'], dev), t32(b['yaw'], dev)[:, None] + t32(g['offs'], dev)[None], wh, g['T'], g['cell'])
        gy, gx = t32(b['gt'][:, 1], dev), t32(b['gt'][:, 2], dev)
        d = torch.sqrt((i[None, :, None] - gy[:, None, None]) ** 2 + (i[None, None, :] - gx[:, None, None]) ** 2) * g['cell']
        near, far = d <= 1.5 * g['cell'], d > excl_m

        def margin(sc):
            sm = sc.amax(1)
            return sm.masked_fill(~near, -1e9).flatten(1).amax(1), sm.masked_fill(~far, -1e9).flatten(1).amax(1), sm

        with torch.no_grad():
            q = F.grid_sample(m2, grid, mode='bilinear', align_corners=False)
            out['content'].append((q.amax(1) > 0.5).float().mean((1, 2)).cpu())
            tm, nv = templates(2 * q - 1, tg)
            sc = correlate(2 * m2 - 1, tm, nv)
            a, o, _ = margin(sc)
            out['margin'].append(((a - o) / m.shape[1]).cpu())
            p = (sc * beta).flatten(1).softmax(1).view_as(sc).sum(1)
            out['oconf'].append((p * (d <= cfg['conf_radius_m'])).sum((1, 2)).cpu())
            if menc is not None:
                with torch.autocast(device_type=dev.split(':')[0], dtype=torch.bfloat16):
                    mf = menc(m)
                mf = mf.float()
                tm, nv = templates(F.grid_sample(mf, grid, mode='bilinear', align_corners=False), tg)
                a, o, sm = margin(correlate(mf, tm, nv))
                out['zmargin'].append(((a - o) / sm.flatten(1).std(1).clamp(min=1e-6)).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# ---------------------------------------------------------------- summaries

def by_group(rec, tau, ok_m):
    """metrics() for all frames, per place (city, village, ...) and per condition (day, night)."""
    out = {'all': metrics(rec, tau, ok_m)}
    for key in ('place', 'cond'):
        for v in sorted({r[key] for r in rec}):
            out[v] = metrics([r for r in rec if r[key] == v], tau, ok_m)
    return out


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)
