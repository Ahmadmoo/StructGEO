"""Phase 2 helpers: camera footprint geometry, map windows, BEV templates, FFT matching, heads, metrics."""
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import RGB, TH

CH = ['major_road', 'minor_road', 'building', 'building_edge', 'water']
PALETTE = np.array([[230, 70, 50], [235, 235, 235], [240, 165, 70], [110, 55, 20], [70, 130, 230]], np.float32)
# query kind -> (frame list it uses, adapter branch). 'sat' = satellite crop rendered at a thermal frame's footprint,
# 'vis2th' = the drone RGB frame taken with a thermal frame, resampled into that thermal frame's footprint (p2_pair.py).
KIND = {'thermal': ('thermal', TH), 'sat': ('thermal', RGB), 'visible': ('visible', RGB), 'vis2th': ('thermal', RGB)}


def load_frames(cfg):
    return json.load(open(f"{cfg['p2_cache']}/frames.json"))


def load_osm(cfg):
    """The map raster named by map_name (default osm: OSM roads, buildings, water; other names add DSM buildings or
    land cover, see p2_osm.py and p2_landcover.py)."""
    P, name = cfg['p2_cache'], cfg.get('map_name') or 'osm'
    return np.load(f'{P}/{name}.npy', mmap_mode='r'), json.load(open(f'{P}/{name}_meta.json'))


def cache_key(frames, kind, cfg=None):
    """Identifies the frame list (and, for satellite crops, the footprints) a cached image/feature file was made from."""
    import hashlib
    fl = frames[KIND[kind][0]]
    rows = [[f['path']] + ([round(f['E'], 2), round(f['N'], 2), round(f['yaw'], 3), round(f['gsd'], 5)] if kind == 'sat' else [])
            + ([f.get('vis')] if kind == 'vis2th' else []) for f in fl]
    if kind == 'vis2th':
        rows.append((cfg or {}).get('vis2th_splits'))
    return hashlib.md5(json.dumps(rows).encode()).hexdigest()[:12]


def work_size(cfg, mod):
    return tuple(cfg['thermal_size']) if mod == 'thermal' else tuple(cfg['visible_size'])


def match_geo(cfg, mod):
    """Sizes of the matcher, in cells. T: template side, S: search range (shifts 0..S), N = T + S: map window side."""
    cell, res = cfg['cell_m'], cfg['osm_res']
    assert abs(cell / res - 2) < 1e-6, 'the map encoder works at stride 2: set cell_m = 2 * osm_res'
    T = int(round(cfg['tmpl_m' if mod == 'thermal' else 'tmpl_m_visible'] / cell))
    S = int(round(2 * cfg['search_m'] / cell))
    K = int(round(cfg['yaw_range'] / cfg['yaw_step']))
    return dict(cell=cell, res=res, T=T, S=S, N=T + S, npx=2 * (T + S), K=K, step=cfg['yaw_step'],
                offs=(np.arange(2 * K + 1) - K) * cfg['yaw_step'])


# ---------------------------------------------------------------- footprint geometry (flat ground, nadir view)
# Image up = heading. yaw = degrees clockwise from north. right = (cos, -sin), forward = (sin, cos) in (E, N).

def axes(yaw):
    t = np.radians(yaw)
    return np.array([np.cos(t), -np.sin(t)]), np.array([np.sin(t), np.cos(t)])


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def img2world(f, wh, yaw=None, scale=1.0, dE=0.0, dN=0.0):
    """3x3 affine: pixel (u, v) of the working-size image (continuous, origin at the top-left corner) -> (E, N)."""
    W, H = wh
    r, fw = axes(f['yaw'] if yaw is None else yaw)
    g = f['gsd'] * scale
    A = np.eye(3)
    A[:2, 0], A[:2, 1] = g * r, -g * fw
    A[:2, 2] = np.array([f['E'] + dE, f['N'] + dN]) - g * W / 2 * r + g * H / 2 * fw
    return A


def world2pix(tr):
    return np.linalg.inv(np.array([[tr.a, tr.b, tr.c], [tr.d, tr.e, tr.f], [0, 0, 1.0]]))


def render_sat(src, f, wh, **kw):
    """Satellite (rasterio dataset) resampled into the camera frame of frame f."""
    from rasterio.windows import Window
    W, H = wh
    P = world2pix(src.transform) @ img2world(f, wh, **kw)
    c = P @ np.array([[0, W, 0, W], [0, 0, H, H], [1, 1, 1, 1]], float)
    c0, r0 = np.floor(c[:2].min(1)).astype(int) - 2
    c1, r1 = np.ceil(c[:2].max(1)).astype(int) + 2
    im = src.read([1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True, fill_value=0).transpose(1, 2, 0)
    if im.dtype != np.uint8:
        hi = np.percentile(im[im > 0], 99.5) if (im > 0).any() else 1
        im = np.clip(im.astype(np.float32) / hi * 255, 0, 255).astype(np.uint8)
    P = np.array([[1, 0, -c0], [0, 1, -r0], [0, 0, 1.0]]) @ P
    k = int(np.hypot(P[0, 0], P[1, 0]))  # source pixels per output pixel
    if k >= 2:
        h0, w0 = im.shape[:2]
        im = cv2.resize(im, (max(1, w0 // k), max(1, h0 // k)), interpolation=cv2.INTER_AREA)
        P = np.diag([im.shape[1] / w0, im.shape[0] / h0, 1.0]) @ P
    h = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1.0]])  # pixel index <-> continuous coordinate
    P = np.linalg.inv(h) @ P @ h
    return cv2.warpAffine(np.ascontiguousarray(im), P[:2], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def warp_down(im, P, wh):
    """Resample image im into a W x H output; P maps output pixel coordinates (continuous) to im pixel coordinates.
    Pre-shrinks im by an integer factor first when it is much finer than the output (avoids aliasing)."""
    W, H = wh
    k = int(np.hypot(P[0, 0], P[1, 0]))
    if k >= 2:
        h0, w0 = im.shape[:2]
        im = cv2.resize(im, (max(1, w0 // k), max(1, h0 // k)), interpolation=cv2.INTER_AREA)
        P = np.diag([im.shape[1] / w0, im.shape[0] / h0, 1.0]) @ P
    h = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1.0]])
    P = np.linalg.inv(h) @ P @ h
    return cv2.warpAffine(np.ascontiguousarray(im), P[:2], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def vis_map(f, wh, vis_wh, src_wh, shift):
    """3x3: thermal-frame pixel (u, v) of frame f -> pixel of its paired RGB file (size src_wh), from both metadata poses;
    the scene point at thermal (u + dx, v + dy) is taken, shift = (dx, dy)."""
    v = f['vis']
    fv = dict(E=v['E'], N=v['N'], yaw=v['yaw'], gsd=v['gsd'] * vis_wh[0] / src_wh[0])  # ground size of one file pixel
    return np.linalg.inv(img2world(fv, src_wh)) @ img2world(f, wh) @ np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def render_vis(f, wh, root, vis_wh):
    """The drone RGB frame paired with thermal frame f (f['vis'], written by p2_pair.py) resampled into f's footprint,
    with the RGB-to-thermal pixel offset p2_pair.py found (vis dx, dy). Zeros if f has no pair."""
    W, H = wh
    v = f.get('vis')
    if not v:
        return np.zeros((H, W, 3), np.uint8)
    im = read_rgb(os.path.join(root, v['path']))
    return warp_down(im, vis_map(f, wh, vis_wh, im.shape[1::-1], (v.get('dx', 0.0), v.get('dy', 0.0))), wh)


def load_gray(path, wh):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    g = cv2.resize(g.astype(np.float32), wh, interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def load_rgb(path, wh):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), wh, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- map windows and batches

def crop(osm, om, E, N, n):
    """n x n pixels of the OSM raster around the pixel corner nearest (E, N). Returns crop and its exact center."""
    res = om['res']
    cc, rc = int(round((E - om['x0']) / res)), int(round((om['y0'] - N) / res))
    c0, r0 = cc - n // 2, rc - n // 2
    out = np.zeros((osm.shape[0], n, n), np.uint8)
    a0, a1, b0, b1 = max(r0, 0), min(r0 + n, osm.shape[1]), max(c0, 0), min(c0 + n, osm.shape[2])
    if a1 > a0 and b1 > b0:
        out[:, a0 - r0:a1 - r0, b0 - c0:b1 - c0] = osm[:, a0:a1, b0:b1]
    return out, om['x0'] + cc * res, om['y0'] - rc * res


def batch(fb, osm, om, g, rng=None, cfg=None):
    """Map windows centered on a position prior. Training (rng given) draws fresh priors; evaluation uses stored ones."""
    n = len(fb)
    if rng is not None:
        pr = np.c_[rng.uniform(-1, 1, (n, 2)) * cfg['prior_m'], rng.uniform(-1, 1, n) * cfg['yaw_noise']]
    else:
        pr = np.array([f['prior'] for f in fb], float)
    maps, rel = [], []
    for f, p in zip(fb, pr):
        m, Ew, Nw = crop(osm, om, f['E'] + p[0], f['N'] + p[1], g['npx'])
        maps.append(m)
        rel.append([f['E'] - Ew, f['N'] - Nw])
    rel = np.array(rel)
    S, K, cell = g['S'], g['K'], g['cell']
    gt = np.c_[np.clip(np.round(-pr[:, 2] / g['step']) + K, 0, 2 * K),
               np.clip(np.round(-rel[:, 1] / cell + S / 2), 0, S),
               np.clip(np.round(rel[:, 0] / cell + S / 2), 0, S)].astype(int)
    yaw = np.array([f['yaw'] for f in fb], float)
    return dict(map=np.stack(maps), rel=rel, yaw=yaw, yaw_prior=yaw + pr[:, 2],
                gsd=np.array([f['gsd'] for f in fb], float), gt=gt, prior=pr)


# ---------------------------------------------------------------- torch geometry

def cam_grid(dE, dN, yaw, gsd, wh, hw, ext):
    """grid_sample grid over a north-up map window (side ext m, centered at 0) for an h x w grid of image points.
    dE, dN: image-center ground point relative to the window center; yaw (deg), gsd (m/px at working size)."""
    W, H = wh
    h, w = hw
    dev = dE.device
    u = (torch.arange(w, device=dev) + 0.5) * W / w - W / 2
    v = H / 2 - (torch.arange(h, device=dev) + 0.5) * H / h
    du, dv = u[None, None, :] * gsd[:, None, None], v[None, :, None] * gsd[:, None, None]
    t = torch.deg2rad(yaw)[:, None, None]
    e = dE[:, None, None] + du * t.cos() + dv * t.sin()
    n = dN[:, None, None] - du * t.sin() + dv * t.cos()
    return torch.stack([e, -n], -1) / (ext / 2)


def tmpl_grid(gsd, yaw, wh, T, cell):
    """Grid over the image for a north-up T x T template centered on the image-center ground point.
    gsd (B,), yaw (B, A) degrees -> (B, A, T, T, 2)."""
    W, H = wh
    c = (torch.arange(T, device=gsd.device, dtype=torch.float32) + 0.5 - T / 2) * cell
    e, n = c[None, :].expand(T, T), -c[:, None].expand(T, T)
    t = torch.deg2rad(yaw)[..., None, None]
    r, fw = e * t.cos() - n * t.sin(), e * t.sin() + n * t.cos()
    g = gsd.view(-1, 1, 1, 1)
    return torch.stack([2 * r / (g * W), -2 * fw / (g * H)], -1)


def templates(feat, grid):
    """feat (B, C, h, w) in the image frame -> rotated north-up templates (B, A, C, T, T) and valid cell counts (B, A)."""
    B, A, T = grid.shape[:3]
    x = feat.float()[:, None].expand(-1, A, -1, -1, -1).flatten(0, 1)
    g = grid.flatten(0, 1)
    v = (g.abs() <= 1).all(-1)[:, None].float()
    t = F.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False) * v
    return t.view(B, A, -1, T, T), v.view(B, A, -1).sum(-1).clamp(min=1)


def correlate(mapf, tm, nv):
    """Exhaustive correlation by FFT. mapf (B, C, N, N), tm (B, A, C, T, T) -> scores (B, A, N-T+1, N-T+1).
    Shift (sy, sx) puts the template center at window cell (sy + T/2, sx + T/2)."""
    N, T = mapf.shape[-1], tm.shape[-1]
    Mf = torch.fft.rfft2(mapf.float())
    Tf = torch.fft.rfft2(tm.float(), s=(N, N))
    s = torch.fft.irfft2(torch.einsum('bcxy,bacxy->baxy', Mf, Tf.conj()), s=(N, N))
    return s[..., :N - T + 1, :N - T + 1] / nv[..., None, None]


def decode(logits, g, r_conf):
    """Joint argmax over (yaw bin, y, x). Returns (dE, dN) of the image center relative to the window center,
    yaw offset relative to the prior, confidence (probability mass within r_conf m of the peak) and log-probs."""
    B, A, S1, _ = logits.shape
    lp = logits.float().flatten(1).log_softmax(1).view(B, A, S1, S1)
    k = lp.flatten(1).argmax(1)
    a, sy, sx = k // (S1 * S1), (k // S1) % S1, k % S1
    pxy = lp.exp().sum(1)
    i = torch.arange(S1, device=logits.device)
    d2 = (i[None, :, None] - sy[:, None, None]) ** 2 + (i[None, None, :] - sx[:, None, None]) ** 2
    conf = (pxy * (d2 * g['cell'] ** 2 <= r_conf ** 2)).sum((1, 2))
    pos = torch.stack([(sx - g['S'] / 2) * g['cell'], -(sy - g['S'] / 2) * g['cell']], 1)
    return pos, (a - g['K']).float() * g['step'], conf, lp


# ---------------------------------------------------------------- trained parts

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class MapEncoder(nn.Module):
    """Small UNet on the rasterized map (1 m) -> matching features at 2 m."""

    def __init__(s, cfg):
        super().__init__()
        c = cfg['map_width']
        s.e0, s.e1, s.e2, s.e3 = block(cfg.get('map_ch', len(CH)), c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 4 * c)
        s.d2, s.d1 = block(8 * c, 4 * c), block(6 * c, 2 * c)
        s.out = nn.Conv2d(2 * c, cfg['match_dim'], 1)

    def forward(s, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = s.e1(F.max_pool2d(s.e0(x), 2))
        x2 = s.e2(F.max_pool2d(x1, 2))
        x3 = s.e3(F.max_pool2d(x2, 2))
        y2 = s.d2(torch.cat([x2, up(x3)], 1))
        return s.out(s.d1(torch.cat([x1, up(y2)], 1)))


class BEVHead(nn.Module):
    """On the adapter token grid, in the image frame: matching features x per-cell weight, and layout logits."""

    def __init__(s, cfg):
        super().__init__()
        d, w = cfg['dim'], cfg['bev_width']
        s.net = nn.Sequential(nn.Conv2d(d, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU(),
                              nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU())
        s.feat, s.conf = nn.Conv2d(w, cfg['match_dim'], 1), nn.Conv2d(w, 1, 1)
        s.layout = nn.Conv2d(w, cfg.get('map_ch', len(CH)), 1)

    def forward(s, z):
        x = s.net(z.permute(0, 3, 1, 2))
        return s.feat(x) * torch.sigmoid(s.conf(x)), s.layout(x)


def layout_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits.float(), target)
    p = torch.sigmoid(logits.float())
    dice = 1 - (2 * (p * target).sum((2, 3)) + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------- metrics and pictures

def edges(im):
    g = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def ncc(a, b, m):
    if m.sum() < 100:
        return float('nan')
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = np.argsort(np.argsort(score, kind='stable'), kind='stable') + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pick_tau(rec, ok_m, target):
    """Lowest confidence threshold whose accepted set still has precision >= target (validation)."""
    e, c = np.array([r['err'] for r in rec]), np.array([r['conf'] for r in rec])
    o = np.argsort(-c)
    prec = np.cumsum(e[o] <= ok_m) / np.arange(1, len(o) + 1)
    k = np.flatnonzero(prec >= target)
    return float(c[o][k.max()]) if len(k) else float(c.max() + 1e-6)


def metrics(rec, tau, ok_m):
    if not rec:
        return {}
    e, c = np.array([r['err'] for r in rec]), np.array([r['conf'] for r in rec])
    y = np.array([r['yaw_err'] for r in rec])
    acc, good = c >= tau, e <= ok_m
    r4 = lambda x: round(float(x), 4)
    au = auroc(c, good)
    out = dict(n=len(e), median_m=r4(np.median(e)), mean_m=r4(e.mean()),
               **{f'r@{t}m': r4((e <= t).mean()) for t in (5, 10, 25, 50)},
               yaw_median_deg=r4(np.median(y)), auroc=None if np.isnan(au) else r4(au), tau=r4(tau), coverage=r4(acc.mean()),
               precision=r4(good[acc].mean()) if acc.any() else None,
               median_accepted_m=r4(np.median(e[acc])) if acc.any() else None)
    return out


def colorize(x):
    x = np.asarray(x, np.float32)
    x = x / 255 if x.max() > 1 else x
    out = np.full((*x.shape[1:], 3), 25, np.float32)
    for i in (4, 2, 3, 1, 0):
        a = np.clip(x[i], 0, 1)[..., None]
        out = out * (1 - a) + PALETTE[i] * a
    return out.astype(np.uint8)


def footprint_px(rel, yaw, gsd, wh, npx, res):
    """Image corners at a pose -> pixel coordinates in the map window."""
    W, H = wh
    r, fw = axes(yaw)
    pts = []
    for u, v in ((0, 0), (W, 0), (W, H), (0, H)):
        en = np.asarray(rel) + gsd * ((u - W / 2) * r + (H / 2 - v) * fw)
        pts.append([npx / 2 + en[0] / res, npx / 2 - en[1] / res])
    return np.array(pts)


def figure(path, qimg, mcrop, pxy, rel, pos, yaw, gsd, wh, g, text):
    m = colorize(mcrop).copy()
    npx, k = m.shape[0], g['cell'] / g['res']
    side = int(round(g['S'] * k))
    hm = cv2.resize(pxy / max(pxy.max(), 1e-12), (side + 1, side + 1), interpolation=cv2.INTER_LINEAR)
    a0 = npx // 2 - side // 2
    heat = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32)
    sub = m[a0:a0 + side + 1, a0:a0 + side + 1].astype(np.float32)
    al = np.clip(hm, 0, 1)[..., None] * 0.8
    m[a0:a0 + side + 1, a0:a0 + side + 1] = (sub * (1 - al) + heat * al).astype(np.uint8)
    cv2.rectangle(m, (a0, a0), (a0 + side, a0 + side), (255, 255, 255), 1)
    cv2.polylines(m, [np.round(footprint_px(rel, yaw, gsd, wh, npx, g['res'])).astype(np.int32)], True, (60, 220, 60), 1)
    p = lambda en: (int(round(npx / 2 + en[0] / g['res'])), int(round(npx / 2 - en[1] / g['res'])))
    cv2.drawMarker(m, (npx // 2, npx // 2), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    cv2.circle(m, p(rel), 6, (60, 220, 60), 2)
    cv2.drawMarker(m, p(pos), (255, 40, 40), cv2.MARKER_TILTED_CROSS, 14, 2)
    q = cv2.resize(np.ascontiguousarray(qimg), (round(qimg.shape[1] * npx / qimg.shape[0]), npx))
    out = np.concatenate([q, m], 1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
