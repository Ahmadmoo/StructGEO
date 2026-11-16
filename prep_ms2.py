import argparse, json, os, warnings
import multiprocessing as mp
import cv2
import numpy as np
import torch
from common import load_cfg, load_backbone, encode, to_input

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
R, C, p, L, dev, bbn = cfg['ms2_root'], cfg['cache_dir'], cfg['patch'], cfg['layers'], cfg['device'], cfg['backbone']
SENS = cfg['sensors']
PLANE = {s: cfg['lidar_plane'] if s == 'lidar' else s for s in SENS}
PL = sorted({'thr', *PLANE.values()})
NEAR = cv2.INTER_NEAREST
os.makedirs(f'{C}/figs', exist_ok=True)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def path(seq, n, kind, q):
    if kind == 'img':
        return f'{R}/sync_data/{seq}/{q}/img_left/{n}.png'
    return f'{R}/proj_depth/{seq}/{q}/{kind}/{n}.png'


def needs(seq, n):
    out = [path(seq, n, 'depth_filtered', q) for q in PL] + [path(seq, n, 'img', s) for s in SENS if s != 'lidar']
    if 'lidar' in SENS:
        ld = cfg['lidar_depth']
        out += [path(seq, n, ld, cfg['lidar_plane']), path(seq, n, 'intensity' + ld[5:], cfg['lidar_plane'])]
    return out


frames = []
for split, f in cfg['splits'].items():
    for seq in [l.strip() for l in open(f'{R}/{f}') if l.strip()]:
        names = sorted(x[:-4] for x in os.listdir(f'{R}/sync_data/{seq}/thr/img_left') if x.endswith('.png'))
        for n in names[cfg['frame_step'] // 2::cfg['frame_step']]:
            if all(os.path.exists(x) for x in needs(seq, n)):
                frames.append(dict(id=len(frames), seq=seq, name=n, split=split, cond=split.replace('test_', '')))

rd = lambda f: cv2.imread(f, cv2.IMREAD_UNCHANGED)
f0 = frames[0]
cal = np.load(f'{R}/sync_data/{f0["seq"]}/calib.npy', allow_pickle=True).item()
K0 = {q: np.asarray(cal[f'K_{q}L'], float) for q in PL}
E = {'nir': np.eye(4)}
for q in ('rgb', 'thr'):
    E[q] = np.eye(4)
    E[q][:3, :3], E[q][:3, 3] = cal[f'R_nir2{q}'], np.ravel(cal[f'T_nir2{q}']) * 1e-3
sc, off, hw, gs = {}, {}, {}, {}
for q in PL:
    h, w = rd(path(f0['seq'], f0['name'], 'depth_filtered', q)).shape[:2]
    sc[q] = K0['thr'][0, 0] / K0[q][0, 0] if cfg['match_scale'] else 1.0
    wr, hr = round(w * sc[q]), round(h * sc[q])
    hw[q] = (hr // p * p, wr // p * p)
    off[q] = np.array([(wr - hw[q][1]) // 2, (hr - hw[q][0]) // 2])
    gs[q] = (hw[q][0] // p, hw[q][1] // p)
GH, GW = gs['thr']


def T(b, a):
    return E[b] @ np.linalg.inv(E[a])


def move(b, a, X):
    M = T(b, a)
    return X @ M[:3, :3].T + M[:3, 3]


def lift(K, uv, z):
    return np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T * z[:, None]


def project(K, X):
    x = X @ K.T
    return x[:, :2] / np.maximum(x[:, 2:], 1e-6)


def to_cont(q, u):
    return sc[q] * (u + 0.5) - off[q]


def to_orig(q, x):
    return (x + off[q]) / sc[q] - 0.5


def centers(q):
    jj, ii = np.meshgrid(np.arange(gs[q][1]), np.arange(gs[q][0]))
    return np.stack([(jj + .5) * p, (ii + .5) * p], -1).reshape(-1, 2)


def inside(q, xy):
    return (xy[:, 0] >= 0) & (xy[:, 0] < hw[q][1]) & (xy[:, 1] >= 0) & (xy[:, 1] < hw[q][0])


def fit(q, im, interp=cv2.INTER_AREA):
    if sc[q] != 1:
        im = cv2.resize(im, (round(im.shape[1] * sc[q]), round(im.shape[0] * sc[q])), interpolation=interp)
    return im[off[q][1]:off[q][1] + hw[q][0], off[q][0]:off[q][0] + hw[q][1]]


def stretch(g, m=None):
    v = g[m] if m is not None else g
    if v.size == 0:
        return np.zeros(g.shape, np.uint8)
    lo, hi = np.percentile(v, (1, 99))
    return np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def fill(x, it):
    for _ in range(it):
        x = np.where(x > 0, x, cv2.dilate(x, np.ones((7, 3), np.uint8)))
    return x


def tok_depth(q, d):
    t = fit(q, d, NEAR).reshape(gs[q][0], p, gs[q][1], p).transpose(0, 2, 1, 3).reshape(-1, p * p)
    t = np.where(t > 0, t, np.nan)
    lo, med, hi = np.nanpercentile(t, (10, 50, 90), axis=1)
    ok = (np.isfinite(t).sum(1) >= cfg['depth_min_px']) & (hi - lo <= cfg['depth_max_spread'] * med)
    return np.nan_to_num(med), ok


def depth_at(d, uv, r):
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    u = np.clip(np.round(uv[:, :1]) + ox.ravel(), 0, d.shape[1] - 1).astype(int)
    v = np.clip(np.round(uv[:, 1:]) + oy.ravel(), 0, d.shape[0] - 1).astype(int)
    s = d[v, u]
    return np.nanmedian(np.where(s > 0, s, np.nan), 1)


def image(fr, s):
    q = PLANE[s]
    if s == 'lidar':
        ld = cfg['lidar_depth']
        d = rd(path(fr['seq'], fr['name'], ld, q)).astype(np.float32) / 256
        inv = fill(np.where(d > 0, 1 / np.maximum(d, 1e-3), 0).astype(np.float32), cfg['lidar_fill_iters'])
        ch = [stretch(inv, inv > 0)]
        if cfg['lidar_intensity']:
            it = fill(rd(path(fr['seq'], fr['name'], 'intensity' + ld[5:], q)).astype(np.float32), cfg['lidar_fill_iters'])
            ch.append(stretch(it, it > 0))
        return fit(q, np.stack([ch[0], ch[-1], ch[0]], -1), NEAR)
    im = rd(path(fr['seq'], fr['name'], 'img', s))
    if s == 'rgb':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.repeat(stretch(g.astype(np.float32))[..., None], 3, -1)
    return fit(q, im)


CT = centers('thr')


def work(fr):
    dep = {q: rd(path(fr['seq'], fr['name'], 'depth_filtered', q)).astype(np.float32) / 256 for q in PL}
    zt, okt = tok_depth('thr', dep['thr'])
    Xt = lift(K0['thr'], to_orig('thr', CT), zt)
    fwd, fok, st, bwd, bok = [], [], [], [], []
    for s in SENS:
        q = PLANE[s]
        if q == 'thr':
            xy, ok, n = CT, np.ones(len(CT), bool), (0, 0, np.nan)
        else:
            Xq = move(q, 'thr', Xt)
            uq = project(K0[q], Xq)
            dq = depth_at(dep[q], uq, cfg['occ_win'])
            h, w = dep[q].shape
            seen = okt & (Xq[:, 2] > 0) & (uq[:, 0] >= 0) & (uq[:, 0] <= w - 1) & (uq[:, 1] >= 0) & (uq[:, 1] <= h - 1) & np.isfinite(dq)
            rel = np.abs(np.nan_to_num(dq) - Xq[:, 2]) / np.maximum(Xq[:, 2], 1e-6)
            ok = seen & (rel <= cfg['occ_tol'])
            d0 = depth_at(dep[q], uq, 0)
            e0 = np.abs(d0 - Xq[:, 2]) / np.maximum(Xq[:, 2], 1e-6)
            xy = to_cont(q, uq)
            ok &= inside(q, xy)
            n = (int(seen.sum()), int(ok.sum()), float(np.nanmedian(e0[seen])) if np.isfinite(e0[seen]).any() else np.nan)
        fwd.append(np.where(ok[:, None], xy / [hw[q][1], hw[q][0]] * 2 - 1, -2.0))
        fok.append(ok)
        st.append(n)
        cq = centers(q)
        if q == 'thr':
            bwd.append(cq)
            bok.append(np.ones(len(cq), bool))
            continue
        zq, okq = tok_depth(q, dep[q])
        X = move('thr', q, lift(K0[q], to_orig(q, cq), zq))
        xy = to_cont('thr', project(K0['thr'], X))
        okb = okq & (X[:, 2] > 0) & inside('thr', xy)
        bwd.append(np.where(okb[:, None], xy, -1.0))
        bok.append(okb)
    return [image(fr, s) for s in SENS], np.stack(fwd).reshape(-1, GH, GW, 2), np.stack(fok).reshape(-1, GH, GW), bwd, bok, np.array(st)


def write_feats(path_, imgs, bb):
    if os.path.exists(path_):
        return
    arr, B = None, cfg['cache_batch']
    with torch.no_grad(), torch.autocast(dev.split(':')[0], dtype=torch.bfloat16):
        for s in range(0, len(imgs), B):
            t, (h, w) = encode(bb, to_input(imgs[s:s + B], bb['norm'], dev), n_last=L)
            if arr is None:
                arr = np.lib.format.open_memmap(path_ + '.tmp', mode='w+', dtype=np.float16, shape=(len(imgs), h, w, t.shape[-1]))
            arr[s:s + B] = t.float().reshape(-1, h, w, t.shape[-1]).cpu().numpy()
    arr.flush()
    del arr
    os.replace(path_ + '.tmp', path_)


N, S = len(frames), len(SENS)
if not os.path.exists(f'{C}/ms2_frames.json'):
    mm = lambda n, dt, sh: np.lib.format.open_memmap(f'{C}/{n}.npy', mode='w+', dtype=dt, shape=(N, *sh))
    img = {s: mm(f'ms2_img_{s}', np.uint8, (*hw[PLANE[s]], 3)) for s in SENS}
    fw, fo = mm('ms2_fwd', np.float16, (S, GH, GW, 2)), mm('ms2_fwd_ok', bool, (S, GH, GW))
    bw = {s: mm(f'ms2_bwd_{s}', np.float16, (*gs[PLANE[s]], 2)) for s in SENS}
    bo = {s: mm(f'ms2_bwd_ok_{s}', bool, gs[PLANE[s]]) for s in SENS}
    st = np.zeros((N, S, 3))
    with mp.get_context('fork').Pool(cfg['workers']) as pool:
        for i, (ims, f, o, b, k, n) in enumerate(pool.imap(work, frames, chunksize=4)):
            for j, s in enumerate(SENS):
                img[s][i], bw[s][i], bo[s][i] = ims[j], b[j].reshape(*gs[PLANE[s]], 2), k[j].reshape(gs[PLANE[s]])
            fw[i], fo[i], st[i] = f, o, n
    for x in (*img.values(), fw, fo, *bw.values(), *bo.values()):
        x.flush()
    meta = dict(sensors=SENS, planes=PLANE, hw={q: list(hw[q]) for q in PL}, grid={q: list(gs[q]) for q in PL},
                scale={q: float(sc[q]) for q in PL}, offset={q: off[q].tolist() for q in PL},
                n_frames={sp: sum(f['split'] == sp for f in frames) for sp in cfg['splits']},
                fwd_valid={s: round(float(fo[:, j].mean()), 4) for j, s in enumerate(SENS)},
                all_valid=round(float(fo.all(1).mean()), 4),
                depth_agree={s: round(float(st[:, j, 1].sum() / max(1, st[:, j, 0].sum())), 4)
                             for j, s in enumerate(SENS) if PLANE[s] != 'thr'},
                depth_err_median={s: round(float(np.nanmedian(st[:, j, 2])), 4) for j, s in enumerate(SENS) if PLANE[s] != 'thr'},
                bwd_valid={s: round(float(bo[s].mean()), 4) for s in SENS})
    json.dump(meta, open(f'{C}/ms2_meta.json', 'w'), indent=1)
    json.dump(frames, open(f'{C}/ms2_frames.json', 'w'))
    print(json.dumps(meta))

    rng = np.random.default_rng(0)
    for i in [frames[0]['id']] + [f['id'] for f in frames if f['split'] != 'train'][:3]:
        ok = np.flatnonzero(fo[i].all(0).ravel())
        sel = rng.choice(ok, min(40, len(ok)), replace=False)
        col = rng.integers(0, 256, (len(sel), 3)).tolist()
        tiles = []
        for j, s in enumerate(SENS):
            q = PLANE[s]
            k = hw['thr'][0] / hw[q][0]
            im = cv2.resize(np.ascontiguousarray(img[s][i]), (round(hw[q][1] * k), hw['thr'][0]))
            xy = (fw[i, j].reshape(-1, 2)[sel].astype(float) + 1) / 2 * [hw[q][1], hw[q][0]] * k
            for c, (x, y) in zip(col, xy):
                cv2.circle(im, (int(x), int(y)), 4, c, -1)
            tiles.append(im)
        cv2.imwrite(f'{C}/figs/corr_{i}.png', cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))

bb = load_backbone(cfg, bbn, dev)
for s in SENS:
    write_feats(f'{C}/ms2_feat_{bbn}_L{L}_{s}.npy', np.load(f'{C}/ms2_img_{s}.npy', mmap_mode='r'), bb)
