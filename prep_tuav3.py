import argparse, json, os, re
import multiprocessing as mp
import cv2
import numpy as np
import torch
from common import load_cfg, load_backbone, encode, to_input
from p2_common import CH, PALETTE, cache_key, load_frames, load_osm, work_size, img2world, warp_down

LC = {'lc_tree': (0, 100, 0), 'lc_grass_shrub': (255, 187, 34), 'lc_crop': (240, 150, 255), 'lc_built': (250, 0, 0),
      'lc_bare': (180, 180, 180), 'lc_water': (0, 100, 200)}

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
P, C, p, L, dev, bbn, k = cfg['p2_cache'], cfg['cache_dir'], cfg['patch'], cfg['layers'], cfg['device'], cfg['backbone'], cfg['map_down']
os.makedirs(f'{C}/figs', exist_ok=True)
fall = load_frames(cfg)
fr = fall['thermal']
for kind in ('thermal', 'sat'):
    side = f'{P}/p2_img_{kind}.key'
    assert os.path.exists(side) and open(side).read() == cache_key(fall, kind, cfg), f'{kind}: Phase 2 cache stale or missing, run p2_cache.py'
osm, om = load_osm(cfg)
CHN = om['channels']
W, H = work_size(cfg, 'thermal')
gh, gw, w4, h4, r = H // p, W // p, W // k, H // k, p // k
assert W % p == 0 and H % p == 0 and p % k == 0
N = len(fr)
A0 = np.array([[1 / om['res'], 0, -om['x0'] / om['res']], [0, -1 / om['res'], om['y0'] / om['res']], [0, 0, 1.0]])


def layers(f):
    A = A0 @ img2world(f, (W, H)) @ np.diag([k, k, 1.0])
    c = A @ np.array([[0, w4, 0, w4], [0, 0, h4, h4], [1, 1, 1, 1]], float)
    c0, r0 = np.floor(c[:2].min(1)).astype(int) - 2
    c1, r1 = np.ceil(c[:2].max(1)).astype(int) + 2
    win = np.zeros((len(CHN), r1 - r0, c1 - c0), np.uint8)
    a0, a1, b0, b1 = max(r0, 0), min(r1, osm.shape[1]), max(c0, 0), min(c1, osm.shape[2])
    if a1 > a0 and b1 > b0:
        win[:, a0 - r0:a1 - r0, b0 - c0:b1 - c0] = osm[:, a0:a1, b0:b1]
    A = np.array([[1, 0, -c0], [0, 1, -r0], [0, 0, 1.0]]) @ A
    out = [warp_down(np.ascontiguousarray(win[i:i + 3].transpose(1, 2, 0)), A, (w4, h4)).reshape(h4, w4, -1)
           for i in range(0, len(CHN), 3)]
    return np.concatenate(out, -1).transpose(2, 0, 1)


def map_image(lay):
    x = np.full((h4, w4, 3), 25, np.float32)
    for nm, col in LC.items():
        if nm in CHN:
            a = lay[CHN.index(nm)][..., None] / 255
            x = x * (1 - a) + np.array(col, np.float32) * a
    for i in (4, 2, 3, 1, 0):
        if CH[i] in CHN:
            a = lay[CHN.index(CH[i])][..., None] / 255
            x = x * (1 - a) + PALETTE[i] * a
    return cv2.resize(x.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)


def work(f):
    lay = layers(f)
    c = (lay.max(0) > 0).reshape(gh, r, gw, r).any((1, 3)).astype(np.uint8)
    d = cfg['map_valid_dilate']
    c = cv2.dilate(c, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) if d > 0 else c
    return lay, map_image(lay), c.astype(bool)


place = lambda s: (re.match(r'[A-Za-z]+', s).group(0).lower() if re.match(r'[A-Za-z]+', s) else s)
frames = [dict(id=f['id'], split=f['split'], cond=f"{place(f['scene'])}-{f['cond']}") for f in fr]
S = ['thr', 'sat', 'map']
files = dict(thr=dict(img=f'{P}/p2_img_thermal.npy', feat=P + '/p2_feat_{bbn}_L{L}_thermal.npy'),
             sat=dict(img=f'{P}/p2_img_sat.npy', feat=P + '/p2_feat_{bbn}_L{L}_sat.npy'),
             map=dict(img=f'{C}/tuav_img_map.npy', feat=C + '/tuav_feat_{bbn}_L{L}_map.npy', layers=f'{C}/tuav_maplayers.npy'))
if not os.path.exists(f'{C}/tuav_frames.json'):
    mm = lambda n, dt, sh: np.lib.format.open_memmap(f'{C}/{n}.npy', mode='w+', dtype=dt, shape=(N, *sh))
    lay, mim = mm('tuav_maplayers', np.uint8, (len(CHN), h4, w4)), mm('tuav_img_map', np.uint8, (H, W, 3))
    fw, fo = mm('tuav_fwd', np.float16, (3, gh, gw, 2)), mm('tuav_fwd_ok', bool, (3, gh, gw))
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    uv = np.stack([(jj + .5) * p, (ii + .5) * p], -1)
    for s in S:
        b = mm(f'tuav_bwd_{s}', np.float16, (gh, gw, 2))
        b[:] = uv
        b.flush()
        bo = mm(f'tuav_bwd_ok_{s}', bool, (gh, gw))
        bo[:] = True
        bo.flush()
    fw[:] = uv / [W, H] * 2 - 1
    sat = np.load(files['sat']['img'], mmap_mode='r')
    with mp.get_context('fork').Pool(cfg['workers']) as pool:
        for i, (l_, im, c) in enumerate(pool.imap(work, fr, chunksize=8)):
            lay[i], mim[i] = l_, im
            fo[i, 0] = True
            fo[i, 1] = (sat[i].max(-1) > 0).reshape(gh, p, gw, p).mean((1, 3)) > 0.5
            fo[i, 2] = c
    for x in (lay, mim, fw, fo):
        x.flush()
    meta = dict(sensors=S, planes={s: 'thr' for s in S}, hw={'thr': [H, W]}, grid={'thr': [gh, gw]}, files=files,
                map_name=cfg.get('map_name') or 'osm', map_channels=CHN, map_down=k,
                n_frames={sp: sum(f['split'] == sp for f in frames) for sp in ('train', 'valid', 'test')},
                fwd_valid={s: round(float(fo[:, j].mean()), 4) for j, s in enumerate(S)})
    json.dump(meta, open(f'{C}/tuav_meta.json', 'w'), indent=1)
    json.dump(frames, open(f'{C}/tuav_frames.json', 'w'))
    print(json.dumps({x: meta[x] for x in ('map_name', 'map_channels', 'n_frames', 'fwd_valid')}))

    thr = np.load(files['thr']['img'], mmap_mode='r')
    te = [f['id'] for f in frames if f['split'] == 'test']
    for i in te[::max(1, len(te) // 4)][:4]:
        m = mim[i].astype(np.float32)
        blend = lambda x: (0.5 * x.astype(np.float32) + 0.5 * m).astype(np.uint8)
        row = np.concatenate([thr[i], sat[i], mim[i], blend(thr[i]), blend(sat[i])], 1)
        cv2.imwrite(f'{C}/figs/align_{i}.png', cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

fm = files['map']['feat'].format(bbn=bbn, L=L)
if not os.path.exists(fm):
    bb = load_backbone(cfg, bbn, dev)
    imgs, arr, B = np.load(files['map']['img'], mmap_mode='r'), None, cfg['cache_batch']
    with torch.no_grad(), torch.autocast(dev.split(':')[0], dtype=torch.bfloat16):
        for s in range(0, N, B):
            t, (h, w) = encode(bb, to_input(imgs[s:s + B], bb['norm'], dev), n_last=L)
            if arr is None:
                arr = np.lib.format.open_memmap(fm + '.tmp', mode='w+', dtype=np.float16, shape=(N, h, w, t.shape[-1]))
            arr[s:s + B] = t.float().reshape(-1, h, w, t.shape[-1]).cpu().numpy()
    arr.flush()
    del arr
    os.replace(fm + '.tmp', fm)
