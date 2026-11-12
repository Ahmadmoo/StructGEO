import argparse, os
import multiprocessing as mp
import cv2
import numpy as np
import torch
from common import load_cfg, load_pairs, work_H, homog, load_backbone, encode, to_input

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--backbone')
ap.add_argument('--option', choices=['same', 'native'])
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
bbn, option = a.backbone or cfg['backbone'], a.option or cfg['option']
C, root, L, dev = cfg['cache_dir'], cfg['metu_root'], cfg['layers'], cfg['device']
pairs, meta = load_pairs(cfg)
H = work_H(cfg, meta)
tw, th = cfg['thermal_size']
rw, rh = cfg['rgb_native_size']
q = homog(np.linalg.inv(H), np.array([[0.0, 0.0], [tw, 0.0]]))
pre = max(1, int(np.linalg.norm(q[1] - q[0]) / tw))


def undist(path, p, k):
    im = cv2.imread(f'{root}/{path}', cv2.IMREAD_COLOR)
    return cv2.undistort(im, np.array(p['K' + k]), np.array(p['d' + k]), None, np.array(p['nK' + k]))


def load_th(p):
    g = cv2.cvtColor(undist(p['th'], p, 't'), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def load_rgb(p):
    im = cv2.cvtColor(undist(p['rgb'], p, 'v'), cv2.COLOR_BGR2RGB)
    if option == 'native':
        return cv2.resize(im, (rw, rh), interpolation=cv2.INTER_AREA)
    small = cv2.resize(im, (im.shape[1] // pre, im.shape[0] // pre), interpolation=cv2.INTER_AREA)
    S = np.diag([im.shape[1] / small.shape[1], im.shape[0] / small.shape[0], 1.0])
    return cv2.warpPerspective(small, H @ S, (tw, th), flags=cv2.INTER_LINEAR)


def write_imgs(path, fn, hw):
    if not os.path.exists(path):
        arr = np.lib.format.open_memmap(path + '.tmp', mode='w+', dtype=np.uint8, shape=(len(pairs), *hw, 3))
        with mp.get_context('fork').Pool(cfg['workers']) as pool:
            for i, im in enumerate(pool.imap(fn, pairs, chunksize=4)):
                arr[i] = im
        arr.flush()
        del arr
        os.replace(path + '.tmp', path)
    return np.load(path, mmap_mode='r')


def write_feats(path, imgs, bb):
    if os.path.exists(path):
        return
    arr, B = None, cfg['cache_batch']
    with torch.no_grad(), torch.autocast(dev.split(':')[0], dtype=torch.bfloat16):
        for s in range(0, len(imgs), B):
            t, (h, w) = encode(bb, to_input(imgs[s:s + B], bb['norm'], dev), n_last=L)
            if arr is None:
                arr = np.lib.format.open_memmap(path + '.tmp', mode='w+', dtype=np.float16, shape=(len(imgs), h, w, t.shape[-1]))
            arr[s:s + B] = t.float().reshape(-1, h, w, t.shape[-1]).cpu().numpy()
    arr.flush()
    del arr
    os.replace(path + '.tmp', path)


img_th = write_imgs(f'{C}/img_th.npy', load_th, (th, tw))
img_rgb = write_imgs(f'{C}/img_rgb_{option}.npy', load_rgb, (th, tw) if option == 'same' else (rh, rw))
bb = load_backbone(cfg, bbn, dev)
write_feats(f'{C}/feat_{bbn}_L{L}_th.npy', img_th, bb)
write_feats(f'{C}/feat_{bbn}_L{L}_rgb_{option}.npy', img_rgb, bb)
