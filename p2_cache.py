"""Phase 2 images and frozen features per query kind:
thermal (Thermal-UAV frames), sat (satellite rendered at each thermal frame's footprint), visible (drone RGB frames)."""
import argparse, glob, json, os
import multiprocessing as mp
import numpy as np
import rasterio
import torch
from common import load_cfg, load_backbone, encode, to_input
from p2_common import KIND, cache_key, work_size, render_sat, load_gray, load_rgb

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--backbone')
ap.add_argument('--kinds', nargs='+')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
bbn, kinds = a.backbone or cfg['backbone'], a.kinds or cfg['p2_kinds']
P, R, L, dev = cfg['p2_cache'], cfg['tuav_root'], cfg['layers'], cfg['device']
frames = json.load(open(f'{P}/frames.json'))
_src = None


def img_job(job):
    global _src
    kind, f = job
    wh = work_size(cfg, KIND[kind][0])
    if kind == 'sat':
        if _src is None:
            _src = rasterio.open(os.path.join(R, cfg['ref_tif']))
        return render_sat(_src, f, wh)
    path = os.path.join(R, f['path'])
    return load_gray(path, wh) if kind == 'thermal' else load_rgb(path, wh)


def write_imgs(kind):
    path = f'{P}/p2_img_{kind}.npy'
    fl = frames[KIND[kind][0]]
    W, H = work_size(cfg, KIND[kind][0])
    if not os.path.exists(path):
        arr = np.lib.format.open_memmap(path + '.tmp', mode='w+', dtype=np.uint8, shape=(len(fl), H, W, 3))
        with mp.get_context('fork').Pool(cfg['workers']) as pool:
            for i, im in enumerate(pool.imap(img_job, [(kind, f) for f in fl], chunksize=8)):
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


missing = [k for k in kinds if KIND[k][0] not in frames]
if missing:
    raise SystemExit(f'{missing}: add their modality to p2_modalities and rerun p2_prep.py')
for k in kinds:  # drop caches made from another frame list or other footprints (e.g. after changing yaw_sign)
    key, side = cache_key(frames, k), f'{P}/p2_img_{k}.key'
    if not os.path.exists(side) or open(side).read() != key:
        for p in glob.glob(f'{P}/p2_img_{k}.npy') + glob.glob(f'{P}/p2_feat_*_{k}.npy'):
            print('stale cache, removing', p)
            os.remove(p)
        open(side, 'w').write(key)
imgs = {k: write_imgs(k) for k in kinds}
bb = load_backbone(cfg, bbn, dev)
for k in kinds:
    write_feats(f'{P}/p2_feat_{bbn}_L{L}_{k}.npy', imgs[k], bb)
    print(k, imgs[k].shape)
