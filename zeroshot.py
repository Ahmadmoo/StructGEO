import argparse, json, os
import cv2
import numpy as np
import torch
from skimage.feature import hog
from common import load_cfg, load_pairs, geometry, rgb_at_th, retrieval, summarize, chance_hit1, pca_panel

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
C, O, dev, L, p = cfg['cache_dir'], cfg['out_dir'], cfg['device'], cfg['layers'], cfg['patch']
os.makedirs(f'{O}/figs', exist_ok=True)
pairs, meta = load_pairs(cfg)
te = [q['id'] for q in pairs if q['test']]
sub = [pairs[i] for i in te]
img_th = np.load(f'{C}/img_th.npy', mmap_mode='r')


def run(name, ft_all, fr_all, grid, img_rgb, local=False):
    res = []
    for s in range(0, len(te), 16):
        idx = list(range(s, min(s + 16, len(te)))) if local else te[s:s + 16]
        ft = torch.from_numpy(np.ascontiguousarray(ft_all[idx])).to(dev).float()
        fr = torch.from_numpy(np.ascontiguousarray(fr_all[idx])).to(dev).float()
        res.append(retrieval(ft, rgb_at_th(fr, grid)))
        if s == 0 and not local:
            for k in range(min(cfg['n_fig'], len(idx))):
                pca_panel(ft[k], fr[k], img_th[idx[k]], img_rgb[idx[k]], f'{O}/figs/{name}_{idx[k]}.png')
    out = summarize(np.concatenate(res), sub, chance_hit1(*ft_all.shape[1:3]))
    json.dump(dict(name=name, step='zero-shot', **out), open(f'{O}/{name}.json', 'w'), indent=1)


def gray(im):
    return cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2GRAY)


def pix(im):
    g = gray(im).astype(np.float32)
    gh, gw = g.shape[0] // p, g.shape[1] // p
    x = g[:gh * p, :gw * p].reshape(gh, p, gw, p).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    return x - x.mean(-1, keepdims=True)


def hogf(im):
    g = gray(im)
    gh, gw = g.shape[0] // p, g.shape[1] // p
    b = hog(g, orientations=9, pixels_per_cell=(p // 2, p // 2), cells_per_block=(2, 2), feature_vector=False)
    return b[0:2 * gh:2, 0:2 * gw:2].reshape(gh, gw, -1)


for bbn in cfg['zs_backbones']:
    th_path = f'{C}/feat_{bbn}_L{L}_th.npy'
    if not os.path.exists(th_path):
        continue
    ft_all = np.load(th_path, mmap_mode='r')
    for opt in cfg['zs_options']:
        rgb_path = f'{C}/feat_{bbn}_L{L}_rgb_{opt}.npy'
        if os.path.exists(rgb_path):
            run(f'zs_{bbn}_{opt}_L{L}', ft_all, np.load(rgb_path, mmap_mode='r'), geometry(cfg, meta, opt, dev)['grid'],
                np.load(f'{C}/img_rgb_{opt}.npy', mmap_mode='r'))

if os.path.exists(f'{C}/img_rgb_same.npy'):
    img_rgb = np.load(f'{C}/img_rgb_same.npy', mmap_mode='r')
    for name, fn in (('pixels', pix), ('hog', hogf)):
        run(f'zs_{name}_same', np.stack([fn(img_th[i]) for i in te]), np.stack([fn(img_rgb[i]) for i in te]),
            None, img_rgb, local=True)
