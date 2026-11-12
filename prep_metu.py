import argparse, glob, json, os
import cv2
import numpy as np
from PIL import Image
from common import load_cfg, homog

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
root, out = cfg['metu_root'], cfg['cache_dir']
os.makedirs(out, exist_ok=True)


def new_K(K, d, path):
    w, h = Image.open(path).size
    return cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0, (w, h))[0], (w, h)


seen, pairs, Hs = set(), [], []
for f in sorted(glob.glob(f'{root}/index/scene_info_*/*.npz')):
    d = np.load(f, allow_pickle=True)
    for p, K, dc in zip(d['image_paths'], d['intrinsics'], d['distortion_coefs']):
        if p is None or (p[0], p[1]) in seen:
            continue
        seen.add((p[0], p[1]))
        parts = p[0].split('/')
        k = next(i for i, x in enumerate(parts) if x.startswith('scene'))
        cond, scene = parts[k - 1], parts[k]
        if cond in cfg['exclude_conds']:
            continue
        Kv, Kt = (np.array(K[i], dtype=float).reshape(3, 3) for i in (0, 1))
        dv, dt = (np.array(dc[i], dtype=float) for i in (0, 1))
        nKv, rgb_size = new_K(Kv, dv, f'{root}/{p[0]}')
        nKt, th_size = new_K(Kt, dt, f'{root}/{p[1]}')
        H = nKt @ np.linalg.inv(nKv)
        Hs.append(H / H[2, 2])
        pairs.append(dict(id=len(pairs), cond=cond, scene=scene, rgb=p[0], th=p[1],
                          same_name=os.path.basename(p[0]) == os.path.basename(p[1]),
                          Kv=Kv.tolist(), dv=dv.tolist(), nKv=nKv.tolist(), Kt=Kt.tolist(), dt=dt.tolist(), nKt=nKt.tolist()))

Hm = np.median(np.stack(Hs), 0)
w, h = th_size
corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
err = np.array([np.abs(homog(Hm, homog(np.linalg.inv(H), corners)) - corners).max() for H in Hs])
meta = dict(H=Hm.tolist(), rgb_size=list(rgb_size), th_size=list(th_size), n_pairs=len(pairs),
            n_name_mismatch=int(sum(not p['same_name'] for p in pairs)),
            H_spread_px_median=float(np.median(err)), H_spread_px_p95=float(np.percentile(err, 95)),
            scenes=sorted({f"{p['cond']}/{p['scene']}" for p in pairs}))
json.dump(pairs, open(f'{out}/pairs.json', 'w'))
json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != 'H'}))
