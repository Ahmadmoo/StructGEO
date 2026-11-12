"""Phase 2 frames: Thermal-UAV metadata -> map coordinates, ground resolution, heading, evaluation priors.
Also writes the Overpass query for the reference-map area."""
import argparse, csv, glob, json, os
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from pyproj import Transformer
from common import load_cfg
from p2_common import work_size

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
cfg = load_cfg(*ap.parse_args().cfg)
R, P = cfg['tuav_root'], cfg['p2_cache']
os.makedirs(P, exist_ok=True)
ref = rasterio.open(os.path.join(R, cfg['ref_tif']))
to_map = Transformer.from_crs('EPSG:4326', ref.crs, always_xy=True)
NUM = ('lat', 'lon', 'abs_height', 'rel_height', 'pitch', 'yaw', 'roll', 'TargetLatitude', 'TargetLongitude')


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def read_meta(split, mod):
    """SCC-Loc metadata JSON if present, else the per-scene <Modality>_info.csv files."""
    M = mod.capitalize()
    js = f'{R}/Data/metadata/{split}_{M}.json'
    if os.path.exists(js):
        return json.load(open(js)), 'json'
    out = []
    for c in sorted(glob.glob(f'{R}/Data/Thermal-UAV/{split}/*/*/{M}_info.csv')):
        d = os.path.dirname(c)
        for row in csv.DictReader(open(c, encoding='utf-8-sig')):
            e = {k: fl(row.get(k)) for k in NUM}
            e.update(cfg['cams'][mod], name=os.path.relpath(f'{d}/{M}/{row["filename"]}', R))
            out.append(e)
    return out, 'csv'


def rolling_median(x, w):
    if w <= 1:
        return x.copy()
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        s = x[max(0, i - w // 2):i + w // 2 + 1]
        s = s[np.isfinite(s)]
        if len(s):
            out[i] = np.median(s)
    return out


def ok(a, b):
    return np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-6 and abs(b) > 1e-6


assert cfg['prior_m'] < cfg['search_m'] and cfg['yaw_noise'] <= cfg['yaw_range'], 'priors must fall inside the search range'
rng = np.random.default_rng(cfg['seed'])
b = ref.bounds
frames, info = {}, dict(crs=ref.crs.to_wkt(), ref_bounds=list(b), ref_res=list(ref.res), sources={})
for mod in cfg['p2_modalities']:
    W, H = work_size(cfg, mod)
    rows = []
    for split in ('train', 'valid', 'test'):
        meta, src = read_meta(split, mod)
        info['sources'][f'{split}_{mod}'] = f'{src}:{len(meta)}'
        for e in meta:
            path = e['name'].replace('\\', '/')
            while path.startswith('./'):
                path = path[2:]
            parts = path.split('/')
            rows.append(dict(split=split, scene=parts[-3], path=path, e=e))
    if not rows:
        raise SystemExit(f'no {mod} metadata found under {R}/Data')
    e0 = rows[0]['e']
    aw, ah = float(e0['width']) / float(e0['height']), W / H
    assert abs(aw / ah - 1) < 0.01, f'{mod}: working size {W}x{H} must keep the sensor aspect {e0["width"]:.0f}x{e0["height"]:.0f}'

    # height above ground: laser range, smoothed with a rolling median along each flight (robust to roof hits)
    rows.sort(key=lambda r: (r['split'], r['scene'], r['path']))
    keys = [(r['split'], r['scene']) for r in rows]
    d_lrf = np.array([fl(r['e']['rel_height']) for r in rows])
    d_lrf[~(d_lrf > 0)] = np.nan
    d = np.full_like(d_lrf, np.nan)
    for k in sorted(set(keys)):
        m = np.array([q == k for q in keys])
        dm = rolling_median(d_lrf[m], cfg['height_win'])
        dm[~np.isfinite(dm)] = np.nanmedian(d_lrf[m]) if np.isfinite(d_lrf[m]).any() else np.nan
        d[m] = dm

    out, drop = [], dict(tilt=0, no_height=0, off_map=0, no_position=0)
    for r, dl, dd in zip(rows, d_lrf, d):
        e = r['e']
        pitch = fl(e['pitch'])
        if not np.isfinite(dd):
            drop['no_height'] += 1
            continue
        if not np.isfinite(pitch) or abs(pitch + 90) > cfg['max_tilt']:
            drop['tilt'] += 1
            continue
        lat, lon = (fl(e.get('TargetLatitude')), fl(e.get('TargetLongitude')))
        if not ok(lat, lon):
            lat, lon = fl(e['lat']), fl(e['lon'])
        if not ok(lat, lon):
            drop['no_position'] += 1
            continue
        E, N = to_map.transform(lon, lat)
        if not (b.left <= E <= b.right and b.bottom <= N <= b.top):
            drop['off_map'] += 1
            continue
        pix = float(e['cam_size']) / np.hypot(float(e['width']), float(e['height']))  # mm per pixel
        f_px = float(e['focal_len']) / pix * W / float(e['width'])                   # focal length at working size
        out.append(dict(id=len(out), split=r['split'], scene=r['scene'],
                        cond='night' if 'night' in r['scene'].lower() else 'day', path=r['path'],
                        E=float(E), N=float(N), yaw=float((cfg['yaw_sign'] * fl(e['yaw']) + cfg['yaw_offset']) % 360),
                        pitch=pitch, d_lrf=float(dl), d=float(dd), gsd=float(dd / f_px),
                        prior=[float(x) for x in np.r_[rng.uniform(-1, 1, 2) * cfg['prior_m'],
                                                       rng.uniform(-1, 1) * cfg['yaw_noise']]]))
    frames[mod] = out
    gsd = np.array([f['gsd'] for f in out])
    diag = gsd * np.hypot(W, H)
    tm = cfg['tmpl_m' if mod == 'thermal' else 'tmpl_m_visible']
    lrf = np.array([f['d_lrf'] for f in out])
    hgt = np.array([f['d'] for f in out])
    rel = np.abs(lrf - hgt) / hgt
    info[mod] = dict(
        n={s: sum(f['split'] == s for f in out) for s in ('train', 'valid', 'test')},
        cond={c: sum(f['cond'] == c for f in out) for c in ('day', 'night')},
        scenes=sorted({f['scene'] for f in out}), dropped=drop,
        height_m=np.percentile(hgt, [0, 50, 100]).round(1).tolist(),
        lrf_off_by_10pct=round(float(np.nanmean(rel > 0.1)), 4),
        footprint_m=dict(width=np.percentile(gsd * W, [0, 50, 100]).round(1).tolist(),
                         height=np.percentile(gsd * H, [0, 50, 100]).round(1).tolist()),
        footprint_diag_over_template=round(float((diag > tm).mean()), 4))
    print(mod, json.dumps(info[mod]))
    if info[mod]['footprint_diag_over_template'] > 0.05:
        print(f'WARNING: {mod}: some footprints are larger than the template ({tm} m); their corners are cut. '
              f'Raise tmpl_m{"" if mod == "thermal" else "_visible"} if this is common.')

json.dump(frames, open(f'{P}/frames.json', 'w'))
json.dump(info, open(f'{P}/p2_meta.json', 'w'), indent=1)

m = cfg['osm_margin_m']
w, s, e, n = transform_bounds(ref.crs, 'EPSG:4326', b.left - m, b.bottom - m, b.right + m, b.top + m, densify_pts=21)
q = f'''[out:xml][timeout:900][maxsize:2000000000][bbox:{s:.6f},{w:.6f},{n:.6f},{e:.6f}];
(
  way["highway"];
  way["building"];
  relation["building"];
  way["natural"="water"];
  relation["natural"="water"];
  way["waterway"];
  relation["waterway"="riverbank"];
  way["landuse"~"^(reservoir|basin)$"];
  relation["landuse"~"^(reservoir|basin)$"];
  way["water"];
  relation["water"];
);
(._;>;);
out body;
'''
open(f'{P}/overpass_query.txt', 'w').write(q)
print(f'\nOSM for the map area (lon {w:.4f}..{e:.4f}, lat {s:.4f}..{n:.4f}), either:\n'
      f'  curl -X POST --data-binary @{P}/overpass_query.txt -o changsha.osm https://overpass-api.de/api/interpreter\n'
      f'  osmium extract -b {w:.5f},{s:.5f},{e:.5f},{n:.5f} china-latest.osm.pbf -o changsha.osm   # Geofabrik extract\n'
      f'then set osm_files in configs/p2_base.yaml')
