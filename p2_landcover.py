"""Experiment 4: add free global land cover (ESA WorldCover 2021, 10 m) to a map raster as extra channels.
map_name (new raster) = lc_base (an existing raster, e.g. osm or osm_dsm) + one channel per land-cover group.
Fields, trees, bare ground and water look different in thermal, and these layers cover the villages where OSM is thin.
Without worldcover_tifs set, prints the tiles to download and stops."""
import argparse, json, os
import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from common import load_cfg

# channel -> WorldCover classes (10 tree, 20 shrub, 30 grass, 40 crop, 50 built-up, 60 bare/sparse, 70 snow/ice,
# 80 water, 90 herbaceous wetland, 95 mangrove, 100 moss/lichen)
LC = [('lc_tree', (10, 95)), ('lc_grass_shrub', (20, 30, 90, 100)), ('lc_crop', (40,)), ('lc_built', (50,)),
      ('lc_bare', (60, 70)), ('lc_water', (80,))]
LC_RGB = np.array([[0, 100, 0], [255, 187, 34], [240, 150, 255], [250, 0, 0], [180, 180, 180], [0, 100, 200]], np.uint8)
URL = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif'

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--force', action='store_true')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
P, base, name = cfg['p2_cache'], cfg['lc_base'], cfg['map_name']
assert name != base, 'map_name must name the new raster (e.g. osm_wc), lc_base the one it extends'
om = json.load(open(f'{P}/{base}_meta.json'))
H, W, res, x0, y0 = om['height'], om['width'], om['res'], om['x0'], om['y0']
tifs = sorted(cfg.get('worldcover_tifs') or [])
SRC = dict(base=base, base_source=om.get('source'), base_channels=om['channels'], worldcover=tifs,
           groups=[[n, list(c)] for n, c in LC])
if not a.force and os.path.exists(f'{P}/{name}.npy') and os.path.exists(f'{P}/{name}_meta.json'):
    if json.load(open(f'{P}/{name}_meta.json')).get('source') == SRC:
        print(f'{name}.npy is up to date (use --force to rebuild)')
        raise SystemExit(0)

w, s, e, n = transform_bounds(CRS.from_wkt(om['crs']), 'EPSG:4326', x0, y0 - H * res, x0 + W * res, y0, densify_pts=21)
need = [f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
        for la in range(int(np.floor(s / 3) * 3), int(np.floor(n / 3) * 3) + 1, 3)
        for lo in range(int(np.floor(w / 3) * 3), int(np.floor(e / 3) * 3) + 1, 3)]
if not tifs:
    print(f'map box lon {w:.4f}..{e:.4f}, lat {s:.4f}..{n:.4f} needs WorldCover tile(s) {need}. Download, e.g.:')
    for t in need:
        print(f'  wget {URL.format(t)}')
    print('then list the files in worldcover_tifs (configs/p2_exp.yaml or your run config) and rerun.')
    raise SystemExit(1)

lc = np.zeros((H, W), np.uint8)
dst = dict(dst_transform=from_origin(x0, y0, res, res), dst_crs=CRS.from_wkt(om['crs']))
for p in tifs:
    tmp = np.zeros((H, W), np.uint8)
    with rasterio.open(p) as src:
        reproject(rasterio.band(src, 1), tmp, src_nodata=0, dst_nodata=0, resampling=Resampling.nearest, **dst)
    lc = np.where(lc == 0, tmp, lc)
    del tmp
if not (lc > 0).any():
    raise SystemExit(f'the WorldCover files {tifs} do not overlap the map box; tiles needed: {need}')

src_map = np.load(f'{P}/{base}.npy', mmap_mode='r')
C0 = src_map.shape[0]
out = np.lib.format.open_memmap(f'{P}/{name}.npy.tmp.npy', mode='w+', dtype=np.uint8, shape=(C0 + len(LC), H, W))
for c in range(C0):
    out[c] = src_map[c]
for j, (_, codes) in enumerate(LC):
    out[C0 + j] = np.isin(lc, codes).astype(np.uint8) * 255
out.flush()
mg = int(cfg['osm_margin_m'] / res)
inner = lc[mg:H - mg, mg:W - mg]
cov_ref = {nm: round(float(np.isin(inner, codes).mean()), 4) for nm, codes in LC}
del out
os.replace(f'{P}/{name}.npy.tmp.npy', f'{P}/{name}.npy')
meta = dict(om, channels=list(om['channels']) + [nm for nm, _ in LC], source=SRC,
            coverage_ref_area=dict(om.get('coverage_ref_area', {}), **cov_ref),
            landcover_unknown_ref_area=round(float((inner == 0).mean()), 4))
json.dump(meta, open(f'{P}/{name}_meta.json', 'w'), indent=1)
k = max(1, int(np.ceil(max(H, W) / 2048)))
small = cv2.resize(lc, (W // k, H // k), interpolation=cv2.INTER_NEAREST)
rgb = np.full((*small.shape, 3), 25, np.uint8)
for (nm, codes), col in zip(LC, LC_RGB):
    rgb[np.isin(small, codes)] = col
os.makedirs(f"{cfg['out_dir']}/p2_figs", exist_ok=True)
cv2.imwrite(f"{cfg['out_dir']}/p2_figs/{name}_landcover_preview.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print(f'{name}: {C0} + {len(LC)} channels; land cover in the reference-map area', cov_ref)
