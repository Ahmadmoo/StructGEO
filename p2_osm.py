"""Rasterize OpenStreetMap (roads, buildings, water) at osm_res over the reference-map area.
Optional extra building sources: Overture buildings (GeoParquet) and buildings from the DSM (coarse)."""
import argparse, json, os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from common import load_cfg
from p2_common import CH, colorize

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', nargs='+', required=True)
ap.add_argument('--force', action='store_true')
a = ap.parse_args()
cfg = load_cfg(*a.cfg)
R, P, res = cfg['tuav_root'], cfg['p2_cache'], cfg['osm_res']
os.makedirs(P, exist_ok=True)
SRC = dict(osm_files=cfg['osm_files'], overture=cfg.get('overture_buildings'), dsm_buildings=bool(cfg.get('dsm_buildings')),
           res=res, margin=cfg['osm_margin_m'], edge_m=cfg['edge_m'])
if not a.force and os.path.exists(f'{P}/osm.npy') and os.path.exists(f'{P}/osm_meta.json'):
    if json.load(open(f'{P}/osm_meta.json')).get('source') == SRC:
        print('osm.npy is up to date (use --force to rebuild)')
        raise SystemExit(0)
os.makedirs(f"{cfg['out_dir']}/p2_figs", exist_ok=True)
ref = rasterio.open(os.path.join(R, cfg['ref_tif']))
b, mg = ref.bounds, cfg['osm_margin_m']
x0, y0 = float(np.floor((b.left - mg) / res) * res), float(np.ceil((b.top + mg) / res) * res)
W, H = int(np.ceil((b.right + mg - x0) / res)), int(np.ceil((y0 - b.bottom + mg) / res))
to_map = Transformer.from_crs('EPSG:4326', ref.crs, always_xy=True)
L = np.zeros((len(CH), H, W), np.uint8)
SH = 2  # fixed-point bits for sub-pixel drawing
print(f'raster {W} x {H} at {res} m')

# class -> (channel, width in m)
ROAD = dict(motorway=(0, 20), trunk=(0, 16), primary=(0, 14), secondary=(0, 12), tertiary=(0, 10),
            motorway_link=(0, 8), trunk_link=(0, 8), primary_link=(0, 8), secondary_link=(0, 7), tertiary_link=(0, 7),
            residential=(1, 7), unclassified=(1, 6), living_street=(1, 5), road=(1, 6), busway=(1, 7), service=(1, 4),
            pedestrian=(1, 5), track=(1, 3), footway=(1, 2), path=(1, 2), cycleway=(1, 2), steps=(1, 2), bridleway=(1, 2))
RIVER = dict(river=20, canal=12, stream=4, drain=3, ditch=2)
cnt = dict(road_major=0, road_minor=0, building=0, water_area=0, water_line=0, overture=0)


def project(lon, lat):
    x, y = to_map.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.stack([(x - x0) / res, (y0 - y) / res], -1)


def fx(p):
    return np.round(p * (1 << SH)).astype(np.int32)


def px(m):
    return max(1, int(round(m / res)))


def hidden(t):
    try:
        under = int(str(t.get('layer', '0')).split(';')[0]) < 0
    except ValueError:
        under = False
    return t.get('tunnel', 'no') != 'no' or t.get('location') == 'underground' or under


def water_area(t):
    return (t.get('natural') == 'water' or t.get('waterway') == 'riverbank'
            or t.get('landuse') in ('reservoir', 'basin') or 'water' in t)


def building(t):
    return t.get('building', 'no') != 'no'


def poly(ch, outers, inners):
    cv2.fillPoly(L[ch], outers, 255, cv2.LINE_8, SH)
    if inners:
        cv2.fillPoly(L[ch], inners, 0, cv2.LINE_8, SH)
    if ch == 2:
        cv2.polylines(L[3], outers + inners, True, 255, px(cfg['edge_m']), cv2.LINE_8, SH)


def parse(path):
    nodes, ways, rels = {}, [], []
    for _, el in ET.iterparse(path, events=('end',)):
        if el.tag == 'node':
            nodes[el.get('id')] = (float(el.get('lon')), float(el.get('lat')))
        elif el.tag == 'way':
            ways.append((el.get('id'), [n.get('ref') for n in el.iter('nd')], {t.get('k'): t.get('v') for t in el.iter('tag')}))
        elif el.tag == 'relation':
            rels.append(([(m.get('type'), m.get('ref'), m.get('role')) for m in el.iter('member')],
                         {t.get('k'): t.get('v') for t in el.iter('tag')}))
        else:
            continue
        el.clear()
    return nodes, ways, rels


def assemble(segs):
    """Join member ways of a multipolygon into rings by matching end nodes."""
    out = {'outer': [], 'inner': []}
    for role in out:
        todo = [list(r) for rl, r in segs if (rl or 'outer') == role and len(r) >= 2]
        while todo:
            ring, grown = todo.pop(), True
            while ring[0] != ring[-1] and grown:
                grown = False
                for i, s in enumerate(todo):
                    if s[0] == ring[-1]:
                        ring = ring + s[1:]
                    elif s[-1] == ring[-1]:
                        ring = ring + s[::-1][1:]
                    elif s[-1] == ring[0]:
                        ring = s + ring[1:]
                    elif s[0] == ring[0]:
                        ring = s[::-1] + ring[1:]
                    else:
                        continue
                    todo.pop(i)
                    grown = True
                    break
            if len(ring) >= 4:
                out[role].append(ring)
    return out


nodes, ways, rels = {}, [], []
for p in cfg['osm_files'] or []:
    n, w, r = parse(p)
    nodes.update(n)
    ways.extend(w)
    rels.extend(r)
    print(f'{p}: {len(n)} nodes, {len(w)} ways, {len(r)} relations')
ids = list(nodes)
xy = project(*np.array([nodes[i] for i in ids]).T) if ids else np.zeros((0, 2))
pos = dict(zip(ids, range(len(ids))))


def line(refs):
    ix = [pos[r] for r in refs if r in pos]
    return fx(xy[ix]) if len(ix) >= 2 else None


wayref = {}
for wid, refs, t in ways:
    wayref[wid] = refs
    if not t or hidden(t):
        continue
    p = line(refs)
    if p is None:
        continue
    closed = refs[0] == refs[-1]
    if t.get('highway') in ROAD:
        ch, wd = ROAD[t['highway']]
        cv2.polylines(L[ch], [p], False, 255, px(wd), cv2.LINE_8, SH)
        cnt['road_major' if ch == 0 else 'road_minor'] += 1
    if building(t) and closed:
        poly(2, [p], [])
        cnt['building'] += 1
    if water_area(t) and closed:
        poly(4, [p], [])
        cnt['water_area'] += 1
    elif t.get('waterway') in RIVER:
        cv2.polylines(L[4], [p], False, 255, px(RIVER[t['waterway']]), cv2.LINE_8, SH)
        cnt['water_line'] += 1

for members, t in rels:
    if t.get('type') != 'multipolygon' or hidden(t) or not (building(t) or water_area(t)):
        continue
    rings = assemble([(role, wayref[ref]) for typ, ref, role in members if typ == 'way' and ref in wayref])
    outers = [p for p in map(line, rings['outer']) if p is not None]
    inners = [p for p in map(line, rings['inner']) if p is not None]
    if outers:
        poly(2 if building(t) else 4, outers, inners)
        cnt['building' if building(t) else 'water_area'] += 1

if cfg.get('overture_buildings'):
    import pyarrow.parquet as pq
    import shapely
    tb = pq.read_table(cfg['overture_buildings'], columns=['geometry'])
    for gm in shapely.from_wkb(tb.column('geometry').to_numpy(zero_copy_only=False)):
        for pg in getattr(gm, 'geoms', [gm]):
            if pg.geom_type == 'Polygon':
                poly(2, [fx(project(*np.asarray(pg.exterior.coords)[:, :2].T))],
                     [fx(project(*np.asarray(r.coords)[:, :2].T)) for r in pg.interiors])
                cnt['overture'] += 1

if cfg.get('dsm_buildings'):
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    dsm = np.full((H, W), np.nan, np.float32)
    with rasterio.open(os.path.join(R, cfg['dsm_tif'])) as s:
        reproject(rasterio.band(s, 1), dsm, dst_transform=from_origin(x0, y0, res, res), dst_crs=ref.crs,
                  resampling=Resampling.bilinear, src_nodata=s.nodata, dst_nodata=np.nan)
    ok = np.isfinite(dsm)
    if ok.any():
        z = np.where(ok, dsm, np.nanmin(dsm)).astype(np.float32)
        k = px(cfg['dsm_ground_win_m']) | 1
        ground = cv2.morphologyEx(z, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        bm = (((z - ground) > cfg['dsm_min_height']) & ok).astype(np.uint8) * 255
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        L[2] = np.maximum(L[2], bm)
        L[3] = np.maximum(L[3], cv2.morphologyEx(bm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
        del z, ground, bm
    del dsm

cov = {c: round(float((L[i] > 0).mean()), 4) for i, c in enumerate(CH)}
inner = (slice(None), slice(int(mg / res), H - int(mg / res)), slice(int(mg / res), W - int(mg / res)))
cov_ref = {c: round(float((L[inner][i] > 0).mean()), 4) for i, c in enumerate(CH)}
np.save(f'{P}/osm.npy', L)
json.dump(dict(x0=x0, y0=y0, res=res, width=W, height=H, crs=ref.crs.to_wkt(), channels=CH, counts=cnt,
               coverage=cov, coverage_ref_area=cov_ref, source=SRC), open(f'{P}/osm_meta.json', 'w'), indent=1)
s = max(1, int(np.ceil(max(H, W) / 2048)))
small = np.stack([cv2.resize(c, (W // s, H // s), interpolation=cv2.INTER_AREA) for c in L])
cv2.imwrite(f"{cfg['out_dir']}/p2_figs/osm_preview.png", cv2.cvtColor(colorize(small), cv2.COLOR_RGB2BGR))
print('counts', cnt)
print('coverage in the reference-map area', cov_ref)
if cov_ref['building'] < 0.05:
    print('NOTE: few buildings in OSM here. Consider overture_buildings or dsm_buildings (see README).')
