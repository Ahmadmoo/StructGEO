# StructGeo — proof of concept

Phase 1 tests one claim: **RGB and thermal drone images share the same structure in self-supervised features, and a small JEPA-trained adapter makes that space shared.** Phase 2 uses that space to place a thermal frame on OpenStreetMap. The full plan, design choices and Phase 2 extension are in [PLAN.md](PLAN.md).

## Setup (HPC)
```bash
bash setup_hpc.sh          # clones I-JEPA + DINOv3 code, downloads I-JEPA ViT-H/16-448, installs requirements
```
Then edit `configs/base.yaml`:
- `metu_root`, `cache_dir`
- `dino_web` / `dino_sat`: your DINOv3 ViT-L/16 checkpoints in Meta's `.pth` format (from the DINOv3 download links, not the Hugging Face format). `dino_sat` is optional.
- `test_scenes`: after `prep_metu.py`, pick one scene with both cloudy and sunny, and one cloudy-only scene.

Data: [METU-VisTIR](https://drive.google.com/file/d/1Sj_vxj-GXvDQIMSg-ZUJR0vHBLIeDrLg/view?usp=sharing) (CC BY-NC-SA 4.0), unzipped so that `metu_root/index/` and `metu_root/cloudy/` exist.

## Run
```bash
bash run_phase1.sh                      # prep → cache → zero-shot → 4 main runs → table
BACKBONES="ijepa dino_web" bash run_phase1.sh   # if you have no DINOv3 SAT weights
bash run_ablations.sh                   # after setting the best backbone/option in configs/a*.yaml
```
After `prep_metu.py`, check `cache_dir/meta.json`: `n_name_mismatch` should be 0 and `H_spread_px_p95` should be below one token (16 px).

## Outputs
| Path | What |
|---|---|
| `results/zs_*.json` | Step 1 (zero-shot) retrieval scores per backbone and resolution option, plus pixel and HOG baselines |
| `results/r*.json`, `results/a*.json` | Step 2 (trained adapter) scores; `.pt` holds adapter, EMA adapter and predictor |
| `results/table.md` | All runs in one table |
| `results/figs/*.png` | Shared-PCA images: RGB, RGB features, thermal, thermal features |

## Files
| File | Role |
|---|---|
| `prep_metu.py` | Same-shot RGB–thermal pairs and the fixed RGB→thermal homography |
| `cache.py` | Preprocessed images and full-image frozen features per backbone and option |
| `zeroshot.py` | Step 1: retrieval + PCA figures + baselines |
| `train.py` | Step 2: adapter + predictor with masked cross-sensor JEPA, NCE and variance losses |
| `report.py` | Collects all results into `results/table.md` |
| `common.py` | Backbones, geometry, masking, model parts, metrics |

Tested end to end on a small synthetic dataset with random small backbones on CPU (geometry checked: warped RGB vs. thermal pixel correlation 0.99). It has not yet run on the real data and weights, so expect small fixes on the first run.

---

## Phase 2: place a thermal frame on OpenStreetMap

A frame plus its height, heading and lens data is matched against an OpenStreetMap raster (roads, buildings, water) within ±128 m and ±10° of a prior. Output: position, heading, and a confidence, or "unsure". Design in [PLAN.md §8](PLAN.md#8-phase-2-thermal-frame--openstreetmap).

### Data
- **Thermal-UAV** ([Hugging Face `FloralHercules/Thermal-UAV`](https://huggingface.co/datasets/FloralHercules/Thermal-UAV), or the Baidu link in the SCC-Loc README). Unpack so that `tuav_root/Data/Thermal-UAV/{train,valid,test}/changsha/<scene>/Thermal/*.JPG` and `tuav_root/Data/Reference_map/changsha/{ref.tif,dsm.tif}` exist. If `tuav_root/Data/metadata/*_Thermal.json` is missing, `p2_prep.py` reads the per-scene `Thermal_info.csv` files instead.
- **OpenStreetMap** for the reference-map box: `p2_prep.py` prints the box and two ways to get it (Overpass `curl`, or `osmium extract` from the Geofabrik China file). Put the `.osm` path in `osm_files`.
- Optional, if OSM buildings are thin there: Overture buildings for the same box (`pip install overturemaps`, then `overturemaps download --bbox=W,S,E,N -f geoparquet --type=building -o buildings.parquet`) → `overture_buildings`; or `dsm_buildings: true` (coarse, from the 5.3 m DSM).
- Check the Thermal-UAV license before publishing results.

### Setup
```bash
pip install -r requirements.txt     # adds rasterio, pyproj, shapely (pyarrow only for Overture)
```
Edit `configs/p2_base.yaml`: `tuav_root`, `p2_cache`, `osm_files`, and `backbone` + `p1_ckpt` (the best Phase 1 run with that backbone; `results/r4_dino_native.pt` by default).

### Run
```bash
bash run_phase2.sh       # prep → OSM raster → images + features → checks → m1, m2, m3, a1 → table
RUNS="p2_a2_jepa" bash run_phase2.sh   # other runs (steps already done are skipped by the caches)
```
Read the `p2_check.py` lines before trusting any run: the geometry check must pick `gt` (if not, it says whether to flip `yaw_sign`, set `yaw_offset: 180` or look at heights), and the matcher check on OSM-rendered queries must be about one cell (2 m) or better.

### Outputs
| Path | What |
|---|---|
| `results/p2_check.json` | Geometry check, matcher check, OSM content per footprint, error of the prior alone |
| `results/p2_<run>.json` | Test metrics per query (thermal, satellite crop), day / night, validation; `p2_<run>_frames.json` per frame; `.pt` weights |
| `results/p2_table.md` | Checks and all runs in one table |
| `results/p2_figs/` | `osm_preview.png`; `check_*`: frame, satellite at the metadata pose, OSM in the camera frame, overlay; run figures: frame, map window with probability, prior (+), truth (o), prediction (x), footprint |

### Files
| File | Role |
|---|---|
| `p2_prep.py` | Metadata → map coordinates, ground resolution, heading, fixed evaluation priors; writes the Overpass query |
| `p2_osm.py` | OSM (and optional Overture / DSM buildings) → 5-channel raster at 1 m |
| `p2_cache.py` | Images and frozen features: thermal frames, satellite crops at each frame's footprint, drone RGB (optional) |
| `p2_check.py` | Geometry and matcher checks, map content, prior-only error |
| `p2_train.py` | Adapter (from Phase 1) + BEV head + map encoder + FFT matcher; pose NLL + layout + NCE (+ JEPA) |
| `p2_report.py` | `results/p2_table.md` |
| `p2_common.py` | Footprint geometry, map windows, templates, matching, heads, metrics, figures |

Disk: with DINOv3 ViT-L at 640×512, features are about 31 GB per query kind (thermal + satellite ≈ 62 GB) and images about 12 GB per kind.

Tested end to end on synthetic Thermal-UAV-style data (CPU, random small backbones, metadata as JSON and as CSV): the geometry check picks the true pose (edge correlation 0.80, wrong settings ≤ 0.14), the matcher finds OSM-rendered queries to 0.7 m median, and all six runs train and report. Not yet run on the real data.
