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

---

## Experiments for the preview (branch `phase2-exp`)

Seven experiments on top of Phase 2. Why each one is there is in [PLAN.md §12](PLAN.md#12-poc-experiments-for-the-nov-17-preview-branch-phase2-exp). Most only evaluate the runs you already have (`results/p2_m1_thermal.pt`, `results/p2_m2_sat_only.pt`); four are new training runs.

### Setup
- Keep your `configs/p2_base.yaml` (paths, `p1_ckpt`). Experiment settings are in `configs/p2_exp.yaml`; `run_exp.sh` loads it after `p2_base.yaml`.
- `pip install matplotlib` (figures).
- Experiment 4b/4c: download the ESA WorldCover 2021 tile for the map (Changsha: `N27E111`), then set `worldcover_tifs` in `configs/p2_exp.yaml`. `python p2_landcover.py --cfg configs/base.yaml configs/p2_base.yaml configs/p2_exp.yaml configs/p2_m6_wc.yaml` prints the exact link if it is not set:
  `wget https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N27E111_Map.tif`
- Experiment 5 needs the drone RGB frames (`Visible/` folders and `Visible_info.csv` or `*_Visible.json`). The public Thermal-UAV release seems to hold thermal only; skip `rgb_prep` / `rgb_eval` if you do not have them.

### Run
```bash
bash run_exp.sh maps                 # first: map rasters osm_dsm, osm_wc, osm_dsm_wc (osm.npy is not touched)
bash run_exp.sh rgb_prep             # first, only with the RGB frames: pair frames, cache RGB in the thermal footprints

# then any of these in parallel, one GPU each, e.g. CUDA_VISIBLE_DEVICES=0 bash run_exp.sh trust_m1 &
bash run_exp.sh trust_m1             # 1 + 2 on m1: fusion and protection levels (evaluation only)
bash run_exp.sh trust_m2             # 1 + 2 on m2
bash run_exp.sh loc                  # 3: map-only distinctiveness, AUROC vs m1/m2 failures, area map
bash run_exp.sh m5_dsm               # 4a: m1 on OSM + DSM buildings        (training, like m1)
bash run_exp.sh m6_wc                # 4b: m1 on OSM + WorldCover           (training)
bash run_exp.sh m7_dsm_wc            # 4c: both                             (training)
bash run_exp.sh m2r_random_adapter   # 6: m2 with a random frozen adapter   (training)
bash run_exp.sh rgb_eval             # 5: m2 and m1 on thermal, satellite crops and drone RGB
bash run_exp.sh a2                   # 7: the JEPA run (backbone online, slower), so partial views can compare with it
bash run_exp.sh partial              # 7: partial views on m1, m2 (and a2 if trained); needs the backbone
bash run_exp.sh trust_wc loc_wc      # after m6_wc: fusion, protection levels and the map score on OSM + land cover
bash run_exp.sh p1_fillin            # 8: Phase 1 fill-in test for the JEPA predictor + figures (needs the backbone)

bash run_exp.sh report               # everything into results/exp_table.md
```
`M1=... M2=...` test other runs (e.g. `M1=p2_m5_dsm bash run_exp.sh trust_m1` after 4a). If the GPU runs out of memory, lower `eval_batch` (default 32) in `configs/p2_exp.yaml`; `fuse_every: 2` halves the time of `trust_*`.

### Outputs
| Path | What |
|---|---|
| `results/exp_table.md` | All experiments in tables (the source for slides) |
| `results/p1_fillin.md`, `.json` | 8: hidden spots, other sensor: JEPA predictor vs DINOv3 mask tokens vs copying the nearest visible spot, by hole depth |
| `results/p1_figs/fillin_<id>.png` | 8: row 1 JEPA (hidden image, truth, fill-in, hits), row 2 the same for a run without JEPA |
| `results/p1_figs/space_<id>.png` | 8: RGB and thermal features in one PCA: rows frozen DINOv3, then each run (one file per row too) |
| `results/p1_figs/match_<id>.png` | 8: where a thermal patch lands in the RGB image: frozen DINOv3 vs r3 |
| `results/exp_trust_<run>_thermal.json` | 1: error and share within 25 m per k (all / city / village / day / night), coverage at 95% precision; 2: protection-level factor, coverage, availability and hazard per alert limit, per map-score bin and per confidence bin. `_frames.json`: every frame and k |
| `results/exp_figs/fusion_<run>_thermal.png` | 1: share within 25 m and median error vs number of frames, city vs village |
| `results/exp_figs/stanford_<run>_thermal.png` | 2: Stanford diagram (error vs protection level), 1 frame vs most frames |
| `results/exp_loc_thermal.json`, `exp_figs/loc_area_thermal_test.png`, `exp_loc_thermal_test.tif` | 3: AUROC of map-only scores vs the runs' own confidence; area map (also as GeoTIFF for QGIS) with test frames that matched / failed |
| `results/exp_figs/loc_binned_<run>_thermal.png` | 3: failure rate per bin of map distinctiveness |
| `results/p2_m5_dsm.json`, `p2_m6_wc.json`, `p2_m7_dsm_wc.json`, `p2_m2r_random_adapter.json` | 4, 6: training runs, same format as the other Phase 2 runs (also in `p2_table.md` after `p2_report.py`) |
| `results/exp_evalq_<run>.json` | 5: the run on each query kind, per day / night and city / village |
| `results/exp_partial.json` | 7: full view vs part hidden at the backbone input (masked) vs part predicted vs part dropped only from the template (cut); a 0.0 in `partial_frac` checks the masked path against the full view |
| `results/exp_loc_thermal_osm_wc.json` (and `_test.tif`, figures) | 3 on the OSM + land-cover map (`loc_wc`); outputs on maps other than `osm` carry the map name |

### Files
| File | Role |
|---|---|
| `p2_trust.py` | 1 + 2: simulated GPS loss with odometry drift, fusion of the last k frames, protection levels with split-conformal factors, figures |
| `p2_loc.py` | 3: map-only distinctiveness per frame and over the area; AUROC vs runs |
| `p2_landcover.py` | 4: adds WorldCover land-cover channels to a map raster |
| `p2_pair.py` | 5: pairs thermal and RGB frames, finds the RGB-to-thermal pixel offset |
| `p2_evalq.py` | 5: evaluates a run on any query kinds (thermal, sat, vis2th) |
| `p2_partial.py` | 7: hides part of the frame at the backbone input; optional fill-in by the Phase 1 predictor |
| `p2_exp_report.py` | `results/exp_table.md` |
| `p2_exp_common.py`, `p2_exp_plots.py` | Run loading, matcher with any prior, probability-map tools, map-only score; figures |

Changes to Phase 2 files: maps are picked by `map_name` (default `osm`, so earlier runs load as before), the map encoder and layout head take the map's channel count, a new query kind `vis2th`, `p2_prep.py --only visible` (adds RGB frames without touching the thermal list or its caches), and `p2_cache.py` can limit a kind to some splits.

Tested end to end on the synthetic data (CPU, small random backbones): k = 1 reproduces each run's single-frame errors exactly, `p2_pair.py` recovers a planted 3 m camera offset (10 px), and every job above runs and reports. Not yet run on the real data.

## Phase 1 with three sensors (MS2)
Same claim with RGB, thermal and LiDAR: **can one adapter put all three sensors in one shared space, and does a JEPA-style objective do it without a contrastive loss?**

### Setup
Edit `configs/ms2.yaml`: `ms2_root` (the folder with `sync_data/`, `proj_depth/` and the `*_list.txt` split files; if the lists are missing, copy them from `MS2dataset/` in the MS2 GitHub repo) and `cache_dir`. DINOv3 paths come from `configs/base.yaml`.

```bash
bash run_ms2.sh      # prep → zero-shot → 5 runs → SIGReg weight sweep → results_ms2/table.md
```

### Check after `prep_ms2.py`
- `cache_dir/ms2_meta.json`:
  - `depth_err_median` (per sensor) should be close to 0, well under 0.01. A larger value means the extrinsics are applied in the wrong direction.
  - `depth_agree` is the share of projected points that agree with the other camera's own depth. It should be high.
  - `n_frames` per split. Feature cache size is about 4 MB per frame with ViT-L (three sensors), so 5,000 frames ≈ 20 GB. Raise `frame_step` to use less.
- `cache_dir/figs/corr_*.png`: thermal | RGB | LiDAR. Dots of the same color must sit on the same object in all three images.

### How the data is built
- **Reference grid:** thermal tokens. Each thermal token gets a depth from MS2's `depth_filtered` (median of the patch; skipped if too few points or the depth spreads too much).
- **Token matching:** thermal token + depth → 3D → the other camera with MS2's calibration. A match is kept only if that camera's own projected depth agrees (`occ_tol`), which removes occluded spots.
- **LiDAR as an image:** the single-scan depth (`lidar_depth: depth`) and intensity, rendered in the NIR camera (`lidar_plane: nir`), holes filled. Channels: inverse depth, intensity, inverse depth. It sits in its own camera, so it is not on the thermal grid.
- **Scale:** RGB and NIR images are resized so one token covers about the same angle as one thermal token (`match_scale`).

### Runs
| Run | Loss | Question |
|---|---|---|
| `zs_dino_web_L1` | none (raw DINOv3) | How shared is the frozen space already? |
| `m3_nce` | NCE over all sensor pairs + variance | The Phase 1 anchor, as the baseline |
| `m3_inv` | invariance + SIGReg | Do the sensors meet with no negatives (LeJEPA-style)? |
| `m3_lejepa` | invariance + hidden-spot prediction + SIGReg | Full JEPA: same spot → one embedding; hidden spots predicted from another sensor |
| `m3_hid` | hidden-spot prediction of the shared target + SIGReg | Is prediction alone enough? |
| `m3_hid_own` | hidden-spot prediction of each sensor's own embedding (sensor-aware queries) + SIGReg | The Phase 1 "two dialects" setup |
| `m3_inv_sig*` | `m3_inv` with SIGReg weight 0.01 / 0.05 | Sensitivity to the one knob |

- **Invariance:** at each matched spot, every sensor's embedding is pulled to the mean of the sensors there.
- **SIGReg:** keeps each sensor's embeddings close to an isotropic Gaussian, so they cannot collapse (no EMA teacher, no negatives).
- **Hidden spots:** blocks of the thermal grid are hidden. One sensor is picked as context, and its tokens over those spots are masked in the DINOv3 pass (no leak). A predictor gets that sensor's visible tokens at their thermal positions and predicts the shared embedding at the hidden spots. The target is stop-gradient (`hid_detach`); without it, the hidden loss collapsed the space in tests.

### Outputs
| Path | What |
|---|---|
| `results_ms2/*.json` | hit@1 and median rank for all 6 sensor directions, per condition (day / night / rain) |
| `results_ms2/table.md` | All runs in one table |
| `results_ms2/figs/*.png` | Shared PCA over the three sensors: image + feature colors per sensor |

### Files
| File | Role |
|---|---|
| `prep_ms2.py` | Frames, images (RGB, thermal, LiDAR), depth-based token matching, checks, feature cache |
| `train3.py` | Adapter (+ predictor) training with NCE / invariance / SIGReg / hidden-spot losses; evaluation; `--zeroshot` for raw features |
| `report3.py` | `results_ms2/table.md` |

Tested end to end on a synthetic MS2-format dataset with a random small DINOv3 on CPU. Token matching gave a 0.99 / 0.98 pixel correlation for RGB / LiDAR, and flipped extrinsics were caught. With longer training on that toy data, `m3_nce`, `m3_inv` and `m3_lejepa` learned the alignment, while `m3_hid` and `m3_hid_own` stayed near chance. Those numbers say nothing about real data. It has not run on real MS2 yet, so expect small fixes on the first run.

## The map as a third view (Thermal-UAV)
Thermal, satellite RGB and the OSM map, all seen straight down from the same footprint, are three views of one place. **Can one adapter put all three in one shared space with the LeJEPA-style objective, so that matching a frame to the map works like matching two sensors?**

### Setup
Nothing new to download: this reuses the Phase 2 cache (`frames.json`, `p2_img_{thermal,sat}.npy`, `p2_feat_*_{thermal,sat}.npy` from `p2_cache.py`, and the map raster from `p2_osm.py` / `p2_landcover.py`). Use the same Phase 2 config you ran with, and set `cache_dir` in `configs/tuav3.yaml`. `map_name: osm_wc` uses OSM + land cover; set `osm` for OSM only.

```bash
bash run_tuav3.sh    # prep → zero-shot → 6 runs → results_tuav3/table.md
```

### Check after `prep_tuav3.py`
- `cache_dir/figs/align_*.png`: thermal | satellite | map | thermal + map | satellite + map. Roads and buildings must line up.
- `tuav_meta.json` → `fwd_valid`: the share of tokens with map content (`map_valid_dilate` grows it by one token). With land cover it is close to 1.
- Disk: the rendered map images (`tuav_img_map.npy`) are the size of `p2_img_thermal.npy`, and their DINOv3 features the size of `p2_feat_*_thermal.npy`.

### How the map view is built
- **Rendering:** the map raster (1 m) is warped into each thermal frame with the Phase 2 footprint geometry (`img2world`). Raw layers are kept at 1/4 resolution (`map_down`) for the CNN option. A colored image is made at full size for DINOv3.
- **Same grid:** all three views sit on the thermal token grid, so no matching step is needed.
- **Valid spots:** a map token counts only if a layer has content near it. The satellite crop counts where it lies inside the reference image.

### Runs
| Run | Views | Loss | Question |
|---|---|---|---|
| `zs_dino_web_L1` | thermal, sat, map | none | How shared is raw DINOv3, including on a rendered map? |
| `t3_ts_nce` | thermal, sat | NCE + variance | Baseline without the map |
| `t3_ts_lejepa` | thermal, sat | invariance + hidden spots + SIGReg | JEPA without the map |
| `t3_tsm_nce` | thermal, sat, map | NCE over all pairs | Contrastive with the map |
| `t3_tsm_lejepa` | thermal, sat, map | invariance + hidden spots + SIGReg | The map as a full view |
| `t3_tsm_split` | thermal, sat, map | same, split embedding | The map shares only the first 128 dims (structure); the other 128 are shared by the sensors only (appearance) |
| `t3_tsm_split_cnn` | thermal, sat, map | same, map through a small trained CNN on the raw layers | Is a trained map encoder better than DINOv3 on a rendered map? |

- **Split embedding:** `struct_dim` and `struct_only`. Pairs that include the map are scored on the shared dims only.
- **`sig_valid: [map]`:** SIGReg on the map uses only tokens with map content.
- **Hidden spots with the map as context:** the map tokens are masked in the DINOv3 pass. With the CNN, the hidden region of the layers is set to zero.

Outputs are the same as for MS2, with hit@1 per sensor pair and per place and condition (city/village × day/night).

Tested end to end on a synthetic Phase 2 cache with a random small DINOv3 on CPU: the rendered map lines up with the frames, and every run trains and reports. Those numbers say nothing about real data. It has not run on the real Thermal-UAV cache yet.
