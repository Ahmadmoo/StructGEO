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
