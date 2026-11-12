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
