#!/usr/bin/env bash
# PoC experiments (README "Experiments"). Usage:  bash run_exp.sh <job> [<job> ...]
# Run "maps" (and "rgb_prep" if you have the drone RGB images) first; every other job is independent and can run in
# parallel, one GPU each. EXTRA=path/to/override.yaml adds one more config at the end. M1 / M2 name the runs to test.
set -eo pipefail
B="configs/base.yaml configs/p2_base.yaml configs/p2_exp.yaml ${EXTRA:-}"
M1=${M1:-p2_m1_thermal}
M2=${M2:-p2_m2_sat_only}
[ $# -gt 0 ] || { grep -E '^  [a-z_0-9|]+\)' "$0" | sed 's/).*//'; exit 1; }
R=$(python - $B <<'PY'
import sys, yaml
c = {}
for p in sys.argv[1:]:
    c.update(yaml.safe_load(open(p)) or {})
print(c.get('out_dir', 'results'))
PY
)
mkdir -p $R
for job in "$@"; do
  echo "== $job"
  case $job in
  maps)              # 4: map rasters for the map variants (osm.npy is not touched)
    python p2_osm.py --cfg $B configs/p2_m5_dsm.yaml
    python p2_landcover.py --cfg $B configs/p2_m6_wc.yaml
    python p2_landcover.py --cfg $B configs/p2_m7_dsm_wc.yaml ;;
  rgb_prep)          # 5: drone RGB frames -> pairs -> RGB views in the thermal footprints (valid + test)
    python p2_prep.py --cfg $B --only visible
    python p2_pair.py --cfg $B
    python p2_cache.py --cfg $B --kinds vis2th ;;
  trust_m1)          # 1 + 2: sequence fusion and protection levels
    python p2_trust.py --cfg $B --ckpt $R/$M1.pt | tee $R/exp_trust_$M1.log ;;
  trust_m2)
    python p2_trust.py --cfg $B --ckpt $R/$M2.pt | tee $R/exp_trust_$M2.log ;;
  loc)               # 3: map-only distinctiveness vs failures, area map
    python p2_loc.py --cfg $B --ckpt $R/$M1.pt --runs $M1 $M2 | tee $R/exp_loc.log ;;
  trust_wc)          # 1 + 2 on the land-cover run (after m6_wc)
    python p2_trust.py --cfg $B --ckpt $R/p2_m6_wc.pt | tee $R/exp_trust_p2_m6_wc.log ;;
  loc_wc)            # 3 on the OSM + land-cover map, vs the land-cover run and m1 (after m6_wc)
    python p2_loc.py --cfg $B --ckpt $R/p2_m6_wc.pt --runs p2_m6_wc $M1 | tee $R/exp_loc_wc.log ;;
  m5_dsm|m6_wc|m7_dsm_wc|m2r_random_adapter)   # 4 + 6: training runs
    python p2_train.py --cfg $B configs/p2_$job.yaml | tee $R/p2_$job.log ;;
  rgb_eval)          # 5: the same runs on thermal, satellite crops and drone RGB
    python p2_evalq.py --cfg $B --ckpt $R/$M2.pt --queries thermal sat vis2th | tee $R/exp_evalq_$M2.log
    python p2_evalq.py --cfg $B --ckpt $R/$M1.pt --queries thermal sat vis2th | tee $R/exp_evalq_$M1.log ;;
  a2)                # 7: the JEPA run, so partial views can compare with it (backbone online, slower)
    python p2_train.py --cfg $B configs/p2_a2_jepa.yaml | tee $R/p2_a2_jepa.log ;;
  partial)           # 7: partial views (uses a2 too if it has been trained)
    CK="$R/$M1.pt $R/$M2.pt"
    [ -f $R/p2_a2_jepa.pt ] && CK="$CK $R/p2_a2_jepa.pt"
    python p2_partial.py --cfg $B --ckpts $CK | tee $R/exp_partial.log ;;
  p1_fillin)         # 8: Phase 1 fill-in, what only the JEPA predictor can do, + figures (needs r3, a1, a2 .pt)
    python p1_fillin.py --cfg configs/base.yaml ${EXTRA:-} | tee $R/p1_fillin.log ;;
  report)            # all of the above into results/exp_table.md
    python p2_exp_report.py --cfg $B ;;
  *) echo "unknown job $job"; exit 1 ;;
  esac
done
