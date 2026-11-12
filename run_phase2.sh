#!/usr/bin/env bash
# Phase 2: prep -> OSM raster -> images + features -> checks -> runs -> table.
# EXTRA=path/to/override.yaml adds one more config after configs/p2_base.yaml.
set -e
B="configs/base.yaml configs/p2_base.yaml ${EXTRA:-}"
RUNS=${RUNS:-"p2_m1_thermal p2_m2_sat_only p2_m3_no_p1 p2_a1_no_layout"}
mkdir -p results
python p2_prep.py --cfg $B
python - $B <<'PY'
import os, sys, yaml
c = {}
for p in sys.argv[1:]:
    c.update(yaml.safe_load(open(p)) or {})
miss = [p for p in (c.get("osm_files") or []) if not os.path.exists(p)]
if miss or not c.get("osm_files"):
    sys.exit(f"OSM file missing: {miss or 'osm_files is empty'}. Download it with the command printed above, then rerun.")
PY
python p2_osm.py --cfg $B
python p2_cache.py --cfg $B
python p2_check.py --cfg $B
for r in $RUNS; do
  python p2_train.py --cfg $B configs/$r.yaml | tee results/$r.log
done
python p2_report.py --cfg $B
