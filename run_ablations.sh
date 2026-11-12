#!/usr/bin/env bash
set -e
B=configs/base.yaml
python cache.py --cfg $B configs/a3_last4.yaml
for r in a1_jepa_only a2_nce_only a3_last4; do
  python train.py --cfg $B configs/$r.yaml | tee results/$r.log
done
python report.py --cfg $B
