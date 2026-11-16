#!/usr/bin/env bash
set -e
B="configs/base.yaml configs/p2_base.yaml configs/tuav3.yaml"
mkdir -p results_tuav3
python prep_tuav3.py --cfg $B
python train3.py --cfg $B --zeroshot
for r in t3_ts_nce t3_ts_lejepa t3_tsm_nce t3_tsm_lejepa t3_tsm_split t3_tsm_split_cnn; do
  python train3.py --cfg $B configs/$r.yaml | tee results_tuav3/$r.log
done
python report3.py --cfg $B
