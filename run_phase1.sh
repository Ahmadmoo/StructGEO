#!/usr/bin/env bash
set -e
B=configs/base.yaml
BACKBONES=${BACKBONES:-"ijepa dino_web dino_sat"}
mkdir -p results
python prep_metu.py --cfg $B
for bb in $BACKBONES; do
  for opt in same native; do python cache.py --cfg $B --backbone $bb --option $opt; done
done
python zeroshot.py --cfg $B
for r in r1_ijepa_same r2_ijepa_native r3_dino_same r4_dino_native; do
  python train.py --cfg $B configs/$r.yaml | tee results/$r.log
done
python report.py --cfg $B
