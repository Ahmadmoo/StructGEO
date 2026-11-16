#!/usr/bin/env bash
set -e
B="configs/base.yaml configs/ms2.yaml"
mkdir -p results_ms2
python prep_ms2.py --cfg $B
python train3.py --cfg $B --zeroshot
for r in m3_nce m3_inv m3_lejepa m3_hid m3_hid_own; do
  python train3.py --cfg $B configs/$r.yaml | tee results_ms2/$r.log
done
for ws in 0.01 0.05; do
  printf "name: m3_inv_sig$ws\nw_sig: $ws\n" > results_ms2/sig$ws.yaml
  python train3.py --cfg $B configs/m3_inv.yaml results_ms2/sig$ws.yaml | tee results_ms2/m3_inv_sig$ws.log
done
python report3.py --cfg $B
