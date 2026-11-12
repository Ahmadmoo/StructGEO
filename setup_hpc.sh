#!/usr/bin/env bash
set -e
mkdir -p third_party weights
[ -d third_party/ijepa ] || git clone --depth 1 https://github.com/facebookresearch/ijepa third_party/ijepa
[ -d third_party/dinov3 ] || git clone --depth 1 https://github.com/facebookresearch/dinov3 third_party/dinov3
wget -c -P weights https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar
pip install -r requirements.txt
