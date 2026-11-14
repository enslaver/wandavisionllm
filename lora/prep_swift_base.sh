#!/bin/zsh
# prep_swift_base.sh - build ~/lora/base/sonnet-swift-1.5-27b-4bit for sonnet.yaml (~20-40 min, 55 GB of BF16
# downloaded into the shared HF cache, ~15 GB written). Safe to rerun; remove the BF16 afterwards with
#   hf cache rm model/ukisai/Swift-1.5-Qwen3.8-27b --yes
set -e
OUT=$HOME/lora/base/sonnet-swift-1.5-27b-4bit
[[ -f $OUT/config.json ]] && { print -r -- "$OUT exists"; exit 0; }
PY=$HOME/lora/.venv/bin/python
$PY -m mlx_lm.convert --hf-path ukisai/Swift-1.5-Qwen3.8-27b --mlx-path "$OUT" -q --q-bits 4 --q-group-size 64
print -r -- "built $OUT"
