#!/usr/bin/env bash
# unsloth_queue.sh NAME [MAX_HOURS] [unsloth_vision.py flags] — one unattended haiku image run on the GPU box
# (Git Bash on Windows, or Linux), from the repo clone:
#   nohup bash lora/unsloth_queue.sh haiku-v1 8 --data ~/lora/data/haiku-v1.jsonl > ~/lora/haiku-v1.queue.log 2>&1 &
# 1. Probes 12 steps for sec/step and peak VRAM. It waits for the GPU first: no other run holding ~/lora/gpu.lock,
#    and --min-free-gb of VRAM free (unload models in Unsloth Studio, ComfyUI or Ollama).
# 2. Trains 1 epoch (flags after MAX_HOURS win, e.g. --epochs 2), capped at MAX_HOURS (default 8) by the probe's rate.
#    A crash (the intermittent Windows Triton 0xC0000005) retries up to 3 times, resuming from the last checkpoint.
# 3. Stages the result for the Mac and writes STAGED last; the Mac waits for that marker before copying the merge
#    to its ~/lora/merges/NAME.
#    Without LORA_STAGE, the merge stays in ~/lora/NAME/merged. With LORA_STAGE set to a folder the Mac can read
#    (a share both mount, say), merged + adapter are copied to $LORA_STAGE/NAME. README, "Ship it on the Mac".
# UNSLOTH_PY overrides the python (default: Unsloth Studio's venv).
set -u
NAME=${1:?usage: unsloth_queue.sh NAME [MAX_HOURS] [unsloth_vision.py flags]}; shift
HOURS=8
case ${1:-} in [0-9]*) HOURS=$1; shift ;; esac
OUT=$HOME/lora/$NAME
TRAIN="$(cd "$(dirname "$0")" && pwd)/unsloth_vision.py"
export PYTHONIOENCODING=utf-8 TORCHDYNAMO_DISABLE=1 UNSLOTH_COMPILE_DISABLE=1
log() { echo "$(date '+%F %T') $*"; }

if [ -n "${UNSLOTH_PY:-}" ]; then
  PY=$UNSLOTH_PY
else
  for PY in "$HOME/.unsloth/studio/unsloth_studio/Scripts/python.exe" "$HOME/.unsloth/studio/unsloth_studio/bin/python"; do
    [ -x "$PY" ] && break
  done
fi
[ -x "$PY" ] || { log "no python at $PY: install Unsloth Studio or set UNSLOTH_PY"; exit 1; }
[ -e "$OUT/merged" ] && { log "$OUT/merged exists; pick a new name"; exit 1; }
mkdir -p "$OUT"

log "probe (waits for the GPU)"
"$PY" -X faulthandler "$TRAIN" "$OUT/probe" --4bit --wait-gpu --probe 12 "$@" > "$OUT/probe.log" 2>&1
sps=$(grep -o 'probe_sec_per_step [0-9.]*' "$OUT/probe.log" | awk '{print $2}')
grep -E "gpu guard|rows,|peak VRAM|probe_sec" "$OUT/probe.log"
[ -n "$sps" ] || { log "probe failed; tail of $OUT/probe.log:"; tail -20 "$OUT/probe.log"; exit 1; }

for try in 1 2 3; do
  resume=""; ls -d "$OUT"/ckpt/checkpoint-* >/dev/null 2>&1 && resume="--resume"
  log "train try $try $resume (sec/step $sps, max ${HOURS}h)"
  "$PY" -X faulthandler "$TRAIN" "$OUT" --4bit --wait-gpu --epochs 1 --max-hours "$HOURS" --sec-per-step "$sps" \
    $resume "$@" >> "$OUT/train.log" 2>&1 && break
  log "train exited non-zero; tail:"; tr '\r' '\n' < "$OUT/train.log" | grep -v '^\s*$' | grep -v '^0x' | tail -5
  [ "$try" = 3 ] && { log "giving up"; exit 1; }
done
tr '\r' '\n' < "$OUT/train.log" | grep -E "train_loss|merged ->" | tail -2

DEST=$OUT
if [ -n "${LORA_STAGE:-}" ]; then
  DEST=$LORA_STAGE/$NAME
  log "staging to $DEST"
  rm -f "$DEST/STAGED"
  mkdir -p "$DEST/adapter"
  cp -R "$OUT/merged/." "$DEST/" && rm -rf "$DEST/.cache" && cp -R "$OUT/adapter/." "$DEST/adapter/" \
    || { log "copy to $DEST failed"; exit 1; }
fi
# STAGED last: a copy in flight can show files at full size, so the Mac gates its pickup on this marker, not on sizes.
{ date '+%F %T'; ls -l "$DEST" | grep -v '^total'; } > "$DEST/STAGED" && log "DONE $NAME staged at $DEST (STAGED written)"
