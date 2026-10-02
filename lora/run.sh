#!/bin/zsh
# run.sh <name> [train.py flags] — one LoRA round, unattended:
#   sample (make_dataset.py) -> train (train.py) -> held-out loss per checkpoint (valloss.py) -> repeat rate (eval.py)
# TIER=sonnet (default) or opus picks <tier>.yaml, that tier's traces, and the tier that samples and judges.
# The base view is LORA_BASE, else ~/lora/base/<tier>-4bit (README, Setup).
# Run on the Mac with the stack up (sampling uses the live tier through llama-swap), from the repo:
#   nohup lora/run.sh v1 --iters 100 --save-every 25 > ~/lora/runs/v1.out 2>&1 &
#   TIER=opus nohup lora/run.sh opus-v1 > ~/lora/runs/opus-v1.out 2>&1 &
# Logs: ~/lora/data/<name>.log, ~/lora/runs/<name>.log, ~/lora/runs/<name>-valloss.log, ~/lora/runs/<name>-eval.log
set -e
N=${1:?usage: [TIER=sonnet|opus] run.sh <name> [train.py flags]}; shift
TIER=${TIER:-sonnet}
cd "${0:A:h}"
CFG=$TIER.yaml
[[ -f $CFG ]] || { print -r -- "run.sh: no $CFG (tiers set up for training: ${(j:, :)${(@)${(f)$(print -l *.yaml)}%.yaml}})"; exit 1; }
BASE=${LORA_BASE:-$HOME/lora/base/$TIER-4bit}
SWAP=${LLAMA_SWAP:-http://127.0.0.1:8001}
PY=~/lora/.venv/bin/python
D=~/lora/data/$N
R=~/lora/runs/$N
[[ -e $R ]] && { print -r -- "run.sh: $R exists; pick a new name"; exit 1; }
mkdir -p ~/lora/data ~/lora/runs

print -r -- "$(date '+%T') sampling ($TIER) -> $D"
$PY make_dataset.py ~/.ultron/traces/*.json(N) ~/lora/traces/*.json(N) --out $D --tier $TIER > $D.log 2>&1
tail -1 $D.log

# Training, valloss and eval load the model next to the tiers: opus at 8k peaks ~36 GB and an
# opus+sonnet+haiku set rests at ~44 GB, so unload every tier but the helper (the next request reloads them).
print -r -- "$(date '+%T') unloading the big tiers for training"
CONF=~/.litellm/tiers.conf
HELPER=$(awk -F' *= *' '$1 == "helper" {print $2}' $CONF 2>/dev/null)
for t in ${(f)"$(awk -F'[][]' -v h="${HELPER:-haiku}" '/^\[/ && $2 != "routing" && $2 != h {print $2}' $CONF 2>/dev/null)"}; do
  curl -s -X POST $SWAP/api/models/unload/$t -o /dev/null || true
done

print -r -- "$(date '+%T') training ($CFG) -> $R"
$PY train.py --config $CFG --model $BASE --data $D --adapter-path $R "$@" > $R.log 2>&1 \
  || { print -r -- "TRAIN FAILED"; tail -5 $R.log; exit 1; }
grep -E "Val loss" $R.log | tail -8 || true

print -r -- "$(date '+%T') held-out loss per checkpoint"
$PY valloss.py $D/valid.jsonl $R --base $BASE > $R-valloss.log 2>&1
grep "val loss" $R-valloss.log || tail -5 $R-valloss.log

print -r -- "$(date '+%T') repeat rate, base vs final adapter"
$PY eval.py $D/valid.jsonl --adapter $R --base $BASE --n 4 > $R-eval.log 2>&1
grep -E "^(base|adapter)" $R-eval.log || tail -5 $R-eval.log
print -r -- "$(date '+%T') done"
