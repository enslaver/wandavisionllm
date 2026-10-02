#!/bin/zsh
# tier-fable.sh <port> — the "fable" tier: a dense 27B for the hardest problems, thinking on. Started and stopped
# by llama-swap (~/.llama-swap/config.yaml, built from tiers.conf). Swap the model: change MODEL below.
# It doesn't share the Mac with opus (tiers.conf resident: loading one unloads the other). Optional: on
# smaller Macs drop the [fable] section from tiers.conf.
#
# Runtime: TensorFold (github.com/ashhart/TensorFold, venv at ~/.tensorfold/venv). Its MLX lane kernels and
# DFlash2 draft trees decode a 27B about twice as fast as plain MTP; output is byte-identical to serial decoding.
# Model: Qwen3.8-27B, MLX 4-bit groups of 64 (TensorFold's reference checkpoint), ~16 GB, 262k context.
# Get it before the first request (llama-swap's health check would time out on a download):
#   ~/.tensorfold/venv/bin/tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
# The DFlash2 drafter (~3.9 GB) is required: without it TensorFold still prints "drafts: on" but decodes about
# one token a round, so this script refuses to start until it is pulled.
#
# tiers.conf for this tier: vision = no (TensorFold serves images only with its [vision] extra and --vision),
# think_in_content = no (the Qwen3.8 template reads only reasoning_content), chat_only = yes
# (TensorFold 0.3.4.x has no /v1/responses). TensorFold has no /v1/mtplx/snapshot, so Wanda and the
# admission hook count this tier's work from LiteLLM's in-flight list.
# TensorFold's output (startup lines, one `done req-…` line per request with prompt/cached/ttft) also goes,
# timestamped, to ~/.tensorfold/<tier>.log, minus /health and snapshot polls; llama-swap only keeps a small
# ring buffer.
PORT=${1:?usage: tier-fable.sh <port> [extra tensorfold serve flags]}; shift
EXTRA=("$@")        # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-fable}  # llama-swap sets TIER from tiers.conf
MODEL="Vontra/Qwen3.8-27B-MLX-4bit"   # Hugging Face id; TensorFold reads the Hugging Face cache
DRAFTER="z-lab/Qwen3.8-27B-DFlash2"   # what TensorFold's --drafter auto picks for this family; it never downloads it
TF="$HOME/.tensorfold/venv"
LOG="$HOME/.tensorfold/$TIER.log"

export TENSORFOLD_NO_UPDATE_CHECK=1

[[ -f $LOG ]] && (( $(stat -f %z "$LOG") > 50000000 )) && mv -f "$LOG" "$LOG.1"
zmodload zsh/datetime
exec > >(
  exec 3>>"$LOG"
  while IFS= read -r line; do
    print -r -- "$line"                          # llama-swap still gets every line (/logs/stream/upstream)
    # polls are ~90% of the output: llama-swap's /health (200) and Wanda's /v1/mtplx/snapshot (404 here)
    [[ $line == *'"GET /health HTTP/'*'" 200 '* || $line == *'"GET /v1/mtplx/snapshot HTTP/'* ]] && continue
    strftime -s ts '%F %T' $EPOCHSECONDS
    print -r -u3 -- "$ts $line"
  done
) 2>&1

# the same check TensorFold's auto mode makes (cached + weights complete); 0.2 s
"$TF/bin/python" -c 'import sys; from tensorfold import hub; p = hub.cached(sys.argv[1]); sys.exit(0 if p and hub._cached_weights_complete(p) else 1)' "$DRAFTER" \
  || { print -r -- "tier-$TIER: draft model $DRAFTER missing or incomplete; not serving without it (fix: $TF/bin/tensorfold pull $DRAFTER)"; exit 1; }

exec "$TF/bin/tensorfold" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" `# loopback only, behind llama-swap` \
  --name "$TIER" \
  --context 262144 `# prompt + reply; tiers.conf caps input at 200000 so 32k+ is left for thinking + answer` \
  --max-tokens 32768 `# only the default when a request sets none; TensorFold clamps every request to context - prompt` \
  --thinking --reasoning-effort medium `# medium adds no system text, so system-block snapshots match` \
  --temperature 0.6 --top-p 0.95 --top-k 20 `# Qwen "thinking, precise coding" preset` \
  --prompt-cache-gib 4 --mlx-cache-gib 6 `# not a RAM cap past ~61k tokens: KV is 64 KiB/token and TF keeps the newest entry (250k = 15.4 GiB)` \
  "${EXTRA[@]}"
