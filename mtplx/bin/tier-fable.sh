#!/bin/zsh
# tier-fable.sh <port> — the "fable" tier: a dense 27B for the hardest problems, thinking on, no vision. Started and
# stopped by llama-swap (~/.llama-swap/config.yaml, built from tiers.conf; ready check /ready). Swap the model: change
# MODEL below. It doesn't share the Mac with opus (tiers.conf resident: loading one unloads the other). Optional: on
# smaller Macs drop the [fable] section from tiers.conf.
#
# Runtime: Splash (github.com/incoai/splash, `brew install incoai/tap/splash`, 1.3.2). Model: Qwen3.8-27B, MLX
# 4-bit, with DFlash2 speculative decoding (Splash picks the family's draft model itself). The tier ran on
# TensorFold first; that script is tier-fable-tensorfold.sh (rollback steps in its header).
# What differs from TensorFold:
#  - reasoning_effort is read per request (it reaches the chat template as enable_thinking + reasoning_effort);
#    there is no thinking_budget. --default-reasoning-effort medium = TensorFold's --reasoning-effort medium.
#  - reasoning_content in assistant history is read back (LiteLLM keeps it for this tier: tiers.conf
#    reasoning_content = yes).
#  - no server-side sampling defaults: temperature defaults to 1.0 (TensorFold ran 0.6), so the LiteLLM
#    admission hook sends temperature 0.6 and max_tokens 32768 when a request sets none. top_p 0.95 / top_k 20
#    are Splash's own defaults.
#  - /health is 200 as soon as the server answers, /ready only once the engine is up: llama-swap checks /ready.
#  - own status/metrics: GET /status, /metrics (Prometheus); the prefix cache can live on SSD across restarts.
#
# tiers.conf for this tier: vision = no, chat_only = yes, reasoning_content = yes, reasoning_effort = yes.
# Splash has no /v1/mtplx/snapshot, so Wanda and the admission hook count this tier's work from LiteLLM's
# in-flight list.
# Pin the model with --revision <commit> once you have it cached: no Hub lookups, no surprise model change when
# the repo moves.
PORT=${1:?usage: tier-fable.sh <port> [extra splash serve flags]}; shift
EXTRA=("$@")         # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-fable}  # llama-swap sets TIER from tiers.conf
MODEL="mlx-community/Qwen3.8-27B-4bit"   # Hugging Face id; Splash reads the Hugging Face cache
SPLASH=${SPLASH:-/opt/homebrew/bin/splash}
LOG="$HOME/.splash/$TIER.log"

mkdir -p "$HOME/.splash"
[[ -f $LOG ]] && (( $(stat -f %z "$LOG") > 50000000 )) && mv -f "$LOG" "$LOG.1"
zmodload zsh/datetime
exec > >(
  exec 3>>"$LOG"
  while IFS= read -r line; do
    print -r -- "$line"                          # llama-swap still gets every line (/logs/stream/upstream)
    [[ $line == *'GET /v1/mtplx/snapshot'* ]] && continue   # Wanda's per-second poll, a 404 Error line each
    strftime -s ts '%F %T' $EPOCHSECONDS
    print -r -u3 -- "$ts $line"
  done
) 2>&1

exec "$SPLASH" serve \
  --model "$MODEL" \
  --language-only `# the checkpoint has no vision tower; images reroute to the vision tier in ultron_admit` \
  --host 127.0.0.1 --port "$PORT" --no-webui `# loopback only, behind llama-swap` \
  --served-model-name "$TIER" `# the id llama-swap forwards` \
  --default-reasoning-effort medium `# medium adds no system text, so system-block snapshots match (as TensorFold)` \
  --thinking-token-budget 16384 `# cap on thinking; a request's thinking_token_budget overrides it` \
  --thinking-end-text $'I have thought about this long enough; the answer follows.\n</think>' `# turns the model toward its answer when the budget runs out (<=64 tokens)` \
  --idle-release off `# llama-swap owns the lifecycle (ttl 600); Splash would unwire the weights after 10 min itself` \
  --max-memory 34G `# Metal budget: ~18.6 GB fixed + KV at 32 KiB/token (256K = 8.5 GB); lower it on a smaller Mac, since sonnet and haiku share the box` \
  --max-cache-disk 48G --persistent-cache `# prompts survive llama-swap unloads and restarts (~/Library/Caches/Splash)` \
  "${EXTRA[@]}"
