#!/bin/zsh
# tier-sonnet.sh <port> — the "sonnet" tier: Swift 1.5 Qwen3.8-27B (ukisai, GSQ-RCO IQ3_XXS GGUF with an MTP head,
# ~10.4 GB), everyday coding and tool calling. Served by Splash like the opus tier (see tier-opus.sh for how Splash
# differs from mtplx). Text only: it has no vision tower, so tiers.conf says vision = no and an image aimed at
# sonnet is answered by the [routing] vision tier. Started and stopped by llama-swap (ttl 600 s, check_endpoint
# /ready). Swap the model: change MODEL below. The mtplx alternate (Qwen3.5-9B, sees images) is tier-sonnet-mtplx.sh;
# its header says how to switch back.
#
# On the reference Mac Swift scored 8/8 reasoning, 8/8 tools, 8/8 coding at 109 tok/s on litellm/bench.py; the
# Qwen3.5-9B it replaced scored 4/8, 8/8, 3/8 at 146 tok/s, and 3 of 10 runs of a stall test hit max_tokens
# where Swift hit it once.
# Differences from an mtplx tier: reasoning_effort is read by Splash itself (tiers.conf reasoning_effort = yes; no
# mtplx_effort.py wrapper, no loop guard, no session bank), reasoning_content rides in the history
# (reasoning_content = yes), there are no server-side sampling defaults (the admission hook sends temperature 0.6,
# max_tokens 32768 and repetition_penalty 1.1 when a request sets none), and the context is limited by --max-memory
# (about 128K at 16G; the cache is not kept on disk).
# Get the model before the first request (llama-swap's health check would time out on a download):
#   uvx --from huggingface_hub hf download ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF
# Pin the model with --revision <commit> once you have it cached: no Hub lookups, no surprise model change when
# the repo moves.
PORT=${1:?usage: tier-sonnet.sh <port> [extra splash serve flags]}; shift
EXTRA=("$@")           # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-sonnet}   # llama-swap sets TIER from tiers.conf
MODEL="ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_XXS-mtp"   # Hugging Face repo and GGUF variant; Splash reads the Hugging Face cache
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
  --language-only `# the model has no vision tower; images reroute to the vision tier in ultron_admit` \
  --host 127.0.0.1 --port "$PORT" --no-webui `# loopback only, behind llama-swap` \
  --served-model-name "$TIER" `# the id llama-swap forwards` \
  --default-reasoning-effort medium \
  --thinking-token-budget 16384 `# cap on thinking, as opus; a request's thinking_token_budget overrides it` \
  --thinking-end-text $'I have thought about this long enough; the answer follows.\n</think>' `# turns the model toward its answer when the budget runs out (<=64 tokens)` \
  --idle-release off `# llama-swap owns the lifecycle (ttl 600)` \
  --max-memory 16G `# Metal budget: ~11 GB weights + KV at 32 KiB/token (about 128K context); opus (<=34G) and haiku (~8 GB) share the box` \
  "${EXTRA[@]}"
