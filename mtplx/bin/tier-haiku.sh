#!/bin/zsh
# tier-haiku.sh <port> — the "haiku" tier: small, fast, tool-calling; always loaded, so it also answers the
# media hook's yes/no checks. Started by llama-swap (~/.llama-swap/config.yaml, built from tiers.conf).
# Swap the model: change MODEL below.
#
# Runtime: mtplx (native MTP speculative decoding). Model: Qwen3.5-4B as an MTPLX Optimized-Speed pack:
# 4-bit body plus MTP head, text only, ~2.6 GB, 262k context. Parallel tool calls work.
# Get it before the first request:
#   uvx --from huggingface_hub hf download Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed \
#     --local-dir ~/.mtplx/models/Youssofal--Qwen3.5-4B-MTPLX-Optimized-Speed
# Want haiku to see images too? Forge your own pack from Qwen/Qwen3.5-4B with its vision tower
# (`mtplx forge build --repo Qwen/Qwen3.5-4B ...`, see `mtplx help forge`), point MODEL at it, and set
# vision = yes for this tier in tiers.conf.
#
# tiers.conf for this tier: vision = no, think_in_content = yes, chat_only = no.
# Sampling = Qwen3.5 model card "thinking, precise coding": temp 0.6 top_p 0.95 top_k 20 presence 0.0.
# Other official presets (clients can send them per request):
#   thinking general 1.0/0.95/20 presence 1.5 · non-thinking 0.7/0.8/20 presence 1.5 (+ enable_thinking=false)
# A presence penalty of 0.5-1.5 helps if the small model starts repeating itself.
PORT=${1:?usage: tier-haiku.sh <port> [extra mtplx serve flags]}; shift
EXTRA=("$@")         # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-haiku}  # llama-swap sets TIER from tiers.conf
MODEL="Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"   # mtplx resolves Hugging Face ids from ~/.mtplx/models

# Session-bank RAM cap. Each mtplx process otherwise plans for most of the Mac's memory, and three tiers would swap.
export MTPLX_SESSION_BANK_MAX_BYTES=5G     # a long agent session (~90k tokens) needs ~3.6 GB
export MTPLX_SESSION_BANK_IDLE_TTL_S=300   # drop idle warm sessions after 5 min (the SSD tier keeps them)
export MTPLX_LOOP_GUARD=1                  # in-reply repetition steering

exec "$HOME/.mtplx/bin/mtplx" serve \
  --host 127.0.0.1 --port "$PORT" \
  --no-auth `# loopback only, behind llama-swap; the network-facing auth is LiteLLM's master key` \
  --model "$MODEL" --model-id "$TIER" \
  --profile sustained `# the pack's recommended profile; MTP depth comes from its mtplx_runtime.json` \
  --tool-prompt-mode native `# the model's own trained tool format (qwen3_coder XML)` \
  --chat-template-profile tokenizer `# the bundled Qwen3.5 template` \
  --reasoning auto --preserve-thinking auto `# thinking on; history scoped to the active agent round` \
  --default-temperature 0.6 --default-top-p 0.95 --default-top-k 20 \
  --default-presence-penalty 0.0 \
  --ssd-session-cache on \
  --yes "${EXTRA[@]}"
