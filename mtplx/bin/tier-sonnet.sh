#!/bin/zsh
# tier-sonnet.sh <port> — the "sonnet" tier: fast tool-calling coder that can see images. Started and stopped
# by llama-swap (~/.llama-swap/config.yaml, built from tiers.conf). Swap the model: change MODEL below.
#
# Runtime: mtplx (native MTP speculative decoding). Model: Qwen3.5-9B as an MTPLX Optimized-Speed pack:
# 4-bit body, MTP head and vision tower, ~8.7 GB, 262k context. Qwen3.5's gated-DeltaNet layers keep the
# KV cache small, so the full context is affordable next to the other tiers. Native tool calling.
# Get it before the first request (llama-swap's health check would time out on a download):
#   uvx --from huggingface_hub hf download Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed \
#     --local-dir ~/.mtplx/models/Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed
#
# tiers.conf for this tier: vision = yes, think_in_content = yes, chat_only = no.
# Sampling = Qwen3.5 model card "thinking, precise coding": temp 0.6 top_p 0.95 top_k 20 presence 0.0.
# Thinking stays ON. With it off, the 9B stopped emitting tool calls under pressure and wrote the next
# command in a ```bash block instead (replays: thinking off 0/3 tool calls, on 4/4, at 70-300 thinking
# tokens). Plain prompts sometimes think for 4k tokens, so tool turns get a hard cap below.
PORT=${1:?usage: tier-sonnet.sh <port> [extra mtplx serve flags]}; shift
EXTRA=("$@")          # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-sonnet}  # llama-swap sets TIER from tiers.conf
MODEL="Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"   # mtplx resolves Hugging Face ids from ~/.mtplx/models

# Session-bank RAM cap. Each mtplx process otherwise plans for most of the Mac's memory, and three tiers would swap.
export MTPLX_SESSION_BANK_MAX_BYTES=6G
export MTPLX_SESSION_BANK_IDLE_TTL_S=300   # drop idle warm sessions after 5 min (the SSD tier keeps them)
export MTPLX_LOOP_GUARD=1                  # in-reply repetition steering
export MTPLX_THINKING_BUDGET=2048          # force-close </think> at 2048 tokens on tool-carrying requests

exec "$HOME/.mtplx/bin/mtplx" serve \
  --host 127.0.0.1 --port "$PORT" \
  --no-auth `# loopback only, behind llama-swap; the network-facing auth is LiteLLM's master key` \
  --model "$MODEL" --model-id "$TIER" \
  --profile sustained `# the pack's recommended profile; MTP depth comes from its mtplx_runtime.json` \
  --tool-prompt-mode native `# the model's own trained tool format (qwen3_coder XML)` \
  --reasoning auto --preserve-thinking auto `# thinking on; history scoped to the active agent round` \
  --default-temperature 0.6 --default-top-p 0.95 --default-top-k 20 \
  --ssd-session-cache on \
  --yes "${EXTRA[@]}"
