#!/bin/zsh
# tier-opus.sh <port> — the "opus" tier: a big MoE coder, thinking on, long context. Started and stopped by
# llama-swap (~/.llama-swap/config.yaml, built from tiers.conf). Swap the model: change MODEL below.
#
# Runtime: mtplx (native MTP speculative decoding). Model: Qwen3.6-35B-A3B as an MTPLX Optimized-Speed pack,
# ~21 GB, 262k context. MoE: 35B weights, 3B active per token, so it decodes like a small model. Hybrid
# attention: 10 of 40 layers are full attention with 2 KV heads, so the KV cache is ~20 KiB/token (256k is
# about 5 GiB) and it stays resident beside sonnet + haiku on 64 GB. Native tool calling.
# Get it before the first request (llama-swap's health check would time out on a download):
#   uvx --from huggingface_hub hf download Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed \
#     --local-dir ~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed
#
# tiers.conf for this tier: vision = no (this pack is text only; a pack with a vision tower can say yes),
# think_in_content = yes (the Qwen3.6 template reads earlier reasoning back from a <think> block),
# chat_only = yes (mtplx rejects images and reasoning items on /v1/responses).
# Sampling = Qwen model card "thinking, precise coding": temp 0.6 top_p 0.95 top_k 20.
PORT=${1:?usage: tier-opus.sh <port> [extra mtplx serve flags]}; shift
EXTRA=("$@")        # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-opus}  # llama-swap sets TIER from tiers.conf
MODEL="Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"   # mtplx resolves Hugging Face ids from ~/.mtplx/models

# Session-bank RAM cap. Each mtplx process otherwise plans for most of the Mac's memory, and the tiers would swap.
export MTPLX_SESSION_BANK_MAX_BYTES=6G
export MTPLX_SESSION_BANK_IDLE_TTL_S=300   # drop idle warm sessions after 5 min (the SSD tier keeps them)
export MTPLX_LOOP_GUARD=1                  # in-reply repetition steering

exec "$HOME/.mtplx/bin/mtplx" serve \
  --host 127.0.0.1 --port "$PORT" \
  --no-auth `# loopback only, behind llama-swap; the network-facing auth is LiteLLM's master key` \
  --model "$MODEL" --model-id "$TIER" \
  --profile sustained `# the pack's recommended profile; MTP depth comes from its mtplx_runtime.json` \
  --tool-prompt-mode native `# the model's own trained tool format (qwen3_coder XML)` \
  --reasoning auto --preserve-thinking auto `# thinking on; the Qwen3.6 template keeps it across turns` \
  --default-temperature 0.6 --default-top-p 0.95 --default-top-k 20 \
  --ssd-session-cache on \
  --yes "${EXTRA[@]}"
