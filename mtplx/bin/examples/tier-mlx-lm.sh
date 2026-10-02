#!/bin/zsh
# tier-mlx-lm.sh <port> — a tier on mlx-lm's server (MLX safetensors, text only). Copy it over a tier script
# (e.g. mtplx/bin/tier-haiku.sh), set TIER's default and MODEL, then ./deploy.py push mtplx.
#
# Install: uv tool install mlx-lm   (puts mlx_lm.server in ~/.local/bin)
# Model: Qwen3.5-4B, mlx-community 4-bit (~3.1 GB). Download it first:
#   uvx --from huggingface_hub hf download mlx-community/Qwen3.5-4B-MLX-4bit
#
# mlx_lm.server answers only to the model it was started with under the name "default_model" (any other
# name in a request is loaded as a new model path). Set upstream_model = default_model for this tier in
# tiers.conf, so llama-swap rewrites the model name on the way in (its useModelName).
# tiers.conf for this tier: vision = no (mlx-lm is text only), think_in_content = yes (the
# Qwen3.5 template reads earlier reasoning back from a <think> block), chat_only = yes (no /v1/responses).
PORT=${1:?usage: tier-mlx-lm.sh <port> [extra mlx_lm.server flags]}; shift
EXTRA=("$@")         # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-haiku}  # llama-swap sets TIER from tiers.conf (mlx_lm.server itself doesn't use it)
MODEL="mlx-community/Qwen3.5-4B-MLX-4bit"   # Hugging Face id or a local folder

exec "${MLX_LM_SERVER:-$HOME/.local/bin/mlx_lm.server}" \
  --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" `# loopback only, behind llama-swap` \
  --max-tokens 32768 `# the server's default reply cap is 512` \
  --temp 0.6 --top-p 0.95 --top-k 20 `# Qwen "thinking, precise coding" preset` \
  --prompt-cache-bytes 4GB `# RAM cap for cached conversation prefixes` \
  "${EXTRA[@]}"
