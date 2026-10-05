#!/bin/zsh
# tier-judge.sh <port> — the image judge (tiers.conf [judge], routed = no): SkyJM-Gen-4B, a Qwen3.5-4B
# fine-tune (skylenage-ai, Apache-2.0) that compares two generated images against their prompt and ends
# with \boxed{A} or \boxed{B}. Vision/judge/rank.py is its client. Started by llama-swap on request,
# unloaded after ttl (120 s) idle.
#
# Runtime: mlx-vlm's server (MLX, text + images). Install: uv tool install mlx-vlm==0.6.13
#   (or pipx; either puts mlx_vlm.server in ~/.local/bin)
# Model: SkyJM-Gen-4B converted to MLX 8-bit (~4.8 GB on disk, ~5.7 GB resident). Build it once:
#   uvx --from huggingface_hub hf download skylenage-ai/SkyJM-Gen-4B
#   chmod u+w "$(readlink -f ~/.cache/huggingface/hub/models--skylenage-ai--SkyJM-Gen-4B/snapshots/*/tokenizer.json)"
#   ~/.local/bin/mlx_vlm.convert --hf-path skylenage-ai/SkyJM-Gen-4B --mlx-path ~/models/SkyJM-Gen-4B-mlx-8bit \
#     -q --q-bits 8 --q-group-size 64
# The chmod matters: the Hugging Face cache is read-only, convert copies tokenizer.json with its mode, and
# saving the processor then fails with "Permission denied" after the weights are written but before
# config.json gets its quantization block.
#
# mlx_vlm.server loads whatever path a request's `model` names, so tiers.conf sets upstream_model to the
# same path as MODEL and llama-swap rewrites "judge" to it. Settings follow SkyJM-RM: temperature 0 (rank.py
# sends it), thinking off (the server's default).
PORT=${1:?usage: tier-judge.sh <port> [extra mlx_vlm.server flags]}; shift
EXTRA=("$@")   # benchmarking / one-off overrides; llama-swap passes none
MODEL="$HOME/models/SkyJM-Gen-4B-mlx-8bit"   # keep in step with upstream_model in tiers.conf

exec "${MLX_VLM_SERVER:-$HOME/.local/bin/mlx_vlm.server}" \
  --host 127.0.0.1 --port "$PORT" `# loopback only, behind llama-swap` \
  --model "$MODEL" \
  --max-tokens 4096 `# the rubric, both scores and the verdict` \
  "${EXTRA[@]}"
