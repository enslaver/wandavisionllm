#!/bin/zsh
# tier-llama-server.sh <port> — a tier on llama.cpp's llama-server (GGUF). Copy it over a tier script
# (e.g. mtplx/bin/tier-sonnet.sh), set TIER's default and MODEL, then ./deploy.py push mtplx.
#
# Install: brew install llama.cpp (llama-swap's LaunchAgent PATH includes Homebrew's bin).
# Model: Qwen3.5-9B, unsloth's Q4_K_M GGUF (~5.7 GB) plus its vision projector (~0.9 GB), which -hf
# downloads automatically. Download before the first real request, so llama-swap's health check doesn't
# time out: run this script once by hand (`zsh tier-llama-server.sh 18999`), wait for "server is listening",
# then Ctrl-C. Files land in llama.cpp's cache.
#
# tiers.conf for this tier: vision = yes (the mmproj file is there; --no-mmproj turns it off),
# think_in_content = yes (the Qwen3.5 template reads earlier reasoning back from a <think> block),
# chat_only = yes for builds without /v1/responses (current llama.cpp has it; then chat_only = no).
PORT=${1:?usage: tier-llama-server.sh <port> [extra llama-server flags]}; shift
EXTRA=("$@")          # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-sonnet}  # llama-swap sets TIER from tiers.conf
MODEL="unsloth/Qwen3.5-9B-GGUF:Q4_K_M"   # <user>/<repo>[:quant] for -hf; use -m /path/file.gguf for a local file

exec "${LLAMA_SERVER:-llama-server}" \
  -hf "$MODEL" \
  --host 127.0.0.1 --port "$PORT" `# loopback only, behind llama-swap` \
  --alias "$TIER" `# the model id clients (LiteLLM) send` \
  --jinja `# the model's own chat template, needed for tool calls` \
  -c 131072 -np 1 `# one request at a time with the whole window; KV memory grows with -c` \
  --cache-ram 4096 `# prompt-cache RAM cap in MiB` \
  --temp 0.6 --top-p 0.95 --top-k 20 `# Qwen "thinking, precise coding" preset` \
  --no-webui \
  "${EXTRA[@]}"
