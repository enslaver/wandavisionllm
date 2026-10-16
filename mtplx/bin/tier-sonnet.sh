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
# `mtplx serve` has no --agent-thinking-budget flag (only the server it execs does), so the cap stays an env var here.
PORT=${1:?usage: tier-sonnet.sh <port> [extra mtplx serve flags]}; shift
EXTRA=("$@")          # benchmarking / one-off overrides; llama-swap passes none
TIER=${TIER:-sonnet}  # llama-swap sets TIER from tiers.conf
MODEL="Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"   # mtplx resolves Hugging Face ids from ~/.mtplx/models

# Session-bank RAM cap. Each mtplx process otherwise plans for most of the Mac's memory, and three tiers would swap.
export MTPLX_SESSION_BANK_MAX_BYTES=4G   # was 6G (see tier-opus.sh); ~0.3% of sonnet's session entries are over 4G (~48 KiB/token including the gated-DeltaNet checkpoints)
export MTPLX_SESSION_BANK_IDLE_TTL_S=300   # drop idle warm sessions after 5 min (the SSD tier keeps them). A 1 h TTL kept idle sessions in the bank, sonnet grew by several GB, and mtplx's prefill admission shed live-session restores to a short prefix (tens of thousands of tokens re-prefilled per turn)
# SSD session cache: save less. Each save copies the whole entry into CPU memory first (~1.9 GB per save), and
# macOS keeps the freed pages in the process footprint, which pushed the Mac under mtplx's shed floor. Only
# sessions of 32k+ tokens are saved (about 70% of saves, 80% of the bytes), and one save waits at a time: the
# writer admits a lone entry when its queue is empty and skips the rest while one is queued (default backlog 4G).
# MTPLX_SSD_INCREMENTAL_ENCODE would copy only new blocks but can restore wrong state on a fingerprint
# collision, so it stays off.
export MTPLX_SSD_WRITER_BACKLOG_BYTES=512M
export MTPLX_LOOP_GUARD=1                  # in-reply repetition steering
export MTPLX_THINKING_BUDGET=2048          # force-close </think> at 2048 tokens on tool-carrying requests; mtplx_effort.py moves it to the server's --agent-thinking-budget so a request's reasoning_effort can override it

# mtplx_effort.py = mtplx plus per-request reasoning_effort -> thinking budget (tiers.conf: effort = yes); same CLI.
# Roll back: exec "$HOME/.mtplx/bin/mtplx" serve ... (same flags)
export PYTHONPYCACHEPREFIX="$HOME/Library/Caches/MTPLX/PythonBytecode"   # as ~/.mtplx/bin/mtplx sets it
exec "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" "$HOME/.mtplx/bin/mtplx_effort.py" serve \
  --host 127.0.0.1 --port "$PORT" \
  --no-auth `# loopback only, behind llama-swap; the network-facing auth is LiteLLM's master key` \
  --model "$MODEL" --model-id "$TIER" \
  --profile sustained `# the pack's recommended profile; MTP depth comes from its mtplx_runtime.json. A/B on the reference Mac: mtplx's turbo profile (4-bit verify matmuls, validated but not bit-exact) was no consistent win over sustained, so stay on sustained` \
  --tool-prompt-mode native `# the model's own trained tool format (qwen3_coder XML)` \
  --reasoning auto --preserve-thinking auto `# thinking on; history scoped to the active agent round` \
  --default-temperature 0.6 --default-top-p 0.95 --default-top-k 20 \
  --ssd-session-cache on --ssd-session-cache-min-prefix-tokens 32768 `# see MTPLX_SSD_WRITER_BACKLOG_BYTES` \
  --yes "${EXTRA[@]}"
