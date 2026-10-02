# Tier script examples

The four tier scripts in `mtplx/bin/` start mtplx (opus, sonnet, haiku) and TensorFold (fable). These start
the same kind of tier on another runtime. They are not deployed; copy one over a tier script to use it.

| Script | Runtime | Example model | Vision |
|---|---|---|---|
| `tier-llama-server.sh` | llama.cpp `llama-server` (GGUF) | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` | yes (mmproj) |
| `tier-mlx-lm.sh` | `mlx_lm.server` (MLX) | `mlx-community/Qwen3.5-4B-MLX-4bit` | no |

Every tier script follows the same contract: `tier-<name>.sh <port> [extra flags]`, listens on
127.0.0.1 only, takes its model id from `$TIER` (llama-swap sets it), and has exactly one `MODEL="..."`
line (Wanda shows its last path part).

To switch a tier:

1. Copy the example over the tier's script, e.g. `cp mtplx/bin/examples/tier-llama-server.sh mtplx/bin/tier-sonnet.sh`.
2. Set `TIER=${TIER:-sonnet}` and `MODEL` in the copy, and download the model (the script's header says how).
3. In `tiers.conf`, set that tier's keys to what the runtime supports (each script's header lists them):
   - `vision`: no for a text-only model; images then go to the vision tier instead.
   - `think_in_content`: yes when the model's chat template reads earlier reasoning back from a
     `<think>` block in assistant content (Qwen3.5); no when it reads only `reasoning_content` (Qwen3.8).
   - `chat_only`: yes when the runtime has no `/v1/responses`; LiteLLM then translates to chat completions.
   - `upstream_model`: `default_model` for mlx_lm.server only (llama-swap sends that name instead of
     the tier name, its `useModelName`).
   - `context`: the window the script starts the server with.
4. `./deploy.py push mtplx llama-swap litellm`.
