# judge — rank generated images (SkyJM-Gen-4B)

Not deployed. Part of [Vision](../README.md). `rank.py` asks the Mac's image judge, `ultron/judge`, which of several
generated images best matches the prompt they came from, and prints a ranking. Python 3.9, standard library only.

```bash
Vision/judge/rank.py "a red fox sitting in snow, golden hour" take1.png take2.png take3.png
Vision/judge/rank.py -v --base https://your-mac.example.ts.net/llm "prompt" a.jpg b.jpg   # from another machine; LITELLM_KEY set
Vision/judge/rank.py --json "prompt" *.png
```

`--base` defaults to `http://127.0.0.1:4000` (LiteLLM on the Mac itself). From another machine use the Mac's
Caddy route `https://your-mac.example.ts.net/llm` or `http://your-mac:4000`. The key is `$LITELLM_KEY`, else
`LITELLM_MASTER_KEY` from `~/.litellm/env`.

## How it works

| Piece | Where |
|---|---|
| Model | `skylenage-ai/SkyJM-Gen-4B` (Qwen3.5-4B fine-tune, Apache-2.0), MLX 8-bit at `~/models/SkyJM-Gen-4B-mlx-8bit` (4.8 GB) |
| Server | `mtplx/bin/tier-judge.sh` runs `~/.local/bin/mlx_vlm.server` (mlx-vlm 0.6.13: `uv tool install mlx-vlm==0.6.13`, or pipx) |
| Config | `[judge]` in `litellm/tiers.conf` with `routed = no`: llama-swap serves it (`ttl 120`, `evict_cost 1`) and LiteLLM exposes it only as `ultron/judge`. It is never a routing target, has no cloud overflow, and doesn't count against a tier when the admission hook decides whether that tier fits |
| Memory | `[routing] resident = (fable \| opus \| judge) & sonnet & haiku`: the judge takes the big tier's slot, next to sonnet and haiku. Loading fable or opus unloads it (evict cost 1) |
| Admission | `ultron_admit` holds a judge request until the tier it evicts is idle and no main conversation used it in the last 5 min (at most `EVICT_WAIT_MAX_S` = 300 s), then for the memory gate (`ULTRON_MEM_WAIT_S`); after either limit it goes anyway |
| Media hook | `ultron_media` skips `routed = no` models: two images and a prompt can read like an edit request |

The prompt is SkyJM-RM's `GEN_TEMPLATE`, verbatim: the judge picks 3-5 weighted dimensions for the
prompt, scores Image A and Image B 0-4 on each, and ends with `\boxed{A}` or `\boxed{B}`. Settings
follow SkyJM-RM: temperature 0, thinking off (mlx_vlm.server's default).

Each pair is judged in both orders. A pair is a win only when both orders pick the same image,
else a tie. Score = wins + ties / 2, so N images take N*(N-1) calls, run one at a time.

Images longer than `--max-edge` (1024) px are shrunk with `sips` (macOS) first: the processor's
`max_pixels` is 16.7 MP, so a 4K image would otherwise be ~16k tokens. Elsewhere they are sent as they are.

## Measured (2026-10-02, the reference Mac)

- **fox vs fox_extra_head** (prompt "a red fox in the snow"): the clean take won in both orders.
  - Both verdicts named the pasted head ("a disembodied fox head floating in the air").
  - Cold call 13 s (about 5 s of it loading the judge), warm call 8 s.
- **3 images** (clean fox, fox with an extra head, an off-prompt lighthouse): 6 calls, 45 s.
  - The clean fox won both of its pairs.
  - extra-head vs lighthouse was a tie: each order picked Image A. That is position bias, and the both-orders rule turned it into a tie instead of a wrong winner.
- **Memory:** `mlx_vlm.server` RSS is 5.7 GB.
- **Swap-back:** a sonnet load evicts only the judge (`evict=[judge] cost=1` in llama-swap's log).

## Build the model

```bash
uv tool install mlx-vlm==0.6.13
uvx --from huggingface_hub hf download skylenage-ai/SkyJM-Gen-4B
chmod u+w "$(readlink -f ~/.cache/huggingface/hub/models--skylenage-ai--SkyJM-Gen-4B/snapshots/*/tokenizer.json)"
~/.local/bin/mlx_vlm.convert --hf-path skylenage-ai/SkyJM-Gen-4B --mlx-path ~/models/SkyJM-Gen-4B-mlx-8bit -q --q-bits 8 --q-group-size 64
```

The `chmod` matters: HF's cache files are read-only, `convert` copies `tokenizer.json` with its mode,
and then `processor.save_pretrained` fails with `Permission denied` after the weights are written
but before config.json gets its `quantization` block.

## Gotchas

- `mlx_vlm.server` loads whatever path the request's `model` names (`get_cached_model`), so
  tiers.conf sets `upstream_model = ~/models/SkyJM-Gen-4B-mlx-8bit` and llama-swap sends that path
  instead of `judge`.
- While a main conversation is using the big tier, each judge load waits up to 5 min. Rank when
  that tier is quiet.
- SkyJM-Gen is for text-to-image. Edit pairs (original + two edits) need SkyJM-Edit-4B and its
  `EDIT_TEMPLATE`, which aren't set up.
