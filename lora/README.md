# lora/ — fine-tuning the local tiers on their own agent failures

Optional and experimental. Not deployed (no `COMPONENTS` entry): scripts live here; artifacts live on
the Mac's local SSD under `~/lora/` (venv, base views, traces, datasets, runs, packs). Traces and
datasets hold tool output (file contents, command output), so they never go in a repo. Everything
below runs on the Mac that serves the tiers.

Set up for the **sonnet** (Qwen3.5-9B) and **opus** (Qwen3.6-35B-A3B) example tiers, both MTPLX packs.
Other Qwen3.5/3.6 sizes should work the same; other model families need changes to `chunked_delta.py`.

## Quick start: train a round

1. Do the [one-time setup](#one-time-setup) and let traces build up for a few days.
2. Check you have new traces: `ls ~/.ultron/traces | wc -l`
3. Start the round (sample → train → evaluate, ~2 h for sonnet, unattended), from the repo:
   ```bash
   nohup lora/run.sh v1 --iters 100 --save-every 25 > ~/lora/runs/v1.out 2>&1 &
   ```
   `TIER=opus lora/run.sh opus-v1` does the same for opus.
4. When `tail ~/lora/runs/v1.out` says `done`, [read the results](#read-the-results).
5. If a checkpoint beats base, [ship it](#ship-an-adapter).

`run.sh` unloads every tier except the `[routing] helper` before training, and the next request
reloads them next to the training process. Don't train while an agent is using the local tiers, or
set route mode `auto` in Wanda for the run so its requests overflow to the cloud.

## How it works

A LoRA here teaches a tier to **recover from a failed tool call** instead of repeating it: the
failure `loop_breaker` catches after the fact. The training data comes from the tier's own work.

```
LiteLLM ultron_admit trace tap
  -> ~/.ultron/traces/*.json   latest request of every agent conversation on a local tier
  -> make_dataset.py           find failure points -> the tier samples 6 next turns -> keep good ones
  -> train.py                  QLoRA on the served 4-bit weights (mlx_lm + memory patches)
  -> valloss.py / eval.py      held-out loss per checkpoint; repeat rate, base vs adapter
  -> fuse_pack.py              adapter -> bf16 weights -> 4-bit MTPLX pack (MTP head + vision kept)
  -> mtplx/bin/tier-<tier>.sh  MODEL= the new pack; deploy; the tier serves it
```

### 1. Collecting traces

With `~/.ultron/trace-mode` = `on` (Wanda: Controls & routing, Trace tap), `ultron_admit`'s
deployment hook writes each local-tier request that carries tools to `~/.ultron/traces/`, after
`normalize_history`, so it's exactly what the model saw. One file per conversation, overwritten every
request: the newest file holds the whole history. Each trace records its `tier`. Captured
`/v1/messages` bodies convert with LiteLLM's interpreter:
`~/.local/share/uv/tools/litellm/bin/python anthropic_to_trace.py body.json ~/lora/traces/name.json`
(no `tier` field, so every `--tier` keeps them).

### 2. Building the dataset (`make_dataset.py`)

For every trace of `--tier` (default `sonnet`):

1. **Repair.** `ultron_admit.repair_split_tool_calls` rejoins tool calls split by the LiteLLM stream
   bug (see `litellm/README.md`, Split tool calls). A point whose prompt still holds
   `__unparsedToolInput` (the model started copying the broken format) is skipped as `poisoned`.
2. **Find failure points.** A tool result that is an error (`Exit code 1`, `<tool_use_error>`,
   `No such file`, …) or the same call returning the same result again. At most 2 points per
   identical failed call. Subagents and resumed sessions share history, so a point already seen in
   another trace file is skipped (`duplicate`).
3. **Compress.** System prompt cut to 10k chars, plus the goal, the last 12 messages and the tools it
   used, so a prompt is ~6–7k tokens.
4. **Sample.** The tier (through llama-swap, not LiteLLM) proposes 6 next turns at T=0.8.
5. **Filter.** A candidate is kept if it is a well-formed tool call, repeats none of the failed calls,
   and the tier, asked as a judge, says it responds to the error. One kept candidate per point
   becomes an example: `{"messages": prompt + [turn], "tools": [...]}`. An example longer than
   `max_seq_length` in `<tier>.yaml` is dropped (`too_long`): mlx_lm would truncate the trained turn
   away, and one such row makes every mlx_lm `Val loss` print `nan`.
6. **Split.** 10% valid, 90% train. Samples and verdicts are cached in `<out>/cache.jsonl`, so a
   rerun resumes and `--no-sample` rebuilds from cache only.

The last line prints the counts: `traces`, `other_tier`, `points`, `kept`, `no_pass`, `poisoned`,
`duplicate`, `too_long`.

### 3. Training (`train.py`, `<tier>.yaml`)

QLoRA on the same 4-bit weights the tier serves (a text-only view of the MTPLX pack, see setup).
`train.py` takes every `mlx_lm.lora` flag (`--iters`, `--save-every`, `--learning-rate`, …).
Checkpoints land in `runs/<name>/00000NN_adapters.safetensors`.

| | sonnet (`sonnet.yaml`) | opus (`opus.yaml`) |
|---|---|---|
| Adapted | all 32 blocks | attention, linear attention and the shared expert (not the 256 routed experts or the router) |
| Rank / scale / LR | 16 / 20 / 2e-5 | 16 / 20 / 1e-5 |
| Trainable | — | 19.2M |
| Peak memory at 2k / 4k / 8k tokens | — / — / ~23 GB | 21.8 / 25.6 / 36.4 GB |
| Speed | ~28 s/iteration | ~30 s/iteration |

`memprobe.py <tier>.yaml` measures the peak on your Mac.

### 4. Evaluating

- `valloss.py DATA/valid.jsonl RUN --base BASE` gives the loss on held-out turns for base and every
  checkpoint. Lower is better. It's the stop signal: v0 overfit from the first checkpoint.
- `eval.py DATA/valid.jsonl --adapter RUN --base BASE` samples 4 turns per held-out point and reports
  `repeat` (calls a failed call again; lower is better), `valid` (well-formed call to a real tool) and
  `no_call`, base vs adapter.

### 5. Fusing (`fuse_pack.py`)

mtplx can't load a LoRA at serve time, so the adapter goes into the weights: for each adapted layer
W + scale·BA is computed from the **bf16** checkpoint and re-quantized to the pack's 4-bit g64. The MTP
head, vision tower, tokenizer and template are kept (hard-linked), and `mtplx_runtime.json` gets a
`lora` block. Sharded packs work: each tensor is looked up in its own shard, and only shards with an
adapted layer are rewritten. It checks first that quantizing bf16 alone reproduces the pack
(≥ 0.98, or it refuses).

## Read the results

`run.sh` prints them; full logs are in `~/lora/runs/<name>*.log`.

1. **Held-out loss**: `cat ~/lora/runs/v1-valloss.log`. Pick the checkpoint with the lowest loss.
   If base is lowest, the round didn't help: don't ship it.
2. **Repeat rate**: `grep -E "^(base|adapter)" ~/lora/runs/v1-eval.log`. `eval.py` scores the final
   adapter. To score another checkpoint, make it an adapter dir and rerun:
   ```bash
   mkdir ~/lora/runs/v1-ck50 && cp ~/lora/runs/v1/adapter_config.json ~/lora/runs/v1-ck50/ \
     && cp ~/lora/runs/v1/0000050_adapters.safetensors ~/lora/runs/v1-ck50/adapters.safetensors
   ~/lora/.venv/bin/python lora/eval.py ~/lora/data/v1/valid.jsonl --adapter ~/lora/runs/v1-ck50
   ```
3. Ship only if held-out loss is at or below base **and** repeats drop. The valid split is by point,
   not by conversation, so the repeat rate is optimistic: live use is the real check.

## Ship an adapter

About 15 minutes, plus live use.

1. Fuse (~3 min for sonnet):
   ```bash
   ~/lora/.venv/bin/python lora/fuse_pack.py ~/.mtplx/models/<sonnet pack> \
     ~/lora/runs/v1-ck50 ~/lora/packs/v1-ck50 --bf16 <bf16 snapshot dir>
   ```
   Its last line should say `bf16 alone reproduces the pack: 1.0000`.
2. In `mtplx/bin/tier-sonnet.sh`, set `MODEL="$HOME/lora/packs/v1-ck50"`.
3. Deploy: `./deploy.py push mtplx`
4. Reload the tier: Wanda → sonnet card → Unload. The next request loads the new pack.
5. Validate: `cd litellm && uvx --with pyyaml python3 suite.py baseline --tool`, then use it on a real
   task. Watch MTP acceptance on the tier card: the MTP head isn't retrained, so drafts may be
   accepted less often and tok/s drop.

**Roll back:** restore the old `MODEL=` line, `./deploy.py push mtplx`, unload the tier in Wanda.

Opus is the same with its own pack, run and bf16 checkpoint; a fused opus pack is ~20 GB.

## One-time setup

```bash
uv venv --python 3.12 ~/lora/.venv && uv pip install --python ~/lora/.venv/bin/python "mlx-lm[train]" pytest
# A text-only view of each served pack: a folder of symlinks to its config, model weights, tokenizer
# and chat template, with a weight index that leaves out the vision tower and MTP head. The scripts
# use ~/lora/base/<tier>-4bit (LORA_BASE overrides it).
mkdir -p ~/lora/base/sonnet-4bit ~/lora/base/opus-4bit
# The bf16 checkpoint each pack was forged from (its model card names it), for fusing:
~/lora/.venv/bin/hf download <bf16 base repo> --include "model*.safetensors" --include "*.json"
echo on > ~/.ultron/trace-mode   # start collecting traces
```

If a tier's model changes, the base view and the bf16 checkpoint must match the new model. Old
adapters don't carry over.

Other tiers: **haiku** has the same shape as sonnet at a smaller size (needs a base view, `haiku.yaml`
and its bf16 checkpoint). **fable** runs on TensorFold, not an MTPLX pack, so `fuse_pack.py` doesn't
apply.

## Housekeeping

- Delete traces you don't want trained on: `~/.ultron/traces/*.json`. They're overwritten per
  conversation but never pruned.
- Stop collecting: `echo off > ~/.ultron/trace-mode`.
- Disk: each sonnet run is ~1 GB of checkpoints, each sonnet pack ~6 GB.

## Steps by hand

`run.sh` is these commands (plus unloading the big tiers):

```bash
cd lora
~/lora/.venv/bin/python make_dataset.py ~/.ultron/traces/*.json ~/lora/traces/*.json --out ~/lora/data/v1 --tier sonnet
~/lora/.venv/bin/python train.py --config sonnet.yaml --model ~/lora/base/sonnet-4bit \
    --data ~/lora/data/v1 --adapter-path ~/lora/runs/v1 --iters 100 --save-every 25
~/lora/.venv/bin/python valloss.py ~/lora/data/v1/valid.jsonl ~/lora/runs/v1 --base ~/lora/base/sonnet-4bit
~/lora/.venv/bin/python eval.py ~/lora/data/v1/valid.jsonl --adapter ~/lora/runs/v1 --base ~/lora/base/sonnet-4bit
```

Timing on the reference Mac: sampling ~26 s per failure point, training ~28 s per iteration, eval
~5 min for 8 points.

## Why the patches

- **chunked_delta.py** — mlx_lm trains Qwen3.5's linear-attention layers (24 of 32) with a per-token
  Python loop; at 8k tokens that exceeds Metal's 499000-buffer limit. The chunked (WY) form does
  64-token blocks. Its in-chunk solve is a blocked forward substitution: the power-product shortcut
  overflowed to NaN on real activations. `test_chunked_delta.py` checks it against the loop (exact
  on CPU; Metal's fp32 matmul is ~1e-3 off on its own).
- **train.py** — logits over a 248k vocab are chunked under `mx.checkpoint`; the train step isn't
  `mx.compile`d (43 GB vs ~20 GB at 8k); the MLX buffer cache is capped (`LORA_CACHE_GB`, default 2)
  because Metal's wired limit (~48 GB of 64) is shared with the serving tiers.
- **fuse_pack.py --bf16** — fusing into the 4-bit pack itself keeps only ~20% of the delta (the
  weights sit on the quantization grid); from bf16 it keeps it on average.

## Results so far

- **v0, sonnet** (2026-10-01, reference Mac): one agent trace (a sync command that kept failing), 87
  failure points (2 per failing call), 6 samples each; 86 kept, 78 train / 8 valid. 200 iters, ~1.5 h,
  peak ~23 GB. A pipeline check, not a fix:
  - held-out loss (`valloss.py`) rose from the first checkpoint: base 0.307, iter 50 0.415,
    iter 200 0.658. Train loss fell to ~0.06–0.1, so it memorized the trace.
  - repeats on held-out points (`eval.py`, 32 samples): base 18.8%, iter 200 0%. In-distribution:
    held-out points share failing calls with training points.
  - On unseen command-line tasks (5 samples each), iter 200 stopped repeating on the trained task
    but did worse on the others; iter 50 was within noise. Neither is worth serving.
  - Applied since: data from many conversations (trace tap), `--iters 100 --save-every 25`, stop on
    held-out loss. Still open: split train/valid by conversation instead of by point.
- **opus smoke run** (2026-10-01, the reference Mac's Qwen3.6-35B-A3B build): 10 iterations on the v0 data; held-out loss 0.590 → 0.550 (5) →
  0.525 (10); bf16 re-quantized reproduces the pack (0.9998); the fused pack serves and calls tools.
