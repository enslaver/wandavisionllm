# lora/ — fine-tuning the local tiers on their own agent failures

Optional and experimental. Not deployed (no `COMPONENTS` entry): scripts live here; artifacts live on
the Mac's local SSD under `~/lora/` (venv, base views, traces, datasets, runs, packs). Traces and
datasets hold tool output (file contents, command output), so they never go in a repo. Everything
below runs on the Mac that serves the tiers, except image training for haiku, which runs on a separate
PC with an NVIDIA GPU (the GPU box).

Set up for the **sonnet** (Qwen3.5-9B), **opus** (Qwen3.6-35B-A3B) and **haiku** (Qwen3.5-4B) example
tiers, all MTPLX packs. Text rounds train on the Mac with mlx_lm for any of the three. Haiku can also
learn from images: Unsloth trains that LoRA on the GPU box, and the Mac forges and serves the result
([The haiku tier](#the-haiku-tier-text-and-images)). Other Qwen3.5/3.6 sizes should work the same;
other model families need changes to `chunked_delta.py`.

## Quick start: train a round

1. Do the [one-time setup](#one-time-setup) and let traces build up for a few days.
2. Check you have new traces: `ls ~/.ultron/traces | wc -l`
3. Start the round (sample → train → evaluate, ~2 h for sonnet, unattended), from the repo:
   ```bash
   nohup lora/run.sh v1 --iters 100 --save-every 25 > ~/lora/runs/v1.out 2>&1 &
   ```
   `TIER=opus lora/run.sh opus-v1` does the same for opus, `TIER=haiku lora/run.sh haiku-v1` for haiku.
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
6. **Split.** ~10% valid (`--valid`), whole conversations at a time: one trace file is one
   conversation, and points of one conversation share most of their prompt. A conversation that would
   take valid past twice its share stays in train (one coding-agent session held 85 of 148 passing
   points). Samples and verdicts are cached in `<out>/cache.jsonl`, so a rerun resumes and
   `--no-sample` rebuilds from cache only (`CACHE_FROM=<round> lora/run.sh ...` starts a round from an
   earlier round's cache that way: same points, fresh split, no sampling).

The last line prints the counts: `traces`, `other_tier`, `points`, `kept`, `no_pass`, `poisoned`,
`duplicate`, `too_long`, `conversations`.

### 3. Training (`train.py`, `<tier>.yaml`)

QLoRA on the same 4-bit weights the tier serves (a text-only view of the MTPLX pack, see setup).
The loss covers the chosen turn only and stops at its last token: mlx_lm's own mask also trains the
pad after it (see [Results so far](#results-so-far), "The loss mask"; `test_train_loss.py` pins it).
`train.py` takes every `mlx_lm.lora` flag (`--iters`, `--save-every`, `--learning-rate`, …).
Checkpoints land in `runs/<name>/00000NN_adapters.safetensors`.

| | sonnet (`sonnet.yaml`) | opus (`opus.yaml`) |
|---|---|---|
| Adapted | all 32 blocks | attention, linear attention and the shared expert (not the 256 routed experts or the router) |
| Rank / scale / LR | 16 / 20 / 2e-5 | 16 / 20 / 1e-5 |
| Trainable | — | 19.2M |
| Peak memory at 2k / 4k / 8k tokens | — / — / ~23 GB | 21.8 / 25.6 / 36.4 GB |
| Speed | ~28 s/iteration | ~30 s/iteration |

`haiku.yaml` uses sonnet's settings on the 4B (all 32 blocks, rank 16, scale 20, LR 2e-5); its
memory and speed aren't measured yet. `memprobe.py <tier>.yaml` measures the peak on your Mac.

### 4. Evaluating

- `valloss.py DATA/valid.jsonl RUN --base BASE` gives the loss on held-out turns for base and every
  checkpoint. Lower is better. It's the stop signal: v0 and v1 overfit from the first checkpoint.
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

Opus and haiku are the same with their own pack, run and bf16 checkpoint; a fused opus pack is ~20 GB.

## One-time setup

```bash
uv venv --python 3.12 ~/lora/.venv && uv pip install --python ~/lora/.venv/bin/python "mlx-lm[train]" pytest
# A text-only view of each served pack: a folder of symlinks to its config, model weights, tokenizer
# and chat template, with a weight index that leaves out the vision tower and MTP head. The scripts
# use ~/lora/base/<tier>-4bit (LORA_BASE overrides it).
mkdir -p ~/lora/base/sonnet-4bit ~/lora/base/opus-4bit ~/lora/base/haiku-4bit
# The bf16 checkpoint each pack was forged from (its model card names it), for fusing:
~/lora/.venv/bin/hf download <bf16 base repo> --include "model*.safetensors" --include "*.json"
echo on > ~/.ultron/trace-mode   # start collecting traces
```

If a tier's model changes, the base view and the bf16 checkpoint must match the new model. Old
adapters don't carry over.

Other tiers: **haiku** trains text like sonnet; for images see [The haiku tier](#the-haiku-tier-text-and-images).
**fable** runs on TensorFold, not an MTPLX pack, so `fuse_pack.py` doesn't apply.

## The haiku tier (text and images)

Haiku (Qwen3.5-4B) trains two ways:

- **Text**, like sonnet: `TIER=haiku lora/run.sh haiku-v1` with `haiku.yaml` and a base view in
  `~/lora/base/haiku-4bit`, shipped with `fuse_pack.py` as in [Ship an adapter](#ship-an-adapter).
  A pack forged from stock Qwen3.5-4B fuses from `Qwen/Qwen3.5-4B`.
- **Images**: Unsloth on the GPU box trains a LoRA and merges it into the bf16 weights; the Mac forges
  the merge into a pack with the vision tower and serves it. The rest of this section is that path.

| Piece | What it is |
|---|---|
| `Qwen/Qwen3.5-4B` | the bf16 checkpoint and base of every haiku fine-tune: 426 language-model, 297 vision and 15 `mtp.*` tensors |
| `recipe-4b-vision.json` | the forge recipe: 4-bit g64 body, MTP head kept bf16; forge restores the source's bf16 vision tower |
| `unsloth_vision.py`, `unsloth_queue.sh` | train and merge on the GPU box: one run, or one unattended run with retries |
| `add_mtp.py MERGED BASE` | copies the base's MTP head into a merge that lost it (checked on a stripped copy of Qwen3.5-4B: the 15 tensors come back byte-identical) |

### A vision pack first

The example haiku pack, `Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed`, is text only: haiku can't take
images until it serves a pack with the vision tower, and an image fine-tune needs a stock baseline to
beat. Forge a vision pack of stock `Qwen/Qwen3.5-4B` first (the header of `mtplx/bin/tier-haiku.sh`
says the same):

```bash
~/.mtplx/bin/mtplx forge build --repo Qwen/Qwen3.5-4B --recipe "$(cat lora/recipe-4b-vision.json)" \
  --out ~/.mtplx/forge/out --run-id haiku-vision --branded-name Qwen3.5-4B-Vision-MTPLX --model-root ~/.mtplx/models
```

Then swap it in as in [Swapping a model](../docs/configuration.md#swapping-a-model): `MODEL=` in
`tier-haiku.sh`, `vision = yes` under `[haiku]` in `litellm/tiers.conf` (and `[routing] vision = haiku`
if haiku should answer images for the tiers that can't see them). On the reference Mac this pack runs
at 234 tok/s at MTP depth 3. Every image fine-tune below is forged with the same recipe and compared
against it.

### Train on the GPU box (Unsloth)

The reference GPU box is a Windows 11 PC with an RTX 4080 Laptop GPU (12 GB). Install Unsloth Studio
there with `python gpu-box\deploy.py push unsloth` (see [../gpu-box/README.md](../gpu-box/README.md))
and clone this repo on it; the scripts run from the clone. Training reads the Hugging Face weights
(`unsloth/Qwen3.5-4B`), **not a GGUF**: GGUF is an inference export, Unsloth can't train it, and haiku
runs MLX.

`unsloth_vision.py` trains, saves the adapter to `OUT/adapter`, and merges it into 16-bit weights in
`OUT/merged`. Run it in Unsloth Studio's venv with Studio's chat model unloaded. From the clone, in
PowerShell:

```powershell
$env:TORCHDYNAMO_DISABLE='1'; $env:UNSLOTH_COMPILE_DISABLE='1'
& "$env:USERPROFILE\.unsloth\studio\unsloth_studio\Scripts\python.exe" -X faulthandler `
  lora\unsloth_vision.py "$env:USERPROFILE\lora\haiku-<name>" --4bit --data <rows>.jsonl
```

Without `--data` it trains 50 steps on 200 rows of `unsloth/LaTeX_OCR`: a pipeline smoke run, not
something to ship. `--data` takes a JSONL file, one row per line:
`{"image": "img/0001.jpg", "prompt": "...", "answer": "...", "think": "..."}`. Image paths are
relative to the file; `think` is optional (see the data notes below). A row without `image` is
text-only: mix a few copies of a text task the tier already does (a yes/no gate, say) into an image
round so the adapter doesn't wear it down. With `--data` the default LR is 1e-4; `--epochs X` sets
the step count from the row count.

For a long run, `unsloth_queue.sh` (Git Bash on Windows, or Linux) does it unattended:

```bash
nohup bash lora/unsloth_queue.sh haiku-<name> 8 --data ~/lora/data/<rows>.jsonl > ~/lora/haiku-<name>.queue.log 2>&1 &
```

It waits for the GPU, probes 12 steps for sec/step and peak VRAM, trains 1 epoch capped at 8 hours,
retries a crash up to 3 times from the last checkpoint, then writes a `STAGED` marker. With
`LORA_STAGE` set to a folder the Mac can read (a share both machines mount), it first copies the merge
and adapter to `$LORA_STAGE/<name>` and writes the marker there; otherwise the merge stays in
`~/lora/<name>/merged`.

- **GPU guard:** the script refuses to start while another run holds `~/lora/gpu.lock`, or when less
  than `--min-free-gb` (default 7) of VRAM is free; free VRAM drops when Unsloth Studio, ComfyUI or
  Ollama holds a model. `--wait-gpu` (the queue uses it) waits instead. It then caps PyTorch at free
  minus `--headroom-gb` (0.5), so an overrun raises CUDA OOM instead of the Windows driver spilling
  into shared RAM. It prints peak VRAM at the end.
- **`--4bit`:** a 12 GB GPU with ~3 GB held by the desktop can't fit the 9.3 GB of bf16 weights, so
  the run is QLoRA. The merge is still lossless: Unsloth's `merge_and_overwrite_lora` folds the LoRA
  into the *original* 16-bit shards.
- **Windows Triton crash:** on the first step Triton's JIT can die with `0xC0000005` in
  `libtriton.pyd`, though a trivial kernel compiles fine. It crashed twice and then went through once
  the env vars above were set (the kernel cache had warmed by then), so it's intermittent: rerun. The
  queue retries on its own.
- **Compile cache:** Unsloth writes compiled modules to the working directory unless told otherwise;
  the script points it at `~/lora/unsloth_compiled_cache` so it stays out of the clone.
- **Smoke run (2026-10-02):** 50 steps at LR 2e-5 on 200 LaTeX_OCR rows, 190 s, loss 0.17. The merge
  changes only language-model linear layers, all of linear attention's `in_proj_*` included (max
  relative delta 9e-4). Embeddings, norms, the vision tower and the MTP head are byte-identical to base.

The recipe, for anyone writing their own trainer. Keep the base identical to haiku's:

```python
from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained("unsloth/Qwen3.5-4B", load_in_4bit=True)
model = FastVisionModel.get_peft_model(model,
    finetune_vision_layers=False,   # the tower stays the one haiku ships
    finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0)
    # no modules_to_save=["lm_head", "embed_tokens"]: the embeddings are tied and feed the MTP head
# ... SFT on Qwen chat-format rows with images, loss on the answers only ...
model.save_pretrained_merged("haiku-<name>", processor, save_method="merged_16bit")
```

Data notes:

- Train the answers haiku should give, in the length and format you want back.
- Haiku serves with thinking on, and clients can turn it off. A row without `think` trains an empty
  `<think></think>` (a thinking-off answer); a row with one trains a short reasoning block first. An
  adapter trained only on empty blocks got worse with thinking on; untrained, the model reasons at
  length about images and can run to max_tokens. A short reasoning block (1–3 sentences, ending in the
  answer) on about half the rows held up in both modes.
- Use the Qwen3.5 chat template Unsloth ships for the model; don't swap templates.
- Images open per row: decoding a few thousand 1280 px images up front took ~13 GB of RAM.

### Ship it on the Mac

1. **Copy the merge to the Mac's local disk**, into `~/lora/merges/<name>`, once `STAGED` exists.
   Forging straight off a network share died in mlx-lm's convert with `[METAL] Command buffer
   execution failed: GPU Timeout Error`, most likely because lazy SMB reads stall the command buffer.
   Copying the ~9 GB merge took ~3 min.
2. **Fix the non-weight files** from haiku's real base, `Qwen/Qwen3.5-4B`:
   - Tokenizer: copy `tokenizer.json`, `tokenizer_config.json` and `vocab.json`. Move Unsloth's
     `chat_template.jinja` aside: it would override the template inside the base's
     `tokenizer_config.json`. Then the forged tokenizer files are byte-identical to the stock vision
     pack's.
   - Image processor: an Unsloth merge carries only `processor_config.json`. Without
     `preprocessor_config.json`, forge's vision graft fails ("has no preprocessor_config.json; the
     runtime cannot decode images without it"). `unsloth_vision.py` copies it and
     `video_preprocessor_config.json` into the merge; the loop below copies Qwen's either way.
3. **MTP head:** `add_mtp.py` (see below).
4. **Forge** with the recipe, when the stack is quiet (below).

```bash
M=~/lora/merges/haiku-<name>; mkdir -p ~/lora/merges
scp -r <gpu box>:lora/haiku-<name>/merged $M     # or: rsync -a <LORA_STAGE on the Mac>/haiku-<name>/ $M/
Q=$(~/lora/.venv/bin/hf download Qwen/Qwen3.5-4B -q)   # prints the snapshot folder
for f in tokenizer.json tokenizer_config.json vocab.json preprocessor_config.json video_preprocessor_config.json; do
  cp -L $Q/$f $M/$f; done
[ -e $M/chat_template.jinja ] && mv $M/chat_template.jinja $M.unsloth-chat_template.jinja
/usr/bin/python3 lora/add_mtp.py $M $Q     # no-op on Unsloth ≥ 2026.9
~/.mtplx/bin/mtplx forge build --repo $M --recipe "$(cat lora/recipe-4b-vision.json)" \
  --out ~/.mtplx/forge/out --run-id haiku-<name> --branded-name Qwen3.5-4B-Vision-<name>-MTPLX --model-root ~/.mtplx/models
```

- **`--recipe` takes the JSON text** (or a preset name), not a file path: mtplx 2.12.0 runs
  `json.loads` on the argument, so a path fails with `--recipe must be JSON or a named preset ...:
  Expecting value`.
- **MTP head:** Unsloth ≥ 2026.9 merges keep it (they stream the original shards, so `mtp.*` comes
  through), and `add_mtp.py` prints `already has 15 mtp.* tensors; nothing to do`. Older or
  plain-transformers saves keep `mtp_num_hidden_layers: 1` in config.json but carry no `mtp.*`
  tensors, and forge refuses them (`no_mtp_heads`); `add_mtp.py` copies the base's head back. The head
  was trained on the base's hidden states, so MTP acceptance (tok/s) may drop a little; forge's
  verification reports it.
- **Tokenizer warning:** forge logs a transformers `fix_mistral_regex` warning, with the base
  tokenizer too. Every forge writes the old Qwen2 pre-tokenizer; see
  [Pre-tokenizer regex](#pre-tokenizer-regex).
- **Quiet stack:** forge verification loads the model next to the tiers (an opus+sonnet+haiku set
  leaves ~7 GB on a 64 GB Mac). Wait until no `ultron/*` request has been in flight for ~60 s; a wait
  loop does it. A forge with opus, sonnet and haiku resident and 11 GB free ran without trouble.
- **Clean up a failed forge first:** it leaves a partial pack under the branded name, and the next run
  writes `<name>-1` instead. Delete `~/.mtplx/models/<branded name>*` and
  `~/.mtplx/forge/out/<run-id>` before rerunning.

Then serve it: `MODEL=` in `mtplx/bin/tier-haiku.sh` with a dated comment, `./deploy.py push mtplx`,
unload haiku in Wanda (the next request reloads it), and run
`cd litellm && uvx --with pyyaml python3 suite.py vision confirm`: haiku is also the helper tier, so
`confirm` checks its MEDIA/OTHER answers. Roll back as for sonnet.

Wanda's LoRA panel lists each `~/lora/merges/<name>` as a run trained on the GPU box, and each pack in
`~/.mtplx/models` whose `mtplx_runtime.json` `base_trunk` points into `~/lora` (a pack forged from a
merge there) as a LoRA pack.

When you score a fine-tune with many one-shot image requests, send the `x-mtplx-cache-mode: bypass`
header. Without it, haiku's active memory grew ~85 MB per request until Metal ran out (HTTP 507 from
about request 570); with it, memory stayed at 3.4–5.6 GB.

**Verified end to end on 2026-10-02** with the smoke run's merge:

- Forge: exit 0, verdict `mtp_depth_wins`, vision tower restored (297 tensors, 667 MB). MTP
  calibration came back "inconclusive", so it kept the family default. The pack is 3.1 GB.
- Verify table, smoke vs the stock vision pack: 234 vs 234 tok/s at depth 3, acceptance
  0.93/0.77/0.68 vs 0.96/0.76/0.59, `quality_passed` at every depth for both. The trunk LoRA didn't
  hurt the base's MTP head.
- Served as haiku: `suite.py vision confirm` 8/8 and 9/9. On a 16-image check (thinking on / off) the
  smoke pack scored 12/16 and 14/16, stock 14/16 and 13/16 the same day. Different items fail each run
  at temperature 0.6, so that's within noise.

## Housekeeping

- Delete traces you don't want trained on: `~/.ultron/traces/*.json`. They're overwritten per
  conversation but never pruned.
- Stop collecting: `echo off > ~/.ultron/trace-mode`.
- Disk: each sonnet run is ~1 GB of checkpoints, each sonnet pack ~6 GB. A haiku image merge is ~9 GB
  on the GPU box and again on the Mac, its pack ~3 GB; a failed forge leaves a partial pack behind.

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

Timing on the reference Mac: sampling ~26 s per failure point (~55 s in v1), training ~28 s per
iteration, eval ~5 min for 8 points.

## Pre-tokenizer regex

Checked on 2026-10-02. **Finding:** every Qwen3.5/3.6/3.8 tokenizer re-saved by transformers 5.x
carries the older Qwen2 split regex. On the reference Mac that covered every MTPLX pack, the
`~/lora/base/*` views, and fable's MLX weights; the example haiku pack
(`Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed`) fails `--check` too. The affected regex is `\p{L}+` and
`[^\s\p{L}\p{N}]`, with ByteLevel `trim_offsets: true`. Qwen's own regex is `[\p{L}\p{M}]+` and
`[^\s\p{L}\p{M}\p{N}]`.

- The rewrite happens in `mtplx forge build`, `mlx_lm.convert` and Unsloth saves; transformers logs a
  `fix_mistral_regex` warning when it does.
- Vocab and merges are unchanged. The packs also carry 7 extra audio special tokens (ids
  248070-248076), which are harmless.
- The right regex is still in each pack's `tokenizer_config.json` (`pretokenize_regex`).

**Rewriting tokenizer.json alone doesn't fix it.** transformers 5.14's `Qwen2Tokenizer` class rebuilds
the Qwen2 regex at load time, even for the pristine `Qwen/Qwen3.5-4B` snapshot, so the tiers serve the
old split. A Hindi+Thai prompt cost 62 prompt tokens on haiku and on sonnet, against 46 under Qwen's
regex; English cost 27 either way.

**Impact.** The Qwen2 regex makes every combining mark a pre-token boundary:

- **Corpus test (32 samples):** Hindi +43% tokens, Bengali/Tamil +52%, Thai +92%, and the splits are
  ones the model never trained on. Pointed Arabic/Hebrew is +2%, and VS16 emoji differ by a token.
- **Unchanged:** English, code, CJK, Korean and precomposed or NFD Latin (the NFC normalizer composes
  NFD Latin first).
- **Real traffic:** zero difference on the reference Mac. All 10,853 text fields in its traces (6.9M
  tokens) tokenize identically under both regexes. This only matters if non-Latin-script prompts reach
  the tiers.

**Fix:** `fix_pretokenizer.py` restores the pre-tokenizer and decoder from `pretokenize_regex`, and sets
`tokenizer_class` to `TokenizersBackend` (Qwen2Tokenizer's base class, which loads tokenizer.json as
written; Unsloth ships Qwen3.6 that way). mtplx has no forge flag for this; `forge build --help` in
2.12.0 has none.

```bash
/usr/bin/python3 lora/fix_pretokenizer.py ~/.mtplx/models/<pack> --check                  # exit 1 = needs the fix
/usr/bin/python3 lora/fix_pretokenizer.py ~/.mtplx/models/<pack> --out ~/lora/<pack>-tokfix  # symlinked view, fixed tokenizer
```

**A/B for haiku (2026-10-02).** Both runs used a temporary `mtplx serve` on a spare port, 10 greedy
prompts, the original pack and then the `--out` view:

- **Loading:** mtplx loads the view, and AutoTokenizer ids match the source on 32/32 samples.
- **Code and English:** byte-identical outputs and identical MTP acceptance (67.1% and 44.5%).
- **Hindi and Thai:** prompt tokens 120 → 85 and 75 → 46. Acceptance 41.3 → 40.3% for Hindi and
  41.8 → 44.6% for Thai, within noise; the outputs differ, as expected.
- **Not tested:** the sonnet and opus packs (same files, same fix, no A/B yet) and fable (TensorFold).

**Applying it** is optional: for English and code prompts nothing changes, and the reference Mac
leaves its tiers on the forged tokenizer. To apply, point `MODEL=` in `mtplx/bin/tier-*.sh` at a
`--out` view, or run `--in-place` (keeps `.orig` copies). Then `./deploy.py push mtplx`, reload the
tier, and run `suite.py`. Re-run `--check` after every forge or `fuse_pack.py`: both copy the pack's
tokenizer.

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
    held-out loss, and (since v2) a train/valid split by conversation instead of by point.
- **opus smoke run** (2026-10-01, the reference Mac's Qwen3.6-35B-A3B build): 10 iterations on the v0 data; held-out loss 0.590 → 0.550 (5) →
  0.525 (10); bf16 re-quantized reproduces the pack (0.9998); the fused pack serves and calls tools.
- **haiku image smoke run** (2026-10-02): see [Ship it on the Mac](#ship-it-on-the-mac). Proves the
  GPU box → Mac pipeline; not meant to ship.
- **v1, sonnet** (2026-10-04): 87 traces from the trace tap, 600 failure points; 111 kept, 100 train /
  11 valid. Of the rest: 396 no_pass (none of the 6 samples passed the filter), 45 duplicate, 36
  too_long, 12 poisoned. Sampling took 7.5 h (~55 s per point, twice v0's rate), training 47 min.
  **Not shipped:**
  - held-out loss rose again: base 0.256, then 0.370 / 0.383 / 0.370 / 0.371 at 25 / 50 / 75 / 100
    iterations.
  - repeats on held-out points fell from 20.5% to 11.4% (44 samples), but that split is
    in-distribution.
  - Lesson: more conversations alone didn't stop the overfit; it happens by the first checkpoint, as in
    v0. Next: a much lower LR or far fewer iterations, and split by conversation. Two of three failure
    points yield no passing sample, so plan on weeks of traces per hundred rows.
- **opus-v1** (2026-10-04): 357 traces, 499 failure points; 239 kept, 216 train / 23 valid (176
  too_long, 78 no_pass, 6 poisoned). Sampling 5 h (~41 s per point), training 47 min. **Not shipped:**
  held-out loss base 0.324, then 0.394 / 0.424 / 0.414 / 0.409 at 25 / 50 / 75 / 100; repeats 18.5% →
  8.7% (92 samples), same in-distribution caveat.
- **The loss mask** (found 2026-10-04): v0, v1 and opus-v1 trained on one target too many per row.
  mlx_lm's `default_loss` masks `steps <= length`, which includes predicting the pad token 0 (`!`)
  after `<|im_end|>\n`. That target costs ~23 nats, so on v1's valid rows it was ~42% of the training
  loss (0.437 with it, 0.257 without). `valloss.py` never counted it, so the held-out numbers above
  stand; what was wrong was the training signal, mostly spent learning the pad. `train.py` now stops at
  `length - 1` and `test_train_loss.py` pins that. It fits every round "overfitting" from the first
  checkpoint; the next sonnet round (v1's samples via `CACHE_FROM`, split by conversation) tests it.
