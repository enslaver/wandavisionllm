# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-04

### Added

- **Vision**, the image side of WandaVision, in `Vision/`: the media hook (moved from `litellm/`;
  it still deploys to `~/.litellm/`), ComfyUI API workflows (Z-Image-Turbo generate, Flux.2 Klein
  edit, MiniMax H3 text → video with sound), Open WebUI image and video settings with a
  `generate_video` tool, and an image judge.
- Image judge: an example `[judge]` tier (SkyJM-Gen-4B, 8-bit MLX, on `mlx_vlm.server`) that
  compares two generated images against their prompt. `Vision/judge/rank.py` ranks a set of images
  pairwise with it.
- `routed = no` in `tiers.conf`: a tier served only when asked for by name (`ultron/<name>`). It
  is never a routing target, overflow or substitute, the media hook leaves it alone, and it
  doesn't count against a routed tier's fit. In enforce mode its request first waits for the tier
  it unloads to go idle, then for the memory gate (rule `direct`).
- `gpu-box/`: an optional Windows PC with an NVIDIA GPU. ComfyUI runs behind Caddy, with
  scheduled tasks, pinned custom nodes and models, an idle unload, and Unsloth Studio. It is
  installed and refreshed from a clone with `python gpu-box\deploy.py status|push`, and its HTTPS
  name is set in `GPU_BOX_HOSTNAME`.
- `lora/`, the haiku tier:
  - `haiku.yaml` for text rounds (`TIER=haiku lora/run.sh`).
  - An image-LoRA path: `unsloth_vision.py` and `unsloth_queue.sh` train Qwen3.5-4B with
    Unsloth on the GPU box (GPU lock, VRAM guard, resume, `--data` JSONL).
  - The Mac forges the merge into an mtplx pack with `recipe-4b-vision.json`; `add_mtp.py`
    restores a missing MTP head.
- `lora/fix_pretokenizer.py` checks for, and restores, the Qwen3.5 pre-tokenizer that
  transformers 5.x re-saves as Qwen2's. Without the fix, Hindi and Thai text takes 43–92 % more
  tokens; English and code are unchanged.
- PDFs and text documents in a request to a local tier are turned into text: PDFs through macOS
  PDFKit, at most 200k characters per document.
- Prompt caching for cloud overflow: `enable_anthropic_prompt_caching`, plus a system-prompt cache
  breakpoint on every `cloud/*` entry. Clients that don't mark their own breakpoints still get
  cache hits.
- Media hook: `audio_chat_prefixes` in `~/.ultron/media.json` sends transcription to models that
  take audio through chat.
- Wanda:
  - The memory lamp names the top 3 processes by memory.
  - The LoRA panel lists merges copied from the GPU box (`~/lora/merges/`) and forged mtplx packs.
- Live suite: checks that every always-loaded tier (`preload`, `ttl = 0`) is resident and that the
  loaded tiers form an allowed set.

### Changed

- Cloud is the last resort:
  - A new conversation joins its local tier's queue up to `max_waiting`, which now defaults to
    20; the example haiku tier allows 50.
  - A reload that would unload a busy tier waits up to 300 s (was 60 s) before that one request
    goes to the cloud.
  - A conversation that overflowed only because its tier was busy goes back to local once the
    tier fits. An explicit cloud pin stays.
  - Cloud no longer falls back to local, so overflow only goes one way.
- Memory guard:
  - It trips only at critical kernel pressure (`ULTRON_PRESSURE_TRIP`, default 4). Warn (2) is
    what a 64 GB Mac reads all day with three tiers resident.
  - After tripping, it holds 20 s (was 120 s) and then until headroom is back over
    `ULTRON_MEM_CLEAR_GB` (5).
- Example tiers: sonnet is always loaded (`ttl = 0`, `preload`), and the judge shares the big
  tier's slot: `resident = (fable | opus | judge) & sonnet & haiku`. For Macs under 64 GB, see
  `docs/hardware.md`.
- `/v1/messages` requests that overflow to the cloud endpoint go through LiteLLM's
  chat-completions bridge instead of `/v1/responses`. `ULTRON_CHAT_BRIDGE=off` restores the old
  path.
- One-shot local requests skip the mtplx session bank (`x-mtplx-cache-mode: bypass`), so they no
  longer fill it. These are requests with no tools and no assistant turn yet, images rerouted to the
  vision tier, and the helper's media check.
- `deploy.py`: a component can list a file kept in another folder as a `(repo path, name)` pair
  (how the media hook ships from `Vision/media/`), and `render` lays files out as on the Mac.

### Fixed

- haiku: the example script turns off mtplx's SSD session cache, which leaks memory in mtplx
  2.12.0 (fixed in 2.12.1).
- Wanda read the wrong command for a tier whose llama-swap entry has a trailing comment.
- `upstream_model = ~/…` in `tiers.conf` is expanded to the home directory.

## [0.1.0] - 2026-10-01

First public release.

### Added

- Local model tiers behind LiteLLM with Claude-compatible names and wildcards (`claude-opus-*` →
  `ultron/opus`, and so on), served by llama-swap with per-tier idle unload and a memory matrix.
  All tiers are defined in one file, `litellm/tiers.conf`; `deploy.py` builds the llama-swap and
  LiteLLM entries from it, and the hooks and Wanda read it directly.
- Four example tiers for a 64 GB Mac: **fable** (Qwen3.8-27B on TensorFold, 200k input), **opus**
  (Qwen3.6-35B-A3B MoE on mtplx, 256k), **sonnet** (Qwen3.5-9B on mtplx, vision) and **haiku**
  (Qwen3.5-4B on mtplx, always loaded, the helper tier). Fable and opus take turns; the rest stay
  resident. Example scripts for llama-server and mlx_lm.server in `mtplx/bin/examples/`.
- `loop_breaker` hook: stops agents repeating the same tool call (warn → force a different call →
  end the turn), with thresholds derived from ~430k real tool calls. The force note names the
  repeated calls.
- `ultron_admit` hook: local-first admission with cloud overflow, per-conversation pins, vision
  rerouting, eviction safety, a memory guard that overflows before macOS starts swapping, and a
  memory gate that holds a local-only request until no other tier is serving one
  (`ULTRON_MEM_WAIT_S`). It also keeps agent rounds' reasoning for Qwen templates
  (`think_in_content`) and fills in tool schemas that would skip LiteLLM's input-size check.
- `ultron_media` hook: image, image-edit, video, web search and transcription prompts answered by
  OmniRoute instead of a text tier; agent requests are confirmed by the helper tier first.
- `ultron_rescue` hook: when a local tier writes the shell command it meant to run in a trailing
  ```` ```bash ```` block instead of calling a tool, the reply becomes a real tool call.
- `ultron_stats` hook: per-request stats and a live in-flight list for every backend.
- Parallel tool calls from mtplx tiers arrive as one `tool_use` block each on `/v1/messages`;
  histories that already hold split calls are repaired.
- Wanda: realtime panel with tier cards, throughput trace, traffic flow, per-model stats, a live
  switch per hook mode, 1 h / 24 h routing, loop, media and rescue tiles, a LoRA section, logs and
  service links.
- `deploy.py`: status / push / pull / render, with an in-place-edit guard, idle waits before
  restarts (`--no-wait` to skip), and `__HOME__` / `__HOSTNAME__` templating.
- `lora/` (experimental, not deployed): fine-tune the sonnet or opus tier on its own agent failures
  from the opt-in trace tap, evaluate it, and fuse a pack. `lora/run.sh` runs a whole round.
- Live integration suite (`litellm/suite.py`: health, swap, vision, classify, confirm, image, video,
  baseline, context) and unit tests that run without a network.
- Docs: why it exists, reference hardware and benchmarks, install, configuration, architecture.
- CI (tests on Python 3.9 and 3.12, ruff, shell syntax, config rendering and validation) and a
  tag-driven release workflow.

[Unreleased]: https://github.com/enslaver/wandavisionllm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/enslaver/wandavisionllm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/enslaver/wandavisionllm/releases/tag/v0.1.0
