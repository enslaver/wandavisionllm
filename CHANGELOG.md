# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/enslaver/wandavisionllm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/enslaver/wandavisionllm/releases/tag/v0.1.0
