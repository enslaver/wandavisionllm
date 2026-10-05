# Architecture

## Processes and ports

| Process | Listens on | Started by | Role |
|---|---|---|---|
| LiteLLM | `0.0.0.0:4000` | LaunchAgent `com.litellm.proxy` | the only network-facing API; Anthropic + OpenAI formats; master key; the hooks |
| llama-swap | `127.0.0.1:8001` | LaunchAgent `com.llama-swap` | starts/stops tier servers; memory matrix; reloads its config on change |
| tier servers | `127.0.0.1:18001+` | llama-swap, via `~/.mtplx/bin/tier-*.sh` | mtplx (opus, sonnet, haiku), TensorFold (fable) and mlx_vlm.server (the image judge); no auth, loopback only |
| Wanda | `127.0.0.1:8790` | LaunchAgent `com.wanda.portal` | samples everything once a second; the panel |
| Caddy | `:443`, `:80` | `brew services` | Wanda at `/`, LiteLLM at `/llm/`, media at `/media/`, your services |
| OmniRoute | elsewhere | you | optional cloud overflow and media generation |
| ComfyUI | the GPU box (optional) | `gpu-box/deploy.py` | local image/video generation for Open WebUI ([Vision](../Vision/README.md)) |

## A request's path

```
Claude Code ──POST /v1/messages {model: claude-sonnet-5, …}──► LiteLLM :4000
  1. ultron_stats   starts the clock, lists the request as in flight
  2. loop_breaker   walks the trailing tool steps; may append a note, set tool_choice: none,
                    or answer the turn itself (mock_response) — no backend call
  3. ultron_media   newest human turn asks for an image/video/search/transcript? →
                    OmniRoute does it; the reply is a link (or extra prompt text)
  4. ultron_admit   first request of this conversation? pick ultron/<tier>, a loaded substitute,
                    or cloud/<tier>, and pin it; later requests follow the pin (except vision
                    reroutes and memory-guard overflow, one request at a time, and an overflow
                    pin, which moves back to local once the tier fits)
  ▼
LiteLLM routes data["model"]:
  ultron/sonnet ──► llama-swap :8001 ──(loads the tier if needed)──► mtplx :1800x
                    (deployment hook: history fixes, PDFs → text, one-shots skip the mtplx bank)
  cloud/sonnet  ──► OmniRoute combo  (never falls back to local: overflow is one-way)
  ▼
response streams back; ultron_rescue turns a trailing ```bash block a local tier wrote instead
of a tool call into a tool_use; ultron_stats logs usage, TTFT, tok/s; x-ultron-route says why
```

## Why two proxies

LiteLLM routes, translates APIs, checks the key and runs the hooks, but it can only route to servers
that are already running. llama-swap starts and stops processes (per-tier `ttl`, `preload`, the
matrix) but knows nothing about conversations, agents or clouds. Each does one job.

## Design decisions

- **Stable tier names.** Clients, LiteLLM and the hooks only know tier names (`fable`/`opus`/`sonnet`/
  `haiku` in the example `tiers.conf`). The model behind a tier is one line in its script.
- **The loop breaker is stateless.** Every request carries the full history, so detection re-reads
  it each time; nothing to lose on restart, nothing to sync. Warnings are appended at the end so the
  backend's prefix cache (often 100k+ tokens) survives.
- **Pins, not per-request routing.** A conversation keeps one backend for life. The two exceptions
  (vision reroute, memory overflow) change one request and leave the pin alone. A pin to the cloud
  that overflow made is soft: once the tier fits again, the conversation comes home.
- **Local first, cloud last.** Queues are deep (`max_waiting` 20) and a reload waits up to 5 minutes
  for the tier it evicts before one request spills to the cloud.
- **Shadow first.** Every hook that changes behavior has a `shadow` mode that logs what it would
  have done. Mode files are re-read per request, so rolling forward or back is one click in Wanda.
- **Fail open for stats, fail closed for media.** A stats bug never fails a request; the media hook
  only intercepts when it is sure (regex + a local-haiku YES for agent traffic).
- **Copy, don't symlink.** `deploy.py` copies files into place, so the services keep starting if
  the repo's disk (a NAS share, in the reference setup) isn't mounted.
- **Standard library only** for everything deployed (Python 3.9, macOS's system Python), so there is
  nothing to install for `deploy.py` or Wanda, and the hooks add nothing to LiteLLM's environment.

## Files on the Mac

| Path | Written by | What |
|---|---|---|
| `~/.litellm/config.yaml`, `start.sh`, `*.py` | deploy | LiteLLM config and hooks |
| `~/.litellm/env` | you | secrets and runtime settings |
| `~/.litellm/pins.sqlite` | ultron_admit | conversation → backend pins |
| `~/.litellm/stats-live.json` | ultron_stats | requests in flight (deploy.py waits on it) |
| `~/.litellm/*.jsonl` | the hooks | loop-breaker, ultron-admit, ultron-stats, media, rescue logs |
| `~/.ultron/*-mode` | Wanda / you | live switches |
| `~/.ultron/media/` | ultron_media | generated images and videos |
| `~/.llama-swap/config.yaml` | deploy | tiers and matrix |
| `~/.mtplx/bin/tier-*.sh` | deploy | tier launch scripts |
| `~/wanda/`, `~/.wanda/token`, `~/.wanda/www/` | deploy / Wanda / you | the panel, its POST token, services + icons |
| `~/.wandavision/deployed.json` | deploy | checksums of the last push (the in-place-edit guard) |
