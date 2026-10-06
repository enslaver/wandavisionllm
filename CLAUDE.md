# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repo.

## What this is

A local-first LLM stack for one Apple Silicon Mac, plus its dashboard. The repo is the source of
truth: change a file here, then `./deploy.py push` copies it into place on the Mac and runs that
component's reload step. Nothing is edited on the Mac directly.

```
LiteLLM 0.0.0.0:4000 (hooks: ultron_stats, loop_breaker, ultron_media, ultron_admit, ultron_rescue, prometheus)
  -> [bili 127.0.0.1:8787, optional billion-context compression, when ~/.ultron/bili-mode = on]
  -> llama-swap 127.0.0.1:8001 -> tiers from litellm/tiers.conf (example: fable on TensorFold; opus, sonnet, haiku on mtplx;
     judge, routed = no, on mlx_vlm.server) on 127.0.0.1:18001+
  -> cloud/* overflow: OmniRoute (optional; local -> cloud only, never back)
Wanda 127.0.0.1:8790 behind Caddy :443/:80 — the dashboard
```

"ultron" is the reference Mac's name: local model ids are `ultron/<tier>` and the hooks are
`ultron_*`. Don't rename them; clients and logs depend on the names.

## Layout — each folder mirrors a place on the Mac

| Folder | On the Mac | Notes |
|---|---|---|
| `litellm/` | `~/.litellm/` | config, `start.sh`, the hooks; tests and tools (`test_*.py`, `conftest.py`, `suite.py`, `replay.py`, `fake_omniroute.py`, `env.example`) are NOT deployed |
| `Vision/` | `~/.litellm/ultron_media.py` | `media/ultron_media.py` deploys with the `litellm` component (a `(repo path, name)` pair in `COMPONENTS`); `comfyui/` goes to the GPU box via `gpu-box/deploy.py`; tests, `judge/`, `open-webui/` are NOT deployed |
| `llama-swap/` | `~/.llama-swap/config.yaml` | reloads itself (`-watch-config`) |
| `mtplx/bin/` | `~/.mtplx/bin/tier-*.sh` | applies next time that tier loads |
| `wanda/` | `~/wanda/` | whole folder minus README and `services.example.json`; `install.sh` restarts it |
| `caddy/` | `/opt/homebrew/etc/Caddyfile` | validated, then reloaded |
| `launchd/` | `~/Library/LaunchAgents/` | LiteLLM, llama-swap and bili agents |
| `bili/` | `~/.bili/` | optional billion-context `start.sh` + `config.json` (providers from `tiers.conf`); restarted when LiteLLM is idle |
| `bin/` | `~/bin/` | `backup-stack.sh` |
| `gpu-box/` | — | NOT deployed from here: `gpu-box/deploy.py` runs on the optional Windows GPU box (ComfyUI, Caddy, Unsloth) |
| `lora/` | — | NOT deployed: optional LoRA fine-tuning scripts (mlx-lm on the Mac; Unsloth on the GPU box for haiku images); artifacts in `~/lora/` |
| `docs/` | — | why, hardware, install, configuration, architecture |
| `scripts/` | — | `check_repo.py` (CI) |

## Conventions

- **Python 3.9, standard library only** in deployed files (`deploy.py`, `wanda/server.py`, the hooks;
  hooks may import `litellm`/`yaml`). `make check` enforces it.
- **No machine-specific values.** Config files use `__HOME__` / `__HOSTNAME__` (filled by
  `deploy.py`); scripts use `$HOME`; endpoints and keys come from `~/.litellm/env`.
- **Secrets never live in the repo.** Configs reference them via `os.environ/...`.
- **Runtime state is not managed:** logs, `pins.sqlite`, `stats-live.json`, `~/.ultron/*-mode`.
- **Files are copied, not symlinked**, so services start without the repo.
- **To add a deployed file**, put it in the matching folder AND add it to `COMPONENTS` in `deploy.py`.
- **Hooks never fail a request**: errors log and pass through; behavior changes ship with `shadow` mode.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## Commands

```bash
make test                          # unit tests (litellm/, Vision/media/, gpu-box/)
make check                         # render templates + config/stdlib/home-path checks
make lint                          # ruff + py3.9 compile
./deploy.py status | push [--dry-run] [component ...] | pull | render DIR
cd litellm && uvx --with pyyaml python3 suite.py     # live suite; needs the stack up
```

After swapping models or changing routing/vision settings, run the live suite.

## Wanda

`wanda/server.py` samples the stack every second and serves `/`, `/api/*`, `/icons/*`;
`wanda/static/index.html` is the whole page (no build, no external assets). API: `GET /api/status`,
`/api/stream` (SSE), `/api/history`, `/api/log?name=swap|litellm|caddy|modes|omniroute|loops|admit|media|rescue|stats`; `POST /api/mode`, `/api/flag`, `/api/tier`
(POSTs need `X-Wanda-Token`). Add a live switch in `MODES`, an on/off flag in `FLAGS`.

## Docs

`README.md` (overview), `docs/*.md`, `litellm/README.md` (hooks), `wanda/README.md` (panel).
